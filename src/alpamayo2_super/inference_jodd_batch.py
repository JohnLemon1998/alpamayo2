# SPDX-License-Identifier: Apache-2.0
"""Export one inference video per JODD scene, with a single shared model and resumable jobs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
from alpamayo2_super.inference_jodd_video import (
    inference_times,
    render_scene_frames,
    write_video_results,
)
from alpamayo2_super.load_jodd import DatasetFiles, prepare_jodd_sample


def scene_times(files, scene, start, end, step):
    """Restrict each scene to the recorded pose coverage needed by the existing adapter."""
    samples = [s for s in files.table("sample") if s["scene_token"] == scene["token"]]
    if not samples:
        raise ValueError("Scene has no samples")
    origin = min(s["timestamp"] for s in samples)
    tokens = {s["token"] for s in samples}
    pose_tokens = {
        f["ego_pose_token"] for f in files.table("sample_data") if f["sample_token"] in tokens
    }
    pose_times = [p["timestamp"] for p in files.table("ego_pose") if p["token"] in pose_tokens]
    if len(pose_times) < 2:
        raise ValueError("Scene has insufficient ego poses")
    step_us = round(step * 1_000_000)
    start_us = round(start * 1_000_000)
    earliest_us = max(1_500_000, min(pose_times) - origin + 1_500_000)
    # Keep the user's timeline grid when a scene starts with incomplete pose coverage.
    first_us = start_us + max(0, (earliest_us - start_us + step_us - 1) // step_us) * step_us
    # The final valid instant is inclusive; inference_times uses an exclusive end.
    stop_us = max(pose_times) - origin - 6_400_000 + 1
    if end is not None:
        stop_us = min(stop_us, round(end * 1_000_000))
    return inference_times(first_us / 1_000_000, stop_us / 1_000_000, step)


def _write_json(path, value):
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _dataset_signature(files):
    """Invalidate completion markers when the local metadata or dataset location changes."""
    digest = hashlib.sha256()
    for name in ("scene", "sample", "sample_data", "sensor", "calibrated_sensor", "ego_pose"):
        with files.path(f"v2.X-train/{name}.json").open("rb") as source:
            digest.update(hashlib.file_digest(source, "sha256").digest())
    return {"root": str(files.root), "metadata_sha256": digest.hexdigest()}


def _is_complete(prefix, job):
    try:
        record = json.loads(Path(f"{prefix}.complete.json").read_text(encoding="utf-8"))
        return (
            record["job"] == job and record["frame_count"] == len(job["times_s"])
            and all(
                Path(f"{prefix}.{suffix}").stat().st_size == record["file_sizes"][suffix] > 0
                for suffix in ("mp4", "jsonl")
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _load_model(model_id):
    import av
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires CUDA. Use --prepare-only to check inputs on a CPU.")
    av.Codec("libx264", "w")
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    return model


def run_batch(
    *, dataset_dir, output_dir, scenes=None, start=2.0, end=None, step=0.5,
    model_id=PUBLIC_MODEL_ID, diffusion_steps=10, seed=42, overwrite=False, prepare_only=False,
):
    """Process scenes sequentially. Failures are recorded and retried on the next invocation."""
    inference_times(start, start + 1 if end is None else end, step)
    if diffusion_steps < 1:
        raise ValueError("diffusion-steps must be positive")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MPLBACKEND", "Agg")
    files = DatasetFiles(dataset_dir)
    available = files.table("scene")
    if scenes is not None:
        missing = set(scenes) - {s["name"] for s in available}
        if missing:
            raise ValueError(f"Unknown scenes: {sorted(missing)}")
        available = [s for s in available if s["name"] in scenes]
    if not available:
        raise ValueError("No scenes selected")
    # Metadata names become output filenames; prevent directory traversal or collisions.
    names = [s["name"] for s in available]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"scene-[\w-]+", n) for n in names):
        raise ValueError("Scene names must be unique safe names beginning with 'scene-'")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _dataset_signature(files)
    summary = {"dataset": signature, "prepare_only": prepare_only, "scenes": []}
    summary_path = output_dir / ("prepare_summary.json" if prepare_only else "batch_summary.json")
    model = None
    print(f"Selected {len(available)} scenes; inference interval {step:g}s.", flush=True)

    for index, scene in enumerate(available):
        name = scene["name"]
        prefix = output_dir / name
        entry = {"scene": name, "status": "running"}
        summary["scenes"].append(entry)
        _write_json(summary_path, summary)
        started = time.perf_counter()
        print(f"\nScene [{index + 1}/{len(available)}]: {name}", flush=True)
        first = None
        frames = None
        try:
            times = scene_times(files, scene, start, end, step)
            job = {
                "format_version": 1, "dataset": signature, "scene_token": scene["token"],
                "times_s": times, "step_s": step, "model_id": model_id,
                "diffusion_steps": diffusion_steps, "seed": seed,
            }
            entry.update({
                "frame_count": len(times), "first_t0_s": times[0], "last_t0_s": times[-1],
            })
            if not prepare_only and not overwrite and _is_complete(prefix, job):
                entry["status"] = "skipped_complete"
                print("Already complete with these settings; skipping.", flush=True)
                continue
            if not prepare_only:
                # A failed regeneration must not retain an old completion marker.
                Path(f"{prefix}.complete.json").unlink(missing_ok=True)
            # Validate boundaries before loading the model; subsequent images stay streaming.
            first = prepare_jodd_sample(name, times[0], files=files)
            if len(times) > 1:
                prepare_jodd_sample(name, times[-1], files=files)
            if prepare_only:
                for t0 in times[1:]:
                    prepare_jodd_sample(name, t0, files=files)
                _write_json(Path(f"{prefix}.input.json"), {"job": job, "status": "inputs_checked"})
                entry["status"] = "inputs_checked"
                continue
            if model is None:
                # A global loading failure must not retry the same costly load once per scene.
                try:
                    model = _load_model(model_id)
                except Exception as error:
                    entry.update({"status": "model_load_failed", "error": str(error)})
                    raise
            from alpamayo2_super.load_jodd import as_torch_sample

            temporary_prefix = output_dir / f"{name}.partial"
            frames = render_scene_frames(
                model, files, name, times, model_id=model_id, diffusion_steps=diffusion_steps,
                seed=seed, first_sample=as_torch_sample(first),
            )
            first = None
            count = write_video_results(temporary_prefix, frames, step)
            if count != len(times):
                raise RuntimeError(f"Expected {len(times)} video frames, wrote {count}")
            # Only mark complete after both files have been finalized and promoted.
            marker = Path(f"{prefix}.complete.json")
            marker.unlink(missing_ok=True)
            for suffix in ("mp4", "jsonl"):
                Path(f"{temporary_prefix}.{suffix}").replace(Path(f"{prefix}.{suffix}"))
            _write_json(marker, {
                "job": job, "frame_count": count,
                "file_sizes": {s: Path(f"{prefix}.{s}").stat().st_size for s in ("mp4", "jsonl")},
            })
            entry.update({"status": "complete", "video": str(Path(f"{prefix}.mp4"))})
            print(f"Saved {prefix}.mp4", flush=True)
        except Exception as error:
            if entry["status"] == "model_load_failed":
                raise
            entry.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {name}: {error}; continuing to the next scene.", flush=True)
        except KeyboardInterrupt:
            entry["status"] = "interrupted"
            raise
        finally:
            if frames is not None:
                frames.close()
            first = None
            entry["elapsed_s"] = round(time.perf_counter() - started, 3)
            _write_json(summary_path, summary)
        if entry["status"] == "failed":
            gc.collect()
            if model is not None:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(f"Batch finished. Summary: {summary_path}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/japan_scenes"))
    parser.add_argument("--scenes", nargs="+", help="Optional subset; defaults to all scenes")
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--end", type=float, help="Exclusive end; default: each scene's valid end")
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument(
        "--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID)
    )
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate completed scenes")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Check inputs without inference"
    )
    args = parser.parse_args()
    try:
        summary = run_batch(**vars(args))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if any(entry["status"] == "failed" for entry in summary["scenes"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
