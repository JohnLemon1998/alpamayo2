# SPDX-License-Identifier: Apache-2.0
"""Run Alpamayo on successive recorded JODD instants and stream the results to MP4."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import time

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID


def inference_times(start: float, end: float, step: float) -> list[float]:
    """Build a uniform, end-exclusive timeline using integer microseconds."""
    if not all(math.isfinite(value) for value in (start, end, step)):
        raise ValueError("start, end and step must be finite")
    if start < 1.5 or end <= start:
        raise ValueError("start must be at least 1.5 s, and end must be greater than start")
    if step < 0.1:
        raise ValueError("step must be at least 0.1 s (the dataset is approximately 10 Hz)")
    times = [
        value / 1_000_000
        for value in range(round(start * 1_000_000), round(end * 1_000_000),
                           round(step * 1_000_000))
    ]
    if not times:
        raise ValueError("Requested interval is shorter than the timestamp precision")
    return times


def write_video_results(prefix: Path, frames, step: float) -> int:
    """Encode frames incrementally; keep one JSONL prediction record per video frame."""
    import av

    fps = Fraction(1_000_000, round(step * 1_000_000))
    count = 0
    stream = None
    with (
        Path(f"{prefix}.jsonl").open("w", encoding="utf-8") as records,
        av.open(str(Path(f"{prefix}.mp4")), mode="w") as container,
    ):
        try:
            for image, result in frames:
                height, width = image.shape[:2]
                if image.ndim != 3 or image.shape[2] != 3 or image.dtype.name != "uint8":
                    raise ValueError("Video frames must be uint8 RGB arrays")
                if height % 2 or width % 2:
                    raise ValueError("H.264 video dimensions must be even")
                if stream is None:
                    stream = container.add_stream("libx264", rate=fps)
                    stream.width, stream.height = width, height
                    stream.pix_fmt = "yuv420p"
                    stream.options = {"crf": "18", "preset": "fast"}
                elif (width, height) != (stream.width, stream.height):
                    raise ValueError("Video frame dimensions changed during inference")
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = count
                frame.time_base = 1 / fps
                for packet in stream.encode(frame):
                    container.mux(packet)
                record = {
                    **result, "video_frame_index": count,
                    "video_time_s": float(count / fps), "video_fps": float(fps),
                }
                records.write(json.dumps(record) + "\n")
                records.flush()
                count += 1
        finally:
            # Finish already encoded frames even if a later sample/inference fails.
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
    if count == 0:
        raise ValueError("No video frames were produced")
    return count


def render_scene_frames(
    model, files, scene_name, times, *, model_id, diffusion_steps, seed, first_sample=None,
):
    """Render successive predictions with a model that can be reused across scenes."""
    import matplotlib.pyplot as plt
    import numpy as np

    from alpamayo2_super.inference_jodd import infer_and_plot
    from alpamayo2_super.load_jodd import as_torch_sample, prepare_jodd_sample

    for index, t0 in enumerate(times):
        started = time.perf_counter()
        data = first_sample if index == 0 and first_sample is not None else as_torch_sample(
            prepare_jodd_sample(scene_name, t0, files=files)
        )
        first_sample = None
        figure, result = infer_and_plot(
            model, data, model_id=model_id, diffusion_steps=diffusion_steps, seed=seed,
        )
        try:
            figure.set_dpi(100)
            figure.suptitle(
                f"{scene_name} | t = {t0:.2f} s | Recorded-drive replay", fontsize=16,
            )
            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        finally:
            plt.close(figure)
        result.update({"t0_s": t0, "replay_mode": "recorded_inputs"})
        elapsed = time.perf_counter() - started
        print(
            f"{scene_name} [{index + 1}/{len(times)}] t={t0:.2f}s, "
            f"ADE={result['ade_xy_m']:.3f}m, processing={elapsed:.1f}s", flush=True,
        )
        del data
        yield image, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene-0668")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--start", type=float, default=2.0, help="Seconds from scene start")
    parser.add_argument("--end", type=float, default=12.0, help="End time in seconds, exclusive")
    parser.add_argument("--step", type=float, default=0.5, help="Seconds between inferences")
    parser.add_argument("--output-prefix", type=Path, default=Path("outputs/japan_video"))
    parser.add_argument(
        "--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID)
    )
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Check inputs without inference"
    )
    args = parser.parse_args()
    try:
        times = inference_times(args.start, args.end, args.step)
    except ValueError as error:
        parser.error(str(error))
    if args.diffusion_steps < 1:
        parser.error("diffusion-steps must be positive")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import torch

    from alpamayo2_super.load_jodd import DatasetFiles, as_torch_sample, prepare_jodd_sample

    if not args.prepare_only and not torch.cuda.is_available():
        parser.error("Inference requires CUDA. Use --prepare-only to check inputs on a CPU.")
    files = DatasetFiles(args.dataset_dir)

    def prepare(t0):
        try:
            return as_torch_sample(prepare_jodd_sample(args.scene, t0, files=files))
        except (ValueError, FileNotFoundError) as error:
            parser.error(f"At {t0:.3f}s: {error}")

    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scene {args.scene}: {len(times)} inferences, step={args.step:g}s", flush=True)
    print("Offline replay: predictions use recorded images and poses at each instant.")
    # Check both temporal boundaries before paying the model loading cost.
    first = prepare(times[0])
    if len(times) > 1:
        prepare(times[-1])
    if args.prepare_only:
        metadata = []
        for index, t0 in enumerate(times):
            data = first if index == 0 else prepare(t0)
            metadata.append({"t0_s": t0, **data["jodd_metadata"]})
        input_path = Path(f"{prefix}.input.json")
        input_path.write_text(json.dumps({"frames": metadata}, indent=2) + "\n", encoding="utf-8")
        print("All video inputs passed. No model inference was performed.")
        print("Saved input metadata:", input_path)
        return

    import av

    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    av.Codec("libx264", "w")  # Check encoder availability before loading weights.
    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    frames = render_scene_frames(
        model, files, args.scene, times, model_id=args.model_id,
        diffusion_steps=args.diffusion_steps, seed=args.seed, first_sample=first,
    )
    del first
    count = write_video_results(prefix, frames, args.step)
    print(f"Saved video: {prefix}.mp4 ({count} frames, {1 / args.step:g} fps)")
    print(f"Saved predictions and source metadata: {prefix}.jsonl")


if __name__ == "__main__":
    main()
