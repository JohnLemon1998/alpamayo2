# SPDX-License-Identifier: Apache-2.0
"""Run pretrained Alpamayo 2 Super on one Japanese driving sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID


def infer_and_plot(model, data, *, model_id: str, diffusion_steps: int, seed: int):
    """Infer one instant and return a CPU figure/record, releasing temporary GPU inputs."""
    import numpy as np
    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.load_jodd import CAMERA_MAPPING
    from alpamayo2_super.visualization import plot_inference_result

    # Future poses are held out; helper only passes images and historical ego poses.
    model_inputs = helper.to_device(
        helper.prepare_model_inputs(data, model.config, model.tokenizer), "cuda",
    )
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, _, extra = model.sample_trajectories_from_data(
            data=model_inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
            diffusion_kwargs={"inference_step": diffusion_steps}, return_extra=True,
        )
    pred_xyz = pred_xyz.detach().cpu().float()
    pred_rot = pred_rot.detach().cpu().float()
    del model_inputs
    figure, result = plot_inference_result(
        data=data, pred_xyz=pred_xyz, extra=extra, model_id=model_id, seed=seed,
    )
    source_titles = [CAMERA_MAPPING[i] for i in result["camera_grid_camera_ids"]]
    for axis, source_name in zip(figure.axes[:6], source_titles):
        axis.set_title(source_name, fontsize=15)
    result.update({
        "figure_style": "jodd_6cam",
        "camera_titles": source_titles,
        "jodd": data["jodd_metadata"],
        "pred_xyz": pred_xyz.numpy().tolist(),
        "pred_rot": pred_rot.numpy().tolist(),
        "ego_future_xyz": data["ego_future_xyz"].numpy().tolist(),
        "ego_history_xyz": data["ego_history_xyz"].numpy().tolist(),
        "ade_xy_m": float(np.linalg.norm(
            pred_xyz.numpy()[0, 0, 0, :, :2] - data["ego_future_xyz"].numpy()[0, 0, :, :2],
            axis=-1,
        ).mean()),
    })
    return figure, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene-0668")
    parser.add_argument(
        "--t0", type=float, default=5.1, help="Seconds from the first scene keyframe"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, help="Existing JODD root; otherwise fetch selected files"
    )
    parser.add_argument(
        "--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID)
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("outputs/japan_sample0"))
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Validate inputs without loading model weights"
    )
    args = parser.parse_args()
    if args.diffusion_steps < 1:
        parser.error("--diffusion-steps must be positive")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import torch

    from alpamayo2_super.load_jodd import as_torch_sample, prepare_jodd_sample

    if not args.prepare_only and not torch.cuda.is_available():
        parser.error("Inference requires CUDA. Use --prepare-only to check the input on a CPU.")
    print(f"Preparing {args.scene} at {args.t0:.3f}s (metadata + 24 camera frames)...", flush=True)
    try:
        data = as_torch_sample(prepare_jodd_sample(args.scene, args.t0, args.dataset_dir))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = data["jodd_metadata"]
    input_path = Path(f"{prefix}.input.json")
    input_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Images:", tuple(data["image_frames"].shape))
    print("Ego history:", tuple(data["ego_history_xyz"].shape))
    print("Camera views are matched by direction; JODD and NVIDIA rigs have different FOVs.")
    print("Saved input metadata:", input_path)
    if args.prepare_only:
        print("Input preparation passed. No model inference was performed.")
        return

    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    import matplotlib.pyplot as plt

    png_path, json_path = Path(f"{prefix}.png"), Path(f"{prefix}.json")
    figure, result = infer_and_plot(
        model, data, model_id=args.model_id, diffusion_steps=args.diffusion_steps, seed=args.seed,
    )
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Chain-of-Causation:", result["cot"])
    print(f"ADE (XY, one sampled trajectory): {result['ade_xy_m']:.4f} meters")
    print("Saved visualization:", png_path)
    print("Saved predictions:", json_path)


if __name__ == "__main__":
    main()
