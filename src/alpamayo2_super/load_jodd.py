# SPDX-License-Identifier: Apache-2.0
"""Adapt one Japan Open Driving Dataset scene for inference, without nuScenes-devkit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from alpamayo2_super.common.constants import CAMERA_INDICES_TO_NAMES


DATASET_ID = "turing-motors/Japan-Open-Driving-Dataset-Sample"
DATASET_REVISION = "2c08f9d"
# Approximate view correspondence, not a claim of identical camera rigs/FOVs.
CAMERA_MAPPING = {
    0: "CAM_FRONT_LEFT",
    1: "CAM_FRONT_WIDE",
    2: "CAM_FRONT_RIGHT",
    3: "CAM_BACK_LEFT",
    5: "CAM_BACK_RIGHT",
    6: "CAM_FRONT",
}
MAX_FRAME_AGE_US = 150_000
MAX_POSE_GAP_US = 200_000


class DatasetFiles:
    """Read an existing local dataset, or fetch only requested files into the HF cache."""

    def __init__(self, dataset_dir: str | Path | None = None):
        self.root = None if dataset_dir is None else Path(dataset_dir).expanduser().resolve()

    def path(self, filename: str) -> Path:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid dataset-relative path: {filename}")
        if self.root is not None:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Dataset path escapes its root: {filename}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing dataset file: {path}")
            return path
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=DATASET_ID, repo_type="dataset", filename=filename,
                revision=DATASET_REVISION,
            )
        )

    def table(self, name: str) -> list[dict[str, Any]]:
        return json.loads(self.path(f"v2.X-train/{name}.json").read_text(encoding="utf-8"))


@dataclass
class _Pose:
    rotation: Rotation
    translation: np.ndarray

    def inv(self) -> _Pose:
        inverse = self.rotation.inv()
        return _Pose(inverse, inverse.apply(-self.translation))

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        return self.rotation.apply(xyz) + self.translation


@dataclass
class _PinholeCamera:
    intrinsic: np.ndarray

    def ray2pixel(self, rays: np.ndarray) -> np.ndarray:
        homogeneous = np.asarray(rays) @ self.intrinsic.T
        return homogeneous[..., :2] / homogeneous[..., 2:3]


def interpolate_poses(
    poses: list[dict[str, Any]], timestamps_us: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    """Interpolate nuScenes wxyz poses, rejecting extrapolation and missing intervals."""
    by_time = {int(pose["timestamp"]): pose for pose in poses}
    times = np.array(sorted(by_time), dtype=np.int64)
    queries = np.asarray(timestamps_us, dtype=np.int64)
    if len(times) < 2 or queries.min() < times[0] or queries.max() > times[-1]:
        raise ValueError("Requested history/future lies outside this scene's ego-pose coverage")
    right = np.searchsorted(times, queries, side="left").clip(1, len(times) - 1)
    gaps = times[right] - times[right - 1]
    if np.any((gaps > MAX_POSE_GAP_US) & (times[right] != queries)):
        raise ValueError("Ego poses have a gap larger than 200 ms around a requested timestamp")
    records = [by_time[int(t)] for t in times]
    xyz = np.array([p["translation"] for p in records], dtype=np.float64)
    quaternions = np.array([p["rotation"] for p in records], dtype=np.float64)
    if not np.isfinite(xyz).all() or not np.isfinite(quaternions).all():
        raise ValueError("Ego poses contain non-finite values")
    # Subtract the epoch before converting to seconds to retain sub-frame precision.
    seconds = (times - times[0]) * 1e-6
    query_seconds = (queries - times[0]) * 1e-6
    positions = np.column_stack([
        np.interp(query_seconds, seconds, xyz[:, axis]) for axis in range(3)
    ])
    rotations = Slerp(seconds, Rotation.from_quat(quaternions, scalar_first=True))(query_seconds)
    return positions, rotations


def select_past_frames(
    frames: list[dict[str, Any]], targets_us: np.ndarray,
) -> list[dict[str, Any]]:
    """Use causal camera frames only; do not borrow a post-t0 image."""
    ordered = sorted(frames, key=lambda frame: frame["timestamp"])
    times = np.array([frame["timestamp"] for frame in ordered], dtype=np.int64)
    indices = np.searchsorted(times, targets_us, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Not enough camera history at the requested time")
    if np.any(targets_us - times[indices] > MAX_FRAME_AGE_US):
        raise ValueError("A requested camera frame is missing or more than 150 ms old")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Four distinct historical frames are required for each camera")
    return [ordered[int(i)] for i in indices]


def prepare_jodd_sample(
    scene_name: str = "scene-0668", t0_s: float = 5.1,
    dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return NumPy inputs and a held-out future for plotting/evaluation only.

    The downloaded files are six metadata tables plus 24 selected JPEGs. No captions,
    LiDAR, object labels, CAN logs, or precomputed future trajectories are needed.
    """
    if not np.isfinite(t0_s) or t0_s < 1.5:
        raise ValueError("--t0 must be finite and leave at least 1.5 s of history")
    files = DatasetFiles(dataset_dir)
    scenes = files.table("scene")
    scene = next((s for s in scenes if s["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"Unknown scene {scene_name!r}; available: {[s['name'] for s in scenes]}")
    samples = sorted(
        (s for s in files.table("sample") if s["scene_token"] == scene["token"]),
        key=lambda sample: sample["timestamp"],
    )
    sample_tokens = {s["token"] for s in samples}
    scene_frames = [f for f in files.table("sample_data") if f["sample_token"] in sample_tokens]
    sensors = {s["token"]: s for s in files.table("sensor")}
    calibrations = {c["token"]: c for c in files.table("calibrated_sensor")}
    pose_tokens = {f["ego_pose_token"] for f in scene_frames}
    poses = [p for p in files.table("ego_pose") if p["token"] in pose_tokens]
    pose_by_token = {p["token"]: p for p in poses}
    start_us = int(samples[0]["timestamp"])
    t0_us = start_us + int(round(t0_s * 1_000_000))
    history_times = t0_us + np.arange(-15, 1, dtype=np.int64) * 100_000
    future_times = t0_us + np.arange(1, 65, dtype=np.int64) * 100_000
    all_xyz, all_rotations = interpolate_poses(poses, np.r_[history_times, future_times])
    t0_xyz, t0_rotation = all_xyz[15], all_rotations[15]
    inverse = t0_rotation.inv()
    local_xyz = inverse.apply(all_xyz - t0_xyz).astype(np.float32)
    local_rot = (inverse * all_rotations).as_matrix().astype(np.float32)
    image_targets = t0_us + np.arange(-3, 1, dtype=np.int64) * 100_000
    images, camera_timestamps, image_paths, camera_calibrations = [], [], [], {}

    for camera_id, channel in CAMERA_MAPPING.items():
        frames = [
            f for f in scene_frames
            if sensors[calibrations[f["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            == channel
        ]
        if not frames:
            raise ValueError(f"Scene {scene_name} is missing camera {channel}")
        chosen = select_past_frames(frames, image_targets)
        loaded = []
        for frame in chosen:
            with Image.open(files.path(frame["filename"])) as image:
                loaded.append(np.asarray(image.convert("RGB")))
        images.append(np.stack(loaded).transpose(0, 3, 1, 2))
        camera_timestamps.append([int(f["timestamp"]) - start_us for f in chosen])
        image_paths.append([f["filename"] for f in chosen])

        # Express the last image's camera in the t0 ego frame, including capture-time motion.
        last = chosen[-1]
        calibration = calibrations[last["calibrated_sensor_token"]]
        if any(abs(float(v)) > 1e-12 for v in calibration.get("distortion", {}).values()):
            raise ValueError(f"{channel} has distortion; this adapter expects rectified images")
        capture_pose = pose_by_token[last["ego_pose_token"]]
        capture_rotation = Rotation.from_quat(capture_pose["rotation"], scalar_first=True)
        camera_rotation = Rotation.from_quat(calibration["rotation"], scalar_first=True)
        camera_world = (
            capture_rotation.apply(calibration["translation"]) + capture_pose["translation"]
        )
        camera_calibrations[camera_id] = {
            "camera_name": channel,
            "camera_model": _PinholeCamera(np.asarray(calibration["camera_intrinsic"])),
            "sensor_pose": _Pose(
                inverse * capture_rotation * camera_rotation,
                inverse.apply(camera_world - t0_xyz),
            ),
        }

    timestamps = np.asarray(camera_timestamps, dtype=np.int64)
    camera_tmin = int(timestamps.min())
    camera_ids = list(CAMERA_MAPPING)
    return {
        "image_frames": np.stack(images),
        "camera_indices": np.array(camera_ids, dtype=np.int64),
        "camera_names": [CAMERA_INDICES_TO_NAMES[i] for i in camera_ids],
        "ego_history_xyz": local_xyz[:16][None, None],
        "ego_history_rot": local_rot[:16][None, None],
        "ego_future_xyz": local_xyz[16:][None, None],
        "ego_future_rot": local_rot[16:][None, None],
        "absolute_timestamps": timestamps,
        "relative_timestamps": (timestamps - camera_tmin).astype(np.float32) * 1e-6,
        "camera_tmin": camera_tmin,
        "clip_id": scene_name,
        "t0_us": t0_us - start_us,
        "camera_calibrations": camera_calibrations,
        "input_profile": {
            "source": DATASET_ID,
            "camera_ids": camera_ids,
            "source_camera_names": list(CAMERA_MAPPING.values()),
            "frame_indices": [0, 1, 2, 3],
            "camera_mapping_is_approximate": True,
        },
        "jodd_metadata": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION if dataset_dir is None else "local",
            "scene_name": scene_name,
            "scene_token": scene["token"],
            "t0_epoch_us": t0_us,
            "camera_mapping": CAMERA_MAPPING,
            "image_paths": image_paths,
            "image_timestamps_relative_to_t0_s": (
                (timestamps - (t0_us - start_us)) * 1e-6
            ).tolist(),
            "history_times_s": (np.arange(-15, 1) * 0.1).tolist(),
            "future_times_s": (np.arange(1, 65) * 0.1).tolist(),
            "coordinate_frame": "ego at t0: x forward, y left, z up; positions in meters",
            "future_and_captions_used_as_model_input": False,
        },
    }


def as_torch_sample(data: dict[str, Any]) -> dict[str, Any]:
    """Convert only arrays to tensors; keep projection objects on the CPU."""
    import torch

    return {
        key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
