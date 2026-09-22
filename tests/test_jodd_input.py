# SPDX-License-Identifier: Apache-2.0
"""Check the coordinate/time boundaries of the Japanese-data inference adapter."""

import json

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from alpamayo2_super.load_jodd import (
    CAMERA_MAPPING,
    DatasetFiles,
    interpolate_poses,
    prepare_jodd_sample,
    select_past_frames,
)


@pytest.fixture
def dataset(tmp_path):
    epoch = 1_740_000_000_000_000
    # Travel north in the world at 2 m/s, facing north. In ego coordinates this is +x.
    yaw = Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True).tolist()
    camera_rotation = Rotation.from_matrix(
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    ).as_quat(scalar_first=True).tolist()
    tables = {name: [] for name in ("scene", "sample", "sample_data", "sensor",
                                     "calibrated_sensor", "ego_pose")}
    tables["scene"] = [{"token": "scene", "name": "scene-test"}]
    for camera_id, channel in CAMERA_MAPPING.items():
        tables["sensor"].append({"token": channel, "channel": channel})
        tables["calibrated_sensor"].append({
            "token": channel, "sensor_token": channel,
            "translation": [1, 0, 1.5], "rotation": camera_rotation,
            "camera_intrinsic": [[40, 0, 32], [0, 40, 20], [0, 0, 1]],
            "distortion": {"k1": 0},
        })
        Image.new("RGB", (64, 40), color=(camera_id * 30, 20, 30)).save(tmp_path / f"{channel}.jpg")
    for i in range(201):
        timestamp = epoch + i * 100_000
        tables["sample"].append({
            "token": f"sample-{i}", "scene_token": "scene", "timestamp": timestamp,
        })
        tables["ego_pose"].append({
            "token": f"pose-{i}", "timestamp": timestamp,
            "translation": [1000, 2000 + i * 0.2, 50], "rotation": yaw,
        })
        for channel in CAMERA_MAPPING.values():
            tables["sample_data"].append({
                "sample_token": f"sample-{i}", "ego_pose_token": f"pose-{i}",
                "calibrated_sensor_token": channel, "timestamp": timestamp,
                "filename": f"{channel}.jpg",
            })
    table_dir = tmp_path / "v2.X-train"
    table_dir.mkdir()
    for name, rows in tables.items():
        (table_dir / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    return tmp_path


def test_pose_and_camera_conversion(dataset):
    data = prepare_jodd_sample("scene-test", 5.1, dataset)
    assert data["image_frames"].shape == (6, 4, 3, 40, 64)
    assert data["camera_indices"].tolist() == [0, 1, 2, 3, 5, 6]
    np.testing.assert_allclose(
        data["ego_history_xyz"][0, 0, :, 0], np.arange(-15, 1) * 0.2, atol=1e-5
    )
    np.testing.assert_allclose(
        data["ego_future_xyz"][0, 0, :, 0], np.arange(1, 65) * 0.2, atol=1e-5
    )
    np.testing.assert_allclose(data["ego_future_xyz"][..., 1:], 0, atol=1e-5)
    np.testing.assert_allclose(data["ego_history_rot"][0, 0, -1], np.eye(3), atol=1e-5)
    assert np.all(data["absolute_timestamps"] <= data["t0_us"])
    calibration = data["camera_calibrations"][1]
    ray = calibration["sensor_pose"].inv().apply(np.array([[11, 0, 1.5]]))
    np.testing.assert_allclose(calibration["camera_model"].ray2pixel(ray), [[32, 20]], atol=1e-5)


def test_future_labels_do_not_change_history(dataset):
    before = prepare_jodd_sample("scene-test", 5.1, dataset)
    path = dataset / "v2.X-train/ego_pose.json"
    poses = json.loads(path.read_text())
    for pose in poses[52:]:
        pose["translation"][0] += 20
    path.write_text(json.dumps(poses))
    after = prepare_jodd_sample("scene-test", 5.1, dataset)
    np.testing.assert_array_equal(before["image_frames"], after["image_frames"])
    np.testing.assert_array_equal(before["ego_history_xyz"], after["ego_history_xyz"])
    np.testing.assert_array_equal(before["ego_history_rot"], after["ego_history_rot"])
    assert not np.array_equal(before["ego_future_xyz"], after["ego_future_xyz"])


def test_camera_frames_never_come_from_the_future():
    frames = [{"timestamp": i * 100_000 + 1000} for i in range(8)]
    selected = select_past_frames(frames, np.arange(3, 7) * 100_000)
    assert [f["timestamp"] for f in selected] == [201000, 301000, 401000, 501000]
    with pytest.raises(ValueError, match="missing|distinct"):
        select_past_frames(frames[:1], np.arange(3, 7) * 100_000)


def test_pose_interpolation_handles_quaternion_sign_and_large_epochs():
    q = Rotation.from_euler("z", 45, degrees=True).as_quat(scalar_first=True)
    epoch = 1_740_000_000_000_000
    poses = [
        {"timestamp": epoch, "translation": [0, 0, 0], "rotation": q.tolist()},
        {"timestamp": epoch + 100000, "translation": [1, 0, 0], "rotation": (-q).tolist()},
    ]
    xyz, rotation = interpolate_poses(poses, np.array([epoch + 50000]))
    np.testing.assert_allclose(xyz, [[0.5, 0, 0]])
    np.testing.assert_allclose(
        rotation.as_matrix()[0], Rotation.from_quat(q, scalar_first=True).as_matrix()
    )
    with pytest.raises(ValueError, match="outside"):
        interpolate_poses(poses, np.array([epoch - 1]))
    poses[1]["timestamp"] += 500000
    with pytest.raises(ValueError, match="gap"):
        interpolate_poses(poses, np.array([epoch + 50000]))


def test_incomplete_scene_fails_before_model_load(dataset):
    with pytest.raises(ValueError, match="outside"):
        prepare_jodd_sample("scene-test", 15, dataset)
    with pytest.raises(ValueError, match="Unknown scene"):
        prepare_jodd_sample("missing", 5.1, dataset)


def test_dataset_files_reject_paths_outside_root(dataset):
    with pytest.raises(ValueError, match="Invalid"):
        DatasetFiles(dataset).path("../outside.json")
