# SPDX-License-Identifier: Apache-2.0
"""Check the coordinate/time boundaries of the Japanese-data inference adapter."""

import json
from pathlib import Path

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


def test_video_prepare_only_checks_successive_instants_without_cuda(dataset, tmp_path, monkeypatch):
    import sys

    from alpamayo2_super.inference_jodd_video import main

    prefix = tmp_path / "video-check"
    monkeypatch.setattr(sys, "argv", [
        "inference_jodd_video", "--dataset-dir", str(dataset), "--scene", "scene-test",
        "--start", "2", "--end", "3.5", "--step", "0.5", "--prepare-only",
        "--output-prefix", str(prefix),
    ])
    main()
    frames = json.loads(Path(f"{prefix}.input.json").read_text())["frames"]
    assert [frame["t0_s"] for frame in frames] == [2, 2.5, 3]
    assert [frame["t0_epoch_us"] for frame in frames] == [
        1_740_000_002_000_000, 1_740_000_002_500_000, 1_740_000_003_000_000,
    ]
    assert all(frame["dataset_revision"] == "local" for frame in frames)
    assert all(not frame["future_and_captions_used_as_model_input"] for frame in frames)


@pytest.fixture
def two_scenes(dataset):
    """A 20-second scene and a separate 9-second scene, sharing only sensor definitions."""
    from copy import deepcopy

    root = dataset / "v2.X-train"
    (root / "scene.json").write_text(json.dumps([
        {"name": "scene-a", "token": "scene"}, {"name": "scene-b", "token": "second"},
    ]))
    for name in ("sample", "sample_data", "ego_pose"):
        path = root / f"{name}.json"
        original = json.loads(path.read_text())
        second = []
        for source in original:
            if source["timestamp"] > 1_740_000_009_000_000:
                continue
            row = deepcopy(source)
            row["timestamp"] += 60_000_000
            for key in ("token", "sample_token", "ego_pose_token"):
                if key in row:
                    row[key] = f"second-{row[key]}"
            if name == "sample":
                row["scene_token"] = "second"
            if name == "ego_pose":
                row["translation"][0] += 100
            second.append(row)
        path.write_text(json.dumps(original + second))
    return dataset


def test_batch_uses_scene_specific_bounds_and_prepare_only_never_loads_model(
    two_scenes, tmp_path, monkeypatch,
):
    from alpamayo2_super import inference_jodd_batch as batch

    def unexpected_load(*args):
        pytest.fail("prepare-only must not load model weights")

    monkeypatch.setattr(batch, "_load_model", unexpected_load)
    result = batch.run_batch(
        dataset_dir=two_scenes, output_dir=tmp_path / "batch", prepare_only=True,
    )
    assert [s["status"] for s in result["scenes"]] == ["inputs_checked", "inputs_checked"]
    assert [s["frame_count"] for s in result["scenes"]] == [24, 2]
    assert [s["last_t0_s"] for s in result["scenes"]] == [13.5, 2.5]
    assert not list((tmp_path / "batch").glob("*.complete.json"))


@pytest.fixture
def batch_predictor(monkeypatch):
    """Deterministic CPU test predictions; exercise the real batch runner and MP4 encoder."""
    pytest.importorskip("av")
    from alpamayo2_super import inference_jodd_batch as batch

    state = {"loads": 0, "models": [], "fail_scene": None}

    def load_model(model_id):
        state["loads"] += 1
        return object()

    def frames(model, files, scene_name, times, **kwargs):
        state["models"].append(model)
        assert kwargs["first_sample"]["clip_id"] == scene_name
        for index, t0 in enumerate(times):
            if state["fail_scene"] == scene_name and index == 1:
                raise RuntimeError("simulated inference failure")
            yield np.full((48, 64, 3), 40 + index * 30, np.uint8), {
                "scene": scene_name, "t0_s": t0, "cot": "CPU test prediction",
            }

    monkeypatch.setattr(batch, "_load_model", load_model)
    monkeypatch.setattr(batch, "render_scene_frames", frames)
    return state


def test_batch_reuses_model_and_resumes_only_matching_complete_outputs(
    two_scenes, tmp_path, batch_predictor,
):
    import av
    from alpamayo2_super import inference_jodd_batch as batch

    output = tmp_path / "batch"
    options = {"dataset_dir": two_scenes, "output_dir": output, "end": 3}
    result = batch.run_batch(**options)
    assert [s["status"] for s in result["scenes"]] == ["complete", "complete"]
    assert batch_predictor["loads"] == 1
    assert batch_predictor["models"][0] is batch_predictor["models"][1]
    for name in ("scene-a", "scene-b"):
        records = [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
        assert [r["t0_s"] for r in records] == [2, 2.5]
        assert all(r["scene"] == name for r in records)
        with av.open(str(output / f"{name}.mp4")) as video:
            assert [f.time for f in video.decode(video=0)] == [0, 0.5]
    skipped = batch.run_batch(**options)
    assert [s["status"] for s in skipped["scenes"]] == ["skipped_complete"] * 2
    assert batch_predictor["loads"] == 1
    changed = batch.run_batch(**options, step=1)
    assert [s["status"] for s in changed["scenes"]] == ["complete", "complete"]
    assert [s["frame_count"] for s in changed["scenes"]] == [1, 1]
    # An interrupted/truncated output must not be treated as complete just because it exists.
    (output / "scene-a.mp4").write_bytes(b"truncated")
    repaired = batch.run_batch(**options, step=1)
    assert [s["status"] for s in repaired["scenes"]] == ["complete", "skipped_complete"]


def test_batch_continues_after_scene_failure_and_retries_it_on_resume(
    two_scenes, tmp_path, batch_predictor,
):
    from alpamayo2_super import inference_jodd_batch as batch

    output = tmp_path / "batch"
    options = {"dataset_dir": two_scenes, "output_dir": output, "end": 3}
    batch_predictor["fail_scene"] = "scene-a"
    result = batch.run_batch(**options)
    assert [s["status"] for s in result["scenes"]] == ["failed", "complete"]
    assert (output / "scene-a.partial.mp4").is_file()
    assert not (output / "scene-a.complete.json").exists()
    assert batch_predictor["loads"] == 1
    batch_predictor["fail_scene"] = None
    retried = batch.run_batch(**options)
    assert [s["status"] for s in retried["scenes"]] == ["complete", "skipped_complete"]
    assert not (output / "scene-a.partial.mp4").exists()
    assert len((output / "scene-a.jsonl").read_text().splitlines()) == 2
    batch_predictor["fail_scene"] = "scene-a"
    batch.run_batch(**options, overwrite=True)
    assert not (output / "scene-a.complete.json").exists()
    batch_predictor["fail_scene"] = None
    regenerated = batch.run_batch(**options)
    assert [s["status"] for s in regenerated["scenes"]] == ["complete", "skipped_complete"]


def test_batch_does_not_retry_global_model_load_failure_for_every_scene(
    two_scenes, tmp_path, monkeypatch,
):
    from alpamayo2_super import inference_jodd_batch as batch

    calls = []

    def failed_load(model_id):
        calls.append(model_id)
        raise RuntimeError("simulated model load failure")

    monkeypatch.setattr(batch, "_load_model", failed_load)
    output = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="model load failure"):
        batch.run_batch(dataset_dir=two_scenes, output_dir=output, end=3)
    assert len(calls) == 1
    summary = json.loads((output / "batch_summary.json").read_text())
    assert len(summary["scenes"]) == 1
    assert summary["scenes"][0]["status"] == "model_load_failed"
