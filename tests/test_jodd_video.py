# SPDX-License-Identifier: Apache-2.0
"""Check encoded playback timing and its correspondence to saved inference records."""

import json

import numpy as np
import pytest

from alpamayo2_super.inference_jodd_video import inference_times, write_video_results


def test_timeline_preserves_recorded_duration_without_end_frame():
    times = inference_times(2, 12, 0.5)
    assert len(times) == 20
    assert times[0] == 2 and times[-1] == 11.5
    assert len(times) * 0.5 == 12 - 2
    assert inference_times(2.1, 2.4, 0.1) == [2.1, 2.2, 2.3]


@pytest.mark.parametrize("start,end,step", [
    (1, 12, 0.5), (2, 2, 0.5), (2, 1, 0.5), (2, 12, 0),
    (2, 12, -1), (2, float("inf"), 0.5), (2, 12, float("nan")),
])
def test_invalid_timeline_is_rejected(start, end, step):
    with pytest.raises(ValueError):
        inference_times(start, end, step)


def _frames():
    for index, t0 in enumerate([2.0, 2.5, 3.0]):
        image = np.full((48, 64, 3), index * 80, dtype=np.uint8)
        yield image, {"t0_s": t0, "cot": f"Prediction {index}"}


def test_encoded_video_matches_prediction_records_and_playback_time(tmp_path):
    av = pytest.importorskip("av")
    prefix = tmp_path / "results"
    assert write_video_results(prefix, _frames(), 0.5) == 3
    records = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
    with av.open(str(prefix.with_suffix(".mp4"))) as video:
        decoded = list(video.decode(video=0))
        assert video.duration / av.time_base == pytest.approx(1.5)
    assert len(decoded) == len(records) == 3
    assert [frame.time for frame in decoded] == [0, 0.5, 1.0]
    assert [record["video_time_s"] for record in records] == [0, 0.5, 1.0]
    assert [record["t0_s"] for record in records] == [2, 2.5, 3]
    for index, (frame, record) in enumerate(zip(decoded, records)):
        assert frame.to_ndarray(format="rgb24").mean() == pytest.approx(index * 80, abs=3)
        assert record["cot"] == f"Prediction {index}"


def test_completed_frames_remain_readable_if_later_inference_fails(tmp_path):
    av = pytest.importorskip("av")

    def interrupted():
        yield from _frames()
        raise RuntimeError("Later inference failed")

    prefix = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="Later inference"):
        write_video_results(prefix, interrupted(), 0.5)
    with av.open(str(prefix.with_suffix(".mp4"))) as video:
        assert len(list(video.decode(video=0))) == 3
    assert len(prefix.with_suffix(".jsonl").read_text().splitlines()) == 3
