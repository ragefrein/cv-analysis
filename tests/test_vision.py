"""Vision invariants that do not depend on accuracy of pretrained detections."""

import json
from pathlib import Path

import pytest

from backend.vision import AnalysisCancelled, CourtRegion, EventProposer, _court_polygon, _owner, analyze_video, probe_video


def player(track_id, center_x, team="home"):
    return {"kind": "player", "track_id": track_id, "team": team,
            "bbox": [center_x - 0.04, 0.30, center_x + 0.04, 0.60], "confidence": 0.9}


def ball(center_x):
    return {"kind": "ball", "track_id": None, "team": "unknown",
            "bbox": [center_x - 0.005, 0.585, center_x + 0.005, 0.60], "confidence": 0.8}


def test_ambiguous_contact_is_unknown():
    assert _owner([player(1, 0.49), player(2, 0.51), ball(0.5)]) is None


def test_continuous_observation_proposes_pass_but_no_goal():
    proposer = EventProposer(5)
    for timestamp, owner_id, x in [(0.0, 1, 0.3), (0.2, 1, 0.3), (0.4, 1, 0.3),
                                   (0.6, 2, 0.7), (0.8, 2, 0.7), (1.0, 2, 0.7)]:
        proposer.consume(timestamp, [player(owner_id, x), ball(x)])
    events = proposer.finish()
    passes = [event for event in events if event["type"] == "pass"]
    assert len(passes) == 1
    assert passes[0]["track_id"] == 1 and passes[0]["receiver_track_id"] == 2
    assert all(event["type"] not in ("goal", "shot") for event in events)


def test_invisible_ball_gap_never_becomes_pass():
    proposer = EventProposer(5)
    for timestamp in [0, 0.2, 0.4]:
        proposer.consume(timestamp, [player(1, 0.3), ball(0.3)])
    for timestamp in [0.6, 0.8, 1.0, 1.2]:
        proposer.consume(timestamp, [player(1, 0.3)])
    for timestamp in [1.4, 1.6, 1.8]:
        proposer.consume(timestamp, [player(2, 0.7), ball(0.7)])
    assert not any(event["type"] == "pass" for event in proposer.finish())


def test_scene_cut_resets_contact_chain():
    proposer = EventProposer(5)
    for timestamp in [0, 0.2, 0.4]:
        proposer.consume(timestamp, [player(1, 0.3), ball(0.3)])
    proposer.reset()
    for timestamp in [0.6, 0.8, 1.0]:
        proposer.consume(timestamp, [player(2, 0.7), ball(0.7)])
    assert not any(event["type"] == "pass" for event in proposer.finish())


def test_opponent_control_is_not_successful_pass():
    proposer = EventProposer(5)
    for timestamp in [0, 0.2, 0.4]:
        proposer.consume(timestamp, [player(1, 0.3), ball(0.3)])
    for timestamp in [0.6, 0.8, 1.0]:
        proposer.consume(timestamp, [player(2, 0.7, "away"), ball(0.7)])
    assert not any(event["type"] == "pass" for event in proposer.finish())


def test_no_team_evidence_does_not_create_team_possession():
    proposer = EventProposer(5)
    for timestamp in [0, 0.2, 0.4]:
        proposer.consume(timestamp, [player(1, 0.3, "unknown"), ball(0.3)])
    assert proposer.finish() == []


def test_auto_court_estimate_filters_outside_and_expires():
    import numpy as np

    image = np.zeros((270, 480, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 180)
    image[30:250, 60:420] = (200, 120, 20)
    polygon = _court_polygon(image)
    assert polygon is not None
    court = CourtRegion([], auto=True)
    court.update(image, 0.0)
    assert court.contains(0.5, 0.5, 0.5) is True
    assert court.contains(0.02, 0.5, 0.5) is False
    assert court.contains(0.5, 0.5, 5.0) is None


def test_auto_court_off_keeps_court_unknown():
    import numpy as np

    image = np.zeros((270, 480, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 180)
    image[30:250, 60:420] = (200, 120, 20)
    court = CourtRegion([], auto=False)
    court.update(image, 0.0)
    assert court.contains(0.5, 0.5, 0.5) is None


def test_reject_invalid_video(tmp_path):
    file = tmp_path / "broken.mp4"
    file.write_text("not a video", encoding="utf-8")
    with pytest.raises(ValueError):
        probe_video(file)


def test_vfr_metadata_and_pts_preserved_in_pipeline(tmp_path, monkeypatch):
    """Stub inference only: the actual video decoder and PTS sampling stay real."""
    import av
    import numpy as np
    from fractions import Fraction
    import sys
    import types
    import backend.vision as vision

    source = tmp_path / "vfr.mp4"
    timestamps = [0, 100, 300, 900, 1000]
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = 64, 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for pts in timestamps:
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 70, dtype=np.uint8), format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def track(self, *args, **kwargs):
            return [types.SimpleNamespace(boxes=None)]

    monkeypatch.setattr(vision, "capabilities", lambda: {"available": True, "model_ready": True, "device": "cpu"})
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    target = tmp_path / "frames.jsonl"
    result = analyze_video(source, target, {"sample_fps": 10}, lambda *_: None, lambda: False)
    observed = [json.loads(line)["time"] for line in target.read_text().splitlines()]
    assert observed == pytest.approx([0, 0.1, 0.3, 0.9, 1.0], abs=0.02)
    assert result["metrics"]["ball_visible_seconds"] == 0


def test_cancel_does_not_write_result(tmp_path, monkeypatch):
    import backend.vision as vision

    monkeypatch.setattr(vision, "capabilities", lambda: {"available": True, "model_ready": True, "device": "cpu"})
    with pytest.raises(AnalysisCancelled):
        analyze_video("unused.mp4", tmp_path / "cancelled.jsonl", {}, lambda *_: None, lambda: True)
    assert not (tmp_path / "cancelled.jsonl").exists()
