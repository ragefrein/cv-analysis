import pytest

from evaluate_tracking import evaluate


def test_perfect_boxes_can_have_wrong_identity():
    tracks = {"frames": [{"p": [[0, 0, 0, 10, 20]]}, {"p": [[1, 0, 0, 10, 20]]}]}
    labels = {"complete_frame_labels": True, "frames": [
        {"index": 0, "players": [{"id": "IDN_5", "box": [0, 0, 10, 20]}]},
        {"index": 1, "players": [{"id": "IDN_5", "box": [0, 0, 10, 20]}]},
    ]}
    metrics = evaluate(tracks, labels)
    assert metrics["detection_precision"] == metrics["detection_recall"] == 1
    assert metrics["identity_f1_on_labelled_frames"] == .5
    assert metrics["identity_switches_between_labelled_observations"] == 1
    tracks["frames"][1]["p"][0][0] = 0
    assert evaluate(tracks, labels)["identity_f1_on_labelled_frames"] == 1


def test_unlabelled_or_partial_reference_cannot_claim_accuracy():
    with pytest.raises(ValueError, match="semua pemain"):
        evaluate({"frames": []}, {"frames": []})
    with pytest.raises(ValueError, match="Belum ada frame"):
        evaluate({"frames": []}, {"complete_frame_labels": True, "frames": []})


def test_missing_detections_and_extra_people_penalize_metrics():
    labels = {"complete_frame_labels": True, "frames": [
        {"index": 0, "players": [{"id": "A", "box": [0, 0, 10, 20]}]},
        {"index": 1, "players": [{"id": "A", "box": [0, 0, 10, 20]}]},
    ]}
    tracks = {"frames": [{"p": [[0, 0, 0, 10, 20], [1, 50, 50, 60, 70]]}, {"p": []}]}
    result = evaluate(tracks, labels)
    assert result["detection_precision"] == result["detection_recall"] == .5
    assert result["identity_f1_on_labelled_frames"] == .5
