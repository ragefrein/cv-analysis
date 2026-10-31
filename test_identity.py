from copy import deepcopy

import cv2
import numpy as np
import pytest

from court import ShotBoundary, confirmed_foot, correct_boundary, detect_court, point_on_court
from identity_review import candidate_links, reconcile_tracklets, refresh_review, split_player, verify_player, visible_players
from players import PlayerMemory, merge_players
from test_vision import analysis_fixture, observation


def test_logo_and_player_holes_do_not_remove_floor():
    frame = np.full((720, 1280, 3), 50, np.uint8)
    frame[250:680, 60:1220] = (254, 209, 195)
    frame[390:470, 500:800] = (255, 255, 255)
    mask, polygon, _ = detect_court(frame)
    assert point_on_court(mask, [640, 430])
    assert not point_on_court(mask, [640, 700])
    assert polygon


def test_missing_floor_never_reuses_previous_camera():
    from vision import court_mask
    previous = np.full((720, 1280), 255, np.uint8)
    mask, polygon, bbox = court_mask(np.full((720, 1280, 3), 50, np.uint8), previous)
    assert not mask.any() and polygon == [] and bbox is None
    assert confirmed_foot(mask, [200, 200, 300, 500]) is None


def test_blue_ad_board_is_not_playable_floor():
    frame = np.full((720, 1280, 3), 50, np.uint8)
    frame[20:240, :] = (254, 209, 195)
    assert not detect_court(frame)[0].any()


def test_truncated_body_and_off_court_feet_rejected():
    mask = np.zeros((720, 1280), np.uint8)
    mask[200:600] = 255
    assert confirmed_foot(mask, [200, 100, 300, 500]) == [250, 500]
    assert confirmed_foot(mask, [200, 300, 300, 650]) is None
    assert confirmed_foot(np.full_like(mask, 255), [200, 300, 300, 720]) is None


def test_cut_detector_does_not_treat_small_motion_as_cut():
    detector = ShotBoundary()
    floor = np.full((720, 1280, 3), (254, 209, 195), np.uint8)
    assert not detector.update(floor)
    floor[400:450, 200:250] = (10, 20, 220)
    assert not detector.update(floor)
    assert detector.update(np.full_like(floor, (10, 20, 220)))


def tracklet_fixture(gap=2, second_segment=0):
    frames = []
    for i in range(20 + gap):
        pid = 0 if i < 10 else 1
        present = i < 10 or i >= 10 + gap
        frames.append({"p": [[pid, 10, 10, 20, 30]] if present else [],
                       "feet": {str(pid): [15, 30]} if present else {}, "poses": {},
                       "anchors": {str(pid): [.3, .5]} if present else {},
                       "segment": 0 if i < 10 else second_segment,
                       "camera": np.eye(3).tolist(), "court": []})
    roster = {str(pid): {"id": pid, "appearance": [[1., 0., 0.]], "team": 0, "number": "",
                        "first_seen": 0 if pid == 0 else 10 + gap, "last_seen": 9 if pid == 0 else len(frames) - 1,
                        "observations": 10, "name": "", "verified": False}
              for pid in (0, 1)}
    return {"frames": frames, "frame_count": len(frames), "fps": 30, "players": roster, "stats": {"unique_ids": 2}}


def test_offline_join_requires_time_motion_and_multiple_observations():
    tracks = tracklet_fixture()
    reconcile_tracklets(tracks)
    assert len(tracks["players"]) == 1
    assert len(tracks["automatic_links"]) == 1
    assert not tracks["players"]["0"]["verified"]


def test_across_cuts_and_long_absence_only_suggest_identity():
    for tracks in (tracklet_fixture(second_segment=1), tracklet_fixture(gap=100)):
        reconcile_tracklets(tracks)
        assert len(tracks["players"]) == 2
        assert len(tracks["review_queue"]) == 1


def test_motion_contradiction_does_not_auto_merge():
    tracks = tracklet_fixture()
    tracks["frames"][12]["feet"]["1"] = [90, 90]
    reconcile_tracklets(tracks)
    assert len(tracks["players"]) == 2


def test_rejected_pair_is_not_suggested_again():
    tracks = tracklet_fixture(second_segment=1)
    tracks["review_decisions"] = [{"action": "different", "ids": [0, 1]}]
    assert candidate_links(tracks) == []


def test_number_verification_conflict_and_roles():
    tracks = tracklet_fixture()
    verify_player(tracks, 0, "05", 0, "Andi")
    assert tracks["players"]["0"]["number"] == "5"
    before = deepcopy(tracks["players"]["1"])
    with pytest.raises(ValueError, match="sudah milik"):
        verify_player(tracks, 1, "5", 0)
    assert tracks["players"]["1"] == before
    verify_player(tracks, 1, "", -1, role="referee")
    assert visible_players(tracks, 12) == []
    assert tracks["stats"]["unique_ids"] == 1


def test_conflicting_jersey_numbers_cannot_be_merged():
    tracks = tracklet_fixture()
    tracks["players"]["0"]["number"] = "5"
    tracks["players"]["1"]["number"] = "9"
    before = deepcopy(tracks)
    with pytest.raises(ValueError, match="Nomor punggung berbeda"):
        merge_players(tracks, 1, 0)
    assert tracks == before


def test_split_repairs_mid_track_switch_and_preserves_previous_identity():
    tracks = tracklet_fixture()
    merge_players(tracks, 1, 0)
    tracks["players"]["0"].update(name="Andi", number="5", verified=True)
    new_id = split_player(tracks, 0, 12)
    assert tracks["frames"][0]["p"][0][0] == 0
    assert tracks["frames"][12]["p"][0][0] == new_id
    assert tracks["players"]["0"]["name"] == "Andi"
    assert tracks["players"][str(new_id)]["name"] == ""
    assert not tracks["players"][str(new_id)]["verified"]


def test_manual_floor_boundary_filters_bench_and_follows_camera():
    tracks = analysis_fixture()
    tracks.update(width=100, height=100)
    tracks["frames"][1]["camera"] = [[1, 0, 20], [0, 1, 0], [0, 0, 1]]
    tracks["frames"][1]["feet"]["1"] = [10, 30]
    removed = correct_boundary(tracks, 0, [[0, 0], [50, 0], [50, 50], [0, 50]])
    assert removed == 1
    assert tracks["frames"][0]["p"]
    assert not tracks["frames"][1]["p"]
    assert "1" not in tracks["frames"][1]["feet"]


def test_precision_mode_does_not_assign_person_from_single_returning_crop():
    memory = PlayerMemory(reidentify=False)
    memory.assign([observation(1, [1, 0])], 0)
    assert memory.assign([observation(2, [1, 0])], 1000) == [1]
