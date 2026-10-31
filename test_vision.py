"""Deterministic checks, independent of model downloads. Run: python -m pytest -q."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from players import PlayerMemory, merge_players
from storage import load_session, save_session
from tactics import court_homography, set_calibration, tactical_positions
from vision import _pick_ball, active_embeddings, court_mask, draw_skeleton, foot_point, match_poses


def test_court():
    frame = np.full((720, 1280, 3), (60, 60, 60), np.uint8)
    frame[250:700, 60:1220] = (254, 209, 195)  # bright blue floor
    mask, poly, bbox = court_mask(frame)
    assert poly and bbox is not None
    assert mask[500, 640] == 255
    assert mask[100, 640] == 0


def test_ball_gate():
    state = {"pos": None, "vel": np.zeros(2), "miss": 0}
    court = (0, 0, 720, 1280)
    first = [400, 600, 420, 620]
    assert _pick_ball([(0.8, first)], court, 720, 1280, state) == first
    near = [430, 600, 450, 620]
    assert _pick_ball([(0.5, near)], court, 720, 1280, state) == near
    far = [80, 100, 100, 120]
    assert _pick_ball([(0.9, far)], court, 720, 1280, state) is None


def observation(track_id, feature, jersey=(80, 130, 140)):
    return {"track_id": track_id, "embedding": np.asarray(feature, np.float32),
            "jersey": np.asarray(jersey, np.float32)}


def test_player_returns_after_tracker_expiry():
    memory = PlayerMemory()
    assert memory.assign([observation(10, [1, 0, 0])], 0) == [0]
    memory.assign([], 1000)
    assert memory.assign([observation(400, [.99, .01, 0])], 10000) == [0]
    assert memory.players[0]["first_seen"] == 0
    assert memory.players[0]["last_seen"] == 10000


def test_boxmot_features_use_exported_identity():
    tracker = SimpleNamespace(active_tracks=[
        SimpleNamespace(id=4, track_id=0, curr_feat=np.asarray([1, 0])),
        SimpleNamespace(id=9, track_id=0, curr_feat=np.asarray([0, 1])),
    ])
    features = active_embeddings(tracker)
    assert set(features) == {4, 9}
    assert np.array_equal(features[4], [1, 0])
    assert np.array_equal(features[9], [0, 1])


def test_player_cannot_be_assigned_twice_in_frame():
    memory = PlayerMemory()
    memory.assign([observation(10, [1, 0, 0])], 0)
    # Unknown track comes first. The existing binding must already be reserved.
    ids = memory.assign([observation(11, [1, 0, 0]), observation(10, [1, 0, 0])], 1)
    assert ids == [1, 0]
    ids = memory.assign([observation(12, [1, 0, 0]), observation(13, [1, 0, 0])], 1000)
    assert len(set(ids)) == 2


def test_ambiguous_appearance_gets_new_id():
    memory = PlayerMemory()
    assert memory.assign([observation(10, [1, 0, 0]), observation(11, [1, .01, 0])], 0) == [0, 1]
    assert memory.assign([observation(90, [1, 0, 0])], 999) == [2]


def test_two_jersey_teams_and_seeded_names():
    memory = PlayerMemory()
    memory.assign([observation(10, [1, 0], (90, 120, 110)),
                   observation(11, [0, 1], (160, 190, 170))], 0)
    roster = memory.finish()
    assert {p["team"] for p in roster.values()} == {0, 1}
    roster["0"].update(name="Andi", team=1, team_manual=True)
    seeded = PlayerMemory(roster)
    assert seeded.assign([observation(98, [1, 0], (90, 120, 110))], 5) == [0]
    refreshed = seeded.finish()
    assert refreshed["0"]["name"] == "Andi"
    assert refreshed["0"]["team"] == 1


def test_goalkeeper_brightness_does_not_collapse_outfield_teams():
    memory = PlayerMemory()
    colors = [(99, 174, 156), (36, 141, 101), (86, 177, 158),
              (41, 144, 95), (98, 181, 163), (202, 141, 143)]
    memory.assign([observation(i, np.eye(len(colors))[i], color) for i, color in enumerate(colors)], 0)
    roster = memory.finish()
    assert roster["0"]["team"] == roster["2"]["team"] == roster["4"]["team"]
    assert roster["1"]["team"] == roster["3"]["team"]
    assert roster["0"]["team"] != roster["1"]["team"]
    assert roster["5"]["team"] == -1


def test_many_short_referee_fragments_do_not_become_a_team():
    memory = PlayerMemory()
    colors = [(84, 170, 129), (99, 178, 135), (89, 171, 127),
              (43, 136, 110), (51, 132, 113), (45, 134, 105)]
    colors += [(207, 119, 208), (203, 109, 193), (232, 110, 212),
               (192, 111, 184), (230, 106, 209), (213, 108, 196)]
    memory.assign([observation(i, np.eye(len(colors))[i], color) for i, color in enumerate(colors)], 0)
    for i, player in memory.players.items():
        player["observations"] = 200 if i < 6 else 4
    roster = memory.finish()
    assert roster["0"]["team"] == roster["1"]["team"] == roster["2"]["team"]
    assert roster["3"]["team"] == roster["4"]["team"] == roster["5"]["team"]
    assert roster["0"]["team"] != roster["3"]["team"]
    assert all(roster[str(i)]["team"] == -1 for i in range(6, 12))


def analysis_fixture():
    return {"pipeline_version": 3, "frame_count": 2, "players": {
        "0": {"id": 0, "name": "Andi", "team": 0, "first_seen": 0, "last_seen": 0, "observations": 1},
        "1": {"id": 1, "name": "", "team": 0, "first_seen": 1, "last_seen": 1, "observations": 1}},
        "frames": [{"p": [[0, 10, 10, 20, 30]], "feet": {"0": [15, 30]},
                    "poses": {"0": [[1, 2, .9]]}, "court": [], "segment": 0, "camera": np.eye(3).tolist()},
                   {"p": [[1, 15, 10, 25, 30]], "feet": {"1": [20, 30]},
                    "poses": {"1": [[1, 2, .9]]}, "court": [], "segment": 0, "camera": np.eye(3).tolist()}],
        "stats": {"unique_ids": 2}}


def test_merge_updates_full_timeline_and_preserves_name():
    tracks = analysis_fixture()
    merge_players(tracks, 1, 0)
    assert len(tracks["players"]) == tracks["stats"]["unique_ids"] == 1
    assert tracks["players"]["0"]["name"] == "Andi"
    assert tracks["frames"][1]["p"][0][0] == 0
    assert "0" in tracks["frames"][1]["feet"]
    assert "0" in tracks["frames"][1]["poses"]


def test_merge_with_saved_but_not_yet_seen_identity():
    tracks = analysis_fixture()
    tracks["players"]["0"].update(first_seen=-1, last_seen=-1, observations=0)
    tracks["frames"][0]["p"] = []
    merge_players(tracks, 1, 0)
    assert tracks["players"]["0"]["first_seen"] == 1


def test_merge_rejects_simultaneous_players_without_mutation():
    tracks = analysis_fixture()
    tracks["frames"][0]["p"].append([1, 30, 10, 40, 30])
    before = deepcopy(tracks)
    with pytest.raises(ValueError, match="bersamaan"):
        merge_players(tracks, 1, 0)
    assert tracks == before


def test_persistence_survives_reopen_and_rejects_replaced_video(tmp_path):
    video = tmp_path / "match.mp4"
    video.write_bytes(b"sample video")
    cache = tmp_path / "session.json.gz"
    tracks = analysis_fixture()
    save_session(video, tracks, cache)
    assert load_session(video, cache) == tracks
    video.write_bytes(b"different video")
    assert load_session(video, cache) is None


def test_projection_uses_feet_and_compensates_camera():
    tracks = analysis_fixture()
    set_calibration(tracks, 0, [[0, 0], [100, 0], [100, 100], [0, 100]])
    points, mode = tactical_positions(tracks, 0)
    assert np.allclose(points[0][1:], [.15, .30])
    # Camera translates right 20 px; player's court location stays the same.
    tracks["frames"][1]["camera"] = [[1, 0, 20], [0, 1, 0], [0, 0, 1]]
    tracks["frames"][1]["feet"]["1"] = [35, 30]
    moved, _ = tactical_positions(tracks, 1)
    assert np.allclose(moved[0][1:], points[0][1:])
    # Across a cut, the previous calibration must no longer be used.
    tracks["frames"][1]["segment"] = 1
    assert tactical_positions(tracks, 1)[0] == []


def test_projection_does_not_pin_outside_players_to_boundary():
    tracks = analysis_fixture()
    set_calibration(tracks, 0, [[0, 0], [100, 0], [100, 100], [0, 100]])
    tracks["frames"][0]["feet"]["0"] = [-50, 20]
    assert tactical_positions(tracks, 0)[0] == []
    with pytest.raises(ValueError):
        court_homography([[0, 0], [100, 100], [100, 0], [0, 100]])


def test_pose_association_and_low_confidence_joints():
    pose = np.tile([20, 20, .9], (17, 1)).astype(np.float32)
    matches = match_poses([[0, 0, 40, 60], [2, 0, 42, 60]], [[0, 0, 40, 60]], [pose])
    assert len(matches) == 1
    assert match_poses([[200, 200, 240, 260]], [[0, 0, 40, 60]], [pose]) == {}
    frame = np.zeros((80, 80, 3), np.uint8)
    pose[:, 2] = .1
    draw_skeleton(frame, pose.tolist(), (255, 100, 100))
    assert not frame.any()
    assert foot_point([0, 0, 40, 60], pose) == [20, 60]
    pose[15:17] = [[10, 55, .9], [30, 58, .9]]
    assert foot_point([0, 0, 40, 60], pose) == [20, 58]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
