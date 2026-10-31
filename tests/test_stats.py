import copy

import pytest

from backend.exports import event_csv, export_json, player_csv, report_html
from backend.stats import calculate, union_seconds


def match_document():
    return {
        "match": {"id": "match", "title": "Pertandingan", "home_name": "Hijau", "away_name": "Kuning", "duration": 120, "status": "review", "is_demo": False, "video_path": "private/video.mp4"},
        "players": [
            {"id": "p1", "name": "Ari", "number": "7", "team": "home", "role": "player"},
            {"id": "p2", "name": "Bima", "number": "8", "team": "home", "role": "player"},
            {"id": "p3", "name": "Candra", "number": "9", "team": "away", "role": "goalkeeper"},
        ],
        "tracks": [], "events": [], "jobs": [], "calibrations": [], "revisions": [],
    }


def event(event_id, kind, *, team="home", player="p1", time=10, status="confirmed", outcome="unknown", **extra):
    return {"id": event_id, "type": kind, "time": time, "team": team, "player_id": player, "receiver_id": None, "status": status, "outcome": outcome, "source": "manual", **extra}


def test_pending_and_rejected_predictions_never_change_final_statistics():
    document = match_document()
    document["events"] = [
        event("a", "goal", status="pending", source="ai"),
        event("b", "goal", team="away", player="p3", status="rejected", source="ai"),
        event("c", "pass", status="pending", outcome="successful", source="ai"),
        event("d", "goal", time=45),
    ]
    stats = calculate(document)
    assert stats["score"] == {"home": 1, "away": 0}
    assert stats["teams"]["home"]["passes"] == 0
    assert stats["pending_count"] == 2
    assert stats["rejected_count"] == 1
    assert stats["confirmed_count"] == 1


def test_unresolved_passes_are_counted_but_excluded_from_accuracy_denominator():
    document = match_document()
    document["events"] = [event("a", "pass", outcome="successful", receiver_id="p2"), event("b", "pass", outcome="failed"), event("c", "pass")]
    stats = calculate(document)
    home = stats["teams"]["home"]
    assert (home["passes"], home["passes_successful"], home["passes_failed"], home["passes_unresolved"]) == (3, 1, 1, 1)
    assert home["pass_accuracy"] == 50
    assert stats["players"][0]["pass_accuracy"] == 50
    assert stats["pass_network"] == [{"from_player_id": "p1", "to_player_id": "p2", "team": "home", "count": 1}]
    assert stats["teams"]["away"]["pass_accuracy"] is None


def test_no_evidence_does_not_display_zero_percent_accuracy_or_possession():
    stats = calculate(match_document())
    assert stats["teams"]["home"]["pass_accuracy"] is None
    assert stats["teams"]["away"]["possession_pct"] is None
    assert stats["coverage"]["possession_unknown_seconds"] == 120


def test_own_goal_credits_opponent_without_individual_goal_or_shot():
    document = match_document()
    document["events"] = [event("own", "own_goal")]
    stats = calculate(document)
    assert stats["score"] == {"home": 0, "away": 1}
    assert stats["teams"]["home"]["own_goals"] == 1
    assert stats["players"][0]["goals"] == 0
    assert stats["players"][0]["own_goals"] == 1
    assert sum(p["shots"] for p in stats["players"]) == 0
    assert sum(t["shots"] for t in stats["teams"].values()) == 0


@pytest.mark.parametrize("shot_outcome", ["unknown", "on_target", "goal"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_explicitly_linked_goal_and_shot_are_one_goal_one_shot(shot_outcome, reverse_order):
    document = match_document()
    document["events"] = [event("shot", "shot", outcome=shot_outcome), event("goal", "goal", time=11, related_event_id="shot")]
    if reverse_order:
        document["events"].reverse()
    stats = calculate(document)
    home = stats["teams"]["home"]
    assert (home["goals"], home["shots"], home["shots_on_target"]) == (1, 1, 1)
    player = stats["players"][0]
    assert (player["goals"], player["shots"], player["shots_on_target"]) == (1, 1, 1)


def test_goal_alone_is_a_shot_and_a_shot_on_target():
    document = match_document()
    document["events"] = [event("goal", "goal")]
    stats = calculate(document)
    assert (stats["teams"]["home"]["goals"], stats["teams"]["home"]["shots"], stats["teams"]["home"]["shots_on_target"]) == (1, 1, 1)


def test_pending_linked_shot_does_not_hide_confirmed_goal():
    document = match_document()
    document["events"] = [event("shot", "shot", status="pending"), event("goal", "goal", related_event_id="shot")]
    home = calculate(document)["teams"]["home"]
    assert (home["goals"], home["shots"], home["shots_on_target"]) == (1, 1, 1)


def test_possession_uses_only_confirmed_intervals_and_exposes_unclassified_time():
    document = match_document()
    document["events"] = [event("a", "possession", time=0, end_time=20), event("b", "possession", time=20, end_time=30, team="away", player="p3"), event("c", "possession", time=30, end_time=100, status="pending")]
    stats = calculate(document)
    assert stats["teams"]["home"]["possession_seconds"] == 20
    assert stats["teams"]["home"]["possession_pct"] == 66.7
    assert stats["teams"]["away"]["possession_pct"] == 33.3
    assert stats["coverage"]["possession_known_seconds"] == 30
    assert stats["coverage"]["possession_unknown_seconds"] == 90


def test_confirmed_team_evidence_survives_missing_player_identity():
    document = match_document()
    document["events"] = [event("a", "goal", player=None)]
    stats = calculate(document)
    assert stats["score"]["home"] == 1
    assert sum(p["goals"] for p in stats["players"]) == 0
    assert any("tanpa identitas" in warning for warning in stats["warnings"])


def test_latest_completed_run_defines_coverage_not_failed_rerun():
    document = match_document()
    document["jobs"] = [{"status": "completed", "metrics": {"analyzed_seconds": 110, "ball_visible_seconds": 40}}, {"status": "failed", "metrics": {"analyzed_seconds": 5}}]
    document["calibrations"] = [{"start_time": 0, "end_time": 10}, {"start_time": 5, "end_time": 20}]
    stats = calculate(document)
    assert stats["coverage"]["analyzed_seconds"] == 110
    assert stats["coverage"]["ball_visible_seconds"] == 40
    assert stats["coverage"]["calibrated_seconds"] == 20


def test_demo_evidence_is_explicitly_labelled_as_simulation():
    document = match_document()
    document["match"]["is_demo"] = True
    assert "SIMULASI" in calculate(document)["warnings"][0]


def test_calculation_is_read_only_and_does_not_mutate_saved_evidence():
    document = match_document()
    document["events"] = [event("a", "pass", outcome="successful", receiver_id="p2")]
    original = copy.deepcopy(document)
    calculate(document)
    calculate(document)
    assert document == original


@pytest.mark.parametrize("intervals,expected", [([], 0), ([(3, 8), (0, 4), (8, 9)], 9), ([(0, 20), (3, 7), (25, 30)], 25)])
def test_union_of_intervals_does_not_double_count_nested_segments(intervals, expected):
    assert union_seconds(intervals) == expected


def test_exports_escape_spreadsheet_formulas_and_html_but_preserve_json_evidence():
    document = match_document()
    document["match"]["title"] = '<script>alert("title")</script>'
    document["players"][0]["name"] = '=HYPERLINK("https://bad.test")'
    document["players"][1]["name"] = '<img src=x onerror=alert(1)>'
    document["events"] = [event("a", "foul", notes="@SUM(1,2)")]
    assert "'=HYPERLINK" in player_csv(document)
    assert "'@SUM" in event_csv(document)
    report = report_html(document)
    assert "<script>" not in report
    assert "<img src=x" not in report
    assert "&lt;script&gt;" in report
    exported = export_json(document)
    assert "private/video.mp4" not in exported
    assert "video_path" not in exported
    assert "HYPERLINK" in exported
