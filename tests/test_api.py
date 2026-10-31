import json

import pytest

from backend import config, jobs, main, store


def add_event(client, match_id, **changes):
    return client.post(f"/api/matches/{match_id}/events", json={"type": "pass", "time": 10, "team": "home", **changes})


def test_match_lifecycle_persists_sqlite_and_hides_filesystem_paths(client, ready_match, isolated_data):
    match_id = ready_match["id"]
    response = client.get(f"/api/matches/{match_id}")
    assert response.status_code == 200
    assert response.json()["match"]["title"] == "Uji futsal"
    assert "video_path" not in response.json()["match"]
    assert (isolated_data / "futsal.sqlite3").is_file()
    with store.connect() as db:
        saved = json.loads(db.execute("SELECT data FROM matches WHERE id=?", (match_id,)).fetchone()[0])
    assert saved["match"]["duration"] == 120
    assert saved["match"]["video_path"].startswith(str(isolated_data))
    assert client.get("/api/matches").json()[0]["id"] == match_id
    assert client.get("/api/matches/nonexistent").status_code == 404


def test_second_and_subsequent_independent_events_can_be_added(client, ready_match):
    match_id = ready_match["id"]
    for kind, time in [("pass", 1), ("pass", 2), ("shot", 3), ("goal", 4), ("foul", 5)]:
        response = add_event(client, match_id, type=kind, time=time)
        assert response.status_code == 201, response.text
    assert len(client.get(f"/api/matches/{match_id}").json()["events"]) == 5


@pytest.mark.parametrize("payload", [{"time": -1}, {"time": 121}, {"time": 10, "end_time": 130}, {"type": "possession", "end_time": 5}, {"type": "possession"}, {"player_id": "foreign-player"}, {"type": "goal", "team": "unknown"}, {"type": "unknown-event"}])
def test_invalid_evidence_is_rejected_without_mutating_match(client, ready_match, payload):
    match_id = ready_match["id"]
    before = store.get(match_id)
    response = add_event(client, match_id, **payload)
    assert response.status_code == 422, response.text
    after = store.get(match_id)
    assert before == after


def test_nonfinite_event_timestamp_rejected(client, ready_match):
    response = client.post(f'/api/matches/{ready_match["id"]}/events', content='{"type":"pass","time":NaN}', headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_roster_team_and_pass_sender_receiver_are_validated(client, ready_match, roster):
    match_id = ready_match["id"]
    home, home2, away = roster
    assert add_event(client, match_id, player_id=away["id"]).status_code == 422
    assert add_event(client, match_id, player_id=home["id"], receiver_id=home["id"]).status_code == 422
    assert add_event(client, match_id, player_id=home["id"], receiver_id=away["id"], outcome="successful").status_code == 422
    valid = add_event(client, match_id, player_id=home["id"], receiver_id=home2["id"], outcome="successful")
    assert valid.status_code == 201, valid.text


def test_player_from_another_match_cannot_be_used_as_actor(client, ready_match):
    another = client.post("/api/matches", json={"title": "Lain", "home_name": "A", "away_name": "B"}).json()
    player = client.post(f'/api/matches/{another["id"]}/players', json={"name": "Asing", "team": "home"}).json()
    assert add_event(client, ready_match["id"], player_id=player["id"]).status_code == 422


def test_pending_ai_requires_review_before_counting_and_finalization(client, ready_match):
    match_id = ready_match["id"]
    response = add_event(client, match_id, type="goal", status="pending")
    assert response.status_code == 201, response.text
    event_id = response.json()["id"]
    with store.edit(match_id) as document:
        document["events"][0].update(source="ai", confidence=0.9)
    assert client.get(f"/api/matches/{match_id}/stats").json()["score"]["home"] == 0
    assert client.patch(f"/api/matches/{match_id}", json={"status": "finalized"}).status_code == 422
    reviewed = client.post(f"/api/matches/{match_id}/events/review", json={"ids": [event_id], "status": "confirmed"})
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json() == {"updated": 1}
    assert client.get(f"/api/matches/{match_id}/stats").json()["score"]["home"] == 1
    assert client.patch(f"/api/matches/{match_id}", json={"status": "finalized"}).status_code == 200
    assert client.patch(f"/api/matches/{match_id}/events/{event_id}", json={"notes": "Tinjau ulang"}).status_code == 200
    assert client.get(f"/api/matches/{match_id}").json()["match"]["status"] == "review"


def test_confirming_overlapping_possession_in_batch_rolls_back_entire_batch(client, ready_match):
    match_id = ready_match["id"]
    ids = []
    for team, start, stop in [("home", 1, 12), ("away", 10, 20)]:
        response = add_event(client, match_id, type="possession", team=team, time=start, end_time=stop, status="pending")
        assert response.status_code == 201, response.text
        ids.append(response.json()["id"])
    before = store.get(match_id)
    response = client.post(f"/api/matches/{match_id}/events/review", json={"ids": ids, "status": "confirmed"})
    assert response.status_code == 422, response.text
    assert store.get(match_id) == before
    assert client.get(f"/api/matches/{match_id}/stats").json()["coverage"]["possession_known_seconds"] == 0


def test_adjacent_possession_intervals_allowed_but_overlap_rejected(client, ready_match):
    match_id = ready_match["id"]
    first = add_event(client, match_id, type="possession", time=0, end_time=10)
    assert first.status_code == 201
    adjacent = add_event(client, match_id, type="possession", team="away", time=10, end_time=20)
    assert adjacent.status_code == 201, adjacent.text
    overlap = add_event(client, match_id, type="possession", time=9, end_time=11)
    assert overlap.status_code == 422
    assert len(store.get(match_id)["events"]) == 2


def test_linked_goal_deduplicates_shot_and_protects_actor_consistency(client, ready_match, roster):
    match_id = ready_match["id"]
    shooter = roster[0]["id"]
    shot = add_event(client, match_id, type="shot", time=15, player_id=shooter, outcome="goal")
    assert shot.status_code == 201, shot.text
    shot_id = shot.json()["id"]
    goal = add_event(client, match_id, type="goal", time=16, player_id=shooter, related_event_id=shot_id)
    assert goal.status_code == 201, goal.text
    home = client.get(f"/api/matches/{match_id}/stats").json()["teams"]["home"]
    assert (home["goals"], home["shots"], home["shots_on_target"]) == (1, 1, 1)
    assert client.patch(f"/api/matches/{match_id}/events/{shot_id}", json={"player_id": roster[1]["id"]}).status_code == 422
    duplicate = add_event(client, match_id, type="goal", time=16, player_id=shooter, related_event_id=shot_id)
    assert duplicate.status_code == 422


def test_revisions_capture_before_after_without_private_path_export(client, ready_match):
    match_id = ready_match["id"]
    created = add_event(client, match_id, type="goal").json()
    response = client.patch(f'/api/matches/{match_id}/events/{created["id"]}', json={"time": 25, "notes": "Koreksi timestamp"})
    assert response.status_code == 200
    exported = client.get(f"/api/matches/{match_id}/export?format=json")
    assert exported.status_code == 200
    assert "video_path" not in exported.json()["match"]
    revisions = exported.json()["revisions"]
    latest = revisions[-1]
    assert latest["action"] == "event_updated"
    assert latest["before"]["time"] == 10
    assert latest["after"]["time"] == 25
    assert latest["before"]["notes"] == ""


def test_later_edits_do_not_rewrite_earlier_audit_snapshots(client, ready_match):
    match_id = ready_match["id"]
    created = add_event(client, match_id, type="goal").json()
    event_id = created["id"]
    for time in (25, 40):
        response = client.patch(f"/api/matches/{match_id}/events/{event_id}", json={"time": time})
        assert response.status_code == 200, response.text
    revisions = client.get(f"/api/matches/{match_id}/export?format=json").json()["revisions"]
    changes = [revision for revision in revisions if revision["action"] in ("event_added", "event_updated")]
    assert [revision["after"]["time"] for revision in changes] == [10, 25, 40]
    assert [revision["before"]["time"] for revision in changes[1:]] == [10, 25]


def test_failed_patch_is_transactional_and_preserves_finalized_status(client, ready_match, roster):
    match_id = ready_match["id"]
    created = add_event(client, match_id, player_id=roster[0]["id"]).json()
    assert client.patch(f"/api/matches/{match_id}", json={"status": "finalized"}).status_code == 200
    before = store.get(match_id)
    response = client.patch(f'/api/matches/{match_id}/events/{created["id"]}', json={"time": 15, "player_id": roster[2]["id"]})
    assert response.status_code == 422
    assert store.get(match_id) == before


def test_referenced_player_cannot_be_deleted_or_moved_to_other_team(client, ready_match, roster):
    match_id = ready_match["id"]
    player_id = roster[0]["id"]
    assert add_event(client, match_id, player_id=player_id).status_code == 201
    assert client.delete(f"/api/matches/{match_id}/players/{player_id}").status_code == 409
    assert client.patch(f"/api/matches/{match_id}/players/{player_id}", json={"team": "away"}).status_code == 422


def test_jersey_number_unique_within_team_but_allowed_across_teams(client, ready_match):
    match_id = ready_match["id"]
    payload = {"name": "Ari", "number": "7", "team": "home"}
    assert client.post(f"/api/matches/{match_id}/players", json=payload).status_code == 201
    assert client.post(f"/api/matches/{match_id}/players", json={**payload, "name": "Bima"}).status_code == 422
    assert client.post(f"/api/matches/{match_id}/players", json={**payload, "team": "away"}).status_code == 201


def test_media_supports_byte_ranges_and_unsatisfiable_range(client, ready_match):
    url = f'/api/matches/{ready_match["id"]}/video'
    response = client.get(url, headers={"Range": "bytes=10-19"})
    assert response.status_code == 206
    assert response.content == bytes(range(10, 20))
    assert response.headers["content-range"] == "bytes 10-19/4096"
    assert response.headers["accept-ranges"] == "bytes"
    assert client.get(url, headers={"Range": "bytes=9999-10000"}).status_code == 416
    assert client.head(url).content == b""


@pytest.mark.parametrize("filename,content", [("not-video.txt", b"anything"), ("empty.mp4", b""), ("broken.mp4", b"not a video container")])
def test_invalid_upload_rejected_and_match_remains_editable(client, filename, content):
    match = client.post("/api/matches", json={"title": "Upload test", "home_name": "A", "away_name": "B"}).json()
    response = client.post(f'/api/matches/{match["id"]}/video', files={"file": (filename, content, "video/mp4")})
    assert response.status_code == 422, response.text
    detail = client.get(f'/api/matches/{match["id"]}').json()
    assert detail["match"]["status"] == "draft"
    assert detail["match"]["video_url"] is None


def test_upload_size_limit_cleans_partial_file(client, monkeypatch):
    match = client.post("/api/matches", json={"title": "Large", "home_name": "A", "away_name": "B"}).json()
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 16)
    response = client.post(f'/api/matches/{match["id"]}/video', files={"file": ("large.mp4", b"x" * 17, "video/mp4")})
    assert response.status_code == 413
    assert not list(config.match_dir(match["id"]).glob("original-*"))


def test_job_requires_video_and_disallows_duplicate_active_job(client, ready_match, monkeypatch):
    submitted = []
    monkeypatch.setattr(jobs, "submit", lambda match_id, job: submitted.append((match_id, job["id"])))
    empty = client.post("/api/matches", json={"title": "Empty", "home_name": "A", "away_name": "B"}).json()
    assert client.post(f'/api/matches/{empty["id"]}/jobs', json={}).status_code == 422
    match_id = ready_match["id"]
    started = client.post(f"/api/matches/{match_id}/jobs", json={"sample_fps": 1})
    assert started.status_code == 202, started.text
    assert started.json()["status"] == "queued"
    assert len(submitted) == 1
    assert client.post(f"/api/matches/{match_id}/jobs", json={}).status_code == 409
    assert client.delete(f"/api/matches/{match_id}").status_code == 409


def test_recovery_marks_running_jobs_interrupted_without_deleting_evidence(client, ready_match):
    match_id = ready_match["id"]
    assert add_event(client, match_id, type="goal").status_code == 201
    with store.edit(match_id) as document:
        document["jobs"].append({"id": "job", "status": "running", "progress": 45})
        document["match"]["status"] = "processing"
    store.recover_jobs()
    document = store.get(match_id)
    assert document["jobs"][0]["status"] == "interrupted"
    assert document["match"]["status"] == "review"
    assert len(document["events"]) == 1


def test_calibration_rejects_degenerate_points_and_overlapping_segments(client, ready_match):
    match_id = ready_match["id"]
    payload = {"start_time": 0, "end_time": 10, "image_points": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    first = client.post(f"/api/matches/{match_id}/calibrations", json=payload)
    assert first.status_code == 201, first.text
    assert client.post(f"/api/matches/{match_id}/calibrations", json={**payload, "start_time": 5, "end_time": 20}).status_code == 422
    assert client.post(f"/api/matches/{match_id}/calibrations", json={**payload, "start_time": 20, "end_time": 30, "image_points": [[0, 0]] * 4}).status_code == 422
    assert client.post(f"/api/matches/{match_id}/calibrations", json={**payload, "start_time": 20, "end_time": 150}).status_code == 422


def test_cross_site_writes_and_dns_rebinding_host_are_blocked(client):
    payload = {"title": "Should not save", "home_name": "A", "away_name": "B"}
    assert client.post("/api/matches", json=payload, headers={"Origin": "https://evil.test"}).status_code == 403
    assert client.get("/api/matches", headers={"Host": "evil.test"}).status_code == 400
    assert client.get("/api/matches").json() == []


def test_delete_match_removes_only_its_own_recording_directory(client, ready_match, isolated_data):
    match_id = ready_match["id"]
    sibling = isolated_data / "matches" / "keep-me"
    sibling.mkdir()
    (sibling / "fixture.txt").write_text("keep", encoding="utf-8")
    assert client.delete(f"/api/matches/{match_id}").status_code == 204
    assert store.get(match_id) is None
    assert not (isolated_data / "matches" / match_id).exists()
    assert (sibling / "fixture.txt").read_text(encoding="utf-8") == "keep"
