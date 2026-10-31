"""Backend tests keep all state in disposable, project-local directories."""
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend import config, jobs, main, store


@pytest.fixture
def isolated_data(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "test-results" / f"api-{uuid4().hex}"
    root.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(store, "DATA_DIR", root)
    monkeypatch.setattr(main, "DATA_DIR", root)
    # Do not shut down the process-wide executor between TestClient contexts.
    monkeypatch.setattr(jobs, "shutdown", lambda: None)
    return root


@pytest.fixture
def client(isolated_data):
    with TestClient(main.app) as connection:
        yield connection


@pytest.fixture
def ready_match(client, isolated_data):
    response = client.post("/api/matches", json={"title": "Uji futsal", "home_name": "Hijau", "away_name": "Kuning"})
    assert response.status_code == 201, response.text
    match = response.json()
    directory = config.match_dir(match["id"])
    video = directory / "range-fixture.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    with store.edit(match["id"], invalidate=False) as document:
        document["match"].update(duration=120.0, width=1280, height=720, fps=30, video_path=str(video), video_url=f'/api/matches/{match["id"]}/video', status="ready")
    return match


@pytest.fixture
def roster(client, ready_match):
    match_id = ready_match["id"]
    players = []
    for name, number, team in [("Ari", "7", "home"), ("Bima", "8", "home"), ("Candra", "9", "away")]:
        response = client.post(f"/api/matches/{match_id}/players", json={"name": name, "number": number, "team": team})
        assert response.status_code == 201, response.text
        players.append(response.json())
    return players
