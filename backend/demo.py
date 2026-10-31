"""Explicit synthetic fixture for learning the UI; never a claimed AI result."""
import json
import math
import shutil

from . import store
from .config import match_dir


def create_demo():
    # Keep repeated button presses idempotent, and never replace edited sample data.
    existing = next((d for d in store.all_documents() if d["match"].get("is_demo") and d["match"].get("video_path")), None)
    if existing:
        return existing
    from .main import new_match
    from .schemas import MatchCreate
    from .vision import create_demo_video, make_poster

    document = new_match(MatchCreate(title="Garuda FC vs Orion · Simulasi", home_name="Garuda FC", away_name="Orion Futsal", home_color="#10b999", away_color="#f97316", venue="Lapangan simulasi", played_at="2026-10-31"), is_demo=True)
    match_id = document["match"]["id"]
    directory = match_dir(match_id)
    video_path = directory / "simulasi.mp4"
    poster_path = directory / "poster.jpg"
    try:
        metadata = create_demo_video(str(video_path))
        make_poster(str(video_path), str(poster_path))
        duration = metadata["duration"]
        names = [("Raka", "1"), ("Bima", "2"), ("Dimas", "3"), ("Arga", "4"), ("Fajar", "5"), ("Rizky", "6"), ("Aditya", "7"), ("Farhan", "8"), ("Reza", "9"), ("Naufal", "10")]
        players = [{"id": store.new_id(), "match_id": match_id, "name": name + " (demo)", "number": number, "team": "home" if index < 5 else "away", "role": "goalkeeper" if index in (4, 9) else "player"} for index, (name, number) in enumerate(names)]
        positions = [(0.22, 0.50), (0.35, 0.28), (0.39, 0.70), (0.49, 0.44), (0.12, 0.50), (0.72, 0.50), (0.62, 0.27), (0.66, 0.72), (0.55, 0.58), (0.88, 0.50)]
        tracks = [{"id": store.new_id(), "match_id": match_id, "track_id": index + 1, "player_id": player["id"], "team": player["team"], "start_time": 0, "end_time": duration, "detection_count": int(duration * 5), "confidence": 1.0} for index, player in enumerate(players)]
        frames_path = directory / "demo-detections.jsonl"
        with frames_path.open("w", encoding="utf-8") as handle:
            for index in range(int(duration * 5)):
                t = index / 5
                objects = []
                for player_index, (x, y) in enumerate(positions):
                    local_index, team_index = player_index % 5, player_index // 5
                    dx = 0.025 * math.sin(t * 0.6 + local_index + team_index)
                    dy = 0.04 * math.cos(t * 0.45 + local_index)
                    objects.append({"track_id": player_index + 1, "kind": "player", "bbox": [x + dx - 0.016, y + dy - 0.034, x + dx + 0.016, y + dy + 0.034], "confidence": 1, "team": players[player_index]["team"], "player_id": players[player_index]["id"]})
                ball_x, ball_y = 0.45 + 0.20 * math.sin(t * 0.4), 0.48 + 0.17 * math.sin(t * 0.65)
                objects.append({"track_id": None, "kind": "ball", "bbox": [ball_x - 0.006, ball_y - 0.009, ball_x + 0.006, ball_y + 0.009], "confidence": 1, "team": "unknown", "player_id": None})
                handle.write(json.dumps({"time": t, "objects": objects}) + "\n")
        events = []

        def event(kind, t, actor, receiver=None, outcome="unknown", status="confirmed", end_time=None):
            value = {"id": store.new_id(), "match_id": match_id, "type": kind, "time": t, "end_time": end_time, "team": players[actor]["team"], "player_id": players[actor]["id"], "receiver_id": players[receiver]["id"] if receiver is not None else None, "outcome": outcome, "status": status, "source": "demo", "confidence": None, "notes": "Fixture simulasi untuk mencoba aplikasi; bukan hasil deteksi AI.", "x": None, "y": None, "related_event_id": None}
            events.append(value)
            return value

        event("pass", 2.1, 0, 1, "successful")
        event("pass", 4.5, 1, 2, "successful")
        event("goal", 6.8, 2)
        event("assist", 6.5, 1)["related_event_id"] = events[-2]["id"]
        event("pass", 8.4, 5, 6, "successful")
        event("pass", 10.1, 6, outcome="failed")
        event("interception", 10.3, 3)
        event("shot", 12.2, 0, outcome="off_target")
        event("pass", 14.3, 7, 5, "successful")
        event("goal", 16.6, 5)
        event("pass", 19.0, 3, 0, "successful")
        event("goal", 22.4, 0)
        event("shot", 24.1, 6, outcome="on_target")
        event("save", 24.5, 4)
        event("pass", 26.2, 1, 3, "successful", "pending")
        event("pass", 27.4, 8, outcome="unknown", status="pending")
        event("possession", 0, 0, end_time=6.7)
        event("possession", 7.7, 5, end_time=10.2)
        event("possession", 10.3, 3, end_time=12.3)
        event("possession", 13.3, 7, end_time=16.7)
        event("possession", 18, 0, end_time=22.5)
        event("possession", 23.6, 6, end_time=24.6)
        with store.edit(match_id, invalidate=False) as current:
            current["match"].update(**{k: metadata[k] for k in ("duration", "width", "height", "fps")}, video_name="SIMULASI-futsal.mp4", video_path=str(video_path), poster_path=str(poster_path), detections_path=str(frames_path), video_url=f"/api/matches/{match_id}/video", poster_url=f"/api/matches/{match_id}/poster", status="review", notes="Seluruh data pertandingan ini sintetis. Jangan dipakai untuk menilai akurasi model.")
            current.update(players=players, tracks=tracks, events=events)
            current["jobs"].append({"id": store.new_id(), "match_id": match_id, "status": "completed", "progress": 100, "stage": "Fixture simulasi, bukan analisis model", "error": None, "created_at": store.now(), "updated_at": store.now(), "options": {}, "metrics": {"analyzed_seconds": duration, "ball_visible_seconds": duration, "identity_seconds": duration, "sampled_frames": int(duration * 5), "source": "synthetic_fixture"}})
        return current
    except Exception:
        store.delete(match_id)
        # Only fixture files created within this known generated match directory.
        base = (directory.parent).resolve()
        if directory.resolve().parent == base and directory.exists():
            shutil.rmtree(directory)
        raise
