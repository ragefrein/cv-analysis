"""Opt-in end-to-end smoke: bundled public test image, actual YOLO, isolated DB.

This proves the processing plumbing, not accuracy on futsal footage.
Run from project root: .venv/Scripts/python.exe scripts/smoke.py
"""
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "test-results" / f"smoke-{uuid4().hex[:12]}"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ["FUTSAL_DATA_DIR"] = str(OUTPUT / "data")
os.environ["YOLO_CONFIG_DIR"] = str(OUTPUT / "ultralytics")
(OUTPUT / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"


def main():
    import av
    import cv2
    import ultralytics
    from fastapi.testclient import TestClient
    from backend.main import app

    asset = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    image = cv2.imread(str(asset))
    if image is None:
        raise RuntimeError("Aset uji bawaan Ultralytics tidak ditemukan.")
    image = cv2.resize(image, (480, 640))
    recording = OUTPUT / "inference-fixture.mp4"
    with av.open(str(recording), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 480, 640, "yuv420p"
        for _ in range(20):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    def check(response, status=200):
        assert response.status_code == status, f"HTTP {response.status_code}: {response.text}"
        return response.json()

    with TestClient(app) as client:
        health = check(client.get("/api/health"))
        assert health["ai"]["model_ready"]
        match = check(client.post("/api/matches", json={"title": "Infrastructure smoke — not futsal", "home_name": "Test home", "away_name": "Test away"}), 201)
        match_id = match["id"]
        base = f"/api/matches/{match_id}"
        with recording.open("rb") as handle:
            imported = check(client.post(base + "/video", files={"file": (recording.name, handle, "video/mp4")}))
        assert 1.9 <= imported["duration"] <= 2.1
        ranged = client.get(base + "/video", headers={"Range": "bytes=0-99"})
        assert ranged.status_code == 206 and len(ranged.content) == 100
        job = check(client.post(base + "/jobs", json={"sample_fps": 2, "confidence": 0.25, "detect_ball": True}), 202)
        started = time.monotonic()
        while True:
            detail = check(client.get(base))
            current = next(j for j in detail["jobs"] if j["id"] == job["id"])
            if current["status"] not in ("queued", "running"):
                break
            assert time.monotonic() - started < 180, "GPU processing did not finish within smoke timeout"
            time.sleep(0.1)
        assert current["status"] == "completed", current
        assert len(detail["tracks"]) >= 2, detail["tracks"]
        assert detail["stats"]["score"] == {"home": 0, "away": 0}
        observations = check(client.get(base + "/detections?time=0.5&window=0.6"))["frames"]
        assert any(f["objects"] for f in observations)
        assert all(0 <= v <= 1 for f in observations for o in f["objects"] for v in o["bbox"])
        players = [check(client.post(base + "/players", json={"name": f"Smoke player {i}", "number": str(i), "team": "home"}), 201) for i in (1, 2)]
        for track, player in zip(detail["tracks"][:2], players):
            check(client.patch(base + f'/tracks/{track["track_id"]}', json={"player_id": player["id"], "team": "home"}))
        check(client.post(base + "/events", json={"type": "pass", "time": 0.4, "team": "home", "player_id": players[0]["id"], "receiver_id": players[1]["id"], "outcome": "successful"}), 201)
        check(client.post(base + "/events", json={"type": "goal", "time": 1.4, "team": "home", "player_id": players[1]["id"]}), 201)
        stats = check(client.get(base + "/stats"))
        assert stats["score"] == {"home": 1, "away": 0}
        assert stats["teams"]["home"]["passes_successful"] == 1
        assert stats["teams"]["home"]["shots"] == 1
        assert stats["coverage"]["identity_seconds"] > 0
        assert len(stats["pass_network"]) == 1
        for format_name in ("json", "csv", "events", "report"):
            export = client.get(base + f"/export?format={format_name}")
            assert export.status_code == 200 and export.content
        calibration = check(client.post(base + "/calibrations", json={"start_time": 0, "end_time": 1.8, "image_points": [[0.01, 0.01], [0.99, 0.01], [0.99, 0.99], [0.01, 0.99]]}), 201)
        projection = check(client.get(base + "/trajectories?coordinates=court"))
        assert projection["points"] and projection["coordinate_system"] == "court"
        final = {"result": "passed", "fixture": "Ultralytics bundled bus.jpg; not a futsal accuracy benchmark", "device": current["metrics"].get("device"), "tracks": len(detail["tracks"]), "sampled_frames": current["metrics"].get("sampled_frames"), "processing_seconds": current["metrics"].get("processing_seconds"), "range_streaming": True, "identity_assignment": True, "events_stats_exports": True, "calibrated_projection": bool(projection["points"]), "output": str(OUTPUT)}
        (OUTPUT / "result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
