import bisect
import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


def browser_video(source: Path, destination: Path):
    """Keep compatible originals; otherwise create an H.264/AAC browser proxy."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        process = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", str(source)], capture_output=True, text=True, timeout=30)
        try:
            stream = json.loads(process.stdout)["streams"][0]
            if source.suffix.lower() in (".mp4", ".m4v") and stream.get("codec_name") == "h264" and stream.get("pix_fmt") in ("yuv420p", "yuvj420p"):
                return source
        except (ValueError, IndexError, KeyError):
            pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("FFmpeg belum tersedia. Pasang FFmpeg untuk membuat video yang dapat diputar browser.")
    process = subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(destination)], capture_output=True, text=True, timeout=7200)
    if process.returncode:
        raise ValueError("Video tidak dapat dikonversi. Pastikan berkas video valid dan dapat diputar.")
    return destination


@lru_cache(maxsize=3)
def _read_frames(path_string: str, modified_ns: int):
    frames = []
    with open(path_string, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                frame = json.loads(line)
                if "time" in frame:
                    frames.append(frame)
    frames.sort(key=lambda frame: frame["time"])
    return frames, [frame["time"] for frame in frames]


def frames_for(document, time=None, window=0.3):
    path_string = document["match"].get("detections_path")
    if not path_string or not Path(path_string).is_file():
        return []
    path = Path(path_string)
    frames, times = _read_frames(str(path), path.stat().st_mtime_ns)
    if time is None:
        return frames
    start = bisect.bisect_left(times, time - window)
    stop = bisect.bisect_right(times, time + window)
    return frames[start:stop]


def assigned_frames(document, frames):
    tracks = {t["track_id"]: t for t in document["tracks"]}
    result = []
    for frame in frames:
        objects = []
        for detection in frame.get("objects", []):
            track = tracks.get(detection.get("track_id"), {}) if detection.get("kind") != "ball" else {}
            objects.append({**detection, "player_id": track.get("player_id"), "team": track.get("team", detection.get("team", "unknown")), "excluded": track.get("excluded", False), "kind": "other" if track.get("excluded") else detection.get("kind", "player")})
        result.append({"time": frame["time"], "objects": objects})
    return result


def trajectory_points(document):
    frames = frames_for(document)
    # Spatial output sampled at <= 1 Hz for players and every sampled frame for
    # the ball, capped for a bounded browser payload.
    sample_step = max(1, len(frames) // 12000)
    points = []
    last_time = -1
    tracks = {t["track_id"]: t for t in document["tracks"]}
    for frame in frames[::sample_step]:
        players_due = frame["time"] - last_time >= 0.9
        if players_due:
            last_time = frame["time"]
        for obj in frame.get("objects", []):
            box = obj["bbox"]
            if obj.get("kind") == "ball":
                points.append({"time": frame["time"], "track_id": None, "kind": "ball", "x": (box[0] + box[2]) / 2, "y": (box[1] + box[3]) / 2, "team": "unknown", "player_id": None})
                continue
            if obj.get("kind") != "player" or not players_due:
                continue
            track = tracks.get(obj.get("track_id"), {})
            if track.get("excluded") or obj.get("in_court") is False:
                continue
            points.append({"time": frame["time"], "track_id": obj["track_id"], "kind": "player", "x": (box[0] + box[2]) / 2, "y": box[3], "team": track.get("team", obj.get("team", "unknown")), "player_id": track.get("player_id")})
    return points


def court_points(document):
    """Only project observations inside manually validated time segments."""
    import cv2
    import numpy as np

    calibrations = document["calibrations"]
    points = trajectory_points(document)
    projected = []
    transforms = [(c, cv2.getPerspectiveTransform(np.asarray(c["image_points"], dtype=np.float32), np.asarray(c["court_points"], dtype=np.float32))) for c in calibrations]
    for point in points:
        calibration = next(((c, h) for c, h in transforms if c["start_time"] <= point["time"] <= c["end_time"]), None)
        if not calibration:
            continue
        segment, matrix = calibration
        image_point = np.asarray([[[point["x"], point["y"]]]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(image_point, matrix)[0, 0]
        if np.isfinite(x) and np.isfinite(y) and 0 <= x <= segment["court_length"] and 0 <= y <= segment["court_width"]:
            projected.append({**point, "x": round(float(x), 3), "y": round(float(y), 3), "calibration_id": segment["id"], "court_length": segment["court_length"], "court_width": segment["court_width"]})
    return projected
