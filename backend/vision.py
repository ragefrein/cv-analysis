"""Offline, uncertainty-aware video detection and conservative event proposals.

Only local model weights are used. A COCO person detection is not proof that a
person is a player, and a tracker ID is not a roster identity. Both are reviewed
in the application. Goal and shot recognition is intentionally not fabricated.

``on_progress(percent, stage)`` receives a float in the range 0..100. Cancellation
raises ``AnalysisCancelled`` and never leaves a completed output file behind.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FUTSAL_DATA_DIR", ROOT / "data")).resolve()
MODEL_PATH = Path(os.environ.get("FUTSAL_MODEL", ROOT / "models" / "yolo11n.pt")).resolve()
TRACKER_PATH = Path(__file__).with_name("tracking.yaml")


class AnalysisCancelled(RuntimeError):
    """The user cancelled a background analysis job."""


@lru_cache(maxsize=1)
def _device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def capabilities() -> dict[str, Any]:
    missing = [name for name in ("ultralytics", "torch", "cv2", "av") if importlib.util.find_spec(name) is None]
    ready = MODEL_PATH.is_file()
    available = not missing
    if missing:
        message = "Dependensi AI belum tersedia: " + ", ".join(missing) + ". Jalankan setup sesuai README."
    elif not ready:
        message = "Bobot model belum tersedia. Jalankan python scripts/setup_model.py."
    else:
        message = "Deteksi pemain/bola dan tracking siap. Identitas serta kandidat kejadian perlu review."
    return {
        "available": available,
        "model_ready": ready,
        "model": MODEL_PATH.name,
        "device": _device() if available else "cpu",
        "message": message,
    }


def probe_video(path: str | Path) -> dict[str, Any]:
    """Read stream metadata without decoding a whole match into memory."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("File tidak memiliki stream video.")
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
            # Some VFR muxers underreport stream.duration while the container
            # timeline retains the later presentation timestamps. Metadata is
            # a progress estimate, never a reason to stop decoding the stream.
            if container.duration is not None:
                duration = max(duration, float(container.duration / av.time_base))
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else 0.0
            width, height = stream.codec_context.width, stream.codec_context.height
            if width < 16 or height < 16:
                raise ValueError("Dimensi video tidak valid.")
            # Validate that at least one actual image can be decoded.
            first = next(container.decode(stream), None)
            if first is None:
                raise ValueError("Video tidak dapat didekode.")
            if duration <= 0 or not math.isfinite(duration):
                raise ValueError("Durasi video tidak dapat dibaca. Konversi ke MP4 lalu coba lagi.")
            return {
                "duration": round(duration, 6),
                "width": width,
                "height": height,
                "fps": round(fps, 6),
                "codec": stream.codec_context.name,
            }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("File bukan video yang dapat dibaca atau file rusak.") from exc


def make_poster(video_path: str | Path, poster_path: str | Path) -> None:
    import av
    import cv2

    target = Path(poster_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        frame = next(container.decode(video=0), None)
        if frame is None:
            raise ValueError("Tidak ada frame untuk membuat poster.")
        image = frame.to_ndarray(format="bgr24")
        if image.shape[1] > 1280:
            image = cv2.resize(image, (1280, round(image.shape[0] * 1280 / image.shape[1])))
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise ValueError("Poster video gagal disimpan.")


def _hex_rgb(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    value = value.lstrip("#")
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


class JerseyClassifier:
    """Small color heuristic. Unknown is preferable to forcing a team."""

    def __init__(self, home_color: str, away_color: str):
        import cv2
        import numpy as np

        self.targets = {}
        for team, value in (("home", home_color), ("away", away_color)):
            rgb = _hex_rgb(value)
            if rgb is not None:
                self.targets[team] = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        if len(self.targets) == 2:
            if float(np.linalg.norm(self.targets["home"] - self.targets["away"])) < 25:
                self.targets.clear()

    def predict(self, frame: Any, bbox: list[float]) -> str:
        import cv2
        import numpy as np

        if len(self.targets) != 2:
            return "unknown"
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        crop = frame[
            max(0, int((y1 + (y2 - y1) * 0.17) * height)):min(height, int((y1 + (y2 - y1) * 0.48) * height)),
            max(0, int((x1 + (x2 - x1) * 0.22) * width)):min(width, int((x1 + (x2 - x1) * 0.78) * width)),
        ]
        if crop.size < 48:
            return "unknown"
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        median = np.median(lab, axis=0)
        distances = sorted((float(np.linalg.norm(median - target)), team) for team, target in self.targets.items())
        if distances[0][0] > 65 or distances[1][0] - distances[0][0] < 14:
            return "unknown"
        return distances[0][1]


def _court_polygon(image: Any) -> Any:
    """Estimate the playable surface from the floor colour near the touchline.

    Advertising boards and the surrounding track usually differ from the floor,
    so the largest matching component is the court. A neutral or ambiguous floor
    returns None: unknown is safer than excluding real players.
    """
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    if width > 480:
        image = cv2.resize(image, (480, max(1, int(round(height * 480 / width)))), interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    patch = hsv[int(height * 0.70):int(height * 0.95), int(width * 0.30):int(width * 0.70)]
    if patch.size < 300:
        return None
    hue, saturation, value = (float(v) for v in np.median(patch.reshape(-1, 3), axis=0))
    if saturation < 20 or value < 30:
        return None
    lower = (int(hue - 12) % 180, max(20, int(saturation * 0.25)), max(20, int(value * 0.45)))
    upper = (int(hue + 12) % 180, 255, 255)
    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, lower[1], lower[2]), (upper[0], 255, 255)),
            cv2.inRange(hsv, (lower[0], lower[1], lower[2]), (179, 255, 255)),
        )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if len(stats) <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ratio = stats[index, cv2.CC_STAT_AREA] / float(width * height)
    if not 0.10 <= ratio <= 0.90:
        return None
    ys, xs = np.where(labels == index)
    hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
    if cv2.contourArea(hull) / float(width * height) < 0.08:
        return None
    polygon = hull.reshape(-1, 2).astype(np.float32)
    polygon[:, 0] /= width
    polygon[:, 1] /= height
    return polygon


class CourtRegion:
    """Manual calibration polygons, plus an automatic floor estimate.

    Manual segments always win. When no manual segment covers a timestamp, the
    latest automatic floor estimate is used if it is recent enough; otherwise
    the court is unknown and every person in frame is counted.
    """

    def __init__(self, calibrations: list[dict[str, Any]] | None, auto: bool = False):
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.segments = []
        for item in calibrations or []:
            points = item.get("image_points") or []
            if len(points) == 4:
                try:
                    polygon = np.asarray(points, dtype=np.float32)
                    self.segments.append((float(item["start_time"]), float(item["end_time"]), polygon))
                except (KeyError, TypeError, ValueError):
                    continue
        self.segments.sort(key=lambda segment: segment[0])
        self.manual_available = bool(self.segments)
        self.auto = bool(auto)
        self.auto_segments: list[tuple[float, float, Any]] = []
        self.next_auto = 0.0
        self.available = self.manual_available

    def update(self, image: Any, timestamp: float) -> None:
        if not self.auto or timestamp + 1e-6 < self.next_auto:
            return
        self.next_auto = timestamp + 1.0
        polygon = _court_polygon(image)
        if polygon is None:
            return
        self.auto_segments.append((timestamp, timestamp + 2.5, polygon))
        cutoff = timestamp - 6.0
        self.auto_segments = [segment for segment in self.auto_segments if segment[1] >= cutoff]
        self.available = True

    def contains(self, x: float, y: float, timestamp: float) -> bool | None:
        for start, end, polygon in self.segments:
            if start - 1e-6 <= timestamp <= end + 1e-6:
                return bool(self.cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0)
        for start, end, polygon in reversed(self.auto_segments):
            if start - 1e-6 <= timestamp <= end + 1e-6:
                return bool(self.cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0)
        return None


def _appearance(frame: Any, bbox: list[float]) -> Any:
    """Coarse upper-body signature used only to disambiguate a track merge."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    crop = frame[
        max(0, int((y1 + (y2 - y1) * 0.05) * height)):min(height, int((y1 + (y2 - y1) * 0.55) * height)),
        max(0, int((x1 + (x2 - x1) * 0.25) * width)):min(width, int((x1 + (x2 - x1) * 0.75) * width)),
    ]
    if crop.size < 128:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm > 0 else None


class IdentityStitcher:
    """Reuse a local track ID after a short loss only when unambiguous.

    BoT-SORT ReID handles occlusion. This adds one conservative rule for a
    tracker reset or a brief loss: exactly one recently lost track may be in the
    same place, otherwise a new ID is assigned. Same-team ambiguity always
    produces a new ID instead of a guess.
    """

    def __init__(self, gap: float = 2.5, distance: float = 0.12):
        self.gap = gap
        self.distance = distance
        self.raw: dict[int, tuple[int, float | None]] = {}
        self.recent: dict[int, dict[str, Any]] = {}
        self.next_id = 1

    def reset(self) -> None:
        # Tracker IDs restart after a scene cut; force a re-match by proximity.
        for raw_id, (local_id, _) in self.raw.items():
            self.raw[raw_id] = (local_id, None)

    def assign(self, raw_id: int, timestamp: float, foot: tuple[float, float], signature: Any) -> int:
        entry = self.raw.get(raw_id)
        if entry and entry[1] is not None:
            local_id = entry[0]
        else:
            local_id = self._match(timestamp, foot)
            self.raw[raw_id] = (local_id, timestamp)
        self.raw[raw_id] = (local_id, timestamp)
        state = self.recent.setdefault(local_id, {})
        state["last_time"] = timestamp
        state["foot"] = foot
        if signature is not None:
            previous = state.get("signature")
            state["signature"] = signature if previous is None else (previous * 0.7 + signature * 0.3)
        return local_id

    def _match(self, timestamp: float, foot: tuple[float, float]) -> int:
        candidates = []
        for local_id, state in self.recent.items():
            last_time = state.get("last_time")
            if last_time is None or not state.get("foot"):
                continue
            if not 0 <= timestamp - last_time <= self.gap:
                continue
            distance = math.hypot(foot[0] - state["foot"][0], foot[1] - state["foot"][1])
            if distance <= self.distance:
                candidates.append((distance, local_id))
        if len(candidates) == 1:
            return candidates[0][1]
        # More than one lost track nearby is ambiguous: never merge on color alone.
        local_id = self.next_id
        self.next_id += 1
        return local_id


def proposed_substitutions(tracks: list[dict[str, Any]], positions: dict[int, tuple[Any, Any]], duration: float) -> list[dict[str, Any]]:
    """Propose sideline exchanges; an outgoing then an incoming track, same team."""
    usable = [t for t in tracks if not t.get("excluded") and t.get("team") in ("home", "away") and t.get("detection_count", 0) >= 4]
    used: set[int] = set()
    events = []
    for outgoing in sorted(usable, key=lambda t: t["end_time"]):
        if outgoing["track_id"] in used:
            continue
        last_foot = (positions.get(outgoing["track_id"]) or (None, None))[1]
        if not last_foot or not (last_foot[0] <= 0.25 or last_foot[0] >= 0.75):
            continue
        options = []
        for incoming in usable:
            if incoming["track_id"] == outgoing["track_id"] or incoming["track_id"] in used:
                continue
            if incoming["team"] != outgoing["team"]:
                continue
            gap = incoming["start_time"] - outgoing["end_time"]
            if not 0 <= gap <= 12:
                continue
            first_foot = (positions.get(incoming["track_id"]) or (None, None))[0]
            if not first_foot:
                continue
            distance = math.hypot(first_foot[0] - last_foot[0], first_foot[1] - last_foot[1])
            if distance <= 0.45:
                options.append((gap, distance, incoming))
        if not options:
            continue
        options.sort(key=lambda item: (item[0], item[1]))
        _, _, incoming = options[0]
        used.update((outgoing["track_id"], incoming["track_id"]))
        if outgoing["end_time"] > duration + 0.05:
            continue
        events.append({
            "type": "substitution", "time": round(outgoing["end_time"], 4),
            "end_time": round(min(incoming["start_time"], duration), 4), "team": outgoing["team"],
            "track_id": outgoing["track_id"], "receiver_track_id": incoming["track_id"],
            "outcome": "unknown", "confidence": 0.4,
            "notes": "Kandidat pergantian: track keluar dan track masuk satu tim berurutan di pinggir lapangan. Hubungkan kedua track ke pemain yang benar sebelum konfirmasi.",
        })
    return events


def _owner(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Require one unambiguous player close to the ball's lower-image contact."""
    balls = [obj for obj in objects if obj["kind"] == "ball" and obj.get("confidence", 0) >= 0.20]
    if len(balls) != 1:
        return None
    ball = balls[0]
    bx1, by1, bx2, by2 = ball["bbox"]
    bx, by = (bx1 + bx2) / 2, (by1 + by2) / 2
    candidates = []
    for obj in objects:
        if obj["kind"] != "player" or obj.get("track_id") is None:
            continue
        x1, y1, x2, y2 = obj["bbox"]
        player_height, player_width = y2 - y1, x2 - x1
        if player_height <= 0 or player_width <= 0:
            continue
        foot_x, foot_y = (x1 + x2) / 2, y2
        dx, dy = abs(bx - foot_x), abs(by - foot_y)
        if dx > player_width * 0.85 or dy > player_height * 0.20:
            continue
        distance = math.hypot(dx / max(player_width, 0.02), dy / max(player_height, 0.02))
        candidates.append((distance, obj))
    candidates.sort(key=lambda item: item[0])
    if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.22):
        return None
    return candidates[0][1]


class EventProposer:
    """Conservative contact/ownership heuristic, producing review proposals only.

    No contact or intention is asserted as ground truth. Long invisible gaps,
    ambiguous player proximity, and camera cuts break candidate continuity.
    """

    def __init__(self, sample_fps: float):
        self.step = 1.0 / sample_fps
        self.max_gap = max(0.45, self.step * 1.6)
        self.events: list[dict[str, Any]] = []
        self.possible: dict[str, Any] | None = None
        self.possible_since = 0.0
        self.possible_count = 0
        self.active: dict[str, Any] | None = None
        self.active_start = 0.0
        self.active_last = 0.0
        self.previous: dict[str, Any] | None = None
        self.last_ball_time: float | None = None
        self.last_time: float | None = None

    def _close(self) -> None:
        if self.active is None:
            return
        duration = self.active_last - self.active_start
        if duration >= max(0.2, self.step * 0.8) and self.active.get("team") in ("home", "away"):
            self.events.append({
                "type": "possession", "time": round(self.active_start, 4),
                "end_time": round(self.active_last, 4), "team": self.active["team"],
                "track_id": self.active["track_id"], "outcome": "unknown",
                "confidence": round(min(float(self.active.get("confidence", 0.5)), 0.65), 3),
                "notes": "Kandidat interval kontrol bola dari kedekatan kaki dan bola; perlu review. Tidak mencakup bola di luar frame.",
            })
        self.previous = {**self.active, "time": self.active_last}
        self.active = None

    def reset(self) -> None:
        self._close()
        self.possible = None
        self.possible_count = 0
        self.previous = None
        self.last_ball_time = None
        self.last_time = None

    def consume(self, timestamp: float, objects: list[dict[str, Any]]) -> None:
        # PTS jumps and sampling gaps cannot be bridged as evidence of a pass.
        if self.last_time is not None and timestamp - self.last_time > self.max_gap:
            self.reset()
        self.last_time = timestamp
        balls = [obj for obj in objects if obj["kind"] == "ball"]
        if len(balls) == 1:
            if self.last_ball_time is not None and timestamp - self.last_ball_time > self.max_gap:
                self._close()
                self.previous = None
                self.possible = None
                self.possible_count = 0
            self.last_ball_time = timestamp
        elif self.last_ball_time is not None and timestamp - self.last_ball_time > self.max_gap:
            self._close()
            self.previous = None
            self.possible = None
            self.possible_count = 0

        owner = _owner(objects)
        if owner is None:
            self._close()
            self.possible = None
            self.possible_count = 0
            return
        if self.active and self.active["track_id"] == owner["track_id"]:
            self.active_last = timestamp
            self.active = owner
            return
        self._close()
        if self.possible and self.possible["track_id"] == owner["track_id"]:
            self.possible_count += 1
        else:
            self.possible = owner
            self.possible_since = timestamp
            self.possible_count = 1
        if self.possible_count < 2 or timestamp - self.possible_since < 0.12:
            return

        self.active = owner
        self.active_start = self.possible_since
        self.active_last = timestamp
        self.possible = None
        self.possible_count = 0
        previous = self.previous
        self.previous = None
        if (
            previous is not None
            and previous["track_id"] != owner["track_id"]
            and 0 < self.active_start - previous["time"] <= 1.8
            and previous.get("team") in ("home", "away")
            and owner.get("team") == previous.get("team")
            and self.last_ball_time is not None
            and timestamp - self.last_ball_time <= self.max_gap
        ):
            self.events.append({
                "type": "pass", "time": round(previous["time"], 4),
                "end_time": round(self.active_start, 4), "team": owner["team"],
                "track_id": previous["track_id"], "receiver_track_id": owner["track_id"],
                "outcome": "successful", "confidence": round(min(
                    float(owner.get("confidence", 0.5)), float(previous.get("confidence", 0.5)), 0.60
                ), 3),
                "notes": "Kandidat passing: kontrol dekat bola berpindah di antara dua track satu tim. Konfirmasi aksi, pengirim, penerima, dan hasilnya dari video.",
            })

    def finish(self) -> list[dict[str, Any]]:
        self._close()
        return sorted(self.events, key=lambda event: (event["time"], event["type"]))


def _cut_score(previous: Any, current: Any) -> bool:
    import cv2
    import numpy as np

    old = cv2.resize(previous, (64, 36))
    new = cv2.resize(current, (64, 36))
    difference = float(np.mean(cv2.absdiff(old, new)))
    histograms = []
    for image in (old, new):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram)
        histograms.append(histogram)
    correlation = cv2.compareHist(histograms[0], histograms[1], cv2.HISTCMP_CORREL)
    return difference > 40 and correlation < 0.30


def analyze_video(
    video_path: str | Path,
    output_path: str | Path,
    options: dict[str, Any],
    on_progress: Callable[[float, str], None],
    is_cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Run real local YOLO inference, writing one sampled PTS frame per line.

    Sampling reduces CPU/GPU load but does not claim every contact is observed.
    ``max_duration`` is a bounded smoke-test option, not the normal match path.
    """
    state = capabilities()
    if not state["available"] or not state["model_ready"]:
        raise RuntimeError(state["message"])
    if is_cancelled():
        raise AnalysisCancelled("Analisis dibatalkan.")
    sample_fps = float(options.get("sample_fps", 5))
    confidence = float(options.get("confidence", 0.25))
    if not 1 <= sample_fps <= 15 or not 0.05 <= confidence <= 0.95:
        raise ValueError("sample_fps harus 1–15 dan confidence 0.05–0.95.")
    # Ultralytics tests writeability before creating its nested directory. Make
    # both paths first so Windows cannot fall back to a global /tmp directory.
    # Honor an explicit config override; the default follows the isolated data
    # root used by the API and test suite.
    config_directory = Path(os.environ.get("YOLO_CONFIG_DIR", DATA_DIR / ".ultralytics")).resolve()
    config_directory.mkdir(parents=True, exist_ok=True)
    (config_directory / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("YOLO_VERBOSE", "false")
    import av
    import numpy as np
    import torch
    from ultralytics import YOLO

    metadata = probe_video(video_path)
    duration = metadata["duration"]
    maximum_duration = float(options["max_duration"]) if options.get("max_duration") is not None else None
    if maximum_duration is not None:
        if maximum_duration <= 0 or not math.isfinite(maximum_duration):
            raise ValueError("max_duration harus lebih besar dari nol.")
        duration = min(duration, maximum_duration)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    started = time.monotonic()
    on_progress(1, "Memuat model lokal dan tracker kamera bergerak")
    # Avoid loading untrusted user-supplied .pt files: MODEL_PATH is configured
    # by the local operator, never obtained from video/upload metadata.
    model = YOLO(str(MODEL_PATH), task="detect")
    jersey = JerseyClassifier(options.get("home_color", "#10b981"), options.get("away_color", "#f97316"))
    court = CourtRegion(options.get("calibrations") or [], auto=bool(options.get("auto_court", True)))
    stitcher = IdentityStitcher()
    proposer = EventProposer(sample_fps)
    summaries: dict[int, dict[str, Any]] = {}
    positions: dict[int, tuple[Any, Any]] = {}
    next_time = 0.0
    first_pts: float | None = None
    previous_frame = None
    sampled = 0
    ball_frames = 0
    jersey_frames = 0
    scene_cuts = 0
    last_sample_time = 0.0
    last_callback = 0.0
    step = 1.0 / sample_fps
    decode_index = 0
    last_decoded_end = 0.0
    try:
        with av.open(str(video_path)) as container, partial.open("w", encoding="utf-8") as output:
            stream = container.streams.video[0]
            for decoded in container.decode(stream):
                if is_cancelled():
                    raise AnalysisCancelled("Analisis dibatalkan.")
                # Actual presentation timestamps preserve variable-frame-rate
                # spacing. Missing PTS falls back only for that frame.
                pts = float(decoded.time) if decoded.time is not None else decode_index / max(metadata["fps"], 1)
                decode_index += 1
                if first_pts is None:
                    first_pts = pts
                timestamp = max(0.0, pts - first_pts)
                if maximum_duration is not None and timestamp > maximum_duration + 1e-5:
                    break
                frame_duration = getattr(decoded, "duration", None)
                frame_seconds = float(frame_duration * decoded.time_base) if frame_duration and decoded.time_base else 1.0 / max(metadata["fps"], 1)
                last_decoded_end = max(last_decoded_end, timestamp + frame_seconds)
                if timestamp + 1e-5 < next_time:
                    continue
                next_time = timestamp + step
                image = decoded.to_ndarray(format="bgr24")
                court.update(image, timestamp)
                if previous_frame is not None and _cut_score(previous_frame, image):
                    scene_cuts += 1
                    proposer.reset()
                    stitcher.reset()
                    predictor = getattr(model, "predictor", None)
                    for tracker in getattr(predictor, "trackers", ()):
                        tracker.reset()
                previous_frame = image
                result = model.track(
                    image, persist=True, tracker=str(TRACKER_PATH),
                    classes=[0, 32] if options.get("detect_ball", True) else [0],
                    conf=confidence, iou=0.6, imgsz=int(options.get("image_size", 960)),
                    device=state["device"], verbose=False, save=False, stream=False,
                )[0]
                height, width = image.shape[:2]
                objects = []
                if result.boxes is not None and len(result.boxes):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
                    for box, score, label, raw_id in zip(boxes, scores, classes, ids):
                        normalized = np.clip(box / np.array([width, height, width, height]), 0, 1).tolist()
                        normalized = [round(float(value), 6) for value in normalized]
                        kind = "ball" if int(label) == 32 else "player"
                        # Large sports-ball predictions and tiny people are
                        # unreliable for this wide-view baseline.
                        if kind == "ball" and ((normalized[2] - normalized[0]) > 0.15 or (normalized[3] - normalized[1]) > 0.2):
                            continue
                        track_id = None
                        team = "unknown"
                        inside = None
                        if kind == "player" and raw_id is not None:
                            raw_id = int(raw_id)
                            foot = ((normalized[0] + normalized[2]) / 2, normalized[3])
                            track_id = stitcher.assign(raw_id, timestamp, foot, _appearance(image, normalized))
                            inside = court.contains(foot[0], foot[1], timestamp)
                            summary = summaries.setdefault(track_id, {
                                "track_id": track_id, "start_time": round(timestamp, 4),
                                "end_time": round(timestamp, 4), "detection_count": 0,
                                "_confidence_sum": 0.0, "_votes": Counter(),
                                "_court_known": 0, "_court_inside": 0,
                            })
                            summary["end_time"] = round(timestamp, 4)
                            summary["detection_count"] += 1
                            summary["_confidence_sum"] += float(score)
                            if inside is not None:
                                summary["_court_known"] += 1
                                if inside:
                                    summary["_court_inside"] += 1
                            first, _ = positions.get(track_id, (None, None))
                            positions[track_id] = (first or foot, foot)
                            predicted_team = jersey.predict(image, normalized)
                            if predicted_team != "unknown":
                                summary["_votes"][predicted_team] += 1
                            votes = summary["_votes"]
                            if votes:
                                winner, count = votes.most_common(1)[0]
                                if count >= 2 and count / sum(votes.values()) >= 0.7:
                                    team = winner
                        objects.append({
                            "track_id": track_id, "kind": kind, "bbox": normalized,
                            "confidence": round(float(score), 5), "team": team,
                            "player_id": None, "in_court": inside,
                        })
                output.write(json.dumps({"time": round(timestamp, 6), "objects": objects}, separators=(",", ":")) + "\n")
                proposer.consume(timestamp, objects)
                sampled += 1
                ball_frames += any(obj["kind"] == "ball" for obj in objects)
                jersey_frames += any(obj["team"] != "unknown" for obj in objects)
                last_sample_time = timestamp
                now = time.monotonic()
                if now - last_callback >= 0.4:
                    on_progress(min(97.0, 2 + timestamp / duration * 94), f"Mendeteksi dan melacak · {timestamp:.0f}/{duration:.0f} detik")
                    last_callback = now
            if sampled == 0:
                raise ValueError("Tidak ada frame yang dapat dianalisis.")
            if is_cancelled():
                raise AnalysisCancelled("Analisis dibatalkan.")
        tracks = []
        for summary in summaries.values():
            votes = summary.pop("_votes")
            known = summary.pop("_court_known", 0)
            inside = summary.pop("_court_inside", 0)
            team = "unknown"
            if votes:
                winner, count = votes.most_common(1)[0]
                if count >= 2 and count / sum(votes.values()) >= 0.7:
                    team = winner
            ratio = inside / known if known else None
            # Automatic estimates can be approximate, so they need stronger
            # evidence before removing a track; short tracks are never excluded.
            threshold = 0.5 if court.manual_available else 0.35
            outside = bool(court.available and ratio is not None and summary["detection_count"] >= 8 and ratio < threshold)
            tracks.append({
                **summary, "team": team,
                "confidence": round(summary["_confidence_sum"] / summary["detection_count"], 5),
                "in_court_ratio": round(ratio, 3) if ratio is not None else None,
                "excluded": outside, "excluded_reason": "outside_court" if outside else None,
            })
            tracks[-1].pop("_confidence_sum")
        elapsed = time.monotonic() - started
        # Finish the real decoded timeline even when stream/container duration
        # was underreported. Explicit bounded smoke tests retain their limit.
        analyzed_seconds = last_decoded_end
        if maximum_duration is not None:
            analyzed_seconds = min(analyzed_seconds, maximum_duration)
        metrics = {
            "analyzed_seconds": round(analyzed_seconds, 3),
            "decoded_duration": round(last_decoded_end, 3),
            "ball_visible_seconds": round(min(analyzed_seconds, ball_frames * step), 3),
            "identity_seconds": 0,
            "team_visible_seconds": round(min(analyzed_seconds, jersey_frames * step), 3),
            "sampled_frames": sampled, "sample_fps": sample_fps,
            "scene_cuts": scene_cuts, "processing_seconds": round(elapsed, 3),
            "model": MODEL_PATH.name, "device": state["device"],
            "tracker": "BoT-SORT sparseOptFlow; appearance ReID on",
            "court_filtered_tracks": sum(1 for track in tracks if track.get("excluded")),
            "limitations": [
                "Model COCO umum; belum fine-tuned pada video futsal pengguna.",
                "Deteksi orang dapat mencakup wasit/official. Tim hanya heuristik warna jersey dan perlu review.",
                "Track ID bukan identitas pemain. Keluar-masuk frame dapat menghasilkan track baru; ID lokal hanya dipakai ulang bila satu kandidat jelas.",
                "Filter di luar lapangan memakai kalibrasi manual; tanpa kalibrasi dipakai estimasi warna permukaan lapangan dan hasilnya bisa tidak tersedia bila warna menyatu dengan sekitar.",
                "Pergantian pemain hanya kandidat dari track keluar-masuk di pinggir lapangan dan harus dihubungkan ke pemain.",
                "Kontrol bola dan passing adalah kandidat berbasis kedekatan gambar, bukan bukti kontak kaki.",
                "Sampling dapat melewatkan one-touch pass, bola blur, atau kejadian singkat.",
                "Shot, gol, assist, dan kejadian tak terlihat perlu anotasi manual; tidak ada tebakan gol otomatis.",
                "Belum ada kalibrasi dinamis otomatis atau pengukuran jarak/kecepatan fisik.",
            ],
        }
        events = proposer.finish()
        events.extend(proposed_substitutions(tracks, positions, analyzed_seconds))
        events.sort(key=lambda event: (event["time"], event["type"]))
        partial.replace(target)
        on_progress(99, "Menyimpan track dan kandidat kejadian untuk review")
        return {"tracks": tracks, "events": events, "metrics": metrics}
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        del model
        if state["device"].startswith("cuda"):
            torch.cuda.empty_cache()


def create_demo_video(path: str | Path) -> dict[str, Any]:
    """Create a clearly labelled, entirely synthetic 30-second court recording.

    This is a UI fixture, never an example of detector accuracy. No third-party
    video or player footage is downloaded.
    """
    import av
    import cv2
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps, duration = 1280, 720, 24, 30
    home = [(0.22, 0.50), (0.35, 0.28), (0.39, 0.70), (0.49, 0.44), (0.12, 0.50)]
    away = [(0.72, 0.50), (0.62, 0.27), (0.66, 0.72), (0.55, 0.58), (0.88, 0.50)]
    try:
        with av.open(str(target), "w") as output:
            stream = output.add_stream("libx264", rate=fps)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "25", "preset": "veryfast"}
            for index in range(duration * fps):
                timestamp = index / fps
                frame = np.full((height, width, 3), (50, 89, 71), dtype=np.uint8)
                frame[70:650, 50:1230] = (62, 112, 89)
                for stripe in range(10):
                    if stripe % 2 == 0:
                        cv2.rectangle(frame, (50 + stripe * 118, 70), (50 + (stripe + 1) * 118, 650), (66, 118, 94), -1)
                white = (202, 228, 216)
                cv2.rectangle(frame, (50, 70), (1230, 650), white, 2)
                cv2.line(frame, (640, 70), (640, 650), white, 2)
                cv2.circle(frame, (640, 360), 90, white, 2)
                cv2.circle(frame, (640, 360), 4, white, -1)
                cv2.rectangle(frame, (50, 225), (210, 495), white, 2)
                cv2.rectangle(frame, (1070, 225), (1230, 495), white, 2)
                cv2.rectangle(frame, (25, 310), (50, 410), white, 2)
                cv2.rectangle(frame, (1230, 310), (1255, 410), white, 2)
                for team_index, positions in enumerate((home, away)):
                    color = (153, 185, 16) if team_index == 0 else (22, 115, 249)
                    for player_index, (x, y) in enumerate(positions):
                        px = int(width * (x + 0.025 * math.sin(timestamp * 0.6 + player_index + team_index)))
                        py = int(height * (y + 0.04 * math.cos(timestamp * 0.45 + player_index)))
                        cv2.ellipse(frame, (px + 4, py + 8), (17, 9), 0, 0, 360, (35, 62, 49), -1)
                        cv2.circle(frame, (px, py), 16, (28, 45, 35), -1)
                        cv2.circle(frame, (px, py), 13, color, -1)
                        label = str((player_index + 1) if team_index == 0 else (player_index + 6))
                        cv2.putText(frame, label, (px - 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 250, 247), 1, cv2.LINE_AA)
                bx = int(width * (0.45 + 0.20 * math.sin(timestamp * 0.4)))
                by = int(height * (0.48 + 0.17 * math.sin(timestamp * 0.65)))
                cv2.circle(frame, (bx, by), 7, (25, 35, 29), -1)
                cv2.circle(frame, (bx, by), 5, (247, 249, 244), -1)
                cv2.rectangle(frame, (0, 0), (width, 51), (25, 39, 31), -1)
                cv2.putText(frame, "FUTSAL VISION  /  SIMULASI - BUKAN HASIL AI", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (224, 240, 231), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{int(timestamp) // 60:02d}:{int(timestamp) % 60:02d}", (1160, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (224, 240, 231), 1, cv2.LINE_AA)
                image = av.VideoFrame.from_ndarray(frame, format="bgr24")
                for packet in stream.encode(image):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    # The browser can request the entire file when faststart remux is absent;
    # FFmpeg makes scrubbing/download behavior more efficient when installed.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        fast = target.with_name(target.stem + ".fast.mp4")
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(target), "-c", "copy", "-movflags", "+faststart", str(fast)],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0:
            fast.replace(target)
        else:
            fast.unlink(missing_ok=True)
    return probe_video(target)
