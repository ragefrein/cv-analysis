"""Futsal tracking pipeline: blue-court filter, stable player IDs, ball track."""
from __future__ import annotations

import os
from collections import defaultdict

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Blue floor is bright; jerseys/boards are darker, so value gate matters.
COURT_LO = np.array((95, 55, 165), np.uint8)
COURT_HI = np.array((122, 255, 255), np.uint8)

PLAYER_CONF = 0.22
IMG_SIZE = 960


def court_mask(frame: np.ndarray, prev: np.ndarray | None = None):
    """Largest bright-blue floor region. Returns (mask, polygon, bbox)."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COURT_LO, COURT_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[idx, cv2.CC_STAT_AREA]
        # Reject a component that swallowed most of the frame (usually ad boards).
        if 0.02 * w * h < area < 0.95 * w * h:
            mask = (labels == idx).astype(np.uint8) * 255
        else:
            mask = np.zeros((h, w), np.uint8)
    else:
        mask = np.zeros((h, w), np.uint8)

    if cv2.countNonZero(mask) < 0.02 * w * h:
        mask = prev.copy() if prev is not None else np.zeros((h, w), np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = []
    bbox = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        bbox = (x, y, bw, bh)
        eps = 0.01 * cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).tolist()
    return mask, polygon, bbox


def _point_in(bbox, x, y, pad_x=0.0, pad_top=0.0):
    x0, y0, w, h = bbox
    return (x0 - pad_x * w <= x <= x0 + w + pad_x * w) and (y0 - pad_top * h <= y <= y0 + h + h * 0.1)


def _ball_ok(box, shape, court_bbox):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    h, w = shape[:2]
    if not (4 <= bw <= 0.14 * w and 4 <= bh <= 0.14 * h):
        return False
    if not (0.45 <= bw / max(1.0, bh) <= 2.2):
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if court_bbox is not None and not _point_in(court_bbox, cx, cy, 0.18, 0.35):
        return False
    return True


def _norm_center(box, court_bbox, width, height):
    x0, y0, bw, bh = court_bbox if court_bbox else (0, 0, width, height)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return np.array([(cx - x0) / max(1, bw), (cy - y0) / max(1, bh)])


def _pick_ball(cands, court_bbox, width, height, state):
    """Pick the candidate that continues the ball path; reject teleports."""
    if not cands:
        state["miss"] += 1
        if state["miss"] > 4:
            state["pos"] = None
        return None

    if state["pos"] is None:
        conf, box = max(cands, key=lambda c: c[0])
        if conf < 0.20:
            return None
        state["pos"] = _norm_center(box, court_bbox, width, height)
        state["vel"] = np.zeros(2)
        state["miss"] = 0
        return box

    pred = state["pos"] + state["vel"]
    gate = 0.08 + 0.02 * min(state["miss"], 4)
    best = None
    for conf, box in cands:
        n = _norm_center(box, court_bbox, width, height)
        dist = float(np.linalg.norm(n - pred))
        score = dist / gate - 0.15 * conf
        if best is None or score < best[0]:
            best = (score, dist, conf, box, n)

    if best is None or best[1] > gate:
        state["miss"] += 1
        if state["miss"] > 4:
            state["pos"] = None
        return None

    _, _, _, box, n = best
    state["vel"] = 0.6 * state["vel"] + 0.4 * (n - state["pos"])
    state["pos"] = n
    state["miss"] = 0
    return box


def _interpolate_ball(frames, max_gap=5):
    """Fill very short detection gaps; never invent long trajectories."""
    prev = None
    for i, fr in enumerate(frames):
        if fr["b"] is not None:
            if prev is not None and 1 < i - prev <= max_gap:
                p, q = frames[prev]["b"], fr["b"]
                for j in range(prev + 1, i):
                    t = (j - prev) / (i - prev)
                    frames[j]["b"] = [p[k] + (q[k] - p[k]) * t for k in range(4)]
            prev = i


def analyze_video(path: str, progress=None, max_frames: int | None = None) -> dict:
    """Detect + track players on the blue court and the ball. Blocking call."""
    from boxmot import BotSort, BotSortConfig, ReIDConfig
    from ultralytics import YOLO

    os.chdir(BASE_DIR)  # local model + ReID weights resolve reliably

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Video tidak bisa dibuka.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if max_frames:
        total = min(total, max_frames)

    model = YOLO(os.path.join(BASE_DIR, "yolo11s.pt"))
    tracker = BotSort(
        config=BotSortConfig(
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.50,
            max_lost_frames=150,
            match_thresh=0.85,
            proximity_thresh=0.50,
            appearance_thresh=0.30,
            use_cmc=True,
            cmc_method="sof",
            use_embeddings=True,
        ),
        reid=ReIDConfig(model="osnet-x0-25-msmt17", device="0"),
    )

    frames = []
    prev_mask = None
    unique_ids = set()
    ball_frames = 0
    court_frames = 0
    ball_state = {"pos": None, "vel": np.zeros(2), "miss": 0}

    while len(frames) < total:
        ok, frame = cap.read()
        if not ok:
            break

        mask, polygon, bbox = court_mask(frame, prev_mask)
        prev_mask = mask
        if polygon:
            court_frames += 1

        result = model.predict(
            frame, classes=[0, 32], conf=PLAYER_CONF, iou=0.6,
            imgsz=IMG_SIZE, device=0, verbose=False,
        )[0]

        dets = []
        ball_cands = []
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(float, box)
            if int(cls) == 32:
                if _ball_ok(box, frame.shape, bbox):
                    ball_cands.append((float(conf), [x1, y1, x2, y2]))
                continue
            foot_x, foot_y = int((x1 + x2) / 2), int(y2)
            if y2 - y1 < 25:
                continue
            if not (0 <= foot_x < width and 0 <= foot_y < height):
                continue
            # The whole point: only feet inside the blue floor count.
            if polygon and mask[foot_y, foot_x] == 0:
                continue
            dets.append([x1, y1, x2, y2, float(conf), 0])

        ball = _pick_ball(ball_cands, bbox, width, height, ball_state)

        dets = np.asarray(dets, np.float32) if dets else np.empty((0, 6), np.float32)
        tracks = tracker.update(dets, frame)

        players = []
        if tracks is not None and len(tracks):
            for row in tracks:
                x1, y1, x2, y2, tid = row[:5]
                # Lost tracks coast for a while; keep showing them only while
                # their feet are still on the blue floor.
                fx, fy = int((x1 + x2) / 2), int(y2)
                if polygon and not (0 <= fx < width and 0 <= fy < height and mask[fy, fx]):
                    continue
                tid = int(tid)
                players.append([tid, int(x1), int(y1), int(x2), int(y2)])
                unique_ids.add(tid)

        if ball is not None:
            ball_frames += 1

        frames.append({
            "p": players,
            "b": [round(v, 1) for v in ball] if ball is not None else None,
            "court": polygon,
        })

        if progress and (len(frames) % 5 == 0 or len(frames) == total):
            progress(len(frames), total, f"Memproses frame {len(frames)}/{total}")

    cap.release()
    _interpolate_ball(frames)
    if progress:
        progress(total, total, "Selesai")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(frames),
        "frames": frames,
        "stats": {
            "unique_ids": len(unique_ids),
            "ball_frames": ball_frames,
            "court_frames": court_frames,
        },
    }


def id_color(tid: int):
    """Stable, high-contrast BGR per identity."""
    hue = int((tid * 47) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_overlay(frame, tracks, idx, show_court=False):
    fr = tracks["frames"][idx] if idx < len(tracks["frames"]) else {"p": [], "b": None, "court": []}
    out = frame.copy()

    if show_court and fr.get("court"):
        poly = np.asarray(fr["court"], np.int32)
        tint = out.copy()
        cv2.fillPoly(tint, [poly], (255, 180, 0))
        out = cv2.addWeighted(tint, 0.14, out, 0.86, 0)
        cv2.polylines(out, [poly], True, (255, 220, 60), 2)

    for tid, x1, y1, x2, y2 in fr["p"]:
        color = id_color(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"P{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ly = max(th + 8, y1)
        cv2.rectangle(out, (x1, ly - th - 8), (x1 + tw + 10, ly), color, -1)
        cv2.putText(out, label, (x1 + 5, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 15, 25), 2, cv2.LINE_AA)

    if fr["b"] is not None:
        trail = []
        j = idx
        while j >= 0 and idx - j <= 10:
            b = tracks["frames"][j]["b"]
            if b is None:
                break
            trail.append((int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)))
            j -= 1
        for k in range(1, len(trail)):
            a, c = trail[k], trail[k - 1]
            cv2.line(out, a, c, (0, 230, 255), max(1, 4 - k // 4))
        cx, cy = trail[0]
        cv2.circle(out, (cx, cy), 11, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 2, (0, 230, 255), -1, cv2.LINE_AA)

    return out
