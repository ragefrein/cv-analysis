"""Futsal detection, pose, persistent player identities, teams and court geometry."""
from __future__ import annotations

import os

import cv2
import numpy as np

from players import PlayerMemory, TEAM_COLORS, jersey_feature, player_label, player_team, portrait
from tactics import CameraMotion
from court import ShotBoundary, apply_boundary, box_foot, confirmed_foot, detect_court
from identity_review import reconcile_tracklets, visible_players

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PLAYER_CONF = 0.22
IMG_SIZE = 960


def court_mask(frame: np.ndarray, prev: np.ndarray | None = None):
    """Compatibility entry point. Missing floor is empty, never a stale mask."""
    return detect_court(frame)


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


def match_poses(boxes, pose_boxes, keypoints):
    """One-to-one IoU association; a skeleton cannot be shared by two players."""
    from scipy.optimize import linear_sum_assignment

    if not len(boxes) or not len(pose_boxes):
        return {}
    boxes, pose_boxes = np.asarray(boxes), np.asarray(pose_boxes)
    low = np.maximum(boxes[:, None, :2], pose_boxes[None, :, :2])
    high = np.minimum(boxes[:, None, 2:], pose_boxes[None, :, 2:])
    intersection = np.prod(np.maximum(0, high - low), axis=2)
    area_a = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    area_b = np.prod(np.maximum(0, pose_boxes[:, 2:] - pose_boxes[:, :2]), axis=1)
    ious = intersection / np.maximum(1, area_a[:, None] + area_b[None, :] - intersection)
    rows, cols = linear_sum_assignment(1 - ious)
    return {int(i): np.asarray(keypoints[j]).round(2).tolist()
            for i, j in zip(rows, cols) if ious[i, j] >= .3}


def foot_point(box, pose):
    fallback = [(float(box[0]) + float(box[2])) / 2, float(box[3])]
    if pose is None or len(pose) < 17:
        return fallback
    ankles = [p for p in pose[15:17] if len(p) >= 3 and p[2] >= .4 and p[0] > 0 and p[1] > 0]
    if not ankles:
        return fallback
    x, y = float(np.mean([p[0] for p in ankles])), float(max(p[1] for p in ankles))
    if box[0] <= x <= box[2] and box[1] + .65 * (box[3] - box[1]) <= y <= box[3] + 5:
        return [round(x, 2), round(y, 2)]
    return fallback


def estimate_poses(model, frame, boxes, device):
    result = model.predict(frame, conf=.20, iou=.6, imgsz=IMG_SIZE, device=device, verbose=False)[0]
    poses = {}
    if result.keypoints is not None:
        poses = match_poses(boxes, result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy())
    # Distant players can be missed by whole-frame pose detection. Infer their
    # padded crops at a larger relative scale; keep genuine confident joints only.
    crops, metadata = [], []
    h, w = frame.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if i in poses:
            continue
        bw, bh = x2 - x1, y2 - y1
        left, top = max(0, int(x1 - bw * .15)), max(0, int(y1 - bh * .08))
        right, bottom = min(w, int(x2 + bw * .15)), min(h, int(y2 + bh * .08))
        crop = frame[top:bottom, left:right]
        if crop.size:
            crops.append(crop)
            metadata.append((i, left, top, [x1 - left, y1 - top, x2 - left, y2 - top]))
    if crops:
        results = model.predict(crops, conf=.20, iou=.6, imgsz=384, rect=False,
                                device=device, verbose=False)
        for result, (i, left, top, box) in zip(results, metadata):
            if result.keypoints is None:
                continue
            match = match_poses([box], result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy())
            if 0 in match:
                points = np.asarray(match[0])
                valid = (points[:, 0] > 0) & (points[:, 1] > 0)
                points[valid, :2] += np.asarray([left, top])
                poses[i] = points.round(2).tolist()
    return poses


def active_embeddings(tracker):
    # BoxMOT 26 exports track.id. track.track_id is an inherited class default
    # (zero), so using it would silently attach every player's feature to P0.
    return {int(track.id): track.curr_feat for track in tracker.active_tracks
            if track.curr_feat is not None}


def analyze_video(path: str, progress=None, max_frames: int | None = None, cancelled=None,
                  initial_roster=None, floor_boundaries=None) -> dict:
    """Blocking analysis. Player galleries live for the entire video, not 150 frames."""
    from boxmot import BotSort, BotSortConfig, ReIDConfig
    from ultralytics import YOLO
    import torch

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Video tidak bisa dibuka.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if max_frames:
        total = min(total, max_frames)

    try:
        return _analyze_capture(cap, width, height, fps, total, progress, cancelled, YOLO,
                                BotSort, BotSortConfig, ReIDConfig, torch.cuda.is_available(), initial_roster, floor_boundaries)
    finally:
        cap.release()


def _analyze_capture(cap, width, height, fps, total, progress, cancelled,
                     YOLO, BotSort, BotSortConfig, ReIDConfig, cuda, initial_roster=None, floor_boundaries=None):
    pose_path = os.path.join(BASE_DIR, "models", "yolo11s-pose.pt")
    if not os.path.isfile(pose_path):
        raise RuntimeError("Model skeleton belum ada. Jalankan: python download_models.py")
    if progress:
        progress(0, total, "Memuat deteksi, skeleton, dan memori pemain…")
    device = 0 if cuda else "cpu"
    model = YOLO(os.path.join(BASE_DIR, "yolo11s.pt"))
    pose_model = YOLO(pose_path)
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
        reid=ReIDConfig(model=os.path.join(BASE_DIR, "models", "osnet_x0_25_msmt17.pt"),
                        device=str(device), allow_download=False),
    )

    frames = []
    memory = PlayerMemory(initial_roster, reidentify=False)
    camera = CameraMotion()
    shot_detector = ShotBoundary()
    shot = 0
    court_frames = 0
    rejected_people = 0
    ball_state = {"pos": None, "vel": np.zeros(2), "miss": 0}

    while len(frames) < total:
        if cancelled and cancelled():
            raise InterruptedError("Analisis dibatalkan.")
        ok, frame = cap.read()
        if not ok:
            break

        if shot_detector.update(frame):
            tracker.reset()
            shot += 1
        mask, polygon, bbox = court_mask(frame)
        if polygon:
            court_frames += 1

        result = model.predict(
            frame, classes=[0, 32], conf=PLAYER_CONF, iou=0.6,
            imgsz=IMG_SIZE, device=device, verbose=False,
        )[0]

        all_people = []
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
            if y2 - y1 < 25:
                continue
            all_people.append([x1, y1, x2, y2, float(conf), 0])

        segment, transform = camera.update(frame, mask, [p[:4] for p in all_people])
        corrected = apply_boundary(frame.shape, polygon, segment, transform, floor_boundaries)
        if corrected is not None:
            mask, polygon, bbox = corrected
        candidates = []
        for person in all_people:
            # This is only a broad candidate gate. Actual admission below uses
            # observed ankles and requires a current validated floor region.
            if not polygon or cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(box_foot(person[:4])), True) < -12:
                rejected_people += 1
                continue
            candidates.append(person)

        candidate_poses = estimate_poses(pose_model, frame, [c[:4] for c in candidates], device) if candidates else {}
        dets, detection_poses, detection_feet = [], {}, {}
        for i, candidate in enumerate(candidates):
            foot = confirmed_foot(mask, candidate[:4], candidate_poses.get(i))
            # Keep near-boundary candidates inside the private tracker so a
            # one-frame admission failure cannot erase an otherwise valid ID.
            index = len(dets)
            dets.append(candidate)
            if foot is None:
                rejected_people += 1
                continue
            detection_feet[index] = foot
            if i in candidate_poses:
                detection_poses[index] = candidate_poses[i]

        ball = _pick_ball(ball_cands, bbox, width, height, ball_state)

        dets = np.asarray(dets, np.float32) if dets else np.empty((0, 6), np.float32)
        tracks = tracker.update(dets, frame)

        visible, poses, observed_feet = [], {}, {}
        if tracks is not None and len(tracks):
            for row in tracks:
                x1, y1, x2, y2, tid = row[:5]
                detection_index = int(row[7])
                if detection_index not in detection_feet:
                    continue
                i = len(visible)
                # Use the observed box/feet, not a coasting Kalman prediction.
                visible.append([int(tid), *map(float, dets[detection_index, :4])])
                observed_feet[i] = detection_feet[detection_index]
                if detection_index in detection_poses:
                    poses[i] = detection_poses[detection_index]
        features = active_embeddings(tracker)
        observations = []
        for i, (tid, x1, y1, x2, y2) in enumerate(visible):
            box = [x1, y1, x2, y2]
            # Avoid adding a mixed appearance crop during heavy overlap.
            clean = True
            for j, other in enumerate(visible):
                if i == j:
                    continue
                left, top = max(x1, other[1]), max(y1, other[2])
                right, bottom = min(x2, other[3]), min(y2, other[4])
                if max(0, right - left) * max(0, bottom - top) > .25 * (x2 - x1) * (y2 - y1):
                    clean = False
                    break
            key = shot * 1000000 + tid
            observations.append({"track_id": key, "embedding": features.get(tid) if clean else None,
                                 "jersey": jersey_feature(frame, box),
                                 "portrait": portrait(frame, box) if key not in memory.bindings else ""})
        public_ids = memory.assign(observations, len(frames))
        players, player_poses, feet, anchors, tracklet_ids = [], {}, {}, {}, {}
        inverse_camera = np.linalg.inv(np.asarray(transform))
        for i, (pid, player) in enumerate(zip(public_ids, visible)):
            players.append([pid, *[int(v) for v in player[1:]]])
            if i in poses:
                player_poses[str(pid)] = poses[i]
            feet[str(pid)] = observed_feet[i]
            anchor = cv2.perspectiveTransform(np.float32([[observed_feet[i]]]), inverse_camera)[0, 0]
            anchors[str(pid)] = (anchor / [width, height]).round(5).tolist()
            tracklet_ids[str(pid)] = shot * 1000000 + player[0]

        frames.append({
            "p": players,
            "b": [round(v, 1) for v in ball] if ball is not None else None,
            "court": polygon,
            "poses": player_poses,
            "feet": feet,
            "segment": segment,
            "camera": transform,
            "anchors": anchors,
            "tracklets": tracklet_ids,
            "shot": shot,
        })

        if progress and (len(frames) % 5 == 0 or len(frames) == total):
            progress(len(frames), total, f"Memproses frame {len(frames)}/{total}")

    _interpolate_ball(frames)
    roster = memory.finish()
    result = {
        "pipeline_version": 3,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(frames),
        "frames": frames,
        "players": roster,
        "calibrations": [],
        "floor_boundaries": floor_boundaries or [],
        "stats": {
            "unique_ids": len(roster),
            "ball_frames": sum(fr["b"] is not None for fr in frames),
            "court_frames": court_frames,
            "no_court_frames": len(frames) - court_frames,
            "rejected_people": rejected_people,
            "camera_cuts": shot,
        },
    }
    if progress:
        progress(total, total, "Mencocokkan fragmen dan menyiapkan tinjauan ID…")
    reconcile_tracklets(result)
    return result


def id_color(tid: int):
    """Stable, high-contrast BGR per identity."""
    hue = int((tid * 47) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


SKELETON = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
            (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4))


def draw_skeleton(image, pose, color, threshold=.35):
    def valid(i):
        return i < len(pose) and len(pose[i]) >= 3 and pose[i][2] >= threshold and pose[i][0] > 0 and pose[i][1] > 0

    for a, b in SKELETON:
        if valid(a) and valid(b):
            start, end = tuple(map(int, pose[a][:2])), tuple(map(int, pose[b][:2]))
            cv2.line(image, start, end, (15, 20, 28), 4, cv2.LINE_AA)
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)
    for i, point in enumerate(pose):
        if valid(i):
            cv2.circle(image, tuple(map(int, point[:2])), 3, (245, 250, 255), -1, cv2.LINE_AA)


def draw_overlay(frame, tracks, idx, show_court=False, show_skeleton=True, selected_id=None):
    fr = tracks["frames"][idx] if idx < len(tracks["frames"]) else {"p": [], "b": None, "court": []}
    out = frame.copy()

    if show_court and fr.get("court"):
        poly = np.asarray(fr["court"], np.int32)
        tint = out.copy()
        cv2.fillPoly(tint, [poly], (255, 180, 0))
        out = cv2.addWeighted(tint, 0.14, out, 0.86, 0)
        cv2.polylines(out, [poly], True, (255, 220, 60), 2)

    players = visible_players(tracks, idx) if idx < len(tracks["frames"]) else []
    for tid, x1, y1, x2, y2 in players:
        color = TEAM_COLORS.get(player_team(tracks, tid), TEAM_COLORS[-1])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if tid == selected_id else 2)
        if show_skeleton:
            draw_skeleton(out, fr.get("poses", {}).get(str(tid), []), color)
        label = player_label(tracks, tid).replace(" · ", " | ")
        label = label.encode("ascii", "replace").decode("ascii")
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
