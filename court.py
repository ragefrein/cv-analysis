"""Strict playable-floor geometry; colour holes must not break player tracks."""
from __future__ import annotations

import cv2
import numpy as np

COURT_LO = np.array((90, 55, 165), np.uint8)
COURT_HI = np.array((122, 255, 255), np.uint8)


def detect_court(frame):
    """Fill the largest blue FLOOR hull, never reuse another shot's floor.

    The hull fills jersey, shoe and centre-logo holes. Admission uses that
    geometry, not the colour of the pixel underneath an ankle. Missing floor
    always means an empty region, including closeups and crowd/bench shots.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COURT_LO, COURT_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty = np.zeros((h, w), np.uint8)
    if not contours:
        return empty, [], None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if not .035 * w * h < area < .97 * w * h:
        return empty, [], None
    hull = cv2.convexHull(contour)
    # A board/banner at the top of the frame is not a playing surface.
    x, y, bw, bh = cv2.boundingRect(hull)
    if y + bh < .55 * h or bw < .30 * w or bh < .15 * h:
        return empty, [], None
    polygon = cv2.approxPolyDP(hull, .002 * cv2.arcLength(hull, True), True)
    cv2.fillConvexPoly(empty, polygon, 255)
    return empty, polygon.reshape(-1, 2).tolist(), (x, y, bw, bh)


def point_on_court(mask, point):
    if point is None or not np.isfinite(point).all():
        return False
    x, y = map(lambda value: int(round(value)), point)
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])


def box_foot(box):
    return [(float(box[0]) + float(box[2])) / 2, float(box[3])]


def confirmed_foot(mask, box, pose=None):
    """Require observed ground contact; truncated bodies are not floor positions."""
    h, w = mask.shape
    x1, y1, x2, y2 = box
    if x2 - x1 <= 5 or y2 - y1 < 25:
        return None
    ankles = [] if pose is None or len(pose) < 17 else [
        p for p in pose[15:17] if len(p) >= 3 and p[2] >= .5
        and x1 <= p[0] <= x2 and y1 + .65 * (y2 - y1) <= p[1] <= y2 + 4
    ]
    if ankles:
        # An ankle joint is ABOVE the shoe's ground contact. Using ankle y
        # admits people standing on the red perimeter next to the blue floor.
        foot = [float(np.mean([p[0] for p in ankles])), float(y2 - 1)]
    else:
        if y2 >= h - 2 or x1 <= 1 or x2 >= w - 1:
            return None
        foot = box_foot(box)
    return foot if point_on_court(mask, foot) else None


class ShotBoundary:
    def __init__(self):
        self.gray = None
        self.histogram = None

    def update(self, frame):
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
        cut = False
        if self.gray is not None:
            difference = float(np.mean(cv2.absdiff(self.gray, gray)))
            distance = cv2.compareHist(self.histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)
            cut = (distance > .38 and difference > 28) or distance > .65
        self.gray, self.histogram = gray, histogram
        return cut


def boundary_mask(shape, polygon):
    polygon = np.asarray(polygon, np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 12:
        raise ValueError("Pilih 3–12 titik batas lapangan secara berurutan.")
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 100:
        raise ValueError("Batas harus berupa poligon cembung tanpa garis bersilangan.")
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    return mask


def apply_boundary(shape, polygon, segment, camera, boundaries):
    for boundary in reversed(boundaries or []):
        if boundary["segment"] == segment:
            matrix = np.asarray(camera) @ np.linalg.inv(np.asarray(boundary["camera"]))
            projected = cv2.perspectiveTransform(np.float32([boundary["polygon"]]), matrix)[0]
            mask = boundary_mask(shape, projected)
            return mask, projected.round(2).tolist(), cv2.boundingRect(projected)
    return None


def correct_boundary(tracks, frame_index, polygon):
    fr = tracks["frames"][frame_index]
    shape = (tracks["height"], tracks["width"])
    boundary_mask(shape, polygon)
    boundary = {"frame": frame_index, "segment": fr["segment"], "camera": fr["camera"], "polygon": polygon}
    boundaries = tracks.setdefault("floor_boundaries", [])
    boundaries[:] = [entry for entry in boundaries if entry["segment"] != fr["segment"]]
    boundaries.append(boundary)
    removed = 0
    for frame in tracks["frames"]:
        corrected = apply_boundary(shape, None, frame["segment"], frame["camera"], [boundary])
        if corrected is None:
            continue
        mask, frame["court"], _ = corrected
        admitted = []
        for player in frame["p"]:
            key = str(player[0])
            if point_on_court(mask, frame.get("feet", {}).get(key, box_foot(player[1:]))):
                admitted.append(player)
            else:
                removed += 1
                for field in ("feet", "poses", "anchors", "tracklets"):
                    frame.get(field, {}).pop(key, None)
        frame["p"] = admitted
    counts = {key: [] for key in tracks["players"]}
    for index, frame in enumerate(tracks["frames"]):
        for player in frame["p"]:
            counts[str(player[0])].append(index)
    for key, indices in counts.items():
        tracks["players"][key].update(observations=len(indices), first_seen=min(indices, default=-1),
                                      last_seen=max(indices, default=-1))
    return removed
