"""Foot-point projection and a 2D futsal board with optional court calibration."""
from __future__ import annotations

import cv2
import numpy as np

from players import TEAM_COLORS, player_label, player_team
from identity_review import visible_players

UNIT_COURT = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def court_homography(points):
    points = np.asarray(points, np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Pilih tepat empat sudut lapangan.")
    contour = points.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 100:
        raise ValueError("Sudut harus membentuk segi empat: kiri atas, kanan atas, kanan bawah, kiri bawah.")
    return cv2.getPerspectiveTransform(points, UNIT_COURT)


def automatic_homography(polygon):
    """Approximate VISIBLE floor only; this is not full-court calibration."""
    if len(polygon) < 3:
        return None
    hull = cv2.convexHull(np.asarray(polygon, np.float32))
    perimeter = cv2.arcLength(hull, True)
    corners = None
    for epsilon in (.015, .025, .04, .06, .09):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(approx) == 4:
            corners = approx
            break
    if corners is None:
        x, y, w, h = cv2.boundingRect(hull)
        corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    # Sort around the centre, starting at the corner nearest top-left.
    center = corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0]))]
    corners = np.roll(corners, -int(np.argmin(corners.sum(axis=1))), axis=0)
    try:
        return court_homography(corners)
    except ValueError:
        return None


class CameraMotion:
    """Scene-local image transforms for calibrated camera pan/zoom.

    A failed estimate starts another segment, invalidating old calibration
    instead of carrying a stale map across cuts or featureless views.
    """

    def __init__(self):
        self.previous = None
        self.previous_mask = None
        self.matrix = np.eye(3)
        self.segment = 0

    def update(self, frame, mask, boxes):
        h, w = frame.shape[:2]
        scale = min(1., 640 / w)
        size = (round(w * scale), round(h * scale))
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size)
        usable = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(usable, (int(x1 * scale), int(y1 * scale)),
                          (int(x2 * scale), int(y2 * scale)), 0, -1)
        warp = None
        if self.previous is not None:
            points = cv2.goodFeaturesToTrack(self.previous, 240, .01, 8, mask=self.previous_mask)
            if points is not None and len(points) >= 12:
                moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
                if moved is not None:
                    back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None)
                    if back is not None:
                        good = (status.ravel() == 1) & (reverse.ravel() == 1)
                        good &= np.linalg.norm(points[:, 0] - back[:, 0], axis=1) < 1.5
                        if good.sum() >= 12:
                            affine, inliers = cv2.estimateAffinePartial2D(
                                points[good, 0], moved[good, 0], method=cv2.RANSAC, ransacReprojThreshold=2.5)
                            if affine is not None and inliers.sum() >= 10 and inliers.mean() >= .55:
                                warp = np.vstack([affine, [0., 0., 1.]])
                                zoom = np.linalg.norm(affine[:, 0])
                                rotation = abs(np.arctan2(affine[1, 0], affine[0, 0]))
                                if not .96 <= zoom <= 1.04 or rotation > .04:
                                    warp = None
            if warp is not None:
                corners = np.float32([[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]])
                projected = cv2.perspectiveTransform(corners, warp)[0]
                ratio = abs(cv2.contourArea(projected)) / (size[0] * size[1])
                if not np.isfinite(projected).all() or not .7 < ratio < 1.4:
                    warp = None
            if warp is None:
                self.segment += 1
                self.matrix = np.eye(3)
            else:
                resize = np.diag([scale, scale, 1.])
                self.matrix = np.linalg.inv(resize) @ warp @ resize @ self.matrix
                self.matrix /= self.matrix[2, 2]
        self.previous, self.previous_mask = gray, usable
        return self.segment, self.matrix.round(8).tolist()


def set_calibration(tracks, frame_index, corners):
    court_homography(corners)  # validate before changing saved data
    fr = tracks["frames"][frame_index]
    calibration = {"frame": frame_index, "segment": fr.get("segment", 0),
                   "corners": np.asarray(corners, float).tolist(),
                   "camera": fr.get("camera", np.eye(3).tolist())}
    calibrations = tracks.setdefault("calibrations", [])
    calibrations[:] = [c for c in calibrations if c["segment"] != calibration["segment"]]
    calibrations.append(calibration)


def projection_for_frame(tracks, idx):
    fr = tracks["frames"][idx]
    for calibration in reversed(tracks.get("calibrations", [])):
        if calibration["segment"] == fr.get("segment", 0):
            try:
                transform = (court_homography(calibration["corners"])
                             @ np.asarray(calibration["camera"])
                             @ np.linalg.inv(np.asarray(fr.get("camera", np.eye(3)))))
                return transform, "Kalibrasi 4 sudut · mengikuti kamera"
            except (ValueError, np.linalg.LinAlgError):
                pass
    return automatic_homography(fr.get("court", [])), "Perkiraan area terlihat · belum dikalibrasi"


def tactical_positions(tracks, idx):
    fr = tracks["frames"][idx]
    matrix, mode = projection_for_frame(tracks, idx)
    if matrix is None:
        return [], "Lapangan belum terdeteksi"
    positions = []
    for pid, x1, y1, x2, y2 in visible_players(tracks, idx):
        foot = fr.get("feet", {}).get(str(pid), [(x1 + x2) / 2, y2])
        xy = cv2.perspectiveTransform(np.float32([[foot]]), matrix)[0, 0]
        # Outside points are omitted, never clamped to a false boundary location.
        if np.isfinite(xy).all() and np.all(xy >= 0) and np.all(xy <= 1):
            positions.append((pid, float(xy[0]), float(xy[1])))
    return positions, mode


def draw_tactical(tracks=None, idx=0, width=600, height=340):
    out = np.full((height, width, 3), (29, 48, 33), np.uint8)
    left, top, right, bottom = 30, 34, width - 30, height - 28
    ink = (150, 190, 163)
    cv2.rectangle(out, (left, top), (right, bottom), ink, 2, cv2.LINE_AA)
    midx, midy = (left + right) // 2, (top + bottom) // 2
    cv2.line(out, (midx, top), (midx, bottom), ink, 1)
    radius = int((bottom - top) * .15)
    cv2.circle(out, (midx, midy), radius, ink, 1, cv2.LINE_AA)
    cv2.circle(out, (midx, midy), 2, ink, -1)
    for x, direction in ((left, 1), (right, -1)):
        reach = int((right - left) * .15)
        half = int((bottom - top) * .3)
        cv2.rectangle(out, (min(x, x + direction * reach), midy - half),
                      (max(x, x + direction * reach), midy + half), ink, 1)
        cv2.rectangle(out, (min(x, x - direction * 9), midy - 23),
                      (max(x, x - direction * 9), midy + 23), ink, 1)
    if tracks is None or idx >= len(tracks["frames"]):
        return out, "Analisis video untuk melihat posisi pemain"
    positions, mode = tactical_positions(tracks, idx)
    for pid, nx, ny in positions:
        point = (round(left + nx * (right - left)), round(top + ny * (bottom - top)))
        color = TEAM_COLORS.get(player_team(tracks, pid), TEAM_COLORS[-1])
        cv2.circle(out, point, 10, (12, 20, 18), -1, cv2.LINE_AA)
        cv2.circle(out, point, 8, color, -1, cv2.LINE_AA)
        label = player_label(tracks, pid).replace(" · ", " ")
        label = label.encode("ascii", "replace").decode("ascii")[:24]
        cv2.putText(out, label, (max(1, point[0] - 10), max(14, point[1] - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (235, 245, 238), 1, cv2.LINE_AA)
    return out, mode
