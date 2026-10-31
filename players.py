"""Match-local player memory, jersey grouping, and editable persistent rosters."""
from __future__ import annotations

import base64
from copy import deepcopy

import cv2
import numpy as np

TEAM_COLORS = {0: (255, 185, 55), 1: (85, 110, 255), -1: (165, 175, 185)}
TEAM_NAMES = {0: "Tim A", 1: "Tim B", -1: "Belum pasti"}


def unit(vector):
    if vector is None:
        return None
    value = np.asarray(vector, np.float32).reshape(-1)
    norm = np.linalg.norm(value)
    return value / norm if np.isfinite(value).all() and norm > 1e-8 else None


def jersey_feature(frame, box):
    """Dominant central torso colour in Lab; suppress blue floor contamination."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    left, right = max(0, int(x1 + .23 * bw)), min(w, int(x2 - .23 * bw))
    top, bottom = max(0, int(y1 + .20 * bh)), min(h, int(y1 + .52 * bh))
    crop = frame[top:bottom, left:right]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    keep = ~((hsv[:, 0] >= 95) & (hsv[:, 0] <= 122) & (hsv[:, 2] > 165))
    saturated = keep & (hsv[:, 1] >= 75)
    if np.count_nonzero(saturated) >= max(10, .20 * len(lab)):
        keep = saturated
    pixels = lab[keep] if np.count_nonzero(keep) >= 10 else lab
    return np.median(pixels, axis=0).astype(np.float32)


def portrait(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if not crop.size:
        return ""
    crop = cv2.resize(crop, (64, 112))
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(encoded).decode("ascii") if ok else ""


class PlayerMemory:
    """Permanent public IDs above the tracker's expiring internal track IDs.

    A gallery never expires during a video. Ambiguous lookalikes get a new ID
    rather than silently inheriting a name. A public ID is used once per frame.
    """

    def __init__(self, roster=None, reidentify=True):
        self.reidentify = reidentify
        self.players = {int(key): deepcopy(value) for key, value in (roster or {}).items()}
        self.bindings = {}
        self.galleries = {pid: [unit(f) for f in p.get("appearance", []) if unit(f) is not None]
                          for pid, p in self.players.items()}
        self.jerseys = {pid: [np.asarray(p["jersey"], np.float32)] if p.get("jersey") else []
                        for pid, p in self.players.items()}
        self.previous_teams = {pid: p.get("team", -1) for pid, p in self.players.items()}
        for player in self.players.values():
            player.update(first_seen=-1, last_seen=-1, observations=0)

    def assign(self, observations, frame_index):
        # Reserve ALL existing bindings first, including tracks processed later.
        assigned = {}
        used = set()
        for i, obs in enumerate(observations):
            pid = self.bindings.get(obs["track_id"])
            if pid is not None and pid not in used:
                assigned[i] = pid
                used.add(pid)

        candidates = []
        for i, obs in enumerate(observations):
            if i in assigned:
                continue
            feature = unit(obs.get("embedding"))
            scores = []
            if feature is not None and self.reidentify:
                for pid, gallery in self.galleries.items():
                    if pid in used or not gallery:
                        continue
                    jersey = obs.get("jersey")
                    old_jerseys = self.jerseys[pid]
                    if jersey is not None and old_jerseys:
                        if np.linalg.norm(jersey - np.median(old_jerseys, axis=0)) > 65:
                            continue
                    distances = sorted(float(1 - np.dot(feature, f)) for f in gallery)
                    scores.append((float(np.mean(distances[:2])), pid))
            scores.sort()
            if scores and scores[0][0] < .22:
                margin = scores[1][0] - scores[0][0] if len(scores) > 1 else 1.0
                if margin >= .045:
                    candidates.append((scores[0][0], i, scores[0][1]))

        for _, i, pid in sorted(candidates):
            if pid not in used:
                assigned[i] = pid
                used.add(pid)

        for i, obs in enumerate(observations):
            if i not in assigned:
                pid = max(self.players, default=-1) + 1
                assigned[i] = pid
                self.players[pid] = {
                    "id": pid, "name": "", "team": -1,
                    "first_seen": frame_index, "last_seen": frame_index,
                    "observations": 0, "portrait": obs.get("portrait", ""),
                    "number": "", "verified": False, "role": "player", "excluded": False,
                }
                self.galleries[pid] = []
                self.jerseys[pid] = []
            pid = assigned[i]
            self.bindings[obs["track_id"]] = pid
            player = self.players[pid]
            if player["first_seen"] < 0:
                player["first_seen"] = frame_index
            player["last_seen"] = frame_index
            player["observations"] += 1
            if not player["portrait"]:
                player["portrait"] = obs.get("portrait", "")
            feature = unit(obs.get("embedding"))
            # Sample periodically; keep the first view and a rolling gallery.
            if feature is not None and (not self.galleries[pid] or player["observations"] % 15 == 0):
                gallery = self.galleries[pid]
                gallery.append(feature)
                if len(gallery) > 16:
                    del gallery[1]
            jersey = obs.get("jersey")
            if jersey is not None and (not self.jerseys[pid] or player["observations"] % 10 == 0):
                self.jerseys[pid].append(jersey)
                self.jerseys[pid] = self.jerseys[pid][-30:]
        return [assigned[i] for i in range(len(observations))]

    def finish(self):
        # Each identity contributes equally, so a long-lived player cannot
        # overwhelm the other team's shorter tracks.
        ids = [pid for pid in self.players if self.jerseys[pid]]
        if len(ids) >= 2:
            samples = np.asarray([np.median(self.jerseys[pid], axis=0) for pid in ids], np.float32)
            # Jersey hue matters more than brightness: a white goalkeeper or
            # a highlight must not split the two outfield teams into one cluster.
            samples = samples * np.float32([.25, 1., 1.])
            # Fit the two main populations with trimmed medoids. Farthest-point
            # seeding often picks a bright referee/keeper as an entire "team".
            distances = np.linalg.norm(samples[:, None] - samples[None, :], axis=2)
            weights = np.asarray([max(1, self.players[pid]["observations"]) for pid in ids], float)
            retain_weight = .85 * weights.sum()
            best = (float("inf"), 0, 1)
            for first in range(len(ids)):
                for second in range(first + 1, len(ids)):
                    residual = np.minimum(distances[:, first], distances[:, second])
                    order = np.argsort(residual)
                    sorted_weights = weights[order]
                    accepted = np.clip(retain_weight - np.r_[0., np.cumsum(sorted_weights)[:-1]], 0, sorted_weights)
                    score = float(np.sum(accepted * residual[order] ** 2) / retain_weight)
                    if score < best[0]:
                        best = (score, first, second)
            _, first, second = best
            centers = samples[[first, second]].copy()
            for _ in range(25):
                labels = np.argmin(np.linalg.norm(samples[:, None] - centers, axis=2), axis=1)
                updated = centers.copy()
                for k in range(2):
                    selected = np.flatnonzero(labels == k)
                    for channel in range(3):
                        ordered = selected[np.argsort(samples[selected, channel])]
                        if len(ordered):
                            median = np.searchsorted(np.cumsum(weights[ordered]), weights[ordered].sum() / 2)
                            updated[k, channel] = samples[ordered[median], channel]
                if np.allclose(centers, updated):
                    break
                centers = updated
            if np.linalg.norm(centers[0] - centers[1]) >= 18:
                votes = sum(1 if self.previous_teams.get(pid) == int(label) else -1
                            for pid, label in zip(ids, labels) if self.previous_teams.get(pid, -1) >= 0)
                if votes < 0:
                    labels = 1 - labels
                    centers = centers[::-1]
                for sample, pid, label in zip(samples, ids, labels):
                    if not self.players[pid].get("team_manual"):
                        distance = np.linalg.norm(sample - centers[label])
                        self.players[pid]["team"] = int(label) if distance <= 32 else -1
        for pid, player in self.players.items():
            player["appearance"] = [f.round(6).tolist() for f in self.galleries[pid]]
            player["jersey"] = np.median(self.jerseys[pid], axis=0).tolist() if self.jerseys[pid] else None
        return {str(pid): player for pid, player in self.players.items()}


def player_label(tracks, pid):
    player = tracks.get("players", {}).get(str(pid), {})
    parts = [f"P{pid}" + (" ?" if tracks.get("pipeline_version", 0) >= 3 and not player.get("verified") else "")]
    if player.get("number"):
        parts.append(f"#{player['number']}")
    if player.get("name", "").strip():
        parts.append(player["name"].strip())
    return " · ".join(parts)


def player_team(tracks, pid):
    return int(tracks.get("players", {}).get(str(pid), {}).get("team", -1))


def merge_players(tracks, source, target):
    """Correct fragmented IDs, refusing to combine simultaneous detections."""
    source, target = int(source), int(target)
    if source == target:
        raise ValueError("Pilih dua ID berbeda.")
    roster = tracks["players"]
    if str(source) not in roster or str(target) not in roster:
        raise ValueError("ID pemain tidak ditemukan.")
    if roster[str(source)].get("number") and roster[str(target)].get("number"):
        if roster[str(source)]["number"] != roster[str(target)]["number"]:
            raise ValueError("Nomor punggung berbeda. Periksa nomor atau pisahkan track yang tertukar lebih dahulu.")
    for fr in tracks["frames"]:
        present = {p[0] for p in fr["p"]}
        if source in present and target in present:
            raise ValueError("Kedua ID terlihat bersamaan. Tidak aman untuk digabung.")
    for fr in tracks["frames"]:
        for player in fr["p"]:
            if player[0] == source:
                player[0] = target
        for field in ("poses", "feet", "anchors", "tracklets"):
            entries = fr.get(field, {})
            if str(source) in entries:
                entries[str(target)] = entries.pop(str(source))
    old, kept = roster.pop(str(source)), roster[str(target)]
    first_seen = [i for i in (old["first_seen"], kept["first_seen"]) if i >= 0]
    kept["first_seen"] = min(first_seen, default=-1)
    kept["last_seen"] = max(old["last_seen"], kept["last_seen"])
    kept["observations"] += old["observations"]
    kept["appearance"] = (kept.get("appearance", []) + old.get("appearance", []))[-32:]
    kept["name"] = kept.get("name") or old.get("name", "")
    kept["number"] = kept.get("number") or old.get("number", "")
    if old.get("verified") and not kept.get("verified"):
        for field in ("verified", "team", "team_manual", "role", "excluded"):
            if field in old:
                kept[field] = old[field]
    tracks["stats"]["unique_ids"] = len(roster)
