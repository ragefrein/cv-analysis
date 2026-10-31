"""Conservative offline tracklet linking and explicit human identity verification."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import cv2

from players import merge_players, unit


def visible_players(tracks, frame_index):
    return [p for p in tracks["frames"][frame_index]["p"]
            if not tracks.get("players", {}).get(str(p[0]), {}).get("excluded", False)
            and not (tracks.get("players", {}).get(str(p[0]), {}).get("low_support")
                     and not tracks["players"][str(p[0])].get("verified"))]


def gallery_distance(first, second):
    a = [unit(f) for f in first]
    b = [unit(f) for f in second]
    a, b = [f for f in a if f is not None], [f for f in b if f is not None]
    if not a or not b:
        return None
    distances = np.maximum(0, 1 - np.asarray(a) @ np.asarray(b).T)
    # Multiple independent views, not a single lucky minimum in a large gallery.
    return float((np.median(distances.min(axis=0)) + np.median(distances.min(axis=1))) / 2)


def _summarize(tracks):
    summaries = {}
    for i, fr in enumerate(tracks["frames"]):
        for pid, *box in fr["p"]:
            value = summaries.setdefault(pid, {"frames": set(), "start": i, "end": i})
            value["frames"].add(i)
            value["end"] = i
    return summaries


def candidate_links(tracks):
    summaries = _summarize(tracks)
    roster, fps = tracks["players"], tracks.get("fps", 30)
    rejected = {tuple(sorted(d["ids"])) for d in tracks.get("review_decisions", []) if d["action"] == "different"}
    candidates = []
    ids = sorted(summaries)
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            a, b = summaries[first], summaries[second]
            p, q = roster[str(first)], roster[str(second)]
            if p.get("excluded") or q.get("excluded") or a["frames"] & b["frames"]:
                continue
            if tuple(sorted((first, second))) in rejected:
                continue
            if p.get("team", -1) >= 0 and q.get("team", -1) >= 0 and p["team"] != q["team"]:
                continue
            if p.get("number") and q.get("number") and p["number"] != q["number"]:
                continue
            score = gallery_distance(p.get("appearance", []), q.get("appearance", []))
            if score is None or score > .32:
                continue
            if a["start"] > b["start"]:
                first_pid, second_pid, a, b = second, first, b, a
            else:
                first_pid, second_pid = first, second
            start, end = b["start"], a["end"]
            if start <= end:  # interleaving but non-simultaneous fragments need a human
                nearby = False
                distance = None
            else:
                old, new = tracks["frames"][end], tracks["frames"][start]
                same_shot = old.get("segment") == new.get("segment")
                old_pos = old.get("feet", {}).get(str(first_pid))
                new_pos = new.get("feet", {}).get(str(second_pid))
                distance = None
                if old_pos and new_pos and same_shot:
                    # Compare in the NEW image, avoiding unstable units from a
                    # cumulative projective transform's original image plane.
                    relative = np.asarray(new.get("camera", np.eye(3))) @ np.linalg.inv(np.asarray(old.get("camera", np.eye(3))))
                    predicted = cv2.perspectiveTransform(np.float32([[old_pos]]), relative)[0, 0]
                    scale = np.asarray([tracks.get("width", 100), tracks.get("height", 100)])
                    distance = float(np.linalg.norm((predicted - new_pos) / scale))
                seconds = (start - end) / fps
                nearby = distance is not None and seconds <= .8 and distance <= .025 + .30 * seconds
            candidates.append({"source": second_pid, "target": first_pid, "score": round(score, 4),
                               "frame": b["start"], "gap_frames": max(0, start - end),
                               "nearby": nearby, "distance": distance,
                               "reason": "Fragmen dekat setelah deteksi terputus" if nearby else "Kemiripan penampilan; perlu konfirmasi nomor/ID"})
    return sorted(candidates, key=lambda item: item["score"])


def reconcile_tracklets(tracks):
    """Automatically join only short, unambiguous, spatially plausible gaps.

    Across camera cuts or long absences, appearance is a suggestion for review,
    not sufficient evidence to silently transfer a person's verified name.
    """
    tracks.setdefault("review_decisions", [])
    merges = []
    while True:
        links = candidate_links(tracks)
        chosen = None
        for link in links:
            if not link["nearby"] or link["score"] > .16:
                continue
            source, target = link["source"], link["target"]
            players = tracks["players"]
            if min(players[str(source)]["observations"], players[str(target)]["observations"]) < 6:
                continue
            competing = [l["score"] for l in links if l is not link and l["nearby"]
                         and ({source, target} & {l["source"], l["target"]})]
            if competing and min(competing) - link["score"] < .05:
                continue
            # Even a plausible automatic join remains unverified until reviewed.
            chosen = link
            break
        if chosen is None:
            break
        merge_players(tracks, chosen["source"], chosen["target"])
        tracks["players"][str(chosen["target"])]["verified"] = False
        merges.append(chosen)
    tracks["automatic_links"] = tracks.get("automatic_links", []) + merges
    for player in tracks["players"].values():
        player["low_support"] = player.get("observations", 0) < 4
    refresh_review(tracks)
    return tracks


def refresh_review(tracks):
    links = candidate_links(tracks)
    selected = []
    counts = {}
    for link in links:
        source = link["source"]
        if counts.get(source, 0) < 3:
            selected.append(link)
            counts[source] = counts.get(source, 0) + 1
    tracks["review_queue"] = selected
    tracks["stats"]["unique_ids"] = sum(not p.get("excluded", False) and p.get("observations", 0) > 0
                                        and (not p.get("low_support") or p.get("verified")) for p in tracks["players"].values())
    tracks["stats"]["unverified_ids"] = sum(not p.get("verified", False) and not p.get("excluded", False)
                                              and not p.get("low_support") and p.get("observations", 0) > 0
                                              for p in tracks["players"].values())
    return selected


def verify_player(tracks, pid, number, team, name="", role="player"):
    number = number.strip()
    if number and (not number.isascii() or not number.isdigit() or not 0 <= int(number) <= 99):
        raise ValueError("Nomor punggung harus 0–99 atau kosong.")
    number = str(int(number)) if number else ""
    player = tracks["players"][str(pid)]
    if role in ("player", "goalkeeper") and team < 0:
        raise ValueError("Pilih Tim A atau Tim B untuk memverifikasi identitas.")
    if number and role in ("player", "goalkeeper"):
        for key, other in tracks["players"].items():
            if int(key) == pid or other.get("excluded"):
                continue
            if other.get("verified") and other.get("team") == team and other.get("number") == number:
                raise ValueError(f"Nomor {number} pada tim ini sudah milik P{key}. Gunakan koreksi/gabung ID ke pemain tersebut.")
    player.update(name=name.strip(), number=number, team=int(team), role=role,
                  team_manual=True, verified=True, excluded=role in ("referee", "other"))
    refresh_review(tracks)


def split_player(tracks, pid, frame_index):
    if not 0 <= frame_index < len(tracks["frames"]):
        raise ValueError("Frame tidak valid.")
    before = any(p[0] == pid for fr in tracks["frames"][:frame_index] for p in fr["p"])
    after = any(p[0] == pid for fr in tracks["frames"][frame_index:] for p in fr["p"])
    if not before or not after:
        raise ValueError("Pisahkan pada frame setelah kemunculan pertama dan sebelum kemunculan terakhir ID ini.")
    new_id = max(map(int, tracks["players"]), default=-1) + 1
    player = deepcopy(tracks["players"][str(pid)])
    player.update(id=new_id, name="", number="", verified=False, excluded=False)
    tracks["players"][str(new_id)] = player
    for fr in tracks["frames"][frame_index:]:
        for row in fr["p"]:
            if row[0] == pid:
                row[0] = new_id
        for field in ("poses", "feet", "anchors", "tracklets"):
            if str(pid) in fr.get(field, {}):
                fr[field][str(new_id)] = fr[field].pop(str(pid))
    summaries = _summarize(tracks)
    for identity in (pid, new_id):
        summary = summaries[identity]
        tracks["players"][str(identity)].update(first_seen=summary["start"], last_seen=summary["end"], observations=len(summary["frames"]))
    tracks.setdefault("review_decisions", []).append({"action": "split", "ids": [pid, new_id], "frame": frame_index})
    refresh_review(tracks)
    return new_id
