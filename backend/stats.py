"""Deterministic statistics from confirmed evidence; pending AI never changes scores."""
from collections import Counter

COUNT_FIELDS = ("goals", "own_goals", "passes", "passes_successful", "passes_failed", "passes_unresolved", "shots", "shots_on_target", "assists", "interceptions", "turnovers", "recoveries", "saves", "blocks", "dribbles", "fouls", "yellow_cards", "red_cards")


def empty_stats():
    return {**dict.fromkeys(COUNT_FIELDS, 0), "pass_accuracy": None, "possession_seconds": 0, "possession_pct": None}


def union_seconds(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(end if end is not None else stop, stop)
    return round(total, 3)


def calculate(document):
    match = document["match"]
    duration = float(match.get("duration") or 0)
    teams = {"home": empty_stats(), "away": empty_stats()}
    players = {p["id"]: {**p, **empty_stats()} for p in document["players"]}
    events = [e for e in document["events"] if e["status"] == "confirmed"]
    by_id = {e["id"]: e for e in events}
    network, timeline = Counter(), {}
    possessions = {"home": [], "away": []}
    player_possessions = {key: [] for key in players}

    def increment(event, field, amount=1, team_override=None, credit_player=True):
        team = team_override or event["team"]
        if team in teams:
            teams[team][field] += amount
        player = players.get(event.get("player_id"))
        if credit_player and player and player["team"] == team:
            player[field] += amount

    for event in events:
        kind = event["type"]
        outcome = event.get("outcome", "unknown")
        if kind == "pass":
            increment(event, "passes")
            field = {"successful": "passes_successful", "failed": "passes_failed"}.get(outcome, "passes_unresolved")
            increment(event, field)
            if outcome == "successful" and event.get("player_id") in players and event.get("receiver_id") in players:
                network[(event["player_id"], event["receiver_id"], event["team"])] += 1
        elif kind in ("shot", "goal"):
            linked = by_id.get(event.get("related_event_id"))
            is_goal = kind == "goal" or (kind == "shot" and outcome == "goal")
            # A goal is also a shot, unless explicitly linked to its separately logged shot.
            if not (kind == "goal" and linked and linked["type"] == "shot"):
                increment(event, "shots")
                if is_goal or outcome == "on_target":
                    increment(event, "shots_on_target")
            elif linked.get("outcome") not in ("on_target", "goal"):
                # Upgrade the linked shot's on-target classification without double counting it.
                increment(event, "shots_on_target")
            if is_goal and not (kind == "goal" and linked and linked.get("outcome") == "goal"):
                increment(event, "goals")
        elif kind == "own_goal":
            increment(event, "own_goals")
            opponent = "away" if event["team"] == "home" else "home"
            increment(event, "goals", team_override=opponent, credit_player=False)
        elif kind == "possession":
            start, stop = event["time"], min(duration, event.get("end_time") or event["time"])
            if event["team"] in possessions and stop > start:
                possessions[event["team"]].append((start, stop))
                if event.get("player_id") in player_possessions:
                    player_possessions[event["player_id"]].append((start, stop))
        else:
            field = {"assist": "assists", "interception": "interceptions", "turnover": "turnovers", "recovery": "recoveries", "save": "saves", "block": "blocks", "dribble": "dribbles", "foul": "fouls", "yellow_card": "yellow_cards", "red_card": "red_cards"}.get(kind)
            if field:
                # Only successful/unknown dribbles are credited; failed attempts do not count as success.
                if kind != "dribble" or outcome != "failed":
                    increment(event, field)
        if kind != "possession" and event["team"] in teams:
            minute = int(event["time"] // 60)
            timeline.setdefault(minute, {"minute": minute, "home": 0, "away": 0})[event["team"]] += 1

    for team, intervals in possessions.items():
        teams[team]["possession_seconds"] = union_seconds(intervals)
    known_possession = sum(t["possession_seconds"] for t in teams.values())
    for key, intervals in player_possessions.items():
        players[key]["possession_seconds"] = union_seconds(intervals)
    for item in [*teams.values(), *players.values()]:
        resolved = item["passes_successful"] + item["passes_failed"]
        item["pass_accuracy"] = round(item["passes_successful"] / resolved * 100, 1) if resolved else None
        item["possession_pct"] = round(item["possession_seconds"] / known_possession * 100, 1) if known_possession else None

    completed = [j for j in document["jobs"] if j["status"] == "completed"]
    metrics = completed[-1].get("metrics", {}) if completed else {}
    calibrated = union_seconds([(c["start_time"], min(c["end_time"], duration)) for c in document["calibrations"]])
    coverage = {"duration": duration, "analyzed_seconds": min(duration, metrics.get("analyzed_seconds", 0)), "ball_visible_seconds": min(duration, metrics.get("ball_visible_seconds", 0)), "identity_seconds": min(duration, metrics.get("identity_seconds", 0)), "calibrated_seconds": calibrated, "possession_known_seconds": round(known_possession, 3), "possession_unknown_seconds": max(0, round(duration - known_possession, 3))}
    warnings = ["Statistik hanya memakai kejadian terkonfirmasi. Bagian di luar kamera tidak dapat dihitung lengkap."]
    if not metrics:
        warnings.append("Video belum selesai dianalisis; kejadian dapat dicatat manual.")
    if duration and known_possession < duration:
        warnings.append("Persentase possession memakai durasi berlabel penguasaan, bukan seluruh pertandingan; sisa durasi belum diklasifikasikan.")
    if any(e["type"] == "pass" and e.get("outcome") == "unknown" for e in events):
        warnings.append("Passing dengan hasil belum diketahui tidak dimasukkan ke denominator akurasi passing.")
    if any(e.get("player_id") is None for e in events if e["type"] not in ("possession", "substitution")):
        warnings.append("Ada kejadian tanpa identitas pemain; angka tim tetap dihitung bila tim diketahui.")
    if match.get("is_demo"):
        warnings.insert(0, "SIMULASI: video, lintasan, dan statistik ini adalah fixture sintetis, bukan hasil analisis pertandingan.")
    return {"score": {team: value["goals"] for team, value in teams.items()}, "teams": teams, "players": list(players.values()), "pending_count": sum(e["status"] == "pending" for e in document["events"]), "confirmed_count": len(events), "rejected_count": sum(e["status"] == "rejected" for e in document["events"]), "coverage": coverage, "pass_network": [{"from_player_id": a, "to_player_id": b, "team": t, "count": count} for (a, b, t), count in network.items()], "event_timeline": sorted(timeline.values(), key=lambda p: p["minute"]), "warnings": warnings}
