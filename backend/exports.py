import csv
import html
import io
import json

from .stats import COUNT_FIELDS, calculate


def public_match(match):
    return {k: v for k, v in match.items() if not k.endswith("_path")}


def export_json(document):
    return json.dumps({"schema_version": 1, "match": public_match(document["match"]), "players": document["players"], "tracks": document["tracks"], "events": document["events"], "calibrations": document["calibrations"], "jobs": document["jobs"], "revisions": document.get("revisions", []), "stats": calculate(document)}, ensure_ascii=False, indent=2, allow_nan=False)


def safe_cell(value):
    # Spreadsheet programs must not evaluate user-entered names or notes as formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def csv_text(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: safe_cell(row.get(key, "")) for key in fields})
    return "\ufeff" + output.getvalue()


def player_csv(document):
    return csv_text(calculate(document)["players"], ["id", "name", "number", "team", "role", *COUNT_FIELDS, "pass_accuracy", "possession_seconds"])


def event_csv(document):
    players = {p["id"]: p["name"] for p in document["players"]}
    rows = [{**e, "player_name": players.get(e.get("player_id"), ""), "receiver_name": players.get(e.get("receiver_id"), "")} for e in sorted(document["events"], key=lambda e: e["time"])]
    return csv_text(rows, ["id", "type", "time", "end_time", "team", "player_name", "receiver_name", "outcome", "status", "source", "confidence", "notes", "x", "y", "related_event_id"])


def report_html(document):
    stats = calculate(document)
    match = document["match"]
    esc = lambda value: html.escape(str(value))
    rows = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in [p["number"], p["name"], match["home_name"] if p["team"] == "home" else match["away_name"], p["goals"], p["assists"], p["passes"], f'{p["pass_accuracy"]}%' if p["pass_accuracy"] is not None else "—", p["shots"], p["shots_on_target"]]) + "</tr>" for p in stats["players"])
    warnings = "".join(f"<li>{esc(w)}</li>" for w in stats["warnings"])
    return f'''<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(match['title'])} · Futsal Vision</title>
<style>body{{font:15px system-ui;color:#15332b;max-width:1050px;margin:50px auto;padding:0 24px}}header{{border-bottom:2px solid #0c8c66;padding-bottom:24px}}h1{{font-size:32px}}.score{{font-size:42px;font-weight:750;margin:24px 0}}table{{border-collapse:collapse;width:100%;margin:28px 0}}th,td{{text-align:left;border-bottom:1px solid #dce5df;padding:12px 8px}}th{{background:#f0f6f2}}li{{margin:8px 0}}button{{background:#087a56;color:white;border:0;border-radius:8px;padding:12px 20px;cursor:pointer}}small{{color:#62736b}}@media print{{button{{display:none}}body{{margin:0;max-width:none}}}}</style>
<header><small>FUTSAL VISION · LAPORAN {'SIMULASI' if match.get('is_demo') else 'PERTANDINGAN'}</small><h1>{esc(match['title'])}</h1><p>{esc(match.get('venue',''))} · {esc(match.get('played_at',''))} · Status: {esc(match['status'])}</p></header>
<div class="score">{esc(match['home_name'])} {stats['score']['home']} — {stats['score']['away']} {esc(match['away_name'])}</div>
<p>{stats['confirmed_count']} kejadian terkonfirmasi · {stats['pending_count']} menunggu review · Durasi video {match.get('duration',0):.1f} detik</p>
<table><thead><tr><th>No.</th><th>Pemain</th><th>Tim</th><th>Gol</th><th>Assist</th><th>Pass</th><th>Akurasi</th><th>Shot</th><th>On target</th></tr></thead><tbody>{rows}</tbody></table>
<ul>{warnings}</ul><p><small>Gol juga dihitung sebagai shot/on target, kecuali ditautkan ke shot yang sudah dicatat. Akurasi pass memakai hasil yang diketahui. Possession memakai interval penguasaan terkonfirmasi; waktu lain belum diklasifikasikan.</small></p><button onclick="window.print()">Cetak / simpan PDF</button></html>'''
