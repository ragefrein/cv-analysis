import asyncio
import copy
import json
import logging
import math
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import jobs, store
from .config import DATA_DIR, FRONTEND_DIST, MAX_UPLOAD_BYTES, match_dir
from .exports import event_csv, export_json, player_csv, public_match, report_html
from .media import assigned_frames, browser_video, court_points, frames_for, trajectory_points
from .schemas import CalibrationInput, EventInput, EventPatch, JobInput, MatchCreate, MatchPatch, PlayerInput, PlayerPatch, ReviewInput, TrackInput
from .stats import calculate

logger = logging.getLogger(__name__)
ALLOWED_ORIGINS = {f"http://{host}:{port}" for host in ("localhost", "127.0.0.1", "[::1]") for port in (8000, 5173, 4173)}


@asynccontextmanager
async def lifespan(app):
    store.init_store()
    store.recover_jobs()
    yield
    jobs.shutdown()


app = FastAPI(title="Futsal Vision", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(ALLOWED_ORIGINS), allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type"])


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    # Do not echo nonfinite values (NaN/Infinity) or large uploaded input back
    # into a strict JSON response. They are rejected, not server errors.
    errors = [{"type": item["type"], "loc": list(item["loc"]), "msg": item["msg"]} for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def local_boundary(request: Request, call_next):
    # Local single-user service: prevent cross-site writes and DNS-rebinding hosts.
    host = request.url.hostname
    if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
        return Response("Host tidak diizinkan", status_code=400)
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin not in ALLOWED_ORIGINS:
        return Response("Origin tidak diizinkan", status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


def require_match(match_id):
    document = store.get(match_id)
    if document is None:
        raise HTTPException(404, "Pertandingan tidak ditemukan.")
    return document


def ensure_idle(document):
    if jobs.active(document) or document["match"]["status"] == "uploading":
        raise HTTPException(409, "Tunggu upload/analisis selesai atau batalkan proses terlebih dahulu.")


def detail(document):
    return {"match": public_match(document["match"]), "players": document["players"], "tracks": document["tracks"], "events": sorted(document["events"], key=lambda e: e["time"]), "jobs": document["jobs"], "calibrations": document["calibrations"], "stats": calculate(document)}


def validate_event(document, event, event_id=None):
    try:
        validated = EventInput.model_validate({k: event.get(k) for k in EventInput.model_fields if k in event})
    except ValidationError as error:
        raise HTTPException(422, str(error)) from error
    event.update(validated.model_dump())
    duration = document["match"].get("duration") or 0
    if event["time"] > duration + 0.05 or (event.get("end_time") or 0) > duration + 0.05:
        raise HTTPException(422, "Waktu kejadian berada di luar durasi video.")
    players = {p["id"]: p for p in document["players"]}
    for field in ("player_id", "receiver_id"):
        player_id = event.get(field)
        if player_id and player_id not in players:
            raise HTTPException(422, "Pemain tidak ada dalam roster pertandingan ini.")
        if player_id and players[player_id]["team"] != event["team"]:
            raise HTTPException(422, "Tim pemain dan tim kejadian harus sesuai.")
    if event["type"] == "pass" and event.get("player_id") and event.get("player_id") == event.get("receiver_id"):
        raise HTTPException(422, "Pengirim dan penerima passing harus berbeda.")
    if event["status"] == "confirmed" and event["type"] in ("goal", "own_goal", "shot") and event["team"] == "unknown":
        raise HTTPException(422, "Pilih tim untuk gol atau tembakan yang dikonfirmasi.")
    if event["status"] == "confirmed" and event["type"] == "possession":
        if event["team"] == "unknown":
            raise HTTPException(422, "Interval penguasaan yang dikonfirmasi harus memiliki tim.")
        for previous in document["events"]:
            if previous["id"] != event_id and previous["type"] == "possession" and previous["status"] == "confirmed" and max(previous["time"], event["time"]) < min(previous.get("end_time") or previous["time"], event["end_time"]) - 0.001:
                raise HTTPException(422, "Interval penguasaan bertumpuk dengan interval terkonfirmasi lain. Koreksi waktunya terlebih dahulu.")
    related_id = event.get("related_event_id")
    if related_id:
        related = next((e for e in document["events"] if e["id"] == related_id), None)
        if not related or related_id == event_id:
            raise HTTPException(422, "Kejadian terkait tidak ditemukan atau mengacu ke diri sendiri.")
        expected = "shot" if event["type"] == "goal" else "goal" if event["type"] == "assist" else None
        if not expected or related["type"] != expected or related["team"] != event["team"]:
            raise HTTPException(422, "Gol harus ditautkan ke shot satu tim; assist ke gol satu tim.")
        if event["type"] == "goal":
            if abs(event["time"] - related["time"]) > 10 or event.get("player_id") != related.get("player_id"):
                raise HTTPException(422, "Shot terkait harus milik pencetak gol yang sama dan berjarak maksimal 10 detik.")
            if any(e["id"] != event_id and e.get("related_event_id") == related_id and e["type"] == "goal" and e["status"] != "rejected" for e in document["events"]):
                raise HTTPException(422, "Shot ini sudah memiliki kejadian gol terkait.")
    # Guard existing links when an upstream shot/goal is edited.
    for dependent in document["events"]:
        if event_id is not None and dependent.get("related_event_id") == event_id and dependent["status"] != "rejected":
            expected = "shot" if dependent["type"] == "goal" else "goal"
            if event["type"] != expected or event["team"] != dependent["team"] or (dependent["type"] == "goal" and event.get("player_id") != dependent.get("player_id")):
                raise HTTPException(422, "Lepas hubungan kejadian terkait sebelum mengubah jenis, tim, atau pemain.")


@app.get("/api/health")
@app.get("/health", include_in_schema=False)
async def health():
    from .vision import capabilities
    return {"status": "ok", "ai": await run_in_threadpool(capabilities), "ffmpeg_available": bool(shutil.which("ffmpeg"))}


@app.get("/api/matches")
def list_matches():
    return [{**public_match(d["match"]), "score": calculate(d)["score"], "summary": {"players": len(d["players"]), "events": len(d["events"]), "pending": sum(e["status"] == "pending" for e in d["events"])}} for d in store.all_documents()]


def new_match(payload, *, is_demo=False):
    match_id = store.new_id()
    match = {"id": match_id, **payload.model_dump(), "status": "draft", "video_name": None, "video_url": None, "poster_url": None, "duration": 0, "width": 0, "height": 0, "fps": 0, "is_demo": is_demo, "created_at": store.now(), "updated_at": store.now(), "notes": ""}
    document = {"match": match, "players": [], "tracks": [], "events": [], "jobs": [], "calibrations": [], "revisions": []}
    store.insert(document)
    return document


@app.post("/api/matches", status_code=201)
def create_match(payload: MatchCreate):
    return public_match(new_match(payload)["match"])


@app.get("/api/matches/{match_id}")
def get_match(match_id: str):
    return detail(require_match(match_id))


@app.patch("/api/matches/{match_id}")
def update_match(match_id: str, payload: MatchPatch):
    require_match(match_id)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    with store.edit(match_id, invalidate=False) as document:
        if changes.get("status") == "finalized":
            ensure_idle(document)
            if not document["match"].get("video_path"):
                raise HTTPException(422, "Unggah video sebelum finalisasi.")
            if any(e["status"] == "pending" for e in document["events"]):
                raise HTTPException(422, "Tinjau seluruh kandidat kejadian sebelum finalisasi.")
        before = public_match(document["match"])
        document["match"].update(changes)
        store.audit(document, "match_updated", before, changes)
    return public_match(document["match"])


@app.delete("/api/matches/{match_id}", status_code=204)
def delete_match(match_id: str):
    with store.LOCK:
        document = require_match(match_id)
        ensure_idle(document)
        # A generated known ID, and a resolved containment check before recursive deletion.
        directory = (DATA_DIR / "matches" / document["match"]["id"]).resolve()
        base = (DATA_DIR / "matches").resolve()
        if directory.parent != base or directory == base:
            raise HTTPException(400, "Lokasi penyimpanan tidak valid.")
        if directory.exists():
            shutil.rmtree(directory)
        store.delete(match_id)
    return Response(status_code=204)


@app.post("/api/matches/{match_id}/video")
async def upload_video(match_id: str, file: UploadFile = File(...)):
    document = require_match(match_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"):
        raise HTTPException(422, "Gunakan video MP4, MOV, MKV, AVI, M4V, atau WebM.")
    with store.edit(match_id, invalidate=False) as current:
        ensure_idle(current)
        if current["match"].get("video_path"):
            raise HTTPException(409, "Pertandingan ini sudah memiliki video. Buat pertandingan baru untuk rekaman lain.")
        current["match"]["status"] = "uploading"
    directory = match_dir(match_id)
    source = directory / f"original-{store.new_id()}{suffix}"
    browser_path = directory / "video.mp4"
    try:
        total = 0
        with source.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Ukuran video maksimal {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
                output.write(chunk)
        if not total:
            raise HTTPException(422, "Video kosong.")
        from .vision import make_poster, probe_video
        metadata = await run_in_threadpool(probe_video, str(source))
        if metadata["duration"] <= 0 or metadata["width"] <= 0 or metadata["height"] <= 0:
            raise ValueError("Metadata video tidak valid.")
        playable = await run_in_threadpool(browser_video, source, browser_path)
        if playable != source:
            # Transcoding may rotate phone footage or normalize timestamps.
            # Overlays must use the dimensions/timeline of the displayed proxy.
            metadata = await run_in_threadpool(probe_video, str(playable))
        poster = directory / "poster.jpg"
        await run_in_threadpool(make_poster, str(playable), str(poster))
        with store.edit(match_id, invalidate=False) as current:
            current["match"].update(**{k: metadata[k] for k in ("duration", "width", "height", "fps")}, video_name=Path(file.filename).name, video_path=str(playable), original_path=str(source), poster_path=str(poster), video_url=f"/api/matches/{match_id}/video", poster_url=f"/api/matches/{match_id}/poster", status="ready")
            store.audit(current, "video_uploaded", after={"name": current["match"]["video_name"], "bytes": total})
        return public_match(current["match"])
    except Exception as error:
        for path in (source, browser_path):
            if path.exists():
                path.unlink()
        with store.edit(match_id, invalidate=False) as current:
            current["match"]["status"] = "draft"
        if isinstance(error, HTTPException):
            raise
        logger.warning("Video import failed: %s", error)
        raise HTTPException(422, str(error)[:1000]) from error
    finally:
        await file.close()


@app.get("/api/matches/{match_id}/video")
@app.head("/api/matches/{match_id}/video")
def video(match_id: str):
    document = require_match(match_id)
    path = document["match"].get("video_path")
    if not path or not Path(path).is_file():
        raise HTTPException(404, "Video belum tersedia.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/matches/{match_id}/poster")
def poster(match_id: str):
    path = require_match(match_id)["match"].get("poster_path")
    if not path or not Path(path).is_file():
        raise HTTPException(404, "Poster belum tersedia.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/matches/{match_id}/players", status_code=201)
def add_player(match_id: str, payload: PlayerInput):
    require_match(match_id)
    with store.edit(match_id) as document:
        if payload.number and any(p["number"] == payload.number and p["team"] == payload.team for p in document["players"]):
            raise HTTPException(422, "Nomor punggung sudah dipakai di tim ini.")
        player = {"id": store.new_id(), "match_id": match_id, **payload.model_dump()}
        document["players"].append(player)
        store.audit(document, "player_added", after=player)
    return player


@app.patch("/api/matches/{match_id}/players/{player_id}")
def update_player(match_id: str, player_id: str, payload: PlayerPatch):
    require_match(match_id)
    with store.edit(match_id) as document:
        player = next((p for p in document["players"] if p["id"] == player_id), None)
        if not player:
            raise HTTPException(404, "Pemain tidak ditemukan.")
        changes = payload.model_dump(exclude_unset=True, exclude_none=True)
        candidate = {**player, **changes}
        if candidate["number"] and any(p["id"] != player_id and p["number"] == candidate["number"] and p["team"] == candidate["team"] for p in document["players"]):
            raise HTTPException(422, "Nomor punggung sudah dipakai di tim ini.")
        if candidate["team"] != player["team"] and (any(e.get("player_id") == player_id or e.get("receiver_id") == player_id for e in document["events"]) or any(t.get("player_id") == player_id for t in document["tracks"])):
            raise HTTPException(422, "Pemain sudah terhubung ke kejadian/lintasan. Lepas hubungan sebelum mengganti tim.")
        before = copy.deepcopy(player)
        player.update(changes)
        store.audit(document, "player_updated", before, player)
    return player


@app.delete("/api/matches/{match_id}/players/{player_id}", status_code=204)
def delete_player(match_id: str, player_id: str):
    require_match(match_id)
    with store.edit(match_id) as document:
        if not any(p["id"] == player_id for p in document["players"]):
            raise HTTPException(404, "Pemain tidak ditemukan.")
        if any(e.get("player_id") == player_id or e.get("receiver_id") == player_id for e in document["events"]) or any(t.get("player_id") == player_id for t in document["tracks"]):
            raise HTTPException(409, "Pemain masih terhubung ke kejadian/lintasan. Lepas hubungan sebelum menghapus.")
        document["players"] = [p for p in document["players"] if p["id"] != player_id]
        store.audit(document, "player_deleted", before={"id": player_id})
    return Response(status_code=204)


@app.patch("/api/matches/{match_id}/tracks/{track_id}")
def assign_track(match_id: str, track_id: int, payload: TrackInput):
    require_match(match_id)
    with store.edit(match_id) as document:
        track = next((t for t in document["tracks"] if t["track_id"] == track_id), None)
        if not track:
            raise HTTPException(404, "Lintasan tidak ditemukan.")
        player = next((p for p in document["players"] if p["id"] == payload.player_id), None)
        if payload.player_id and not player:
            raise HTTPException(422, "Pemain tidak ada dalam pertandingan.")
        before = copy.deepcopy(track)
        track["player_id"] = payload.player_id
        track["team"] = player["team"] if player else payload.team or track["team"]
        if payload.excluded is not None:
            track["excluded"] = payload.excluded
        elif player:
            track["excluded"] = False
        if track.get("excluded"):
            track.update(player_id=None, team="unknown")
        latest_job = next((j for j in reversed(document["jobs"]) if j["status"] == "completed"), None)
        for event in document["events"]:
            # Only pending predictions from the current run inherit identity corrections.
            if event["status"] == "pending" and event["source"] == "ai" and latest_job and event.get("job_id") == latest_job["id"]:
                if event.get("track_id") == track_id:
                    event.update(player_id=payload.player_id, team=track["team"])
                if event.get("receiver_track_id") == track_id:
                    event["receiver_id"] = payload.player_id
                if track.get("excluded") and track_id in (event.get("track_id"), event.get("receiver_track_id")):
                    event["status"] = "rejected"
        if latest_job:
            mapped = {t["track_id"] for t in document["tracks"] if t.get("player_id") and not t.get("excluded")}
            frames = frames_for(document)
            step = 1 / max(1, latest_job.get("options", {}).get("sample_fps", 5))
            identity_seconds = 0
            for index, frame in enumerate(frames):
                if any(o.get("kind") == "player" and o.get("track_id") in mapped for o in frame.get("objects", [])):
                    next_time = frames[index + 1]["time"] if index + 1 < len(frames) else document["match"]["duration"]
                    identity_seconds += min(step, max(0, next_time - frame["time"]))
            latest_job.setdefault("metrics", {})["identity_seconds"] = round(identity_seconds, 3)
        store.audit(document, "track_assigned", before, track)
    return track


@app.get("/api/matches/{match_id}/detections")
def detections(match_id: str, time: float = Query(ge=0), window: float = Query(default=0.3, ge=0, le=2)):
    if not math.isfinite(time):
        raise HTTPException(422, "Waktu tidak valid.")
    document = require_match(match_id)
    return {"frames": assigned_frames(document, frames_for(document, time, window))}


@app.get("/api/matches/{match_id}/trajectories")
def trajectories(match_id: str, coordinates: str = Query(default="image", pattern="^(image|court)$")):
    document = require_match(match_id)
    return {"points": court_points(document) if coordinates == "court" else trajectory_points(document), "coordinate_system": coordinates, "warnings": ["Hanya pengamatan di dalam frame; bukan posisi lengkap pertandingan.", "Kalibrasi manual hanya berlaku selama sudut/pan/zoom tidak berubah dalam interval."]}


@app.post("/api/matches/{match_id}/events", status_code=201)
def add_event(match_id: str, payload: EventInput):
    require_match(match_id)
    with store.edit(match_id) as document:
        event = {"id": store.new_id(), "match_id": match_id, **payload.model_dump(), "source": "manual", "confidence": None}
        validate_event(document, event)
        document["events"].append(event)
        store.audit(document, "event_added", after=event)
    return event


@app.post("/api/matches/{match_id}/events/review")
def review_events(match_id: str, payload: ReviewInput):
    require_match(match_id)
    with store.edit(match_id) as document:
        ids = set(payload.ids)
        if not ids.issubset({e["id"] for e in document["events"]}):
            raise HTTPException(404, "Sebagian kejadian tidak ditemukan.")
        for event in document["events"]:
            if event["id"] in ids:
                before = copy.deepcopy(event)
                event["status"] = payload.status
                validate_event(document, event, event["id"])
                store.audit(document, "event_reviewed", before, event)
    return {"updated": len(ids)}


@app.patch("/api/matches/{match_id}/events/{event_id}")
def update_event(match_id: str, event_id: str, payload: EventPatch):
    require_match(match_id)
    with store.edit(match_id) as document:
        event = next((e for e in document["events"] if e["id"] == event_id), None)
        if not event:
            raise HTTPException(404, "Kejadian tidak ditemukan.")
        before = copy.deepcopy(event)
        event.update(payload.model_dump(exclude_unset=True))
        validate_event(document, event, event_id)
        store.audit(document, "event_updated", before, event)
    return event


@app.delete("/api/matches/{match_id}/events/{event_id}", status_code=204)
def delete_event(match_id: str, event_id: str):
    require_match(match_id)
    with store.edit(match_id) as document:
        event = next((e for e in document["events"] if e["id"] == event_id), None)
        if not event:
            raise HTTPException(404, "Kejadian tidak ditemukan.")
        for other in document["events"]:
            if other.get("related_event_id") == event_id:
                other["related_event_id"] = None
        document["events"] = [e for e in document["events"] if e["id"] != event_id]
        store.audit(document, "event_deleted", before=event)
    return Response(status_code=204)


@app.post("/api/matches/{match_id}/jobs", status_code=202)
def start_job(match_id: str, payload: JobInput):
    require_match(match_id)
    with store.edit(match_id, invalidate=False) as document:
        ensure_idle(document)
        if not document["match"].get("video_path"):
            raise HTTPException(422, "Unggah video terlebih dahulu.")
        job = {"id": store.new_id(), "match_id": match_id, "status": "queued", "progress": 0, "stage": "Menunggu giliran analisis", "error": None, "created_at": store.now(), "updated_at": store.now(), "options": payload.model_dump(), "metrics": {}}
        document["jobs"].append(job)
        document["match"]["status"] = "processing"
    jobs.submit(match_id, job)
    return job


@app.post("/api/matches/{match_id}/jobs/{job_id}/cancel")
def cancel_job(match_id: str, job_id: str):
    require_match(match_id)
    with store.edit(match_id, invalidate=False) as document:
        job = next((j for j in document["jobs"] if j["id"] == job_id), None)
        if not job:
            raise HTTPException(404, "Proses tidak ditemukan.")
        if job["status"] in ("running", "queued"):
            job["stage"] = "Membatalkan proses…"
            if jobs.cancel(job_id):
                job.update(status="cancelled", stage="Dibatalkan sebelum mulai", updated_at=store.now())
                document["match"]["status"] = "review"
    return job


@app.post("/api/matches/{match_id}/calibrations", status_code=201)
def add_calibration(match_id: str, payload: CalibrationInput):
    require_match(match_id)
    with store.edit(match_id) as document:
        if payload.end_time > document["match"]["duration"] + 0.05:
            raise HTTPException(422, "Interval kalibrasi di luar durasi video.")
        if any(max(c["start_time"], payload.start_time) < min(c["end_time"], payload.end_time) for c in document["calibrations"]):
            raise HTTPException(422, "Interval kalibrasi bertumpuk. Gunakan segmen yang tidak bertumpuk.")
        calibration = {"id": store.new_id(), "match_id": match_id, **payload.model_dump()}
        document["calibrations"].append(calibration)
        store.audit(document, "calibration_added", after=calibration)
    return calibration


@app.delete("/api/matches/{match_id}/calibrations/{calibration_id}", status_code=204)
def delete_calibration(match_id: str, calibration_id: str):
    require_match(match_id)
    with store.edit(match_id) as document:
        old = next((c for c in document["calibrations"] if c["id"] == calibration_id), None)
        if not old:
            raise HTTPException(404, "Kalibrasi tidak ditemukan.")
        document["calibrations"] = [c for c in document["calibrations"] if c["id"] != calibration_id]
        store.audit(document, "calibration_deleted", before=old)
    return Response(status_code=204)


@app.get("/api/matches/{match_id}/stats")
def stats(match_id: str):
    return calculate(require_match(match_id))


@app.get("/api/matches/{match_id}/export")
def export(match_id: str, format: str = Query(default="json", pattern="^(json|csv|events|report)$")):
    document = require_match(match_id)
    if format == "report":
        return HTMLResponse(report_html(document))
    content, media_type, suffix = {"json": (export_json, "application/json", "json"), "csv": (player_csv, "text/csv", "csv"), "events": (event_csv, "text/csv", "events.csv")}[format]
    return Response(content(document), media_type=media_type, headers={"Content-Disposition": f'attachment; filename="futsal-{match_id[:8]}.{suffix}"'})


@app.post("/api/demo", status_code=201)
def demo():
    from .demo import create_demo
    return public_match(create_demo()["match"])


# Production build is served by the same local server, including API/media.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    def index():
        return '<html lang="id"><title>Futsal Vision</title><body style="font-family:system-ui;padding:60px"><h1>Futsal Vision API berjalan.</h1><p>Jalankan frontend dengan <code>npm run dev --prefix frontend</code> lalu buka <a href="http://localhost:5173">localhost:5173</a>, atau build frontend untuk menyajikan aplikasi di port 8000.</p><a href="/docs">API documentation</a></body></html>'
