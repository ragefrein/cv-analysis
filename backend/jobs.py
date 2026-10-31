import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import store
from .config import MODEL_PATH, match_dir

logger = logging.getLogger(__name__)
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="futsal-vision")
CANCEL_SIGNALS: dict[str, threading.Event] = {}
FUTURES = {}


def active(document):
    return any(j["status"] in ("queued", "running") for j in document["jobs"])


def _update(match_id, job_id, **changes):
    with store.edit(match_id, invalidate=False) as document:
        job = next(j for j in document["jobs"] if j["id"] == job_id)
        job.update(changes, updated_at=store.now())


def submit(match_id, job):
    signal = threading.Event()
    CANCEL_SIGNALS[job["id"]] = signal
    future = EXECUTOR.submit(_run, match_id, job["id"], signal)
    FUTURES[job["id"]] = future

    def done(_):
        FUTURES.pop(job["id"], None)
        CANCEL_SIGNALS.pop(job["id"], None)

    future.add_done_callback(done)


def cancel(job_id):
    signal = CANCEL_SIGNALS.get(job_id)
    if signal:
        signal.set()
    future = FUTURES.get(job_id)
    return bool(future and future.cancel())


def _run(match_id, job_id, signal):
    temporary = match_dir(match_id) / f"detections-{job_id}.partial.jsonl"
    try:
        from .vision import analyze_video

        document = store.get(match_id)
        job = next(j for j in document["jobs"] if j["id"] == job_id)
        if signal.is_set():
            _update(match_id, job_id, status="cancelled", stage="Dibatalkan")
            return
        _update(match_id, job_id, status="running", stage="Memuat model deteksi", progress=0)

        def progress(value, stage):
            _update(match_id, job_id, progress=max(0, min(99.9, float(value))), stage=stage)

        options = {**job["options"], "home_color": document["match"]["home_color"], "away_color": document["match"]["away_color"], "model_path": str(MODEL_PATH), "calibrations": document.get("calibrations", [])}
        output = analyze_video(document["match"]["video_path"], str(temporary), options, progress, signal.is_set)
        if signal.is_set():
            _update(match_id, job_id, status="cancelled", stage="Dibatalkan; hasil sebelumnya dipertahankan")
            return
        final_path = match_dir(match_id) / f"detections-{job_id}.jsonl"
        os.replace(temporary, final_path)
        with store.edit(match_id, invalidate=False) as current:
            # Never reuse old tracker-to-player mappings after a new tracking run.
            current["tracks"] = [{"id": store.new_id(), "match_id": match_id, "player_id": None, **t} for t in output.get("tracks", [])]
            preserved = [e for e in current["events"] if not (e["source"] == "ai" and e["status"] == "pending")]
            for event in output.get("events", []):
                # Confirmed evidence survives reruns; do not create obvious duplicate candidates.
                if any(e["type"] == event["type"] and abs(e["time"] - event["time"]) < 1 and e["status"] == "confirmed" for e in preserved):
                    continue
                preserved.append({"id": store.new_id(), "match_id": match_id, "end_time": None, "player_id": None, "receiver_id": None, "team": "unknown", "outcome": "unknown", "notes": "", "x": None, "y": None, "related_event_id": None, **event, "source": "ai", "status": "pending", "job_id": job_id})
            current["events"] = preserved
            current["match"].update(detections_path=str(final_path), status="review")
            decoded_duration = output.get("metrics", {}).get("decoded_duration", 0)
            if decoded_duration > current["match"].get("duration", 0) + 0.05:
                current["match"]["duration"] = decoded_duration
            current_job = next(j for j in current["jobs"] if j["id"] == job_id)
            current_job.update(status="completed", progress=100, stage="Selesai. Tinjau identitas pemain dan kandidat kejadian.", error=None, metrics=output.get("metrics", {}), updated_at=store.now())
            store.audit(current, "analysis_completed", after={"job_id": job_id, "track_count": len(current["tracks"]), "candidate_count": len(output.get("events", []))})
    except Exception as error:
        if signal.is_set() or error.__class__.__name__ in ("AnalysisCancelled", "CancelledError"):
            _update(match_id, job_id, status="cancelled", stage="Dibatalkan; hasil sebelumnya dipertahankan")
        else:
            logger.exception("Analysis failed: %s", job_id)
            _update(match_id, job_id, status="failed", stage="Analisis gagal; data review sebelumnya tetap tersimpan", error=str(error)[:2000])
    finally:
        try:
            with store.edit(match_id, invalidate=False) as current:
                if current["match"]["status"] == "processing" and not active(current):
                    current["match"]["status"] = "review"
        except KeyError:
            pass
        CANCEL_SIGNALS.pop(job_id, None)
        if temporary.exists():
            temporary.unlink()


def shutdown():
    for signal in list(CANCEL_SIGNALS.values()):
        signal.set()
    EXECUTOR.shutdown(wait=False, cancel_futures=True)
