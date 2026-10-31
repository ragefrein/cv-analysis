"""Small versioned SQLite document store; each match mutation is transactional."""
import json
import copy
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from .config import DATA_DIR, prepare_directories

LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def new_id():
    return uuid4().hex


@contextmanager
def connect():
    connection = sqlite3.connect(DATA_DIR / "futsal.sqlite3", timeout=30)
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_store():
    prepare_directories()
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS matches (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        db.execute("PRAGMA user_version=1")


def insert(document):
    with LOCK, connect() as db:
        db.execute("INSERT INTO matches(id,data) VALUES (?,?)", (document["match"]["id"], json.dumps(document, allow_nan=False)))


def get(match_id):
    with connect() as db:
        row = db.execute("SELECT data FROM matches WHERE id=?", (match_id,)).fetchone()
    return json.loads(row[0]) if row else None


def all_documents():
    with connect() as db:
        rows = db.execute("SELECT data FROM matches ORDER BY rowid DESC").fetchall()
    return [json.loads(row[0]) for row in rows]


@contextmanager
def edit(match_id, *, invalidate=True):
    with LOCK, connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT data FROM matches WHERE id=?", (match_id,)).fetchone()
        if row is None:
            raise KeyError(match_id)
        document = json.loads(row[0])
        yield document
        document["match"]["updated_at"] = now()
        if invalidate and document["match"]["status"] == "finalized":
            document["match"]["status"] = "review"
        db.execute("UPDATE matches SET data=? WHERE id=?", (json.dumps(document, allow_nan=False), match_id))


def delete(match_id):
    with LOCK, connect() as db:
        db.execute("DELETE FROM matches WHERE id=?", (match_id,))


def audit(document, action, before=None, after=None):
    document.setdefault("revisions", []).append({"id": new_id(), "time": now(), "action": action, "before": copy.deepcopy(before), "after": copy.deepcopy(after)})


def recover_jobs():
    for document in all_documents():
        if any(job["status"] in ("queued", "running") for job in document["jobs"]) or document["match"]["status"] == "uploading":
            with edit(document["match"]["id"], invalidate=False) as current:
                for job in current["jobs"]:
                    if job["status"] in ("queued", "running"):
                        job.update(status="interrupted", stage="Proses terhenti. Jalankan analisis ulang.", updated_at=now())
                current["match"]["status"] = "review" if current["match"].get("video_path") else "draft"
