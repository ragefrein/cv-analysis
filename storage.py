"""Atomic, compressed per-video analysis files (no pickle or executable data)."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = 2
RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "sessions"


def video_signature(video_path):
    path = Path(video_path).resolve()
    stat = path.stat()
    # The sample catches replacement even when a copied file preserves mtime.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(65536))
        stream.seek(max(0, stat.st_size - 65536))
        digest.update(stream.read(65536))
    return {"size": stat.st_size, "modified_ns": stat.st_mtime_ns, "sample": digest.hexdigest()}


def session_path(video_path):
    path = Path(video_path).resolve()
    key = hashlib.sha256(str(path).casefold().encode("utf-8")).hexdigest()[:16]
    return RESULTS_DIR / f"{path.stem[:60]}-{key}.json.gz"


def save_session(video_path, tracks, destination=None):
    destination = Path(destination) if destination else session_path(video_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": SCHEMA_VERSION, "video": video_signature(video_path), "tracks": tracks}
    handle, temporary = tempfile.mkstemp(prefix=".session-", dir=destination.parent)
    os.close(handle)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_session(video_path, source=None):
    source = Path(source) if source else session_path(video_path)
    if not source.exists():
        return None
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("version") != SCHEMA_VERSION or payload.get("video") != video_signature(video_path):
        return None
    tracks = payload["tracks"]
    if tracks["frame_count"] != len(tracks["frames"]) or not isinstance(tracks["players"], dict):
        raise ValueError("Data analisis tidak lengkap.")
    return tracks
