import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FUTSAL_DATA_DIR", ROOT / "data")).resolve()
MODEL_PATH = Path(os.environ.get("FUTSAL_MODEL", ROOT / "models" / "yolo11n.pt")).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("FUTSAL_MAX_UPLOAD_MB", "4096")) * 1024 * 1024
FRONTEND_DIST = ROOT / "frontend" / "dist"


def prepare_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "matches").mkdir(exist_ok=True)


def match_dir(match_id: str) -> Path:
    # IDs are server-generated hex strings and validated before reaching this function.
    directory = DATA_DIR / "matches" / match_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory
