"""Fetch official Ultralytics weights once. Analysis itself never downloads files."""
import hashlib
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models" / "yolo11n.pt"
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION.is_file():
        temporary = DESTINATION.with_suffix(".download")
        print("Downloading official YOLO11n weights (~5.4 MB)...")
        try:
            urllib.request.urlretrieve(URL, temporary)
            if temporary.stat().st_size < 1_000_000:
                raise RuntimeError("Unduhan terlalu kecil; bobot model tidak valid.")
            os.replace(temporary, DESTINATION)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Model: {DESTINATION}")
    print(f"SHA256: {hashlib.sha256(DESTINATION.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
