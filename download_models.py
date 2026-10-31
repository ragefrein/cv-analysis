"""Download the official YOLO11 pose checkpoint into the project."""
from pathlib import Path
from urllib.request import urlretrieve


def main():
    destination = Path(__file__).resolve().parent / "models" / "yolo11s-pose.pt"
    if destination.exists():
        print(f"Model tersedia: {destination}")
        return
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt", temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Model tersimpan: {destination}")


if __name__ == "__main__":
    main()
