# Futsal Vision

Native desktop app: tracks only futsal players whose feet are inside the blue court, keeps one ID per player, and follows the ball.

## Run

```powershell
pip install -r requirements.txt
python app.py
```

Open a video, press **Analisis**, wait for the offline pass, then playback starts automatically. For videos over 3 minutes the app offers to analyze only the first 2 minutes; full-match analysis can take hours at 1080p.

Model files already in the project:

- `yolo11s.pt` — person + sports-ball detector
- `models/osnet_x0_25_msmt17.pt` — BoT-SORT ReID weights

Requires an NVIDIA GPU/CUDA for the current settings (`device=0`).

## Pipeline

1. Per-frame HSV mask keeps the largest bright-blue floor region.
2. Person detections are discarded unless the bottom-center of the box is on that mask. Referee, bench, crowd and red perimeter are excluded.
3. BoT-SORT + OSNet ReID tracks players; `sof` camera-motion compensation handles pan/zoom. Lost tracks are retained 150 frames, so a cut has time to re-acquire by appearance.
4. Ball candidates are gated in court-normalized coordinates: a detection that teleports is rejected instead of drawing a fake trail.
5. Analysis is offline; the annotated result is rendered during playback.

## Limits

- Court detection expects a bright blue floor. If no floor is found, the app counts all person detections instead of guessing (safer fallback).
- Hard cut to a completely different camera angle may start a new ID if appearance changes too much.
- Ball coverage follows the detector; fast motion blur or occlusion drops frames. No invented long trajectories.

Check: `python test_vision.py`
