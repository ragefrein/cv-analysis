# Futsal Vision implementation

Delivery status, 31 October 2026: local assisted-analysis application implemented. See README.md for launch/workflow and VALIDATION.md for passing checks and actual automation boundaries. The research roadmap is not a claim that every advanced statistic has been automated.

Build a local-first, single-user application for uploaded futsal recordings from a moving camera. Preserve the full research roadmap in FUTSAL_ANALYTICS_PLAN.md; implement a usable assisted analysis workflow with real detection and honest uncertainty.

## Delivery

- React + TypeScript + Vite frontend in `frontend/`, a Python FastAPI API in `backend/`, SQLite and media in `data/` (runtime only).
- One match workspace: overview, synced video review with overlays, roster/track assignments, event review and manual annotation, statistics and exports, optional segment calibration.
- Jobs use one background executor, persistent status, cancel/retry, and interrupted-job recovery. AI predictions are always pending; confirmed/manual events determine final counts.
- Demo is an explicitly labelled synthetic fixture, created only on request. Never use sample stats as a real analysis result.
- Test API invariants, aggregation, upload/range serving, pipeline smoke and frontend build plus browser workflows. Document launch and limitations.

## API contract

All endpoints under `/api`. IDs are strings. Timestamps use video seconds. Match detail is `{match, players, tracks, events, jobs, calibrations, stats}`.

Match fields: `id,title,home_name,away_name,home_color,away_color,played_at,venue,status,video_name,video_url,poster_url,duration,width,height,fps,is_demo,created_at,notes`.

Player fields: `id,match_id,name,number,team` (`home` or `away`), `role` (`player` or `goalkeeper`).

Track fields: `id,match_id,track_id` (integer), `player_id` (nullable), `team` (`home`,`away`,`unknown`), `start_time,end_time,detection_count,confidence`.

Event fields: `id,match_id,type` (`pass,shot,goal,assist,interception,turnover,recovery,save,block,dribble,foul,yellow_card,red_card,substitution,kick_in,corner,penalty,possession`), `time,end_time` (nullable), `team` (`home`,`away`,`unknown`), `player_id,receiver_id` (nullable), `outcome` (`successful,failed,unknown,on_target,off_target,blocked`), `status` (`pending,confirmed,rejected`), `source` (`manual,ai,demo`), `confidence` (nullable), `notes,x,y` (nullable position in meters).

Job fields: `id,match_id,status` (`queued,running,completed,failed,cancelled,interrupted`), `progress,stage,error,created_at,updated_at,options,metrics`.

Calibration: `id,match_id,start_time,end_time,court_length,court_width,image_points` (4 normalized points in TL,TR,BR,BL correspondence), `court_points` (4 `[x,y]` meter points), `quality` (`manual`), `notes`.

Routes:

- GET `/health` -> `{status,ai:{available,model_ready,model,device,message},ffmpeg_available}`
- GET `/matches` -> array of matches (with score and summary).
- POST `/matches` JSON `{title,home_name,away_name,home_color?,away_color?,played_at?,venue?}` -> match.
- GET/PATCH/DELETE `/matches/{id}` -> detail / updated match / 204.
- POST `/matches/{id}/video` multipart `file` -> match. Stream upload, validate container, create browser-compatible proxy if needed; original kept private.
- GET `/matches/{id}/video` -> byte range supported video.
- GET `/matches/{id}/poster` -> jpg.
- POST `/demo` -> demo match; explicitly synthetic data and synthetic court recording.
- POST `/matches/{id}/players` -> player; PATCH/DELETE `/matches/{id}/players/{player_id}`.
- PATCH `/matches/{id}/tracks/{track_id}` body `{player_id,team?}` -> track; `track_id` is integer tracker ID.
- GET `/matches/{id}/detections?time=SECONDS&window=0.3` -> `{frames:[{time,objects:[{track_id,kind,bbox:[x1,y1,x2,y2],confidence,team,player_id}]}]}`; normalized bbox.
- GET `/matches/{id}/trajectories` -> `{points:[{time,track_id,x,y,team,player_id}],coordinate_system:'image'}`; sampled image coordinates. Never label as court heatmap.
- POST/PATCH/DELETE `/matches/{id}/events[/{event_id}]` -> event / event / 204; event time/player/team validation.
- POST `/matches/{id}/events/review` `{ids:[...],status:'confirmed'|'rejected'}` -> `{updated:N}`.
- POST `/matches/{id}/jobs` `{sample_fps:5,confidence:0.25,detect_ball:true}` -> job. Reject duplicate active job and matches with no video. `sample_fps` allowed 1..15; pipeline smoke can use 1.
- POST `/matches/{id}/jobs/{job_id}/cancel` -> job.
- POST `/matches/{id}/calibrations` -> calibration; DELETE `/matches/{id}/calibrations/{calibration_id}`.
- GET `/matches/{id}/stats` -> `{score:{home,away},teams:{home:TeamStats,away:TeamStats},players:[PlayerStats],pending_count,confirmed_count,rejected_count,coverage:{duration,analyzed_seconds,ball_visible_seconds,identity_seconds,calibrated_seconds},pass_network:[{from_player_id,to_player_id,count,team}],event_timeline:[{minute,home,away}],warnings:[...]}`.
- TeamStats: `goals,passes,passes_successful,passes_failed,passes_unresolved,pass_accuracy` (null if denominator zero), `shots,shots_on_target,assists,interceptions,turnovers,recoveries,saves,blocks,dribbles,fouls,yellow_cards,red_cards,possession_seconds,possession_pct`.
- PlayerStats same counting fields plus `id,name,number,team,role`.
- GET `/matches/{id}/export?format=json|csv|events|report` -> JSON full export, player stats CSV, events CSV, printable HTML report.

## Vision module boundary

`backend/vision.py` owns `capabilities()`, `probe_video(path)->{duration,width,height,fps}`, `make_poster(video_path,poster_path)`, `analyze_video(video_path,output_path,options,on_progress,is_cancelled)->dict` and `create_demo_video(path)->metadata`.

`analyze_video` writes JSONL frames using normalized bbox as above. Return `{tracks:[{track_id,team,start_time,end_time,detection_count,confidence}],events:[{type,time,end_time?,team,track_id?,receiver_track_id?,outcome,confidence,notes}],metrics:{analyzed_seconds,ball_visible_seconds,identity_seconds,...}}`. Never assign real player IDs automatically. API owns DB persistence; identity mapping through manual review.

Use actual Ultralytics model (default yolo11n.pt under project models/) if installed. No fake fallback detections. Missing dependencies/weights -> explicit actionable failure. Camera compensation is enabled, ball candidates conservative. No unsupported automatic goal claim from mere ball proximity. Calibration is manual per time segment; physical metrics suppressed outside valid segments.

## Work ownership

- Frontend specialist owns `frontend/` only, using the API contract.
- Vision specialist owns `backend/vision.py`, `backend/tracking.yaml`, `tests/test_vision.py`, optional `scripts/` fixtures only. Do not install dependencies or create shared init files.
- Root owns API, persistence, statistics, dependencies, launch scripts, README, integration tests and integration fixes.
