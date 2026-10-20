# VisionShield AI

**An intelligent real-time multi-object detection, tracking and threat-assessment platform using computer vision.**

VisionShield AI processes a recorded video, a webcam / IP camera, or a single image. It detects objects, tracks them across frames, analyses their movement and position relative to configured zones, computes a configurable **explainable threat score (0–100)**, and raises prioritised alerts that are stored in SQLite and shown on a live dashboard.

> **Status: local prototype.** There is no authentication. Do not expose it to a network before securing it.

## Proposed contribution (and what is *not* claimed)

The contribution is the **integrated, configurable, context-aware workflow**, not any single algorithm:

| Question | Answered by |
|---|---|
| What objects are visible? | Detection (pretrained YOLO) |
| Which observations are the same object? | Tracking (ByteTrack) |
| How is it moving? | Movement analysis (speed, direction, approach) |
| Where is it relative to monitored areas? | Zone analysis (inside / near / outside, entry) |
| How long has the condition lasted? | Persistence (dwell time with grace for track loss) |
| How should it be prioritised? | Weighted threat scoring with per-factor explanation |
| What needs attention now? | Alerting with cooldown / de-duplication |
| What happened, when and why? | SQLite event log + dashboard |

Not claimed: YOLO, ByteTrack, OpenCV and FastAPI are existing tools. The YOLO model is a general COCO-pretrained network used for inference only — it is **not** trained for threat detection and no accuracy numbers are reported. Tracking IDs identify objects within one session, not real people. The score weights are engineering defaults, **not** scientifically validated probabilities. Novelty over the wider literature has not been established (no literature comparison was done).

## Architecture

```
Video/Camera/Image → Pre-processing → Detection (YOLO) → Tracking (ByteTrack)
   → Zone + Motion analysis → Threat scoring → Alert manager → SQLite
                                                   ↘ WebSocket → Dashboard
   annotated JPEG (shared latest frame) → MJPEG stream → Dashboard
```

* One background capture loop (`SurveillanceService`) owns the source, so any number of dashboard clients share it; a second session is refused with `409`.
* Inference never runs in API handlers. Frames never travel over the WebSocket (events and small status JSON only); video is served as MJPEG from the latest annotated frame.
* Recorded videos use the **video timeline** (frame index / fps) for movement and persistence, so results do not depend on how fast the CPU processes the file.

### Folder structure

```
app/
  main.py                 FastAPI app factory (uvicorn app.main:app)
  core/                   config (env-validated), logging, exceptions
  vision/                 video_source, image_source, preprocessing, detector, tracker,
                          zone_analyzer, motion_analyzer, annotator, pipeline, types
  assessment/             threat_scorer, alert_manager
  services/               surveillance_service (session lifecycle), event_hub (WebSocket fan-out)
  db/                     database, models, repository (SQLAlchemy + SQLite)
  schemas/                Pydantic request/response models
  api/routes/             health, sessions, objects, events, zones, analytics, stream, uploads, ws
  evaluation/             optional offline dataset evaluation (MOT17, VisDrone, UCF-Crime)
  web/                    dashboard.html, styles.css, dashboard.js
config/zones.example.json example restricted zone
data/videos, data/images, data/datasets    your inputs (git-ignored)
scripts/                  evaluate_dataset.py, make_demo_video.py
tests/                    unit/ and integration/ (no webcam or model download needed)
docs/DATASETS.md          dataset setup guide
```

## Setup

Requires Python 3.11+ (developed on 3.12). The first run downloads the small pretrained weights `yolo11n.pt` (~5 MB) automatically via Ultralytics into the working directory; set `YOLO_MODEL` to use another model or a local path. No GPU is needed (CPU is the default); set `YOLO_DEVICE=0` for a CUDA GPU.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

> If `pip`/`python` in the venv picks up packages from elsewhere (e.g. imports fail with odd version errors), check that the `PYTHONPATH` environment variable is empty: PowerShell `$env:PYTHONPATH=""`, bash `unset PYTHONPATH`.

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> for the dashboard and <http://127.0.0.1:8000/docs> for the API docs. The database (`data/visionshield.db`) is created automatically at startup.

### Quick demo without a camera (recommended path)

1. Put any MP4/AVI with people or vehicles into `data/videos/` (or use **Upload video** in the dashboard). A video you have permission to use is required. No video handy? Generate a pan across any street photo:
   ```bash
   python scripts/make_demo_video.py --image path/to/photo.jpg --out data/videos/demo_pan.mp4
   ```
2. In the dashboard choose **Recorded video**, pick the file, press **Start**.
3. Watch boxes, IDs, scores and the zone overlay; events appear in *Event History*.
4. **Upload image** mode analyses one still image (movement/persistence are reported as *unknown* — a single image has no temporal data — and contribute 0).
5. **Live camera** needs a working webcam (`CAMERA_INDEX`); **IP stream** uses `STREAM_URL` from `.env` (never accepted through the API).

### Input modes

| Mode | How | Temporal analysis |
|---|---|---|
| Recorded video | `POST /api/v1/sessions {"source_type":"video","video_name":"x.mp4"}` | yes |
| Live camera | `{"source_type":"camera"}` (uses `CAMERA_INDEX`) | yes |
| IP/RTSP stream | `{"source_type":"stream"}` (uses `STREAM_URL`) | yes |
| Still image | `POST /api/v1/images/analyze` (multipart `file`) | no |

## Threat scoring

```
Score = 100 × (w1·ObjectRisk + w2·Confidence + w3·Movement + w4·Proximity + w5·Persistence)
```

| Factor | Normalised 0–1 from |
|---|---|
| Object risk | per-class table (`OBJECT_RISK`); unlisted classes use `DEFAULT_OBJECT_RISK` |
| Confidence | detector confidence (not a threat probability) |
| Movement | `0.6·speed/(SPEED_REF_DIAG_PER_S·frame diagonal) + 0.4·[approaching a restricted zone]` |
| Proximity | 1 inside a restricted zone, falling linearly to 0 at `ZONE_NEAR_FRACTION` × diagonal |
| Persistence | seconds in a monitored zone / `DWELL_REF_S` (capped at 1) |

Defaults: weights 0.30 / 0.15 / 0.20 / 0.20 / 0.15 (must sum to 1.0; validated at startup). Levels: **LOW 0–30, MEDIUM 31–60, HIGH 61–100** (fractional scores above 30 / 60 fall into the next band). A person alone scores LOW; high scores need combinations of context. Every event stores each factor's value, weight and point contribution plus a text explanation (`GET /api/v1/events/{id}`).

To tune: edit the `WEIGHT_*`, `OBJECT_RISK`, and reference settings in `.env`, replay the same recorded video, and compare the stored events. To *evaluate* the scores you need your own labelled scenarios; the public datasets do not provide ground truth for this custom score.

Speeds are **image-space pixels per second**, not metres per second (that needs camera calibration).

### Alerts

An event is emitted on a zone entry or when the level differs from the last emitted level. Escalations fire immediately; de-escalations wait for `ALERT_COOLDOWN_S`; levels below `ALERT_MIN_LEVEL` (default MEDIUM) are not alerted; a zone entry and level change on the same frame merge into one event. Dropping out of the alerting band and coming back is reported again once the cooldown passed.

## Zones

Zones are polygons in image coordinates declared against a `reference_size`, then scaled to the processing resolution. A default demo zone is built in. To customise:

```bash
cp config/zones.example.json config/zones.json     # PowerShell: Copy-Item
```

Edit `polygon` points (`restricted: true` zones drive proximity/approach scoring; `restricted: false` zones are monitored for dwell time only). The dashboard overlay and `GET /api/v1/zones` reflect the file after a server restart.

## API summary

`GET /health` · `GET /api/v1/config` · `GET|POST /api/v1/sessions` · `GET /api/v1/sessions/{id}` · `POST /api/v1/sessions/{id}/stop` · `GET /api/v1/objects` · `GET /api/v1/events` (`page`, `page_size`, `level`, `start`, `end`, `session_id`) · `GET /api/v1/events/{id}` · `GET /api/v1/zones` · `GET /api/v1/analytics/summary` · `WS /ws/events` — plus `GET /api/v1/sources`, `GET /api/v1/stream` (MJPEG), `GET /api/v1/frame`, `POST /api/v1/images/analyze`, `POST /api/v1/videos/upload`. Errors use `{"error": {"code", "message"}}`.

## Tests

```bash
python -m pytest
```

The suite (unit + integration) mocks the detector, tracker input and video source, so it needs no webcam, GPU or model download. Covered: score calculation and 0/30/31/60/61/100 boundaries, weight/factor validation, zone membership and entry, movement edge cases, persistence, alert de-duplication, event persistence/pagination/filtering, health and validation endpoints, WebSocket events, and session stop releasing the camera and worker thread.

Verification status of this repository: the test suite passes (122 tests; the dataset runners are exercised only on tiny synthetic datasets, never on MOT17/VisDrone/UCF-Crime), and a manual smoke test with the real `yolo11n.pt` model + ByteTrack on a generated pan video and one photo ran end-to-end (detections, track IDs, scores, zone entries, stored events, annotated frames; ~6 fps on a CPU). A webcam was not tested, and the dashboard was driven in real Chrome (Playwright): start/stop, live feed, telemetry, charts, event filter and image mode all worked (screenshots in `outputs/demo/`, git-ignored; that run exposed and fixed a stuck Start button and a broken-image placeholder).

## Offline dataset evaluation (optional)

Implemented runners for **MOT17** (tracking via `motmetrics`), **VisDrone2019-DET** (precision/recall/F1 at IoU 0.5) and **UCF-Crime** (video-level descriptive analysis). Nothing is downloaded automatically; see [docs/DATASETS.md](docs/DATASETS.md) for layouts and commands. **No evaluation results are included — none have been run on the full datasets.** The threat classification itself is only evaluated where defensible labels exist (none of the three datasets provide them).

## Troubleshooting

* **Camera won't open** – close other apps using it, try another `CAMERA_INDEX`, or use a recorded video. The API returns `503 source_unavailable`.
* **Model errors (`503 model_unavailable`)** – first run needs internet to fetch `yolo11n.pt`, or set `YOLO_MODEL` to a local `.pt` file.
* **Slow processing** – lower `MAX_FRAME_WIDTH`, `INFERENCE_IMGSZ` or `MAX_FPS`; use a GPU via `YOLO_DEVICE`.
* **Dependency/import errors** – activate the venv and check `PYTHONPATH` is empty (see Setup). `motmetrics` is only needed for MOT17 evaluation.
* **Dashboard charts empty/blank** – Chart.js loads from a CDN (jsDelivr); offline machines will lack charts but everything else works.
* **Config rejected at startup** – weights must sum to 1.0 and probabilities must be in [0, 1]; the error names the setting.

## Limitations and future work

* Single session at a time, no authentication, local use only; SQLite only.
* Pretrained COCO classes only; object risk is a hand-set table. No face/identity recognition or re-identification by design.
* Zone checks use the box centroid; no camera calibration, so no real-world distances or speeds.
* Scoring is deterministic and unvalidated; future work: labelled scenarios for tuning, learned scoring, Deep SORT / BoT-SORT adapters (the tracker is behind the `ObjectTracker` interface), per-camera zone editor in the UI.
#   V I S I O N  
 