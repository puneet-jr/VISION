# VisionShield AI

> **An intelligent, real-time multi-object detection, tracking, and
> threat-assessment platform using computer vision.**

VisionShield AI analyses recorded video, a live webcam/IP-camera stream,
or a single image. It detects objects, tracks them across video frames,
evaluates movement and position relative to configured zones, calculates
an explainable **threat score (0--100)**, and raises prioritised alerts.
Events are stored in SQLite and displayed in a browser-based dashboard.

> \[!WARNING\] **Prototype --- local use only.** Authentication is not
> implemented. Keep the application bound to localhost and do not expose
> it to a network before adding appropriate security controls.

## Contents

-   [Overview](#overview)
-   [Features](#features)
-   [How it works](#how-it-works)
-   [Technology stack](#technology-stack)
-   [Project structure](#project-structure)
-   [Requirements](#requirements)
-   [Installation](#installation)
-   [Run the application](#run-the-application)
-   [Demo without a camera](#demo-without-a-camera)
-   [Input modes](#input-modes)
-   [Threat scoring](#threat-scoring)
-   [Alerts and zones](#alerts-and-zones)
-   [API overview](#api-overview)
-   [Testing](#testing)
-   [Optional dataset evaluation](#optional-dataset-evaluation)
-   [Troubleshooting](#troubleshooting)
-   [Limitations and future work](#limitations-and-future-work)
-   [Research contribution and responsible
    use](#research-contribution-and-responsible-use)

## Overview

VisionShield combines established computer-vision components into one
context-aware workflow. Instead of treating each detection as an
isolated result, it uses tracking and scene context to help prioritise
events for review.

  -----------------------------------------------------------------------
  Question                            Component
  ----------------------------------- -----------------------------------
  What objects are visible?           YOLO object detection

  Which detections belong to the same ByteTrack tracking
  object across frames?               

  How is an object moving?            Movement analysis

  Is it inside or approaching a       Zone and proximity analysis
  monitored area?                     

  How long has a condition persisted? Persistence / dwell-time analysis

  How should an event be prioritised? Weighted threat scoring

  What needs attention?               Alert manager with cooldown and
                                      de-duplication

  What happened, when, and why?       SQLite event history and dashboard
  -----------------------------------------------------------------------

## Features

-   **Flexible inputs:** recorded video, webcam, configured IP/RTSP
    stream, and still image.
-   **Object detection:** inference with a pretrained YOLO model.
-   **Multi-object tracking:** ByteTrack IDs across frames within a
    session.
-   **Context analysis:** movement, approach to restricted zones,
    proximity, and dwell time.
-   **Explainable scoring:** configurable weighted factors and
    per-factor explanations.
-   **Prioritised alerts:** configurable alert level, cooldown, and
    duplicate suppression.
-   **Live dashboard:** annotated feed, tracked objects, zone overlays,
    telemetry, charts, and event history.
-   **Persistent event records:** SQLite storage and summary analytics.
-   **API access:** REST endpoints and WebSocket event updates.
-   **Automated tests:** unit and integration tests designed to run
    without a webcam, GPU, or model download.

## How it works

``` text
Video / Camera / IP Stream / Image
                 |
            Pre-processing
                 |
            YOLO Detection
                 |
           ByteTrack Tracking
                 |
      +----------+----------+
      |                     |
  Zone Analysis        Motion Analysis
      |                     |
      +----------+----------+
                 |
        Weighted Threat Score
                 |
            Alert Manager
                 |
          SQLite Event Log
                 |
        Dashboard / WebSocket

Latest annotated frame -----------------> MJPEG stream
```

### Processing design

-   `SurveillanceService` owns one background capture loop. Dashboard
    clients share the active session; a second session is refused with
    HTTP `409`.
-   Inference runs in the background pipeline rather than in API request
    handlers.
-   WebSockets send events and small status payloads, not video frames.
    Annotated video is served through MJPEG.
-   Recorded-video movement and persistence use the source timeline
    (frame index and FPS), not CPU processing speed.
-   A still image has no timeline, so movement and persistence are
    **unknown** and contribute zero to its score.

## Technology stack

  Area                   Technology
  ---------------------- -----------------------------------------
  Language               Python 3.11+
  Computer vision        OpenCV, NumPy
  Detection              Ultralytics YOLO / PyTorch
  Tracking               ByteTrack
  API                    FastAPI, Uvicorn, Pydantic
  Database               SQLAlchemy, SQLite
  Live updates / video   WebSockets, MJPEG
  Dashboard              HTML, CSS, JavaScript, Chart.js
  Testing                pytest
  Optional evaluation    motmetrics and dataset-specific runners

## Project structure

``` text
app/
├── main.py                  # FastAPI entry point
├── core/                    # Configuration, logging, exceptions
├── vision/                  # Sources, preprocessing, detector, tracker,
│                            # zone/motion analysis, annotation, pipeline, types
├── assessment/              # Threat scorer and alert manager
├── services/                # Session lifecycle and event fan-out
├── db/                      # SQLAlchemy models, database, repository
├── schemas/                 # Pydantic request/response schemas
├── api/routes/              # Health, sessions, objects, events, zones,
│                            # analytics, streaming, uploads, WebSocket
├── evaluation/              # Optional offline dataset evaluation
└── web/                     # Dashboard HTML, CSS, JavaScript
config/
└── zones.example.json       # Example zone configuration
data/
├── videos/                  # Local video inputs (git-ignored)
├── images/                  # Local image inputs (git-ignored)
└── datasets/                # Optional datasets (git-ignored)
scripts/
├── evaluate_dataset.py      # Dataset evaluation entry point
└── make_demo_video.py       # Create a pan video from a still image
tests/
├── unit/
└── integration/
docs/
└── DATASETS.md              # Dataset layouts and evaluation instructions
```

## Requirements

-   Python **3.11 or later** (developed with Python 3.12).
-   Internet access on the first run to download the default model,
    unless a local model file is configured.
-   CPU is supported by default; a compatible CUDA setup can be
    configured for GPU inference.
-   A webcam is optional. Recorded video or a still image is enough to
    try the project.

The default model is `yolo11n.pt` (approximately 5 MB). Ultralytics
downloads it when needed. Set `YOLO_MODEL` to select another model or a
local `.pt` file.

## Installation

Run the commands from the repository root.

### Windows (PowerShell)

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux / macOS

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Configuration

Review `.env` before starting. Common settings include:

  -----------------------------------------------------------------------
  Setting                             Purpose
  ----------------------------------- -----------------------------------
  `YOLO_MODEL`                        Model name or local model path

  `YOLO_DEVICE`                       Inference device; CPU by default,
                                      `0` for CUDA GPU 0 where supported

  `CAMERA_INDEX`                      Local camera index

  `STREAM_URL`                        Configured IP/RTSP stream URL

  `WEIGHT_*`                          Threat-score factor weights

  `OBJECT_RISK`,                      Per-class and fallback object-risk
  `DEFAULT_OBJECT_RISK`               values

  `SPEED_REF_DIAG_PER_S`              Reference for movement
                                      normalisation

  `ZONE_NEAR_FRACTION`                Distance threshold for zone
                                      proximity

  `DWELL_REF_S`                       Reference dwell time for
                                      persistence

  `ALERT_MIN_LEVEL`,                  Alert threshold and cooldown
  `ALERT_COOLDOWN_S`                  
  -----------------------------------------------------------------------

Use `.env.example` as the source of truth for the exact setting names
and defaults in your checkout.

**Import troubleshooting:** If imports behave unexpectedly, activate the
virtual environment and check that `PYTHONPATH` is empty. In PowerShell
run `$env:PYTHONPATH=""`; in bash run `unset PYTHONPATH`.

## Run the application

With the virtual environment activated:

``` bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open these URLs in your browser:

-   **Dashboard:** http://127.0.0.1:8000/
-   **Interactive API documentation:** http://127.0.0.1:8000/docs
-   **Health check:** http://127.0.0.1:8000/health

The SQLite database is created automatically at `data/visionshield.db`
during startup.

## Demo without a camera

**Recommended first run:** use a recorded video. You do not need to buy
or connect a separate camera.

1.  Place a video containing people or vehicles in `data/videos/`, or
    use **Upload video** in the dashboard. Only use media you have
    permission to process.

2.  If you do not have a video, create a simple pan video from a still
    image:

    ``` bash
    python scripts/make_demo_video.py --image path/to/photo.jpg --out data/videos/demo_pan.mp4
    ```

3.  Start the server and open the dashboard.

4.  Choose **Recorded video**, select the file, and click **Start**.

5.  Review detections, track IDs, zone overlays, threat scores, and the
    **Event History** panel.

6.  Try **Upload image** to analyse a single frame. Movement and
    persistence will be unknown because a still image has no temporal
    data.

For a webcam, configure `CAMERA_INDEX`. For an IP/RTSP stream, set
`STREAM_URL` in `.env`. The stream URL is not accepted directly through
the API.

## Input modes

  ------------------------------------------------------------------------------------------------
  Mode                    API / configuration                              Temporal analysis
  ----------------------- ------------------------------------------------ -----------------------
  Recorded video          `POST /api/v1/sessions` with                     Yes
                          `{"source_type":"video","video_name":"x.mp4"}`   

  Live camera             `POST /api/v1/sessions` with                     Yes
                          `{"source_type":"camera"}`; uses `CAMERA_INDEX`  

  IP/RTSP stream          `POST /api/v1/sessions` with                     Yes
                          `{"source_type":"stream"}`; uses configured      
                          `STREAM_URL`                                     

  Still image             `POST /api/v1/images/analyze` with multipart     No
                          field `file`                                     
  ------------------------------------------------------------------------------------------------

## Threat scoring

The configurable score is a weighted combination of five normalised
factors:

``` text
Score = 100 × (w1·ObjectRisk + w2·Confidence + w3·Movement
               + w4·Proximity + w5·Persistence)
```

  --------------------------------------------------------------------------------------------------
  Factor                              How it is estimated
  ----------------------------------- --------------------------------------------------------------
  **Object risk**                     Per-class value from `OBJECT_RISK`; unlisted classes use
                                      `DEFAULT_OBJECT_RISK`.

  **Confidence**                      Detector confidence. This is confidence in the detection,
                                      **not** the probability that an object is a threat.

  **Movement**                        `0.6 × normalised speed + 0.4 × approaching-zone indicator`.

  **Proximity**                       1 inside a restricted zone, falling linearly to 0 at the
                                      configured near-zone distance.

  **Persistence**                     Time in a monitored zone divided by `DWELL_REF_S`, capped at
                                      1.
  --------------------------------------------------------------------------------------------------

### Default weights

  Factor            Weight
  ------------- ----------
  Object risk         0.30
  Confidence          0.15
  Movement            0.20
  Proximity           0.20
  Persistence         0.15
  **Total**       **1.00**

Weights must sum to `1.0`; configuration is validated at startup.

### Score levels

      Score Level
  --------- ------------
      0--30 **LOW**
     31--60 **MEDIUM**
    61--100 **HIGH**

Fractional scores above 30 or 60 fall into the next band. A person
detected by itself is intended to remain low-risk; higher scores
generally require a combination of context.

Every stored event includes factor values, weights, point contributions,
and a text explanation. Use `GET /api/v1/events/{id}` to retrieve the
explanation.

**Interpretation limits:** the weights and class-risk values are
engineering defaults, not validated probabilities. Speeds are measured
in **image-space pixels per second**, not metres per second; real-world
speeds require camera calibration. For still images, movement and
persistence are unknown and contribute zero. To tune scoring, change the
relevant settings in `.env`, replay the same video, and compare the
stored events. Proper evaluation requires labelled scenarios; the listed
public datasets do not provide ground truth for this custom score.

## Alerts and zones

### Alert behaviour

-   An event is emitted on zone entry or when the alert level changes.
-   Escalations are emitted immediately; de-escalations wait for
    `ALERT_COOLDOWN_S`.
-   Levels below `ALERT_MIN_LEVEL` (default `MEDIUM`) are not alerted.
-   Zone entry and a level change on the same frame are merged into one
    event.
-   If a condition drops below the alerting band and later returns, it
    can be reported again after the cooldown.

### Zone configuration

Zones are polygons in image coordinates, declared against a
`reference_size` and scaled to the processing resolution.

Create a local configuration file:

**Windows (PowerShell)**

``` powershell
Copy-Item config/zones.example.json config/zones.json
```

**Linux / macOS**

``` bash
cp config/zones.example.json config/zones.json
```

Edit `config/zones.json` following the example schema:

-   `restricted: true` zones affect proximity and approach scoring.
-   `restricted: false` zones are monitored for dwell time only.

The dashboard overlay and `GET /api/v1/zones` reflect the file after the
server restarts.

## API overview

Full request and response schemas are available at `/docs`.

  ------------------------------------------------------------------------------
  Method                  Endpoint                       Purpose
  ----------------------- ------------------------------ -----------------------
  `GET`                   `/health`                      Health status

  `GET`                   `/api/v1/config`               Effective configuration

  `GET`, `POST`           `/api/v1/sessions`             Read active session /
                                                         start a session

  `GET`                   `/api/v1/sessions/{id}`        Session status

  `POST`                  `/api/v1/sessions/{id}/stop`   Stop a session

  `GET`                   `/api/v1/sources`              Available input sources

  `GET`                   `/api/v1/objects`              Current tracked objects

  `GET`                   `/api/v1/events`               List events with
                                                         filters and pagination

  `GET`                   `/api/v1/events/{id}`          Event detail and score
                                                         explanation

  `GET`                   `/api/v1/zones`                Configured zones

  `GET`                   `/api/v1/analytics/summary`    Summary analytics

  `GET`                   `/api/v1/frame`                Latest annotated frame

  `GET`                   `/api/v1/stream`               Annotated MJPEG stream

  `POST`                  `/api/v1/images/analyze`       Analyse an uploaded
                                                         image

  `POST`                  `/api/v1/videos/upload`        Upload a video

  `WS`                    `/ws/events`                   Live event and status
                                                         updates
  ------------------------------------------------------------------------------

The events endpoint supports `page`, `page_size`, `level`, `start`,
`end`, and `session_id` filters. Error responses use
`{"error":{"code":"...","message":"..."}}`.

## Testing

Run the test suite from the repository root:

``` bash
python -m pytest
```

Unit and integration tests mock the detector, tracker input, and video
source. They do not require a webcam, GPU, or model download.

Test coverage includes:

-   Score calculations and boundaries at 0, 30, 31, 60, 61, and 100.
-   Weight and factor validation.
-   Zone membership and entry.
-   Movement edge cases and persistence.
-   Alert de-duplication.
-   Event persistence, pagination, and filtering.
-   Health and request-validation endpoints.
-   WebSocket event delivery.
-   Session shutdown and release of the camera/worker thread.

### Recorded verification status

The supplied project notes report **122 passing tests** at the time the
source README was prepared. They also report a manual CPU smoke test
using `yolo11n.pt`, ByteTrack, a generated pan video, and a photo,
running end-to-end at approximately 6 FPS. A webcam was **not** tested.
A manual Chrome dashboard check covered start/stop, live feed,
telemetry, charts, event filtering, and image mode.

These are previously recorded results, not a guarantee for every
environment or code revision. Re-run the tests and smoke checks against
your current checkout before presenting them as current results.

## Optional dataset evaluation

  Dataset                Evaluation focus
  ---------------------- ------------------------------------------
  **MOT17**              Multi-object tracking using `motmetrics`
  **VisDrone2019-DET**   Precision, recall, and F1 at IoU 0.5
  **UCF-Crime**          Video-level descriptive analysis

Dataset files are not downloaded automatically. See
[`docs/DATASETS.md`](docs/DATASETS.md) for layouts and commands.

**No full-dataset results are included.** The runners have only been
exercised on small synthetic datasets, not on the full MOT17,
VisDrone2019-DET, or UCF-Crime datasets. These datasets also do not
provide ground-truth labels for VisionShield's custom threat score.

## Troubleshooting

  -----------------------------------------------------------------------
  Problem                             What to check
  ----------------------------------- -----------------------------------
  Camera will not open                Close other apps using it, try
                                      another `CAMERA_INDEX`, or test
                                      with recorded video. The API may
                                      return `503 source_unavailable`.

  Model unavailable                   Allow the first run to download
  (`503 model_unavailable`)           `yolo11n.pt`, or set `YOLO_MODEL`
                                      to a local `.pt` file.

  Processing is slow                  Reduce `MAX_FRAME_WIDTH`,
                                      `INFERENCE_IMGSZ`, or `MAX_FPS`;
                                      consider a supported GPU using
                                      `YOLO_DEVICE`.

  Dependency/import errors            Activate the virtual environment
                                      and check that `PYTHONPATH` is
                                      empty. `motmetrics` is only
                                      required for MOT17 evaluation.

  Dashboard charts are blank          Chart.js loads from jsDelivr, so
                                      charts may not load offline. Other
                                      features may still work.

  Configuration rejected at startup   Ensure weights sum to `1.0` and
                                      probability-like values are in
                                      `[0, 1]`. Check the setting named
                                      in the error.
  -----------------------------------------------------------------------

## Limitations and future work

### Current limitations

-   One active surveillance session at a time.
-   No authentication or role-based access control; local use only.
-   SQLite is the only configured persistence backend.
-   The pretrained model uses COCO classes and is not trained
    specifically for threat detection.
-   Object-risk values are manually configured.
-   No face recognition, identity recognition, or cross-session
    re-identification.
-   Zone checks use the bounding-box centroid.
-   No camera calibration; distance and speed are not expressed in
    real-world units.
-   The threat score is deterministic but has not been scientifically
    validated.

### Possible future work

-   Build labelled scenarios to evaluate and tune the score.
-   Add camera calibration for real-world movement estimates.
-   Add a per-camera zone editor.
-   Add Deep SORT or BoT-SORT adapters behind the existing
    `ObjectTracker` interface.
-   Implement authentication, authorisation, audit controls, and
    deployment hardening.
-   Evaluate on full public datasets and publish reproducible metrics.

## Research contribution and responsible use

The intended contribution is the integration of existing components into
a configurable, context-aware workflow: detect objects, track them,
analyse spatial and temporal context, calculate an explainable score,
generate prioritised alerts, and persist events for review.

This project does **not** claim to introduce YOLO, ByteTrack, OpenCV, or
FastAPI. The default YOLO model is used for inference and is not trained
specifically for threat detection. Track IDs refer to objects within a
session; they do not identify real people. The score is not a validated
probability of danger, and novelty relative to the wider literature has
not been established through a completed literature comparison.

Use only media you have permission to process. Treat scores as
**decision-support signals**, not definitive judgements about people.
Validate the system in its intended environment before relying on its
alerts, and add authentication, access controls, retention policies, and
secure deployment practices before any operational or networked use.
