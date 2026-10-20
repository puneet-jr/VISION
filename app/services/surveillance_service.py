"""Controlled background processing of one surveillance session at a time.

The service owns the (single) capture loop, the shared detector and the latest annotated
frame. API handlers only start/stop sessions and read snapshots; they never run inference
or camera loops themselves, and any number of dashboard clients share the one loop.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

from app.assessment.alert_manager import AlertEvent
from app.core.config import Settings
from app.core.exceptions import (
    ConfigError, InvalidFrameError, SessionConflictError, SessionNotFoundError,
    SourceNotFoundError, SourceUnavailableError,
)
from app.db.database import Database
from app.db.models import SurveillanceSession
from app.db.repository import EventRepository, SessionRepository
from app.services.event_hub import EventHub
from app.services.recorder import SessionRecorder, prune_recordings
from app.vision import annotator
from app.vision.detector import ObjectDetector, YoloDetector
from app.vision.image_source import decode_image
from app.vision.pipeline import (
    FrameResult, ImageAnalyzer, ImageResult, VisionPipeline, ZoneProvider, build_pipeline,
    build_scorer, build_zone_provider,
)
from app.vision.preprocessing import FramePreprocessor
from app.vision.browser_source import BrowserFrameSource
from app.vision.video_source import FrameSource, SourceKind, SourceSpec, VideoSource
from app.vision.zone_analyzer import Zone

logger = logging.getLogger(__name__)

DetectorFactory = Callable[[Settings], ObjectDetector]
SourceFactory = Callable[[SourceSpec, Settings], FrameSource]
STATUS_INTERVAL_S = 0.5
STOP_JOIN_TIMEOUT_S = 15.0


def default_source_factory(spec: SourceSpec, settings: Settings) -> FrameSource:
    if spec.kind is SourceKind.BROWSER:
        return BrowserFrameSource()
    return VideoSource(spec, loop=settings.loop_video,
                       reconnect_attempts=settings.reconnect_attempts,
                       reconnect_delay_s=settings.reconnect_delay_s)


@dataclass
class _Worker:
    session_id: int
    thread: threading.Thread
    stop_event: threading.Event
    source: FrameSource


class SurveillanceService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        hub: EventHub,
        detector_factory: DetectorFactory = YoloDetector,
        source_factory: SourceFactory = default_source_factory,
        zone_provider: Optional[ZoneProvider] = None,
    ) -> None:
        self.settings = settings
        self.hub = hub
        self.sessions = SessionRepository(db)
        self.events = EventRepository(db)
        self._detector_factory = detector_factory
        self._source_factory = source_factory
        self.zone_provider = zone_provider or build_zone_provider(settings)
        self._detector: Optional[ObjectDetector] = None
        self._lock = threading.RLock()
        self._worker: Optional[_Worker] = None
        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_plain: Optional[bytes] = None
        self.browser_source: Optional[BrowserFrameSource] = None
        self._source_kind: Optional[str] = None
        self._recording = False
        self._model_state = "idle"  # idle | loading | ready | error
        self._model_error: Optional[str] = None
        self._frame_seq = 0
        # Live state read by API handlers (assigned atomically by the worker thread)
        self._objects: list[dict] = []
        self._state = "idle"
        self._frames = 0
        self._fps = 0.0
        self._fps_t = time.monotonic()
        self._last_error: Optional[str] = None
        self._source_label: Optional[str] = None
        self._session_id: Optional[int] = None

    # -- lifecycle -------------------------------------------------------------------------
    def start_session(self, source_type: str, video_name: Optional[str] = None,
                      record: Optional[bool] = None) -> SurveillanceSession:
        with self._lock:
            if self._worker is not None and self._worker.thread.is_alive():
                raise SessionConflictError("a surveillance session is already running")

            spec = self._build_spec(source_type, video_name)
            detector = self._get_detector()
            warmup = getattr(detector, "warmup", None)
            if callable(warmup):
                warmup()  # fail fast (ModelLoadError) instead of inside the worker thread
            source = self._source_factory(spec, self.settings)
            try:
                source.open()
            except Exception:
                source.close()
                raise

            try:
                row = self.sessions.create(spec.kind.value, spec.label)
                pipeline = build_pipeline(self.settings, detector, self.zone_provider,
                                          session_id=row.id, sinks=[self._on_event])
            except Exception:
                source.close()
                raise

            self.browser_source = source if isinstance(source, BrowserFrameSource) else None
            self._source_kind = spec.kind.value
            recorder = None
            if self.settings.record_sessions if record is None else record:
                folder = self.settings.recordings_path
                prune_recordings(folder, self.settings.recording_keep - 1)
                recorder = SessionRecorder(folder / f"session_{row.id}.avi", self.settings.recording_fps,
                                           self.settings.recording_max_mb * 1024 * 1024)
            self._recording = recorder is not None
            stop_event = threading.Event()
            thread = threading.Thread(target=self._run, args=(row.id, source, pipeline, stop_event, recorder),
                                      name=f"session-{row.id}", daemon=True)
            self._worker = _Worker(row.id, thread, stop_event, source)
            self._session_id, self._source_label = row.id, spec.label
            self._state, self._frames, self._fps, self._last_error = "running", 0, 0.0, None
            self._objects = []
            thread.start()
            logger.info("session %d started (%s)", row.id, spec.label)
            self._publish_status()
            return row

    def warmup_async(self) -> None:
        """Load and prime the detector in the background so the first Start is quick."""
        threading.Thread(target=self._warmup, name="model-warmup", daemon=True).start()

    def _warmup(self) -> None:
        self._model_state = "loading"
        self._publish_status()
        try:
            detector = self._get_detector()
            if not self.is_running:
                prime = getattr(detector, "prime", None) or getattr(detector, "warmup", None)
                if callable(prime):
                    prime()
            self._model_state, self._model_error = "ready", None
            logger.info("detection model ready")
        except Exception as exc:  # a failure here is reported again when a session starts
            self._model_state, self._model_error = "error", str(exc)
            logger.warning("model warm-up failed: %s", exc)
        self._publish_status()

    def stop_session(self, session_id: int) -> SurveillanceSession:
        with self._lock:
            row = self.sessions.get(session_id)
            if row is None:
                raise SessionNotFoundError(f"session {session_id} not found")
            worker = self._worker
            if worker is not None and worker.session_id == session_id:
                worker.stop_event.set()
                if isinstance(worker.source, BrowserFrameSource):
                    worker.source.close()  # its read() blocks waiting for frames
                worker.thread.join(timeout=STOP_JOIN_TIMEOUT_S)
                if worker.thread.is_alive():
                    logger.error("session %d worker did not stop within %.0fs", session_id,
                                 STOP_JOIN_TIMEOUT_S)
                else:
                    self._worker = None
            return self.sessions.get(session_id) or row

    def shutdown(self) -> None:
        with self._lock:
            worker = self._worker
        if worker is not None:
            self.stop_session(worker.session_id)
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def recording_path(self, session_id: int) -> Optional[Path]:
        """Path of a finished session's recording (looked up by session id, never by user-given name)."""
        row = self.sessions.get(session_id)
        if row is None or not row.recording_name or (self.is_running and self._session_id == session_id):
            return None
        folder = self.settings.recordings_path.resolve()
        path = (folder / row.recording_name).resolve()
        return path if path.parent == folder and path.is_file() else None

    @property
    def is_running(self) -> bool:
        w = self._worker
        return w is not None and w.thread.is_alive()

    # -- worker ----------------------------------------------------------------------------
    def _run(self, session_id: int, source: FrameSource, pipeline: VisionPipeline,
             stop_event: threading.Event, recorder: Optional[SessionRecorder] = None) -> None:
        status, error = "completed", None
        min_interval = 1.0 / self.settings.max_fps
        last_status = 0.0
        try:
            while not stop_event.is_set():
                started = time.monotonic()
                packet = source.read()
                if packet is None:  # end of a recorded video
                    break
                try:
                    result = pipeline.process_frame(packet.image, packet.frame_id, packet.timestamp)
                except InvalidFrameError as exc:
                    logger.warning("skipping invalid frame %d: %s", packet.frame_id, exc)
                    continue
                self._store_frame(result)
                if recorder is not None:
                    recorder.write(result.annotated, result.timestamp)
                now = time.monotonic()
                if now - last_status >= STATUS_INTERVAL_S:
                    last_status = now
                    self._publish_status()
                spare = min_interval - (time.monotonic() - started)
                if spare > 0:
                    stop_event.wait(spare)
            if stop_event.is_set():
                status = "stopped"
        except SourceUnavailableError as exc:
            status, error = "failed", str(exc)
            logger.error("session %d source error: %s", session_id, exc)
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            logger.exception("session %d crashed", session_id)
        finally:
            try:
                source.close()
            finally:
                self._finalise(session_id, status, error, recorder)

    def _finalise(self, session_id: int, status: str, error: Optional[str],
                  recorder: Optional[SessionRecorder] = None) -> None:
        name, seconds = None, None
        if recorder is not None:
            recorder.close()
            if recorder.frames > 0:
                name, seconds = recorder.path.name, round(recorder.seconds, 1)
        try:
            self.sessions.finish(session_id, status, self._frames, error, name, seconds)
        except Exception:
            logger.exception("could not record end of session %d", session_id)
        self._state, self._last_error, self._objects = status, error, []
        self.browser_source = None
        self._recording = False
        message = {"completed": "Video finished", "stopped": "Processing stopped"}.get(
            status, "Source unavailable")
        self._set_jpeg(annotator.encode_jpeg(annotator.placeholder_frame(message),
                                             self.settings.jpeg_quality))
        logger.info("session %d ended: %s", session_id, status)
        self._publish_status()

    def _store_frame(self, result: FrameResult) -> None:
        self._frames += 1
        self._objects = [o.to_dict() for o in result.objects]
        plain = (annotator.encode_jpeg(result.plain, self.settings.jpeg_quality)
                 if result.plain is not None else None)
        self._set_jpeg(annotator.encode_jpeg(result.annotated, self.settings.jpeg_quality), plain)
        now = time.monotonic()
        elapsed, self._fps_t = now - self._fps_t, now
        if self._frames > 1 and elapsed > 0:
            inst = 1.0 / elapsed
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst

    def _set_jpeg(self, data: Optional[bytes], plain: Optional[bytes] = None) -> None:
        if data is None:
            return
        with self._frame_lock:
            self._latest_jpeg = data
            self._latest_plain = plain if plain is not None else data
            self._frame_seq += 1

    def _on_event(self, event: AlertEvent) -> None:
        try:
            self.events.create(event)
        except Exception:
            logger.exception("could not persist event; it is still published live")
        self.hub.publish({"type": "event", "event": event.to_dict()})

    # -- read-side API ---------------------------------------------------------------------
    def status(self) -> dict:
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for obj in self._objects:
            if obj.get("level") in counts:
                counts[obj["level"]] += 1
        return {
            "type": "status",
            "state": self._state,
            # Derived from the state, not the thread: the final status is published while the
            # worker thread is still finishing, and must already read "not running".
            "running": self._state == "running",
            "session_id": self._session_id,
            "source": self._source_label,
            "source_kind": self._source_kind,
            "recording": self._recording,
            "model": self._model_state,
            "model_error": self._model_error,
            "browser_frames": self.browser_source.frames_received if self.browser_source else 0,
            "frames_processed": self._frames,
            "fps": round(self._fps, 1),
            "error": self._last_error,
            "object_count": len(self._objects),
            "object_levels": counts,
            "objects": self._objects,
        }

    def _publish_status(self) -> None:
        self.hub.publish(self.status())

    def current_objects(self) -> list[dict]:
        return list(self._objects)

    def live_fps(self, session_id: int) -> Optional[float]:
        return round(self._fps, 1) if self.is_running and self._session_id == session_id else None

    def live_frames(self, session_id: int) -> Optional[int]:
        return self._frames if self.is_running and self._session_id == session_id else None

    def latest_frame(self, view: str = "full") -> tuple[Optional[bytes], int]:
        """Newest JPEG: ``full`` (VisionShield view) or ``plain`` (boxes and labels only)."""
        with self._frame_lock:
            return (self._latest_plain if view == "plain" else self._latest_jpeg), self._frame_seq

    @property
    def session_id(self) -> Optional[int]:
        return self._session_id

    # -- still images ----------------------------------------------------------------------
    def analyze_image(self, data: bytes) -> ImageResult:
        image = decode_image(data)
        analyzer = ImageAnalyzer(self._get_detector(), self.zone_provider, build_scorer(self.settings),
                                 FramePreprocessor(self.settings.max_frame_width))
        return analyzer.analyze(image)

    # -- helpers ---------------------------------------------------------------------------
    def list_videos(self) -> list[str]:
        folder = self.settings.video_path
        if not folder.is_dir():
            return []
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)

    def _build_spec(self, source_type: str, video_name: Optional[str]) -> SourceSpec:
        if source_type == "camera":
            return SourceSpec(SourceKind.CAMERA, camera_index=self.settings.camera_index)
        if source_type == "browser":
            if not self.settings.allow_browser_camera:
                raise ConfigError("browser camera input is disabled (ALLOW_BROWSER_CAMERA=false)")
            return SourceSpec(SourceKind.BROWSER)
        if source_type == "stream":
            if not self.settings.stream_url:
                raise ConfigError("no stream URL is configured (set STREAM_URL in .env)")
            return SourceSpec(SourceKind.STREAM, url=self.settings.stream_url)
        if source_type == "video":
            folder = self.settings.video_path.resolve()
            path = (folder / (video_name or "")).resolve()
            if path.parent != folder or not path.is_file():
                raise SourceNotFoundError(f"video '{video_name}' not found in the video directory")
            return SourceSpec(SourceKind.VIDEO, path=path)
        raise ConfigError(f"unsupported source type '{source_type}'")

    def _get_detector(self) -> ObjectDetector:
        with self._lock:
            if self._detector is None:
                self._detector = self._detector_factory(self.settings)
            return self._detector

    # -- zones (live editing) ----------------------------------------------------------------
    def update_zones(self, zones: list[Zone], ref_size: tuple[float, float]) -> None:
        """Apply new zones to the running pipeline (next frame) and persist them for restarts."""
        self.zone_provider.set_zones(zones, ref_size)
        path = self.settings.resolve(self.settings.zones_file)
        payload = {"reference_size": list(ref_size), "zones": [z.to_dict() for z in zones]}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written zones file

    def reset_zones(self) -> None:
        from app.vision.zone_analyzer import DEFAULT_ZONES_DATA, parse_zones
        zones, ref = parse_zones(DEFAULT_ZONES_DATA)
        self.update_zones(zones, ref)
