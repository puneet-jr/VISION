"""Object detection with a pretrained Ultralytics YOLO model.

The model is a general-purpose COCO-pretrained network used for inference only; it has not
been trained or validated for threat detection. Model initialisation is centralised: the
detector is created once by the service and shared, never per request.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Protocol

import numpy as np

from app.core.config import Settings
from app.core.exceptions import ModelLoadError
from app.vision.types import Detection

logger = logging.getLogger(__name__)


class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> list[Detection]:
        """Stateless detection (used for still images)."""

    def track(self, frame: np.ndarray, frame_id: int, timestamp: float,
              tracker_config: str) -> list[Detection]:
        """Detection plus multi-object tracking; consecutive calls form one tracking session."""

    def reset_tracking(self) -> None:
        """Forget all tracks (called at the start of each session)."""

    def close(self) -> None: ...


class YoloDetector:
    """Ultralytics YOLO wrapper.

    Tracking and still-image detection use two model instances because Ultralytics keeps
    per-model predictor state (including tracker state) that must not leak into one-off
    image analysis. Both are loaded lazily and guarded by their own lock.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.yolo_model
        self._conf = settings.detection_conf
        self._iou = settings.detection_iou
        self._imgsz = settings.inference_imgsz
        self._device: Optional[str] = settings.yolo_device or None
        self._track_model: Any = None
        self._detect_model: Any = None
        self._track_lock = threading.Lock()
        self._detect_lock = threading.Lock()
        self._load_lock = threading.Lock()

    def _load(self) -> Any:
        try:
            from ultralytics import YOLO  # imported lazily: heavy, and optional for tests
        except ImportError as exc:
            raise ModelLoadError("ultralytics is not installed (pip install -r requirements.txt)") from exc
        try:
            logger.info("loading YOLO model %s", self._model_name)
            return YOLO(self._model_name)
        except Exception as exc:  # download failure, corrupt weights, missing file...
            raise ModelLoadError(f"could not load model '{self._model_name}': {exc}") from exc

    def warmup(self) -> None:
        """Load the tracking model eagerly so configuration errors surface early."""
        with self._load_lock:
            if self._track_model is None:
                self._track_model = self._load()

    def _tracker_model(self) -> Any:
        self.warmup()
        return self._track_model

    def _still_model(self) -> Any:
        with self._load_lock:
            if self._detect_model is None:
                self._detect_model = self._load()
            return self._detect_model

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> list[Detection]:
        model = self._still_model()
        with self._detect_lock:
            results = model.predict(frame, conf=self._conf, iou=self._iou, imgsz=self._imgsz,
                                    device=self._device, verbose=False)
        return self._parse(results, frame_id, timestamp)

    def track(self, frame: np.ndarray, frame_id: int, timestamp: float,
              tracker_config: str) -> list[Detection]:
        model = self._tracker_model()
        with self._track_lock:
            results = model.track(frame, persist=True, tracker=tracker_config, conf=self._conf,
                                  iou=self._iou, imgsz=self._imgsz, device=self._device,
                                  verbose=False)
        return self._parse(results, frame_id, timestamp)

    def reset_tracking(self) -> None:
        with self._track_lock:
            if self._track_model is not None:
                # Dropping the predictor discards ByteTrack state; Ultralytics rebuilds it.
                self._track_model.predictor = None

    def close(self) -> None:
        with self._load_lock:
            self._track_model = None
            self._detect_model = None

    @staticmethod
    def _parse(results: Any, frame_id: int, timestamp: float) -> list[Detection]:
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        names = result.names
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
        detections: list[Detection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            detections.append(Detection(
                frame_id=frame_id, timestamp=timestamp, class_id=int(cls[i]),
                class_name=str(names[int(cls[i])]), confidence=float(conf[i]),
                bbox=(x1, y1, x2, y2), track_id=int(ids[i]) if ids is not None else None,
            ))
        return detections
