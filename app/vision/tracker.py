"""Multi-object tracking.

ByteTrack is used through the documented Ultralytics tracking interface
(``model.track(..., tracker="bytetrack.yaml")``). Track ids identify objects only within
one processing session; they are not persistent identities of real people. The
``ObjectTracker`` protocol keeps the integration swappable (e.g. ``botsort.yaml`` or a
Deep SORT adapter) without touching the rest of the pipeline.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.vision.detector import ObjectDetector
from app.vision.types import Detection


class ObjectTracker(Protocol):
    def update(self, frame: np.ndarray, frame_id: int, timestamp: float) -> list[Detection]:
        """Return detections that carry a ``track_id`` (untracked boxes are dropped)."""

    def reset(self) -> None: ...


class UltralyticsTracker:
    """Tracker backed by Ultralytics' built-in ByteTrack (or any tracker YAML it supports)."""

    def __init__(self, detector: ObjectDetector, tracker_config: str = "bytetrack.yaml") -> None:
        self._detector = detector
        self.tracker_config = tracker_config
        self.reset()

    def update(self, frame: np.ndarray, frame_id: int, timestamp: float) -> list[Detection]:
        detections = self._detector.track(frame, frame_id, timestamp, self.tracker_config)
        # ByteTrack leaves low-confidence, not-yet-confirmed boxes without an id.
        return [d for d in detections if d.track_id is not None]

    def reset(self) -> None:
        self._detector.reset_tracking()


ByteTrackTracker = UltralyticsTracker
