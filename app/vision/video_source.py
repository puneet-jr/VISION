"""Frame acquisition from a webcam, a local video file or an IP/RTSP stream.

Acquisition is kept apart from API routes and scoring. A source is owned by exactly one
capture loop (the surveillance service) so several API clients never open duplicates.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from app.core.exceptions import SourceUnavailableError

logger = logging.getLogger(__name__)


class SourceKind(str, Enum):
    CAMERA = "camera"
    VIDEO = "video"
    STREAM = "stream"
    BROWSER = "browser"  # phone / laptop camera pushed from a web page


@dataclass(frozen=True)
class SourceSpec:
    kind: SourceKind
    camera_index: int = 0
    path: Optional[Path] = None
    url: Optional[str] = None

    @property
    def label(self) -> str:
        """Human-readable name that never includes a stream URL (it may hold credentials)."""
        if self.kind is SourceKind.CAMERA:
            return f"camera:{self.camera_index}"
        if self.kind is SourceKind.VIDEO:
            return f"video:{self.path.name if self.path else '?'}"
        if self.kind is SourceKind.BROWSER:
            return "browser:camera"
        return "stream:configured"


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    frame_id: int
    timestamp: float  # seconds on the source timeline (video time for files, elapsed for live)


class FrameSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> Optional[FramePacket]: ...  # None at end of stream
    def close(self) -> None: ...


class VideoSource:
    """OpenCV-backed frame source with bounded reconnection for live inputs."""

    def __init__(
        self,
        spec: SourceSpec,
        loop: bool = False,
        reconnect_attempts: int = 3,
        reconnect_delay_s: float = 2.0,
    ) -> None:
        self.spec = spec
        self.loop = loop
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id = 0
        self._fps = 0.0
        self._t0 = 0.0
        self._offset = 0.0  # accumulated timeline when a file loops

    @property
    def is_live(self) -> bool:
        return self.spec.kind is not SourceKind.VIDEO

    def open(self) -> None:
        self.close()
        if self.spec.kind is SourceKind.CAMERA:
            # CAP_DSHOW opens quickly on Windows; other platforms ignore the preference.
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") and _is_windows() else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.spec.camera_index, backend)
        elif self.spec.kind is SourceKind.VIDEO:
            if self.spec.path is None or not self.spec.path.is_file():
                raise SourceUnavailableError("video file does not exist")
            cap = cv2.VideoCapture(str(self.spec.path))
        else:
            if not self.spec.url:
                raise SourceUnavailableError("no stream URL configured")
            cap = cv2.VideoCapture(self.spec.url)
        if not cap.isOpened():
            cap.release()
            raise SourceUnavailableError(f"cannot open source {self.spec.label}")
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1e-3 and not np.isnan(fps) else 25.0
        self._t0 = time.monotonic()
        logger.info("opened %s (fps=%.1f)", self.spec.label, self._fps)

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            raise SourceUnavailableError("source is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if not self.is_live:
                return self._handle_file_end()
            frame = self._reconnect_and_read()
            if frame is None:
                raise SourceUnavailableError(f"lost connection to {self.spec.label}")
        packet = FramePacket(frame, self._frame_id, self._timestamp())
        self._frame_id += 1
        return packet

    def _timestamp(self) -> float:
        if self.is_live:
            return time.monotonic() - self._t0
        return self._offset + self._frame_id / self._fps

    def _handle_file_end(self) -> Optional[FramePacket]:
        if not self.loop or self._cap is None:
            return None
        self._offset += self._frame_id / self._fps
        self._frame_id = 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        packet = FramePacket(frame, self._frame_id, self._timestamp())
        self._frame_id += 1
        return packet

    def _reconnect_and_read(self) -> Optional[np.ndarray]:
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.warning("%s: read failed, reconnect %d/%d", self.spec.label, attempt,
                           self.reconnect_attempts)
            time.sleep(self.reconnect_delay_s)
            try:
                self.open()
            except SourceUnavailableError:
                continue
            assert self._cap is not None
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")
