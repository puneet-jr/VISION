"""Records the annotated VisionShield view of a session to a video file.

Output is MJPEG in an .avi container: OpenCV can write it without extra codecs on every
platform, VLC/Windows players open it, and the dashboard replays it through the MJPEG endpoint.
Frames are placed on the source timeline (duplicated when processing is slower than the
recording rate), so playback runs at real speed whatever the processing rate was.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MAX_DUPLICATES = 8          # never stretch a gap with more than this many repeats
_SIZE_CHECK_EVERY = 30


class SessionRecorder:
    def __init__(self, path: Path, fps: float = 10.0, max_bytes: int = 300 * 1024 * 1024) -> None:
        self.path = path
        self.fps = fps
        self.max_bytes = max_bytes
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple[int, int]] = None
        self._t0: Optional[float] = None
        self._written = 0
        self._calls = 0
        self.disabled = False

    @property
    def frames(self) -> int:
        return self._written

    @property
    def seconds(self) -> float:
        return self._written / self.fps

    def write(self, frame: np.ndarray, t: float) -> None:
        """Add ``frame`` observed at source time ``t`` (seconds). Never raises."""
        if self.disabled:
            return
        try:
            if self._writer is None and not self._open(frame, t):
                return
            if (frame.shape[1], frame.shape[0]) != self._size:
                frame = cv2.resize(frame, self._size)
            due = int((t - self._t0) * self.fps) + 1 - self._written  # type: ignore[operator]
            for _ in range(max(0, min(_MAX_DUPLICATES, due))):
                self._writer.write(frame)  # type: ignore[union-attr]
                self._written += 1
            self._calls += 1
            if self._calls % _SIZE_CHECK_EVERY == 0 and self.path.exists() \
                    and self.path.stat().st_size > self.max_bytes:
                logger.warning("recording %s reached its size limit; recording stopped", self.path.name)
                self.close()
                self.disabled = True
        except Exception:
            logger.exception("recording failed; continuing without it")
            self.close()
            self.disabled = True

    def _open(self, frame: np.ndarray, t: float) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
        if not writer.isOpened():
            logger.warning("could not open a video writer for %s; recording disabled", self.path.name)
            self.disabled = True
            return False
        self._writer, self._size, self._t0 = writer, (w, h), t
        return True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def prune_recordings(folder: Path, keep: int) -> list[str]:
    """Delete the oldest recordings so that at most ``keep`` remain. Returns removed names."""
    if not folder.is_dir():
        return []
    files = sorted((p for p in folder.glob("session_*.avi") if p.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for old in files[keep:]:
        try:
            os.remove(old)
            removed.append(old.name)
        except OSError:
            logger.warning("could not delete old recording %s", old.name)
    return removed
