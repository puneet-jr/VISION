"""Frame source fed by a browser (a phone or laptop camera) over a WebSocket.

The browser captures with ``getUserMedia`` and pushes JPEG frames; the capture loop pulls the
newest one. Only the newest frames are kept (a tiny bounded buffer), so a slow detector drops
frames instead of building latency or memory.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from app.core.exceptions import SourceUnavailableError
from app.vision.video_source import FramePacket

MAX_JPEG_BYTES = 2 * 1024 * 1024


class BrowserFrameSource:
    def __init__(self, idle_timeout_s: float = 30.0, first_frame_timeout_s: float = 300.0,
                 buffer_size: int = 2) -> None:
        self.idle_timeout_s = idle_timeout_s  # silence after frames started
        self.first_frame_timeout_s = first_frame_timeout_s  # time allowed for the phone to connect
        self._buffer: Deque[bytes] = deque(maxlen=buffer_size)
        self._cond = threading.Condition()
        self._closed = False
        self._frame_id = 0
        self._t0 = 0.0
        self._pushed = 0
        self._dropped = 0

    # -- producer side (WebSocket handler) --------------------------------------------------
    def push(self, jpeg: bytes) -> bool:
        """Offer one JPEG frame; returns False when the source is closed or the frame is rejected."""
        if self._closed or not jpeg or len(jpeg) > MAX_JPEG_BYTES:
            return False
        with self._cond:
            if len(self._buffer) == self._buffer.maxlen:
                self._dropped += 1
            self._buffer.append(jpeg)
            self._pushed += 1
            self._cond.notify()
        return True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def frames_received(self) -> int:
        return self._pushed

    # -- FrameSource ------------------------------------------------------------------------
    def open(self) -> None:
        self._closed = False
        self._t0 = time.monotonic()

    def read(self) -> Optional[FramePacket]:
        """Block until a decodable frame arrives. None once closed; raises if the camera goes silent."""
        last_activity = time.monotonic()
        while True:
            with self._cond:
                while not self._buffer and not self._closed:
                    limit = self.idle_timeout_s if self._pushed else self.first_frame_timeout_s
                    remaining = limit - (time.monotonic() - last_activity)
                    if remaining <= 0:
                        raise SourceUnavailableError("the browser camera is not sending frames")
                    self._cond.wait(min(0.5, remaining))
                if self._closed:
                    return None
                data = self._buffer.pop()  # newest frame; anything older is stale
                self._buffer.clear()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue  # corrupt frame: wait for the next one
            packet = FramePacket(frame, self._frame_id, time.monotonic() - self._t0)
            self._frame_id += 1
            return packet

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._buffer.clear()
            self._cond.notify_all()
