"""Frame validation, resizing and coordinate bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.exceptions import InvalidFrameError


@dataclass(frozen=True)
class PreparedFrame:
    image: np.ndarray  # BGR uint8, what the detector sees
    scale_x: float  # processed / original
    scale_y: float
    original_size: tuple[int, int]  # (width, height)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        """Map a processed-frame coordinate back to the original frame."""
        return x / self.scale_x, y / self.scale_y


class FramePreprocessor:
    def __init__(self, max_width: int = 0) -> None:
        self.max_width = max_width

    @staticmethod
    def validate(frame: object) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise InvalidFrameError("frame is missing or empty")
        if frame.dtype != np.uint8:
            raise InvalidFrameError(f"unsupported frame dtype {frame.dtype}")
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        raise InvalidFrameError(f"unsupported frame shape {frame.shape}")

    def prepare(self, frame: np.ndarray) -> PreparedFrame:
        image = self.validate(frame)
        h, w = image.shape[:2]
        sx = sy = 1.0
        if self.max_width and w > self.max_width:
            new_w = self.max_width
            new_h = max(1, int(round(h * new_w / w)))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sx, sy = new_w / w, new_h / h
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        return PreparedFrame(image, sx, sy, (w, h))
