"""Safe decoding of uploaded still images into frames."""
from __future__ import annotations

import cv2
import numpy as np

from app.core.exceptions import InvalidFrameError

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def decode_image(data: bytes, max_pixels: int = 50_000_000) -> np.ndarray:
    """Decode bytes to a BGR image, rejecting empty, corrupt or absurdly large images."""
    if not data:
        raise InvalidFrameError("empty image upload")
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidFrameError("could not decode image")
    h, w = image.shape[:2]
    if h * w > max_pixels:
        raise InvalidFrameError("image resolution too large")
    return image


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidFrameError("could not read image")
    return image
