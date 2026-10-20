import numpy as np
import pytest

from app.core.exceptions import InvalidFrameError
from app.vision.preprocessing import FramePreprocessor


def test_rejects_invalid_frames():
    p = FramePreprocessor()
    for bad in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((4, 4, 3), np.float32),
                np.zeros((4, 4, 2), np.uint8), "frame"):
        with pytest.raises(InvalidFrameError):
            p.prepare(bad)  # type: ignore[arg-type]


def test_converts_gray_and_bgra_to_bgr():
    p = FramePreprocessor()
    assert p.prepare(np.zeros((10, 20), np.uint8)).image.shape == (10, 20, 3)
    assert p.prepare(np.zeros((10, 20, 4), np.uint8)).image.shape == (10, 20, 3)


def test_resize_keeps_aspect_and_records_scale():
    p = FramePreprocessor(max_width=100)
    prepared = p.prepare(np.zeros((200, 400, 3), np.uint8))
    assert prepared.size == (100, 50)
    assert prepared.scale_x == pytest.approx(0.25) and prepared.scale_y == pytest.approx(0.25)
    assert prepared.to_original(50, 25) == pytest.approx((200, 100))


def test_no_resize_when_small_or_disabled():
    frame = np.zeros((50, 80, 3), np.uint8)
    assert FramePreprocessor(max_width=100).prepare(frame).image is frame
    assert FramePreprocessor(max_width=0).prepare(frame).scale_x == 1.0
