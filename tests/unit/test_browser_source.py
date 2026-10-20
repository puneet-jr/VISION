import threading
import time

import cv2
import numpy as np
import pytest

from app.core.exceptions import SourceUnavailableError
from app.vision.browser_source import BrowserFrameSource


def jpeg(value=0, w=64, h=48) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def test_reads_newest_frame_and_drops_the_backlog():
    src = BrowserFrameSource()
    src.open()
    for v in (10, 100, 200):
        assert src.push(jpeg(v))
    packet = src.read()
    assert packet is not None and packet.image.shape == (48, 64, 3)
    assert abs(int(packet.image.mean()) - 200) < 10      # the newest, not the oldest
    assert packet.frame_id == 0 and src.frames_received == 3


def test_buffer_is_bounded():
    src = BrowserFrameSource(buffer_size=2)
    for v in range(50):
        src.push(jpeg(v))
    assert len(src._buffer) == 2 and src._dropped == 48


def test_close_unblocks_a_waiting_reader():
    src = BrowserFrameSource()
    src.open()
    result = []
    t = threading.Thread(target=lambda: result.append(src.read()))
    t.start()
    time.sleep(0.15)
    src.close()
    t.join(timeout=3)
    assert not t.is_alive() and result == [None]
    assert src.push(jpeg()) is False          # closed sources refuse frames


def test_silence_after_frames_raises():
    src = BrowserFrameSource(idle_timeout_s=0.2)
    src.open()
    src.push(jpeg())
    assert src.read() is not None
    with pytest.raises(SourceUnavailableError):
        src.read()


def test_corrupt_and_oversized_frames_are_ignored():
    src = BrowserFrameSource()
    src.open()
    assert src.push(b"") is False and src.push(b"x" * (3 * 1024 * 1024)) is False
    src.push(b"definitely not a jpeg")
    threading.Timer(0.1, lambda: src.push(jpeg(77))).start()
    packet = src.read()                        # skips the corrupt frame, returns the valid one
    assert packet is not None and abs(int(packet.image.mean()) - 77) < 10
