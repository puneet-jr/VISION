"""Shared fakes: tests never need a webcam or a YOLO model download."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.database import Database
from app.main import create_app
from app.vision.types import Detection
from app.vision.video_source import FramePacket, SourceSpec

FRAME_W, FRAME_H = 320, 240


def blank_frame() -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def person_walk(frame_id: int, track_id: int = 1, conf: float = 0.9, step: float = 5.0) -> Detection:
    """A person walking left-to-right at y=150; the default zone starts at x=160 (at 320x240)."""
    cx = 20 + frame_id * step
    return Detection(frame_id, frame_id / 10.0, 0, "person", conf,
                     (cx - 15, 120.0, cx + 15, 180.0), track_id)


class FakeDetector:
    """Scripted detector. ``script(frame_id)`` returns the detections for that frame."""

    def __init__(self, script: Optional[Callable[[int], list[Detection]]] = None) -> None:
        self.script = script or (lambda fid: [person_walk(fid)])
        self.resets = 0
        self.closed = False
        self.warmed = False
        self.detect_calls = 0

    def warmup(self) -> None:
        self.warmed = True

    def detect(self, frame, frame_id, timestamp):
        self.detect_calls += 1
        return [Detection(0, 0.0, d.class_id, d.class_name, d.confidence, d.bbox, None)
                for d in self.script(frame_id)]

    def track(self, frame, frame_id, timestamp, tracker_config):
        return self.script(frame_id)

    def reset_tracking(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


class FakeSource:
    """Yields blank frames; ``frames=None`` means endless (until closed)."""

    instances: list["FakeSource"] = []

    def __init__(self, spec: SourceSpec, frames: Optional[int] = 60, delay: float = 0.0,
                 fail_open: bool = False) -> None:
        self.spec, self.frames, self.delay, self.fail_open = spec, frames, delay, fail_open
        self.opened = self.closed = False
        self.read_count = 0
        FakeSource.instances.append(self)

    def open(self) -> None:
        if self.fail_open:
            from app.core.exceptions import SourceUnavailableError
            raise SourceUnavailableError("fake camera unavailable")
        self.opened, self.closed = True, False

    def read(self) -> Optional[FramePacket]:
        if self.closed:
            return None
        if self.frames is not None and self.read_count >= self.frames:
            return None
        if self.delay:
            time.sleep(self.delay)
        fid = self.read_count
        self.read_count += 1
        return FramePacket(blank_frame(), fid, fid / 10.0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def video_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "clip.mp4").write_bytes(b"not a real video; FakeSource never decodes it")
    return folder


@pytest.fixture
def settings(video_dir: Path, tmp_path: Path) -> Settings:
    return Settings(_env_file=None, video_dir=video_dir, database_path=tmp_path / "t.db",
                    max_fps=1000.0, alert_cooldown_s=1.0, stream_url="rtsp://user:secret@cam/stream",
                    zones_file=tmp_path / "missing_zones.json", recordings_dir=tmp_path / "recordings")


@pytest.fixture
def make_client(settings: Settings):
    """Factory so tests can choose the fake source/detector; all clients are closed after."""
    opened: list[TestClient] = []
    FakeSource.instances = []

    def _make(source_kwargs: Optional[dict] = None, detector: Optional[FakeDetector] = None,
              **overrides) -> tuple[TestClient, FakeDetector]:
        det = detector or FakeDetector()
        kwargs = source_kwargs or {}
        app = create_app(
            settings.model_copy(update=overrides) if overrides else settings,
            detector_factory=lambda _s: det,
            source_factory=lambda spec, _s: FakeSource(spec, **kwargs),
            database=Database(None),
        )
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        return client, det

    yield _make
    for c in opened:
        c.__exit__(None, None, None)


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
