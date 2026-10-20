import os
import time

import cv2
import numpy as np
from sqlalchemy import create_engine, text

from app.db.database import Database
from app.services.recorder import SessionRecorder, prune_recordings


def frame(value=0, w=64, h=48):
    return np.full((h, w, 3), value, np.uint8)


def frame_count(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def test_frames_are_placed_on_the_source_timeline(tmp_path):
    rec = SessionRecorder(tmp_path / "r.avi", fps=10.0)
    for i in range(10):                       # 10 frames over 5 s of source time (2 fps processing)
        rec.write(frame(i * 20), i * 0.5)
    rec.close()
    assert rec.frames == 5 * 10 - 4 and abs(rec.seconds - 4.6) < 0.2   # gaps filled, real-time speed
    assert frame_count(tmp_path / "r.avi") == rec.frames


def test_faster_than_recording_rate_skips_frames(tmp_path):
    rec = SessionRecorder(tmp_path / "r.avi", fps=10.0)
    for i in range(100):                      # 100 frames in 1 s of source time
        rec.write(frame(), i * 0.01)
    rec.close()
    assert 9 <= rec.frames <= 11


def test_long_gaps_are_not_stretched_indefinitely(tmp_path):
    rec = SessionRecorder(tmp_path / "r.avi", fps=10.0)
    rec.write(frame(), 0.0)
    rec.write(frame(), 600.0)                 # a ten-minute stall
    rec.close()
    assert rec.frames <= 9


def test_size_limit_stops_recording_without_raising(tmp_path):
    rec = SessionRecorder(tmp_path / "r.avi", fps=30.0, max_bytes=2000)
    noisy = np.random.randint(0, 255, (240, 320, 3), np.uint8)
    for i in range(300):
        rec.write(noisy, i / 30.0)
    assert rec.disabled and rec.frames < 300
    rec.write(noisy, 99.0)                    # further writes are ignored
    rec.close()


def test_unwritable_target_disables_recording_quietly(tmp_path):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    rec = SessionRecorder(blocker / "sub" / "r.avi")   # parent is a file: cannot create
    rec.write(frame(), 0.0)
    assert rec.disabled and rec.frames == 0


def test_prune_keeps_newest(tmp_path):
    for i in range(5):
        p = tmp_path / f"session_{i}.avi"
        p.write_bytes(b"x")
        os.utime(p, (time.time() + i, time.time() + i))
    removed = prune_recordings(tmp_path, keep=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["session_3.avi", "session_4.avi"]
    assert len(removed) == 3 and prune_recordings(tmp_path / "nope", 2) == []


def test_old_databases_gain_the_recording_columns(tmp_path):
    path = tmp_path / "old.db"
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    with engine.begin() as conn:   # the schema as created by the first version
        conn.execute(text("CREATE TABLE sessions (id INTEGER PRIMARY KEY, source_type VARCHAR(16), "
                          "source_label VARCHAR(255), status VARCHAR(16), started_at DATETIME, "
                          "ended_at DATETIME, frames_processed INTEGER, error TEXT)"))
    engine.dispose()
    db = Database(path)
    db.init()
    with db.engine.connect() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(sessions)"))}
    assert {"recording_name", "recording_seconds"} <= cols
    db.init()   # idempotent
    db.dispose()
