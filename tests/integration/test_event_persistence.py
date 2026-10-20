from datetime import datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from app.assessment.alert_manager import AlertEvent
from app.assessment.threat_scorer import RiskLevel
from app.db.database import Database
from app.db.repository import EventRepository, SessionRepository

BASE = datetime(2025, 3, 1, 10, 0, 0)


@pytest.fixture
def db():
    database = Database(None)
    database.init()
    yield database
    database.dispose()


def make_event(session_id: int, i: int, level: RiskLevel = RiskLevel.HIGH, cls: str = "person") -> AlertEvent:
    return AlertEvent(
        session_id=session_id, timestamp=BASE + timedelta(minutes=i), source_time_s=float(i),
        event_type="threat_level", track_id=i, class_name=cls, confidence=0.9, score=70.0,
        level=level, zone_id="z1", zone_name="Zone", movement={"speed_px_s": 12.5, "dwell_s": 3.0},
        factors={"object_risk": {"value": 0.4}}, explanation="because", reason=f"reason {i}",
        bbox=[1.0, 2.0, 3.0, 4.0], centroid=[2.0, 3.0],
    )


def test_event_roundtrip_with_observation(db):
    sid = SessionRepository(db).create("video", "video:clip.mp4").id
    repo = EventRepository(db)
    event = make_event(sid, 1)
    repo.create(event)
    assert event.event_id is not None

    row = repo.get(event.event_id)
    assert row is not None
    assert (row.level, row.score, row.track_id, row.zone_name) == ("HIGH", 70.0, 1, "Zone")
    assert row.factors == {"object_risk": {"value": 0.4}} and row.movement["speed_px_s"] == 12.5
    assert row.observation.bbox == [1.0, 2.0, 3.0, 4.0] and row.observation.speed_px_s == 12.5
    assert repo.get(9999) is None


def test_pagination_orders_newest_first_and_counts_total(db):
    sid = SessionRepository(db).create("video", "v").id
    repo = EventRepository(db)
    for i in range(25):
        repo.create(make_event(sid, i))
    page1, total = repo.list(page=1, page_size=10)
    page3, _ = repo.list(page=3, page_size=10)
    assert total == 25 and len(page1) == 10 and len(page3) == 5
    assert page1[0].track_id == 24 and page3[-1].track_id == 0


def test_filters_by_level_date_and_session(db):
    sessions = SessionRepository(db)
    s1, s2 = sessions.create("video", "a").id, sessions.create("video", "b").id
    repo = EventRepository(db)
    repo.create(make_event(s1, 0, RiskLevel.LOW))
    repo.create(make_event(s1, 10, RiskLevel.MEDIUM))
    repo.create(make_event(s1, 20, RiskLevel.HIGH))
    repo.create(make_event(s2, 30, RiskLevel.HIGH))

    assert repo.list(level="HIGH")[1] == 2
    assert repo.list(level="high")[1] == 2
    assert repo.list(session_id=s2)[1] == 1
    rows, total = repo.list(start=BASE + timedelta(minutes=5), end=BASE + timedelta(minutes=25))
    assert total == 2 and {r.track_id for r in rows} == {10, 20}


def test_summary_counts(db):
    sid = SessionRepository(db).create("video", "v").id
    repo = EventRepository(db)
    repo.create(make_event(sid, 0, RiskLevel.LOW))
    repo.create(make_event(sid, 1, RiskLevel.HIGH, cls="car"))
    repo.create(make_event(sid, 90, RiskLevel.HIGH))
    s = repo.summary()
    assert s["total"] == 3 and s["by_level"] == {"LOW": 1, "MEDIUM": 0, "HIGH": 2}
    assert s["by_class"]["person"] == 2 and s["by_event_type"] == {"threat_level": 3}
    assert len(s["timeline"]) == 2 and s["timeline"][0]["HIGH"] == 1
    assert repo.summary(start=BASE + timedelta(minutes=60))["total"] == 1


def test_failed_insert_rolls_back_completely(db):
    repo = EventRepository(db)
    with pytest.raises(IntegrityError):
        repo.create(make_event(session_id=12345, i=1))  # no such session: FK enforced
    assert repo.list()[1] == 0


def test_session_lifecycle_and_orphan_cleanup(db):
    repo = SessionRepository(db)
    a, b = repo.create("camera", "camera:0"), repo.create("video", "video:x")
    repo.finish(a.id, "stopped", 42)
    assert repo.get(a.id).status == "stopped" and repo.get(a.id).frames_processed == 42
    assert repo.close_orphans() == 1
    assert repo.get(b.id).status == "failed" and repo.get(b.id).ended_at is not None
    rows, total = repo.list()
    assert total == 2 and rows[0].id == b.id
