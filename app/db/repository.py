"""Persistence operations for sessions and events."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func, select, update

from app.assessment.alert_manager import AlertEvent
from app.db.database import Database
from app.db.models import ObjectObservation, SurveillanceSession, ThreatEvent, utcnow

LEVELS = ("LOW", "MEDIUM", "HIGH")


class SessionRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, source_type: str, source_label: str) -> SurveillanceSession:
        with self._db.session() as s:
            row = SurveillanceSession(source_type=source_type, source_label=source_label,
                                      status="running", started_at=utcnow())
            s.add(row)
            s.flush()
            return row

    def get(self, session_id: int) -> Optional[SurveillanceSession]:
        with self._db.session() as s:
            return s.get(SurveillanceSession, session_id)

    def list(self, limit: int = 50, offset: int = 0) -> tuple[list[SurveillanceSession], int]:
        with self._db.session() as s:
            total = s.scalar(select(func.count()).select_from(SurveillanceSession)) or 0
            rows = s.scalars(select(SurveillanceSession)
                             .order_by(SurveillanceSession.id.desc()).limit(limit).offset(offset)).all()
            return list(rows), total

    def finish(self, session_id: int, status: str, frames: int, error: Optional[str] = None
               ) -> Optional[SurveillanceSession]:
        with self._db.session() as s:
            row = s.get(SurveillanceSession, session_id)
            if row is None:
                return None
            row.status, row.frames_processed, row.error = status, frames, error
            row.ended_at = utcnow()
            return row

    def close_orphans(self) -> int:
        """Sessions left 'running' by a crash cannot be running after a restart."""
        with self._db.session() as s:
            result = s.execute(update(SurveillanceSession)
                               .where(SurveillanceSession.status == "running")
                               .values(status="failed", ended_at=utcnow(),
                                       error="interrupted by application restart"))
            return result.rowcount or 0


class EventRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, event: AlertEvent) -> ThreatEvent:
        """Insert an event (and its object snapshot) in a single transaction."""
        with self._db.session() as s:
            row = ThreatEvent(
                session_id=event.session_id, timestamp=event.timestamp,
                source_time_s=event.source_time_s, event_type=event.event_type,
                track_id=event.track_id, class_name=event.class_name,
                confidence=event.confidence, score=event.score, level=event.level.value,
                zone_id=event.zone_id, zone_name=event.zone_name, movement=event.movement,
                factors=event.factors, explanation=event.explanation, reason=event.reason[:512],
            )
            centroid = event.centroid or [0.0, 0.0]
            row.observation = ObjectObservation(
                track_id=event.track_id, class_name=event.class_name, bbox=event.bbox,
                centroid_x=centroid[0], centroid_y=centroid[1],
                speed_px_s=float(event.movement.get("speed_px_s", 0.0) or 0.0),
                dwell_s=event.movement.get("dwell_s"),
            )
            s.add(row)
            s.flush()
            event.event_id = row.id
            return row

    def get(self, event_id: int) -> Optional[ThreatEvent]:
        with self._db.session() as s:
            return s.get(ThreatEvent, event_id)

    @staticmethod
    def _filters(level: Optional[str], start: Optional[datetime], end: Optional[datetime],
                 session_id: Optional[int]) -> list:
        conds = []
        if level:
            conds.append(ThreatEvent.level == level.upper())
        if start:
            conds.append(ThreatEvent.timestamp >= start)
        if end:
            conds.append(ThreatEvent.timestamp <= end)
        if session_id is not None:
            conds.append(ThreatEvent.session_id == session_id)
        return conds

    def list(self, page: int = 1, page_size: int = 25, level: Optional[str] = None,
             start: Optional[datetime] = None, end: Optional[datetime] = None,
             session_id: Optional[int] = None) -> tuple[list[ThreatEvent], int]:
        conds = self._filters(level, start, end, session_id)
        with self._db.session() as s:
            total = s.scalar(select(func.count()).select_from(ThreatEvent).where(*conds)) or 0
            rows = s.scalars(
                select(ThreatEvent).where(*conds)
                .order_by(ThreatEvent.timestamp.desc(), ThreatEvent.id.desc())
                .limit(page_size).offset((page - 1) * page_size)
            ).unique().all()
            return list(rows), total

    def summary(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> dict:
        conds = self._filters(None, start, end, None)
        with self._db.session() as s:
            by_level = {lv: 0 for lv in LEVELS}
            for level, n in s.execute(select(ThreatEvent.level, func.count())
                                      .where(*conds).group_by(ThreatEvent.level)):
                by_level[level] = n
            by_class = dict(s.execute(
                select(ThreatEvent.class_name, func.count()).where(*conds)
                .group_by(ThreatEvent.class_name).order_by(func.count().desc()).limit(10)).all())
            by_type = dict(s.execute(
                select(ThreatEvent.event_type, func.count()).where(*conds)
                .group_by(ThreatEvent.event_type)).all())
            bucket = func.strftime("%Y-%m-%dT%H:00:00Z", ThreatEvent.timestamp)
            timeline: dict[str, dict] = {}
            for b, level, n in s.execute(select(bucket, ThreatEvent.level, func.count())
                                         .where(*conds).group_by(bucket, ThreatEvent.level)
                                         .order_by(bucket)):
                timeline.setdefault(b, {"bucket": b, **{lv: 0 for lv in LEVELS}})[level] = n
            return {
                "total": sum(by_level.values()),
                "by_level": by_level,
                "by_class": by_class,
                "by_event_type": by_type,
                "timeline": list(timeline.values()),
            }
