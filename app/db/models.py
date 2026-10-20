"""SQLAlchemy models. Only events and bounded metadata are stored - never frames."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Naive UTC datetime (SQLite does not keep time zones)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class SurveillanceSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(16))
    source_label: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), index=True)  # running|stopped|completed|failed
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recording_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    recording_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    events: Mapped[list["ThreatEvent"]] = relationship(back_populates="session",
                                                       cascade="all, delete-orphan")


class ThreatEvent(Base):
    __tablename__ = "threat_events"
    __table_args__ = (
        Index("ix_events_level_ts", "level", "timestamp"),
        Index("ix_events_session_ts", "session_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, default=utcnow)
    source_time_s: Mapped[float] = mapped_column(Float, default=0.0)
    event_type: Mapped[str] = mapped_column(String(24))
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    class_name: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    level: Mapped[str] = mapped_column(String(8))
    zone_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    zone_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    movement: Mapped[dict] = mapped_column(JSON, default=dict)
    factors: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation: Mapped[str] = mapped_column(Text, default="")
    reason: Mapped[str] = mapped_column(String(512), default="")

    session: Mapped[SurveillanceSession] = relationship(back_populates="events")
    observation: Mapped[Optional["ObjectObservation"]] = relationship(
        back_populates="event", cascade="all, delete-orphan", uselist=False, lazy="joined")


class ObjectObservation(Base):
    """Snapshot of the object at the moment an event fired (one per event; bounded)."""

    __tablename__ = "object_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("threat_events.id", ondelete="CASCADE"),
                                          unique=True)
    track_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(64))
    bbox: Mapped[list] = mapped_column(JSON, default=list)
    centroid_x: Mapped[float] = mapped_column(Float, default=0.0)
    centroid_y: Mapped[float] = mapped_column(Float, default=0.0)
    speed_px_s: Mapped[float] = mapped_column(Float, default=0.0)
    dwell_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    event: Mapped[ThreatEvent] = relationship(back_populates="observation")
