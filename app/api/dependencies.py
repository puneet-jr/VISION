"""FastAPI dependencies: application singletons live on ``app.state``."""
from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.db.database import Database
from app.db.repository import EventRepository, SessionRepository
from app.services.surveillance_service import SurveillanceService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_service(request: Request) -> SurveillanceService:
    return request.app.state.service


def get_event_repo(request: Request) -> EventRepository:
    return EventRepository(request.app.state.db)


def get_session_repo(request: Request) -> SessionRepository:
    return SessionRepository(request.app.state.db)
