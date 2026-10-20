from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import get_service, get_session_repo, get_settings_dep
from app.core.config import Settings
from app.core.exceptions import SessionNotFoundError
from app.db.models import SurveillanceSession
from app.db.repository import SessionRepository
from app.schemas.sessions import SessionCreate, SessionListOut, SessionOut, SourcesOut
from app.services.surveillance_service import SurveillanceService

router = APIRouter(prefix="/api/v1", tags=["sessions"])


def _to_out(row: SurveillanceSession, service: SurveillanceService) -> SessionOut:
    out = SessionOut.model_validate(row)
    live_frames = service.live_frames(row.id)
    if live_frames is not None:
        out.frames_processed = live_frames
        out.live_fps = service.live_fps(row.id)
    return out


@router.get("/sources", response_model=SourcesOut)
def list_sources(settings: Settings = Depends(get_settings_dep),
                 service: SurveillanceService = Depends(get_service)) -> SourcesOut:
    """Configured inputs the dashboard may start (the stream URL itself is never exposed)."""
    return SourcesOut(camera_index=settings.camera_index,
                      stream_configured=settings.stream_url is not None,
                      videos=service.list_videos())


@router.get("/sessions", response_model=SessionListOut)
def list_sessions(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                  repo: SessionRepository = Depends(get_session_repo),
                  service: SurveillanceService = Depends(get_service)) -> SessionListOut:
    rows, total = repo.list(limit=limit, offset=offset)
    return SessionListOut(items=[_to_out(r, service) for r in rows], total=total)


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def start_session(body: SessionCreate,
                  service: SurveillanceService = Depends(get_service)) -> SessionOut:
    row = service.start_session(body.source_type, body.video_name, body.record)
    return _to_out(row, service)


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: int, repo: SessionRepository = Depends(get_session_repo),
                service: SurveillanceService = Depends(get_service)) -> SessionOut:
    row = repo.get(session_id)
    if row is None:
        raise SessionNotFoundError(f"session {session_id} not found")
    return _to_out(row, service)


@router.post("/sessions/{session_id}/stop", response_model=SessionOut)
def stop_session(session_id: int,
                 service: SurveillanceService = Depends(get_service)) -> SessionOut:
    """Idempotent: stopping a session that already ended returns its final state."""
    row = service.stop_session(session_id)
    return _to_out(row, service)
