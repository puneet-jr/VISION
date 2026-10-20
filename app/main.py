"""VisionShield AI application factory.

Local prototype: it has no authentication and must be secured before any network exposure.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException  # base class: also covers router 404/405

from app.api.routes import analytics, events, health, objects, sessions, stream, uploads, ws, zones
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConfigError, InvalidFrameError, ModelLoadError, SessionConflictError, SessionNotFoundError,
    SourceNotFoundError, SourceUnavailableError, VisionShieldError,
)
from app.core.logging import configure_logging
from app.db.database import Database
from app.db.repository import SessionRepository
from app.services.event_hub import EventHub
from app.services.surveillance_service import (
    DetectorFactory, SourceFactory, SurveillanceService, default_source_factory,
)
from app.vision.detector import YoloDetector

logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"

_STATUS_FOR: dict[type, tuple[int, str]] = {
    SessionConflictError: (409, "session_conflict"),
    SessionNotFoundError: (404, "session_not_found"),
    SourceNotFoundError: (404, "source_not_found"),
    SourceUnavailableError: (503, "source_unavailable"),
    ModelLoadError: (503, "model_unavailable"),
    InvalidFrameError: (422, "invalid_image"),
    ConfigError: (400, "configuration_error"),
}


def _error(status: int, code: str, message: str, details: object = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def create_app(
    settings: Optional[Settings] = None,
    detector_factory: DetectorFactory = YoloDetector,
    source_factory: SourceFactory = default_source_factory,
    database: Optional[Database] = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    db = database or Database(settings.database_file)
    hub = EventHub()
    service = SurveillanceService(settings, db, hub, detector_factory, source_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind_loop(asyncio.get_running_loop())
        db.init()
        orphans = SessionRepository(db).close_orphans()
        if orphans:
            logger.warning("marked %d interrupted session(s) as failed", orphans)
        logger.info("VisionShield AI ready (local prototype - no authentication)")
        yield
        await run_in_threadpool(service.shutdown)  # stops workers, releases camera and model
        db.dispose()

    app = FastAPI(title="VisionShield AI", version="1.0.0", lifespan=lifespan,
                  description="Real-time detection, tracking and explainable threat assessment.")
    app.state.settings, app.state.db, app.state.hub, app.state.service = settings, db, hub, service

    @app.exception_handler(VisionShieldError)
    async def _domain_error(_: Request, exc: VisionShieldError) -> JSONResponse:
        for exc_type, (status, code) in _STATUS_FOR.items():
            if isinstance(exc, exc_type):
                return _error(status, code, str(exc))
        return _error(500, "internal_error", str(exc))

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return _error(422, "validation_error", "request validation failed", details)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return _error(500, "internal_error", "internal server error")

    for module in (health, sessions, objects, events, zones, analytics, stream, uploads, ws):
        app.include_router(module.router)

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(WEB_DIR / "templates" / "dashboard.html")

    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn app.main:app_factory --factory``."""
    return create_app()


def __getattr__(name: str):  # lazy ``app.main:app`` so importing the module has no side effects
    if name == "app":
        instance = create_app()
        globals()["app"] = instance
        return instance
    raise AttributeError(name)
