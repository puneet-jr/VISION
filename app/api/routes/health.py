from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_db, get_service, get_settings_dep
from app.core.config import Settings
from app.db.database import Database
from app.services.surveillance_service import SurveillanceService

router = APIRouter()


@router.get("/health", tags=["health"])
def health(db: Database = Depends(get_db), service: SurveillanceService = Depends(get_service),
           settings: Settings = Depends(get_settings_dep)) -> dict:
    db_ok = db.ping()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "unavailable",
        "session_running": service.is_running,
        "model": settings.yolo_model,
        "model_state": service.status()["model"],
    }


@router.get("/api/v1/config", tags=["health"])
def runtime_config(settings: Settings = Depends(get_settings_dep)) -> dict:
    return settings.safe_dict()
