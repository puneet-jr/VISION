"""Environment-driven configuration, validated at startup."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

# Initial engineering defaults, NOT scientifically validated. Override with OBJECT_RISK.
DEFAULT_OBJECT_RISK: dict[str, float] = {
    "person": 0.40,
    "backpack": 0.50,
    "suitcase": 0.50,
    "handbag": 0.30,
    "knife": 0.90,
    "scissors": 0.60,
    "baseball bat": 0.60,
    "car": 0.40,
    "truck": 0.40,
    "bus": 0.35,
    "motorcycle": 0.40,
    "bicycle": 0.25,
    "dog": 0.20,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Server
    app_host: str = "127.0.0.1"
    app_port: int = Field(8000, ge=1, le=65535)
    log_level: str = "INFO"

    # Detection
    yolo_model: str = "yolo11n.pt"
    yolo_device: str = ""
    detection_conf: float = Field(0.35, ge=0.0, le=1.0)
    detection_iou: float = Field(0.5, ge=0.0, le=1.0)
    inference_imgsz: int = Field(640, ge=32, le=2560)

    # Tracking
    tracker_config: str = "bytetrack.yaml"
    trajectory_len: int = Field(30, ge=2, le=300)
    track_stale_s: float = Field(5.0, gt=0)

    # Video input
    camera_index: int = Field(0, ge=0)
    stream_url: Optional[str] = None
    video_dir: Path = Path("data/videos")
    loop_video: bool = False
    max_frame_width: int = Field(960, ge=0)  # 0 disables resizing
    max_fps: float = Field(15.0, gt=0)
    reconnect_attempts: int = Field(3, ge=0)
    reconnect_delay_s: float = Field(2.0, ge=0)
    max_video_upload_mb: int = Field(500, ge=1)
    max_image_upload_mb: int = Field(10, ge=1)
    jpeg_quality: int = Field(80, ge=10, le=100)

    # Database
    database_path: Path = Path("data/visionshield.db")

    # Threat scoring
    weight_object_risk: float = 0.30
    weight_confidence: float = 0.15
    weight_movement: float = 0.20
    weight_proximity: float = 0.20
    weight_persistence: float = 0.15
    object_risk: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_OBJECT_RISK))
    default_object_risk: float = Field(0.20, ge=0.0, le=1.0)
    speed_ref_diag_per_s: float = Field(0.15, gt=0)
    zone_near_fraction: float = Field(0.15, gt=0)
    dwell_ref_s: float = Field(10.0, gt=0)
    zone_exit_grace_s: float = Field(2.0, ge=0)
    approach_min_observations: int = Field(5, ge=2)

    # Alerts
    alert_cooldown_s: float = Field(10.0, ge=0)
    alert_min_level: str = "MEDIUM"

    # Zones
    zones_file: Path = Path("config/zones.json")

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid LOG_LEVEL: {v}")
        return v

    @field_validator("alert_min_level")
    @classmethod
    def _check_min_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("ALERT_MIN_LEVEL must be LOW, MEDIUM or HIGH")
        return v

    @field_validator("stream_url")
    @classmethod
    def _blank_stream_is_none(cls, v: Optional[str]) -> Optional[str]:
        return v or None

    @field_validator("object_risk")
    @classmethod
    def _check_object_risk(cls, v: dict[str, float]) -> dict[str, float]:
        for name, risk in v.items():
            if not 0.0 <= risk <= 1.0:
                raise ValueError(f"object risk for '{name}' must be within [0, 1]")
        return {k.lower(): float(r) for k, r in v.items()}

    @model_validator(mode="after")
    def _check_weights(self) -> "Settings":
        total = sum(self.weights_tuple())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"threat-scoring weights must sum to 1.0 (got {total:.6f})")
        return self

    def weights_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.weight_object_risk,
            self.weight_confidence,
            self.weight_movement,
            self.weight_proximity,
            self.weight_persistence,
        )

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path relative to the project root."""
        return path if path.is_absolute() else (BASE_DIR / path)

    @property
    def video_path(self) -> Path:
        return self.resolve(self.video_dir)

    @property
    def database_file(self) -> Path:
        return self.resolve(self.database_path)

    def safe_dict(self) -> dict:
        """Non-secret runtime configuration for the API (no stream URL, no paths)."""
        return {
            "yolo_model": self.yolo_model,
            "yolo_device": self.yolo_device or "auto",
            "detection_conf": self.detection_conf,
            "detection_iou": self.detection_iou,
            "inference_imgsz": self.inference_imgsz,
            "tracker_config": self.tracker_config,
            "camera_index": self.camera_index,
            "stream_configured": self.stream_url is not None,
            "max_frame_width": self.max_frame_width,
            "max_fps": self.max_fps,
            "loop_video": self.loop_video,
            "weights": {
                "object_risk": self.weight_object_risk,
                "confidence": self.weight_confidence,
                "movement": self.weight_movement,
                "proximity": self.weight_proximity,
                "persistence": self.weight_persistence,
            },
            "object_risk": self.object_risk,
            "default_object_risk": self.default_object_risk,
            "speed_ref_diag_per_s": self.speed_ref_diag_per_s,
            "zone_near_fraction": self.zone_near_fraction,
            "dwell_ref_s": self.dwell_ref_s,
            "alert_cooldown_s": self.alert_cooldown_s,
            "alert_min_level": self.alert_min_level,
            "levels": {"LOW": "0-30", "MEDIUM": "31-60", "HIGH": "61-100"},
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
