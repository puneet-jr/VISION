from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.schemas.common import UtcDatetime

_SAFE_NAME = re.compile(r"^[\w][\w\-. ]{0,199}$")


class SessionCreate(BaseModel):
    """Start a session from a *configured* source. No raw URLs or paths are accepted."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["camera", "video", "stream"]
    video_name: Optional[str] = None

    @field_validator("video_name")
    @classmethod
    def _basename_only(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if ".." in v or "/" in v or "\\" in v or not _SAFE_NAME.match(v):
            raise ValueError("video_name must be a plain file name inside the video directory")
        return v

    @model_validator(mode="after")
    def _video_needs_name(self) -> "SessionCreate":
        if self.source_type == "video" and not self.video_name:
            raise ValueError("video_name is required when source_type is 'video'")
        if self.source_type != "video" and self.video_name:
            raise ValueError("video_name is only valid when source_type is 'video'")
        return self


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_type: str
    source_label: str
    status: str
    started_at: UtcDatetime
    ended_at: Optional[UtcDatetime] = None
    frames_processed: int
    error: Optional[str] = None
    live_fps: Optional[float] = None


class SessionListOut(BaseModel):
    items: list[SessionOut]
    total: int


class SourcesOut(BaseModel):
    camera_index: int
    stream_configured: bool
    videos: list[str]
