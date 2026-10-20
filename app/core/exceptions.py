"""Domain exceptions. The API layer maps these to HTTP status codes."""
from __future__ import annotations


class VisionShieldError(Exception):
    """Base class for application errors."""


class ConfigError(VisionShieldError):
    """Invalid or inconsistent configuration."""


class InvalidFrameError(VisionShieldError):
    """A frame is empty, malformed or has an unsupported layout."""


class SourceUnavailableError(VisionShieldError):
    """A camera, stream or video file could not be opened or read."""


class SourceNotFoundError(VisionShieldError):
    """A requested local video file does not exist."""


class ModelLoadError(VisionShieldError):
    """The detection model could not be loaded."""


class SessionConflictError(VisionShieldError):
    """A surveillance session is already running."""


class SessionNotFoundError(VisionShieldError):
    """No session with the requested id exists."""


class InvalidThreatInputError(VisionShieldError, ValueError):
    """Threat-scoring input is missing, malformed or out of range."""
