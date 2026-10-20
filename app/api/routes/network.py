"""Where a phone can reach this server (for the "connect your phone" panel)."""
from __future__ import annotations

import socket

from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings_dep
from app.core.config import Settings

router = APIRouter(prefix="/api/v1", tags=["network"])


def lan_addresses() -> list[str]:
    """Private IPv4 addresses of this machine (no packets are sent)."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # routing lookup only
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in found:
                found.append(ip)
    except OSError:
        pass
    return [ip for ip in found if ip.startswith(("192.168.", "10.", "172.")) or ip.startswith("100.")]


@router.get("/network")
def network_info(settings: Settings = Depends(get_settings_dep)) -> dict:
    port = settings.public_https_port
    urls = [f"https://{ip}:{port}/phone" for ip in lan_addresses()] if port else []
    return {
        "browser_camera_enabled": settings.allow_browser_camera,
        "https_enabled": bool(port),
        "phone_urls": urls,
        "hint": None if urls else
                "Start the server with `python scripts/run_demo.py` to enable the phone camera "
                "(phones only allow camera access on HTTPS).",
    }
