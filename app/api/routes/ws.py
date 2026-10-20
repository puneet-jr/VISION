from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/events")
async def events_socket(ws: WebSocket) -> None:
    """Streams new events and periodic status updates (small JSON only - no frames)."""
    hub, service = ws.app.state.hub, ws.app.state.service
    await ws.accept()
    queue = hub.subscribe()

    async def sender() -> None:
        while True:
            await ws.send_json(await queue.get())

    async def receiver() -> None:
        while True:  # client messages are ignored; receiving detects disconnects
            await ws.receive_text()

    tasks: list[asyncio.Task] = []
    try:
        await ws.send_json(service.status())
        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("websocket closed with error", exc_info=True)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        hub.unsubscribe(queue)


@router.websocket("/ws/camera")
async def camera_socket(ws: WebSocket) -> None:
    """Receives JPEG frames from a browser camera (phone or laptop) for the active browser session."""
    service = ws.app.state.service
    await ws.accept()
    source = service.browser_source
    if source is None or source.closed:
        await ws.send_json({"type": "error", "message": "no browser-camera session is active"})
        await ws.close(code=4403)
        return
    sent = 0
    try:
        await ws.send_json({"type": "ready"})
        while not source.closed:
            try:
                data = await asyncio.wait_for(ws.receive_bytes(), timeout=2.0)
            except asyncio.TimeoutError:
                continue  # re-check whether the session was stopped
            source.push(data)
            sent += 1
            if sent % 15 == 0:
                await ws.send_json({"type": "ack", "frames": source.frames_received})
        await ws.send_json({"type": "stopped"})
    except WebSocketDisconnect:
        pass
    except (RuntimeError, KeyError):  # text frame instead of bytes / closed socket
        logger.debug("camera socket closed", exc_info=True)
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass
