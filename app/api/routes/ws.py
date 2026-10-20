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
