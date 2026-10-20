"""Fan-out of live messages to WebSocket clients.

Publishers run in worker threads; subscribers live on the asyncio loop. Each subscriber has
a bounded queue and the oldest message is dropped when it falls behind, so a slow browser
can never grow memory or block the capture loop.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

QUEUE_SIZE = 200


class EventHub:
    def __init__(self) -> None:
        self._clients: set[asyncio.Queue] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._clients.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._clients.discard(queue)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def publish(self, message: dict[str, Any]) -> None:
        """Thread-safe; silently a no-op when no loop is bound or nobody listens."""
        loop = self._loop
        if loop is None or loop.is_closed() or not self._clients:
            return
        try:
            loop.call_soon_threadsafe(self._fanout, message)
        except RuntimeError:  # loop shut down between the check and the call
            logger.debug("event hub loop closed; dropping message")

    def _fanout(self, message: dict[str, Any]) -> None:
        for queue in list(self._clients):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass
