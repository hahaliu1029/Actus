"""WebSocket holder implementations for quiesce barrier registration.

See spec §8.5: VNC / takeover shell endpoints register as WebSocketHolder
so that destroy() can gracefully close them during quiesce.

close() contract: (1) cancel tasks, (2) close sockets, (3) await all tasks
done. Must complete within ~5s. Must be idempotent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from starlette.websockets import WebSocket

if TYPE_CHECKING:
    import websockets

logger = logging.getLogger(__name__)

_CLOSE_TIMEOUT = 5.0  # §8.5: 5s best-effort timeout


class TakeoverShellWebSocketHolder:
    """Holder for takeover shell WS endpoint (both WS-direct and HTTP-fallback)."""

    def __init__(
        self,
        session_id: str,
        client_ws: WebSocket,
        sandbox_ws: "websockets.WebSocketClientProtocol | None",
        tasks: list[asyncio.Task],  # type: ignore[type-arg]
        closed_event: asyncio.Event,
    ) -> None:
        self.session_id = session_id
        self._client_ws = client_ws
        self._sandbox_ws = sandbox_ws
        self._tasks = list(tasks)
        self._closed_event = closed_event
        self._close_called = False

    async def close(self) -> None:
        """Idempotent teardown: cancel tasks, close sockets, await tasks done."""
        if self._close_called:
            return
        self._close_called = True

        # 1. Signal closed event so while-loops in forwarding tasks break
        self._closed_event.set()

        # 2. Cancel all forwarding tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # 3. Close upstream sandbox WS
        if self._sandbox_ws is not None:
            try:
                await asyncio.wait_for(self._sandbox_ws.close(), timeout=2.0)
            except Exception:
                pass

        # 4. Close client WS
        try:
            await self._client_ws.close(code=1001, reason="sandbox destroyed")
        except Exception:
            pass

        # 5. Await all tasks to actually finish (the critical missing step)
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=_CLOSE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "TakeoverShellWebSocketHolder.close(): tasks did not finish "
                    "within %ss for session %s",
                    _CLOSE_TIMEOUT,
                    self.session_id,
                )


class VncWebSocketHolder:
    """Holder for VNC WS endpoint."""

    def __init__(
        self,
        session_id: str,
        client_ws: WebSocket,
        sandbox_ws: "websockets.WebSocketClientProtocol",
        tasks: list[asyncio.Task],  # type: ignore[type-arg]
    ) -> None:
        self.session_id = session_id
        self._client_ws = client_ws
        self._sandbox_ws = sandbox_ws
        self._tasks = list(tasks)
        self._close_called = False

    async def close(self) -> None:
        """Idempotent teardown: cancel tasks, close sockets, await tasks done."""
        if self._close_called:
            return
        self._close_called = True

        # 1. Cancel all forwarding tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # 2. Close upstream sandbox WS
        try:
            await asyncio.wait_for(self._sandbox_ws.close(), timeout=2.0)
        except Exception:
            pass

        # 3. Close client WS
        try:
            await self._client_ws.close(code=1001, reason="sandbox destroyed")
        except Exception:
            pass

        # 4. Await all tasks to actually finish
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=_CLOSE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "VncWebSocketHolder.close(): tasks did not finish "
                    "within %ss for session %s",
                    _CLOSE_TIMEOUT,
                    self.session_id,
                )
