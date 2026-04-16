"""In-process sandbox instance registry.

Replaces the ``@alru_cache`` on ``DockerSandbox.get()`` (I4). Tracks live
Sandbox instances, open handles, in-flight tasks, and WebSocket holders
per session. Provides quiesce barrier primitives for two-phase destroy (I6).

All **non-async** mutators are O(1) in-memory operations that assume the
caller holds the per-session lock from ``SandboxLifecycleService``.

``get_sandbox`` / ``get_generation`` are **hints**, not authoritative —
the DB ``session.sandbox_binding`` is always the source of truth (I9).

See spec §8.4 for interface contracts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Protocol

from app.domain.external.sandbox import Sandbox

if TYPE_CHECKING:
    from app.infrastructure.external.sandbox.sandbox_handle import SandboxHandleImpl

logger = logging.getLogger(__name__)


class WebSocketHolder(Protocol):
    """Protocol for long-lived WebSocket connections that need graceful teardown.

    VNC / takeover shell endpoints implement this and register with the
    registry at connection establishment. ``close()`` must be idempotent.
    See spec §8.5.
    """

    session_id: str

    async def close(self) -> None:
        """Gracefully tear down internal resources.

        Implementer is responsible for: (1) cancel forwarding tasks,
        (2) close client WebSocket, (3) close upstream connection,
        (4) await all tasks done. Must be idempotent.
        """
        ...


class SandboxRegistry:
    """Service-internal registry. Only ``SandboxLifecycleService`` should
    call mutator methods (under per-session lock)."""

    # Type alias for a callback that pushes an event into the live SSE stream
    # and returns the stream message ID (used to unify IDs across Redis + PG).
    LiveEventSink = Callable[[Any], Awaitable[Optional[str]]]

    def __init__(self) -> None:
        self._sandboxes: dict[str, Sandbox] = {}
        self._generations: dict[str, int] = {}
        self._open_handles: dict[str, set[SandboxHandleImpl]] = {}
        self._inflight_tasks: dict[str, set[asyncio.Task]] = {}  # type: ignore[type-arg]
        self._drain_events: dict[str, asyncio.Event] = {}
        self._ws_holders: dict[str, set[WebSocketHolder]] = {}
        self._live_event_sinks: dict[str, SandboxRegistry.LiveEventSink] = {}

    # ── Core registry state mutations (O(1) in-memory, no IO) ──

    def register(self, session_id: str, sandbox: Sandbox, generation: int) -> None:
        """Register a live sandbox instance for a session."""
        self._sandboxes[session_id] = sandbox
        self._generations[session_id] = generation
        self._open_handles.setdefault(session_id, set())
        self._inflight_tasks.setdefault(session_id, set())
        self._drain_events[session_id] = asyncio.Event()
        self._ws_holders.setdefault(session_id, set())

    def remove(self, session_id: str) -> None:
        """Remove all registry state for a session after destroy completes."""
        self._sandboxes.pop(session_id, None)
        self._generations.pop(session_id, None)
        self._open_handles.pop(session_id, None)
        self._inflight_tasks.pop(session_id, None)
        self._drain_events.pop(session_id, None)
        self._ws_holders.pop(session_id, None)
        self._live_event_sinks.pop(session_id, None)

    # ── Lookup (hint, not authoritative — I9) ──

    def get_sandbox(self, session_id: str) -> Optional[Sandbox]:
        """Process-local hint; authoritative source is DB sandbox_binding."""
        return self._sandboxes.get(session_id)

    def get_generation(self, session_id: str) -> Optional[int]:
        """Process-local hint; authoritative source is DB sandbox_binding."""
        return self._generations.get(session_id)

    def update_generation(self, session_id: str, generation: int) -> None:
        """Align in-memory generation with authoritative DB value."""
        self._generations[session_id] = generation

    # ── Handle acquisition / release ──

    def acquire_handle(self, session_id: str) -> SandboxHandleImpl:
        """Create and register a new handle for the session's sandbox."""
        from app.infrastructure.external.sandbox.sandbox_handle import (
            SandboxHandleImpl,
        )

        sandbox = self._sandboxes[session_id]
        generation = self._generations[session_id]
        handle = SandboxHandleImpl(
            sandbox=sandbox,
            session_id=session_id,
            generation=generation,
            registry=self,
        )
        self._open_handles.setdefault(session_id, set()).add(handle)
        return handle

    def release_handle(self, session_id: str, handle: SandboxHandleImpl) -> None:
        """Remove a handle from the open set. Idempotent.

        Does NOT require the caller to hold the per-session lock — this is
        a sync O(1) operation that's safe in asyncio single-threaded context
        (eng review decision #4).
        """
        handles = self._open_handles.get(session_id)
        if handles is not None:
            handles.discard(handle)

    # ── In-flight task tracking (quiesce barrier internals) ──

    def enroll_inflight(self, session_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
        """Called by SandboxHandleImpl._checked_call on method entry."""
        self._inflight_tasks.setdefault(session_id, set()).add(task)

    def discharge_inflight(self, session_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
        """Called by SandboxHandleImpl._checked_call in finally block."""
        tasks = self._inflight_tasks.get(session_id)
        if tasks is not None:
            tasks.discard(task)
            # Signal drain event if no more inflight tasks
            if not tasks:
                drain_event = self._drain_events.get(session_id)
                if drain_event is not None:
                    drain_event.set()

    # ── WebSocket holder tracking ──

    def register_ws_holder(self, session_id: str, holder: WebSocketHolder) -> None:
        """Called by WS endpoint at connection establishment."""
        self._ws_holders.setdefault(session_id, set()).add(holder)

    def release_ws_holder(self, session_id: str, holder: WebSocketHolder) -> None:
        """Called by WS endpoint at connection close. Idempotent."""
        holders = self._ws_holders.get(session_id)
        if holders is not None:
            holders.discard(holder)

    # ── Live SSE event sink (PR2 §10) ──

    def register_live_event_sink(
        self, session_id: str, sink: "SandboxRegistry.LiveEventSink"
    ) -> None:
        """Register a callback that pushes lifecycle events into the active SSE stream.

        Called by AgentService when a task starts. The sink pushes serialized
        events into ``task.output_stream`` so the live ``chat()`` SSE loop
        can yield them to the frontend without waiting for PG recovery poll.
        """
        self._live_event_sinks[session_id] = sink

    def release_live_event_sink(self, session_id: str) -> None:
        """Release the live event sink. Idempotent."""
        self._live_event_sinks.pop(session_id, None)

    def get_live_event_sink(
        self, session_id: str
    ) -> Optional["SandboxRegistry.LiveEventSink"]:
        """Return the active sink, or None if no task is streaming."""
        return self._live_event_sinks.get(session_id)

    # ── Quiesce barrier primitives ──
    # Only called by SandboxLifecycleService.destroy() under per-session lock.

    async def cancel_and_drain(self, session_id: str, timeout: float) -> None:
        """Cancel all in-flight tasks + close all WS holders, then await drain.

        Raises ``asyncio.TimeoutError`` if drain doesn't complete within timeout.
        """
        # Cancel in-flight asyncio tasks
        tasks_to_cancel = list(self._inflight_tasks.get(session_id, set()))
        for task in tasks_to_cancel:
            task.cancel()

        # Close WebSocket holders concurrently
        ws_holders = list(self._ws_holders.get(session_id, set()))
        if ws_holders:
            close_coros = [holder.close() for holder in ws_holders]
            await asyncio.gather(*close_coros, return_exceptions=True)

        # Await cancelled tasks to finish (they may still be in CancelledError handling).
        # Tasks enrolled via _checked_call will discharge themselves in finally;
        # raw tasks (e.g., from confirmation sweep) may not, so we gather + force-clear.
        if tasks_to_cancel:
            await asyncio.wait_for(
                asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                timeout=timeout,
            )
        # Force-clear inflight set — any task that didn't self-discharge is now done.
        inflight = self._inflight_tasks.get(session_id)
        if inflight is not None:
            inflight.clear()
            drain_event = self._drain_events.get(session_id)
            if drain_event is not None:
                drain_event.set()

    async def destroy_infra(self, session_id: str) -> None:
        """Call Sandbox.destroy() on the live instance (docker rm + httpx aclose).

        Must run AFTER cancel_and_drain. Exceptions propagate to caller.
        """
        sandbox = self._sandboxes.get(session_id)
        if sandbox is not None:
            await sandbox.destroy()
