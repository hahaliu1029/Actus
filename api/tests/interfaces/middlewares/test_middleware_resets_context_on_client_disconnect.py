"""B5 PR-S1-5 acceptance: client disconnect mid-SSE still resets context.

The third FOLLOW-9 escape route: client closes the connection while
the SSE generator is still emitting frames. ASGI servers signal this
to the app via an ``http.disconnect`` message on ``receive``. The
SSE response (``sse_starlette.EventSourceResponse``) polls
``receive`` and detects the disconnect; the generator winds down
and ``await self.app(scope, receive, send)`` returns. Our
``try/finally`` is a plain async-aware ``finally`` block, so it
ALWAYS runs.

We drive the middleware directly with a custom ``receive`` that
emits ``http.disconnect`` after the first ``http.request`` —
deterministically reproducing the disconnect path without relying
on ``httpx.ASGITransport`` whose disconnect timing varies across
versions. This also frees the test from any ``EventSourceResponse``
implementation detail (frame timing, polling cadence) — the
contract under test is "middleware reset on disconnect", not
"sse_starlette propagates disconnect".

Two assertions:

- The generator did NOT run to completion (frame count below the
  intended 100).
- The contextvar is back to ``None`` after the disconnect path.
"""
from __future__ import annotations

import asyncio

import pytest
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.routing import Route

from app.infrastructure.observability.context import get_trace_context
from app.interfaces.middlewares.observability_middleware import (
    ObservabilityMiddleware,
)


def _http_scope(path: str = "/stream") -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "server": ("test", 80),
        "client": ("test", 65432),
    }


@pytest.mark.anyio
async def test_reset_runs_on_client_disconnect_mid_stream() -> None:
    """Client closes mid-flight; contextvar is still reset to ``None``."""
    frames_emitted: list[int] = []

    async def long_generator():
        for i in range(100):
            frames_emitted.append(i)
            yield {"event": "stage", "data": f"frame-{i}"}
            await asyncio.sleep(0.02)

    async def stream_endpoint(request):
        return EventSourceResponse(long_generator())

    app = Starlette(routes=[Route("/stream", stream_endpoint)])
    app.add_middleware(ObservabilityMiddleware)

    request_sent = False
    disconnect_emitted = False

    async def receive():
        nonlocal request_sent, disconnect_emitted
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }
        if not disconnect_emitted:
            # Sleep long enough for the generator to emit a few
            # frames, then signal client disconnect.
            await asyncio.sleep(0.05)
            disconnect_emitted = True
            return {"type": "http.disconnect"}
        # After disconnect, block forever — the SSE response should
        # already be unwinding.
        await asyncio.Event().wait()

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    assert get_trace_context() is None

    try:
        await app(_http_scope(), receive, send)
    except (asyncio.CancelledError, Exception):
        # Disconnect propagation may surface as cancel or exception
        # group depending on event-loop / sse_starlette internals.
        # The middleware's ``finally`` runs on every path.
        pass

    # Allow any pending generator cleanup / finally chains to settle.
    await asyncio.sleep(0.05)

    # 1. Generator did not run to completion — disconnect short-
    # circuited it. Some frames are emitted before disconnect lands;
    # we just need < 100.
    assert len(frames_emitted) < 100, (
        f"generator emitted all {len(frames_emitted)} frames; "
        "client disconnect did not propagate to the response body"
    )

    # 2. Contextvar reset ran despite the disconnect / cancel path.
    assert get_trace_context() is None, (
        "contextvar leaked past the client-disconnect path"
    )
