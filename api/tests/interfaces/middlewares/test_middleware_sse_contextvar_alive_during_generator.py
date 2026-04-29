"""B5 PR-S1-5 acceptance: ``TraceContext`` survives the SSE generator.

Closes FOLLOW-9 — the original ``BaseHTTPMiddleware`` regression that
broke trace correlation on Actus's primary user path
``/sse/sessions/...``. With ``BaseHTTPMiddleware`` the response object
is constructed synchronously, the ``finally + reset`` fires
immediately, and by the time the SSE generator runs (``await
asyncio.sleep`` for inference, network calls to OpenAI, sandbox
shell) the contextvar is already reset to ``None`` — every emitted
log line / span attribute / event payload would carry ``trace_id="-"``
instead of the bound request's id.

The pure-ASGI middleware in PR-S1-5 fixes this by relying on
``await self.app(scope, receive, send)`` not returning until the
response body — including every SSE frame the generator emits — is
fully drained. We sleep mid-generator to widen the window during
which the regression would otherwise bite, then assert the
contextvar value is identical at every checkpoint.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.routing import Route

from app.infrastructure.observability.context import get_trace_context
from app.interfaces.middlewares.observability_middleware import (
    ObservabilityMiddleware,
)


@pytest.mark.anyio
async def test_contextvar_alive_during_sse_generator() -> None:
    """``get_trace_context()`` returns the same non-None ctx across yields."""
    captures: list[tuple[str, object]] = []

    async def slow_generator():
        captures.append(("pre-sleep", get_trace_context()))
        yield {"event": "stage", "data": "before-sleep"}
        # The original BaseHTTPMiddleware bug surfaces here: by the
        # time we resume, the conventional finally/reset would have
        # fired and ``get_trace_context()`` would return ``None``.
        await asyncio.sleep(0.05)
        captures.append(("post-sleep", get_trace_context()))
        yield {"event": "stage", "data": "after-sleep"}
        await asyncio.sleep(0.05)
        captures.append(("final", get_trace_context()))
        yield {"event": "stage", "data": "final"}

    async def stream_endpoint(request):
        return EventSourceResponse(slow_generator())

    app = Starlette(routes=[Route("/stream", stream_endpoint)])
    app.add_middleware(ObservabilityMiddleware)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream("GET", "/stream") as resp:
            assert resp.status_code == 200
            # Drain the response so the generator runs to completion.
            async for _ in resp.aiter_bytes():
                pass

    assert len(captures) == 3, captures
    # Every checkpoint inside the generator saw a real ``TraceContext``.
    for label, ctx in captures:
        assert ctx is not None, f"contextvar None at {label}"

    # And it was the SAME context object across yields — trace_id
    # / request_id correlate to one request, not three.
    assert captures[0][1].trace_id == captures[1][1].trace_id
    assert captures[1][1].trace_id == captures[2][1].trace_id
    assert captures[0][1].request_id == captures[1][1].request_id
