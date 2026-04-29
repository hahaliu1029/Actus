"""B5 PR-S1-5 acceptance: ``TraceContext`` reset survives downstream raise.

When a route handler or SSE generator raises, the contextvar must
still be reset to ``None`` so the next request on the same task
starts from a clean binding. The pure-ASGI middleware uses a plain
``try/finally`` around ``await self.app(...)``, which is the
canonical Python idiom for "always run on the way out, regardless
of exception path".

Two scenarios are exercised:

- A synchronous route handler that raises before any response is
  written (the exception bubbles up through ``await self.app(...)``
  and Starlette converts it into a 500).
- An SSE generator that raises mid-stream (``sse_starlette``
  swallows generator exceptions and closes the stream gracefully —
  ``await self.app(...)`` returns normally — but the contextvar
  must still be reset).
"""
from __future__ import annotations

import httpx
import pytest
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.infrastructure.observability.context import get_trace_context
from app.interfaces.middlewares.observability_middleware import (
    ObservabilityMiddleware,
)


@pytest.mark.anyio
async def test_reset_runs_when_route_handler_raises() -> None:
    """500 path still resets the contextvar."""

    async def explode_endpoint(request):
        raise RuntimeError("handler crashed before response")

    async def healthy_endpoint(request):
        return JSONResponse({"trace_id": get_trace_context().trace_id})

    app = Starlette(
        routes=[
            Route("/explode", explode_endpoint, methods=["GET"]),
            Route("/echo", healthy_endpoint, methods=["GET"]),
        ]
    )
    app.add_middleware(ObservabilityMiddleware)

    # Pre-condition.
    assert get_trace_context() is None

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        resp = await client.get("/explode")
        assert resp.status_code == 500

        # The next request on the same task starts from a fresh
        # context — the previous request's binding did not leak.
        resp2 = await client.get("/echo")
        first_trace = resp2.json()["trace_id"]
        resp3 = await client.get("/echo")
        second_trace = resp3.json()["trace_id"]

    assert get_trace_context() is None
    assert first_trace != second_trace, (
        "subsequent requests should each get a fresh trace_id; "
        "leak indicates contextvar not reset after exception"
    )


@pytest.mark.anyio
async def test_reset_runs_when_sse_generator_raises() -> None:
    """SSE generator raising mid-stream still resets the contextvar."""

    async def crashing_generator():
        yield {"event": "stage", "data": "first"}
        raise RuntimeError("agent crashed mid-stream")

    async def stream_endpoint(request):
        return EventSourceResponse(crashing_generator())

    app = Starlette(routes=[Route("/stream", stream_endpoint)])
    app.add_middleware(ObservabilityMiddleware)

    assert get_trace_context() is None

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        async with client.stream("GET", "/stream") as resp:
            # Drain whatever sse_starlette emits before / on the
            # exception — it may close the stream gracefully or with
            # a partial body. Either way is acceptable; we only
            # assert the post-condition.
            try:
                async for _ in resp.aiter_bytes():
                    pass
            except Exception:
                pass

    assert get_trace_context() is None, (
        "contextvar leaked past the SSE generator exception path"
    )
