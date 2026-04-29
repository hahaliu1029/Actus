"""B5 PR-S1-5 acceptance: ``ObservabilityMiddleware`` HTTP basics.

Pins the cross-cutting invariants that every other observability
component depends on:

- The middleware is registered as the **outermost** frame on
  ``app.user_middleware`` (Starlette ``insert(0, …)`` semantics →
  the most recently added middleware is at index 0). Anything inner
  (CORS, body limit, exception handlers, route handlers) can read
  ``trace_id`` / ``request_id`` from the contextvar; anything
  outside this position is broken by definition.
- Every HTTP response carries an ``X-Request-ID`` header equal to
  the contextvar's ``request_id``. When the client sends a
  well-formed UUIDv4 the value is propagated for distributed-trace
  correlation; when missing or malformed, a fresh UUID is minted so
  the canonical contract's NEVER-null + UUIDv4 format guarantees
  hold.
- The contextvar is bound for the duration of route-handler code
  and cleared after the response — the next request on the same
  task starts from ``None``.
"""
from __future__ import annotations

import re
import uuid

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.infrastructure.observability.context import get_trace_context
from app.interfaces.middlewares.observability_middleware import (
    ObservabilityMiddleware,
)


_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


async def _echo_endpoint(request):
    """Read the contextvar and echo the values for inspection."""
    ctx = get_trace_context()
    if ctx is None:
        return JSONResponse({"trace_id": None, "request_id": None})
    return JSONResponse(
        {
            "trace_id": ctx.trace_id,
            "request_id": ctx.request_id,
            "event_id": ctx.event_id,
            "session_id": ctx.session_id,
        }
    )


def _build_app() -> Starlette:
    app = Starlette(
        routes=[Route("/echo", _echo_endpoint, methods=["GET"])],
    )
    app.add_middleware(ObservabilityMiddleware)
    return app


@pytest.mark.anyio
async def test_response_carries_x_request_id_header() -> None:
    app = _build_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        resp = await client.get("/echo")

    assert resp.status_code == 200
    assert "x-request-id" in {h.lower() for h in resp.headers.keys()}
    rid = resp.headers["x-request-id"]
    assert _UUID4_RE.match(rid), f"X-Request-ID not UUIDv4: {rid!r}"


@pytest.mark.anyio
async def test_client_supplied_request_id_propagates() -> None:
    """Well-formed inbound X-Request-ID is echoed back verbatim (lowercase)."""
    app = _build_app()
    inbound = str(uuid.uuid4())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/echo", headers={"X-Request-ID": inbound})

    assert resp.headers["x-request-id"] == inbound.lower()
    assert resp.json()["request_id"] == inbound.lower()


@pytest.mark.anyio
async def test_malformed_request_id_replaced_with_fresh_uuid() -> None:
    """Bad inbound X-Request-ID → fresh UUIDv4; client cannot poison contract."""
    app = _build_app()
    bogus = "not-a-uuid"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/echo", headers={"X-Request-ID": bogus})

    body = resp.json()
    assert body["request_id"] != bogus
    assert _UUID4_RE.match(body["request_id"]), body
    assert resp.headers["x-request-id"] == body["request_id"]


@pytest.mark.anyio
async def test_trace_id_is_32_hex_lowercase() -> None:
    """Sprint-1 ``trace_id`` fallback is ``uuid4().hex`` (32 lowercase hex)."""
    app = _build_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/echo")

    body = resp.json()
    assert _TRACE_ID_RE.match(body["trace_id"]), body
    assert body["trace_id"] == body["trace_id"].lower()


@pytest.mark.anyio
async def test_contextvar_populated_inside_handler() -> None:
    """Endpoint handler reads a non-None ``TraceContext`` from the contextvar."""
    app = _build_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/echo")

    body = resp.json()
    assert body["trace_id"] is not None
    assert body["request_id"] is not None
    assert body["event_id"] is not None


@pytest.mark.anyio
async def test_contextvar_reset_between_requests() -> None:
    """Sequential requests get independent ``TraceContext`` instances."""
    app = _build_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = (await client.get("/echo")).json()
        second = (await client.get("/echo")).json()

    assert first["trace_id"] != second["trace_id"]
    assert first["request_id"] != second["request_id"]


@pytest.mark.anyio
async def test_contextvar_cleared_after_request() -> None:
    """After ``await app(...)`` the caller-side contextvar is None again."""
    app = _build_app()

    # Pre-condition.
    assert get_trace_context() is None

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.get("/echo")

    # Post-condition: the middleware reset its bound token in finally.
    assert get_trace_context() is None


def _build_fastapi_app_with_handlers():
    """FastAPI app + ``register_exception_handlers`` + observability.

    Mirrors ``app.main`` wiring at the level needed for the JSON-500
    contract: the ``@app.exception_handler(Exception)`` registered by
    ``register_exception_handlers`` is dispatched by Starlette's
    ``ServerErrorMiddleware`` (which lives OUTSIDE
    ``ObservabilityMiddleware``), so the ``X-Request-ID`` header
    must be attached by the handler itself reading
    ``request.scope["actus_request_id"]``.
    """
    from fastapi import FastAPI

    from app.interfaces.errors.exception_handlers import (
        register_exception_handlers,
    )

    app = FastAPI()

    @app.get("/explode")
    async def _explode():
        raise RuntimeError("crash before response_started")

    register_exception_handlers(app)
    app.add_middleware(ObservabilityMiddleware)
    return app


@pytest.mark.anyio
async def test_500_response_carries_x_request_id_header() -> None:
    """Review-found P1: 500 path keeps the canonical JSON shape AND header.

    The handler registered by ``register_exception_handlers`` runs in
    Starlette's ``ServerErrorMiddleware`` (outside
    ``ObservabilityMiddleware``), so the ``send`` it ultimately
    calls is the original ASGI ``send`` and bypasses our send
    wrapper. The middleware therefore stashes the request_id on
    ``scope["actus_request_id"]`` and the handler reads it from
    ``request.scope`` to attach the ``X-Request-ID`` header itself
    — preserving both the canonical JSON body
    ``{"code":500,"msg":"Internal Server Error","data":{}}`` and
    the trace-correlation header.
    """
    app = _build_fastapi_app_with_handlers()
    inbound = str(uuid.uuid4())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/explode", headers={"X-Request-ID": inbound}
        )

    # Canonical JSON 500 contract.
    assert resp.status_code == 500
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {
        "code": 500,
        "msg": "Internal Server Error",
        "data": {},
    }

    # Client-supplied UUIDv4 propagates verbatim (lowercase) on the
    # JSON-500 response — proving the handler read scope and attached
    # the header even though the response went through ServerError.
    assert resp.headers["x-request-id"] == inbound.lower()


@pytest.mark.anyio
async def test_500_handler_log_carries_trace_context() -> None:
    """Review-found P2: 500 crash log must carry trace_id / request_id.

    The catch-all ``Exception`` handler runs inside Starlette's
    ``ServerErrorMiddleware`` — *outside*
    ``ObservabilityMiddleware`` whose ``finally`` block has already
    reset the contextvar. Without re-binding, the handler's
    ``logger.error(..., exc_info=True)`` would go through
    ``_actus_log_record_factory`` with no bound context and emit
    ``trace_id="-"`` / ``request_id="-"`` — breaking
    ``trace_id``-keyed log join on every 500 even though the JSON
    response and ``X-Request-ID`` header are correct.

    The handler reads the stashed ctx from
    ``request.scope["actus_trace_context"]`` and re-binds it for
    the log emission. We capture root LogRecords here and assert
    the handler's "Unhandled exception" line carries the same
    ``request_id`` the client supplied (and a real, non-default
    32-hex ``trace_id``).
    """
    import logging as _logging
    import re as _re

    captured: list[_logging.LogRecord] = []

    class _CaptureHandler(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            captured.append(record)

    root = _logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    capture = _CaptureHandler(level=_logging.DEBUG)
    root.addHandler(capture)
    if root.level > _logging.DEBUG:
        root.setLevel(_logging.DEBUG)
    try:
        app = _build_fastapi_app_with_handlers()
        inbound = str(uuid.uuid4())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=app, raise_app_exceptions=False
            ),
            base_url="http://test",
        ) as client:
            resp = await client.get(
                "/explode", headers={"X-Request-ID": inbound}
            )

        assert resp.status_code == 500
        assert resp.headers["x-request-id"] == inbound.lower()

        handler_log_records = [
            r
            for r in captured
            if r.name == "app.interfaces.errors.exception_handlers"
            and "Unhandled exception" in r.getMessage()
        ]
        assert handler_log_records, (
            "no 'Unhandled exception' log line captured from the "
            "catch-all handler — test scaffold did not propagate "
            "correctly to the registered exception_handler"
        )
        crash_record = handler_log_records[-1]

        # The whole point of P2 — same request_id as response header.
        assert crash_record.request_id == inbound.lower(), (
            f"crash log request_id={crash_record.request_id!r} does not "
            f"match response X-Request-ID={inbound.lower()!r} — handler "
            "did not re-bind scope TraceContext before logging"
        )
        # trace_id is non-default and matches the canonical 32-hex shape.
        assert crash_record.trace_id != "-", (
            "crash log trace_id is the default placeholder — "
            "TraceContext was not re-bound"
        )
        assert _re.match(r"^[0-9a-f]{32}$", crash_record.trace_id), (
            f"crash log trace_id={crash_record.trace_id!r} is not "
            "32-hex lowercase"
        )
    finally:
        root.removeHandler(capture)
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)


@pytest.mark.anyio
async def test_500_x_request_id_uses_fresh_uuid_when_not_supplied() -> None:
    """500 path still gets a fresh request_id when client did not send one."""
    app = _build_fastapi_app_with_handlers()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        resp = await client.get("/explode")

    assert resp.status_code == 500
    assert resp.json() == {
        "code": 500,
        "msg": "Internal Server Error",
        "data": {},
    }
    rid = resp.headers["x-request-id"]
    assert _UUID4_RE.match(rid), f"X-Request-ID not UUIDv4: {rid!r}"


def test_observability_middleware_registered_outermost() -> None:
    """``app.user_middleware[0]`` is ``ObservabilityMiddleware``.

    Starlette's ``add_middleware`` does ``user_middleware.insert(0,
    …)``, so the most recently added middleware sits at index 0.
    The build-stack reverse-walk wraps it last, making it the
    outermost frame and the first to see every incoming request.
    Production main.py adds it AFTER ``CORSMiddleware`` and the
    ``@app.middleware("http")`` body-limit decorator so it ends up
    at the top of the stack.
    """
    from app.main import app

    assert app.user_middleware, "no middleware registered"
    top = app.user_middleware[0]
    cls_name = (
        top.cls.__name__ if hasattr(top, "cls") else type(top).__name__
    )
    assert cls_name == "ObservabilityMiddleware", (
        f"outermost middleware is {cls_name}, expected ObservabilityMiddleware"
    )
