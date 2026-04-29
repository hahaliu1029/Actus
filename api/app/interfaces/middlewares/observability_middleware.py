"""Pure-ASGI ``ObservabilityMiddleware`` (B5 PR-S1-5, FOLLOW-9).

Installs a per-request ``TraceContext`` on the contextvar carrier
*before* any inner middleware (CORS / body-limit / exception
handlers) runs, and resets it *after* the response body — including
the ``EventSourceResponse`` SSE generator on Actus's main user path
``/sse/sessions/...`` — finishes writing.

Why pure ASGI (and not ``BaseHTTPMiddleware``)
-----------------------------------------------
``BaseHTTPMiddleware.dispatch`` returns a ``Response`` object
**synchronously** for streaming bodies; the conventional
``try/finally + reset`` pattern fires its ``finally`` block the
moment the ``Response`` is constructed, *before* the SSE generator
starts emitting. That severs the contextvar lifetime exactly when
agent work begins emitting events — trace_id correlation is lost on
the project's primary user path.

Pure ASGI middleware ``await self.app(scope, receive, send)`` does
**not** return until the response body (every chunk, including the
final ``more_body=False`` marker the SSE generator sends after its
last frame) is fully emitted. The ``finally`` block therefore fires
*after* the generator completes, so contextvar lifetime spans the
entire SSE response.

Registration
------------
``app.main`` adds this middleware *last* via ``app.add_middleware``.
Starlette's ``add_middleware`` performs ``user_middleware.insert(0,
…)``, so the most recently added middleware sits at index 0; the
``build_middleware_stack`` reverse-walk wraps it last, making it the
outermost frame and the first to see every incoming request.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Awaitable, Callable, MutableMapping

from app.domain.external.observability import TraceContext
from app.infrastructure.observability.context import (
    reset_trace_context,
    set_trace_context,
)

_REQUEST_ID_HEADER: bytes = b"x-request-id"

# Mirrors the canonical contract regex in ``domain.external.observability``
# (``_UUID4_RE``). Defined locally so this interfaces-layer module does
# not reach into a private domain symbol; the contract is small and
# stable.
_UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _extract_request_id(
    headers: list[tuple[bytes, bytes]],
) -> str:
    """Return a canonical-format request_id read from headers, else fresh.

    Header lookup is case-insensitive (per RFC 7230). When the client
    supplies an ``X-Request-ID`` that is well-formed UUIDv4 it is
    propagated as-is for distributed-trace correlation; when missing
    or malformed, a fresh ``uuid.uuid4()`` is minted so the canonical
    contract's NEVER-null + UUIDv4 format guarantees still hold.
    """
    for name, value in headers:
        if name.lower() == _REQUEST_ID_HEADER:
            decoded = value.decode("latin-1", errors="ignore").strip()
            if _UUID4_PATTERN.match(decoded):
                return decoded.lower()
            # Malformed inbound — replace; canonical contract requires
            # UUIDv4. Don't pass it through and risk downstream
            # ``validate_attributes`` rejecting the whole emission.
            break
    return str(uuid.uuid4())


def _build_context_from_scope(scope: MutableMapping[str, Any]) -> TraceContext:
    """Construct the per-request ``TraceContext`` from an ASGI HTTP scope.

    ``trace_id`` is fresh ``uuid4().hex`` (32 lowercase hex, dash-less)
    matching the Sprint-1 fallback path until Sprint-2 wires the real
    OTel TraceContext from the ``traceparent`` header.

    ``event_id`` is fresh per-request UUIDv4; downstream emitters
    (PromptTelemetry, future spans) generate their own ``event_id``
    per emission via ``build_canonical_attributes`` — the
    middleware-bound value is just a placeholder so the contextvar
    is contract-compliant even before any emitter runs.

    ``session_id`` stays ``None`` until a sub-flow binds it via
    ``bind_session_context`` (chat/SSE path resolves session_id from
    URL params; long-running CLI tools may bind explicitly).
    """
    headers = scope.get("headers") or []
    request_id = _extract_request_id(headers)
    return TraceContext(
        trace_id=uuid.uuid4().hex,
        request_id=request_id,
        event_id=str(uuid.uuid4()),
    )


class ObservabilityMiddleware:
    """Pure-ASGI middleware that owns the per-request ``TraceContext``."""

    def __init__(
        self,
        app: Callable[
            [
                MutableMapping[str, Any],
                Callable[..., Awaitable[Any]],
                Callable[..., Awaitable[Any]],
            ],
            Awaitable[Any],
        ],
    ) -> None:
        self.app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[Any]],
        send: Callable[[Any], Awaitable[None]],
    ) -> None:
        # WebSocket / lifespan scopes do not carry a TraceContext;
        # passthrough without binding so we never leak a context onto
        # a long-lived ws connection (the per-message life cycle is
        # handled separately by chat-stream code paths).
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        ctx = _build_context_from_scope(scope)
        # Stash request_id AND the full TraceContext on ``scope``
        # BEFORE binding the contextvar. Global exception handlers
        # registered via ``register_exception_handlers`` read these
        # keys for two distinct reasons:
        #
        # 1. ``actus_request_id`` — the catch-all 500 handler runs
        #    inside ``ServerErrorMiddleware`` (outside this
        #    middleware) and sends the response via the original
        #    ASGI ``send``, bypassing our send wrapper. The handler
        #    attaches ``X-Request-ID`` directly on its
        #    ``JSONResponse`` so the failing client still gets the
        #    correlation header.
        #
        # 2. ``actus_trace_context`` (review-found P2) — by the time
        #    the catch-all handler runs, this middleware's
        #    ``finally`` block has already reset the contextvar.
        #    ``logger.error(..., exc_info=True)`` inside the handler
        #    would therefore go through ``_actus_log_record_factory``
        #    with no bound context and emit ``trace_id="-"`` /
        #    ``request_id="-"``, breaking ``trace_id``-keyed log
        #    join on every 500. The handler temporarily re-binds
        #    this stashed ctx around its log call so the crash log
        #    line carries the same trace_id/request_id as the
        #    response header.
        scope["actus_request_id"] = ctx.request_id
        scope["actus_trace_context"] = ctx

        token = set_trace_context(ctx)
        request_id_bytes = ctx.request_id.encode("latin-1")

        async def send_with_request_id_header(message: Any) -> None:
            if message.get("type") == "http.response.start":
                # Drop any inner-set ``X-Request-ID`` so the value the
                # client sees is exactly the one we bound to the
                # contextvar — single source of truth for correlation.
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != _REQUEST_ID_HEADER
                ]
                headers.append((_REQUEST_ID_HEADER, request_id_bytes))
                message["headers"] = headers
            await send(message)

        try:
            # No inner try/except: route-handler exceptions that
            # match ``Exception`` / ``500`` are dispatched by the
            # outer ``ServerErrorMiddleware``. The send wrapper here
            # still covers (a) every successful response and (b) any
            # ``AppException`` / ``HTTPException`` (or other typed
            # exception) that ``ExceptionMiddleware`` dispatches —
            # those responses *do* travel through the wrapper. The
            # ``Exception`` catch-all bypasses the wrapper but its
            # ``X-Request-ID`` is attached by the registered handler
            # via ``scope["actus_request_id"]``.
            await self.app(scope, receive, send_with_request_id_header)
        finally:
            # ``finally`` runs after the response body is fully drained
            # (SSE generator completion, client disconnect raising
            # ``CancelledError``, or downstream exception). The
            # ``reset_trace_context`` always reverts the contextvar so
            # the next request on this task starts from a clean slate.
            reset_trace_context(token)
