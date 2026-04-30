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

import asyncio
import re
import uuid
from typing import Any, Awaitable, Callable, MutableMapping

from opentelemetry.trace import Status, StatusCode

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


def _build_context_from_scope(
    scope: MutableMapping[str, Any],
    *,
    trace_id: str | None = None,
) -> TraceContext:
    """Construct the per-request ``TraceContext`` from an ASGI HTTP scope.

    ``trace_id`` defaults to fresh ``uuid4().hex`` (32 lowercase hex,
    dash-less) when no caller-supplied value is given. The middleware
    passes the **OTel native trace_id** of the request-root span
    (B5 PR-S2-2 round-3 fix) so the canonical Actus ``trace_id`` and
    the OTel native ``trace_id`` are the SAME 32-hex string — Phoenix
    / Jaeger / Loki UIs that key by native trace can also be filtered
    by Actus canonical attribute lookups, and vice versa.

    ``event_id`` is fresh per-request UUIDv4; downstream emitters
    (PromptTelemetry, span attrs) generate their own ``event_id`` per
    emission via ``build_canonical_attributes`` — the middleware-bound
    value is just a placeholder so the contextvar is contract-compliant
    even before any emitter runs.

    ``session_id`` stays ``None`` until a sub-flow binds it via
    ``bind_session_context`` (chat/SSE path resolves session_id from
    URL params; long-running CLI tools may bind explicitly).
    """
    headers = scope.get("headers") or []
    request_id = _extract_request_id(headers)
    return TraceContext(
        trace_id=trace_id if trace_id is not None else uuid.uuid4().hex,
        request_id=request_id,
        event_id=str(uuid.uuid4()),
    )


def _otel_trace_id_hex(span: Any) -> str | None:
    """Return ``span``'s native OTel trace_id as 32-hex, or ``None``.

    Returns ``None`` when the span carries the ``INVALID_TRACE_ID``
    sentinel (= ``0``) — this happens when no ``TracerProvider`` is
    installed (very early startup, tests that explicitly tear down).
    The middleware then falls back to ``uuid4().hex`` so the canonical
    contract's NEVER-null guarantee still holds.
    """
    try:
        ctx = span.get_span_context()
    except Exception:
        return None
    raw = getattr(ctx, "trace_id", 0)
    if not isinstance(raw, int) or raw == 0:
        return None
    return format(raw, "032x")


def _route_template_from_scope(scope: MutableMapping[str, Any]) -> str | None:
    """Return the matched Starlette route TEMPLATE, or ``None``.

    Starlette sets ``scope["route"]`` to the matched ``Route`` object
    after routing succeeds. ``Route.path`` is the TEMPLATE
    (``/sessions/{session_id}/chat``) — low-cardinality and free of
    path-param PII, which is exactly what the OTel semantic
    convention requires for ``http.route``. We deliberately do NOT
    fall back to ``scope["path"]`` here: the literal URL would defeat
    the whole purpose of the attribute.

    Returns ``None`` when no route matched (404), when ``scope["route"]``
    is not a Starlette Route instance, or when ``Route.path`` is
    missing — the caller then skips the ``http.route`` attribute
    entirely.
    """
    route = scope.get("route")
    if route is None:
        return None
    template = getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    return None


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

        # B5 PR-S2-2 round-3 fix: open an OTel root span around the
        # whole request so every child span (graph node spans + tool
        # spans inside ``traced_node`` / ``OtelToolSpanCallback``)
        # inherits the same native OTel ``trace_id`` via the OTel
        # context propagation chain. Without this root, each
        # ``start_as_current_span`` inside the request would mint its
        # own native trace, fragmenting Phoenix / Jaeger trace trees
        # — even though the canonical attribute ``trace_id`` was
        # consistent (Actus join key only).
        #
        # The Actus canonical ``trace_id`` is sourced from this span's
        # native trace_id (32-hex) so the Actus attr trace_id and the
        # OTel native trace_id are the SAME string — single source of
        # truth across both UI views.
        from app.infrastructure.observability.otel_tracer import OtelTracer

        tracer = OtelTracer()
        method = scope.get("method", "")

        # NOTE: ``http.route`` is intentionally NOT set here. The OTel
        # semantic convention defines ``http.route`` as the matched
        # route TEMPLATE (e.g. ``/sessions/{session_id}/chat``), not
        # the literal request URL. Setting it to ``scope["path"]``
        # would explode cardinality (one dimension value per session
        # id / object id) and leak path-param PII into span attrs.
        # Starlette only writes ``scope["route"]`` AFTER routing has
        # matched the request — we backfill from there in the send
        # wrapper when ``http.response.start`` fires (see below).
        with tracer.start_as_current_span(
            f"http.{method.lower() or 'request'}",
            attributes={
                "http.method": method,
            },
        ) as root_span:
            otel_trace_id = _otel_trace_id_hex(root_span)
            ctx = _build_context_from_scope(scope, trace_id=otel_trace_id)
            # Stash request_id AND the full TraceContext on ``scope``
            # BEFORE binding the contextvar. Global exception handlers
            # registered via ``register_exception_handlers`` read
            # these keys for two distinct reasons:
            #
            # 1. ``actus_request_id`` — the catch-all 500 handler
            #    runs inside ``ServerErrorMiddleware`` (outside this
            #    middleware) and sends the response via the original
            #    ASGI ``send``, bypassing our send wrapper. The
            #    handler attaches ``X-Request-ID`` directly on its
            #    ``JSONResponse`` so the failing client still gets
            #    the correlation header.
            #
            # 2. ``actus_trace_context`` (review-found P2) — by the
            #    time the catch-all handler runs, this middleware's
            #    ``finally`` block has already reset the contextvar.
            #    ``logger.error(..., exc_info=True)`` inside the
            #    handler would therefore go through
            #    ``_actus_log_record_factory`` with no bound context
            #    and emit ``trace_id="-"`` / ``request_id="-"``,
            #    breaking ``trace_id``-keyed log join on every 500.
            #    The handler temporarily re-binds this stashed ctx
            #    around its log call so the crash log line carries
            #    the same trace_id/request_id as the response header.
            scope["actus_request_id"] = ctx.request_id
            scope["actus_trace_context"] = ctx

            token = set_trace_context(ctx)
            request_id_bytes = ctx.request_id.encode("latin-1")
            # B5 PR-S2-2 round-8 P2 fix: track the http.response.start
            # status code in a closure cell so the CancelledError
            # branch knows whether a 5xx has already been emitted. The
            # cancel branch must NOT overwrite a recorded ERROR with
            # OK — see the except clause below for the policy.
            response_started_status: list[int | None] = [None]

            async def send_with_request_id_header(message: Any) -> None:
                if message.get("type") == "http.response.start":
                    # Backfill ``http.route`` from the route template
                    # NOW that Starlette has finished routing. The
                    # template (``/sessions/{session_id}/chat``) is
                    # the low-cardinality, PII-safe value the OTel
                    # semantic convention requires. Unmatched routes
                    # (404 path) leave ``scope["route"]`` unset → we
                    # skip the attribute entirely (better than emitting
                    # the raw URL).
                    template = _route_template_from_scope(scope)
                    if template is not None:
                        try:
                            root_span.set_attribute("http.route", template)
                        except Exception:
                            # Defensive: a recording-disabled or
                            # closed span must not crash the send path.
                            pass
                    # B5 PR-S2-2 round-5 fix: write ``http.status_code``
                    # so root spans can be filtered / alerted on by 2xx
                    # / 4xx / 5xx, and compute server-side error rate.
                    # Use the legacy ``http.status_code`` name to stay
                    # consistent with the legacy ``http.method`` /
                    # ``http.route`` keys we already write — switching
                    # to ``http.response.status_code`` (new convention)
                    # would split the dimension surface across attr
                    # names. Span status: only ``>=500`` flips to
                    # ``ERROR`` (4xx is treated as a successful client-
                    # observable outcome per OTel server-span guidance
                    # — handled exceptions, missing routes, auth
                    # rejections etc. are not server faults).
                    status_code = message.get("status")
                    if isinstance(status_code, int):
                        # Round-8 fix: stash the started-response
                        # status BEFORE we touch span attrs so the
                        # CancelledError branch (if it fires next)
                        # can decide whether to suppress OTel
                        # auto-ERROR or honour an already-recorded
                        # ``>=500`` outcome.
                        response_started_status[0] = status_code
                        try:
                            root_span.set_attribute(
                                "http.status_code", status_code
                            )
                        except Exception:
                            pass
                        if status_code >= 500:
                            try:
                                root_span.set_status(Status(StatusCode.ERROR))
                            except Exception:
                                pass
                    # Drop any inner-set ``X-Request-ID`` so the
                    # value the client sees is exactly the one we
                    # bound to the contextvar — single source of
                    # truth for correlation.
                    headers = [
                        (name, value)
                        for name, value in message.get("headers", [])
                        if name.lower() != _REQUEST_ID_HEADER
                    ]
                    headers.append((_REQUEST_ID_HEADER, request_id_bytes))
                    message["headers"] = headers
                await send(message)

            try:
                # The send wrapper above covers (a) every successful
                # response and (b) any ``AppException`` /
                # ``HTTPException`` (or other typed exception) that
                # ``ExceptionMiddleware`` dispatches — those
                # responses *do* travel through the wrapper.
                #
                # The ``Exception`` catch-all path (a route raising
                # ``RuntimeError`` that no FastAPI exception handler
                # claims) is rendered by ``ServerErrorMiddleware`` AND
                # re-raises the original exception after sending the
                # 500 — so the exception propagates back here. The
                # send wrapper would *typically* still see the 500
                # response, but for cases where the response never
                # started (exception during routing setup / lifecycle
                # bridge / pre-send handler) we backfill on the
                # except branch so the root span always carries
                # ``http.status_code`` for any 5xx outcome (B5
                # PR-S2-2 round-6 P2 fix).
                await self.app(scope, receive, send_with_request_id_header)
            except asyncio.CancelledError:
                # B5 PR-S2-2 round-7 P2 fix: ``asyncio.CancelledError``
                # is control flow (SSE client disconnect, request task
                # cancelled by lifespan shutdown, upstream cancel via
                # ``CancelledError`` propagation), NOT a server fault.
                # In Python 3.8+ ``CancelledError`` inherits from
                # ``BaseException`` (not ``Exception``) precisely so
                # generic ``except Exception:`` won't swallow it.
                #
                # Without this explicit clause, ``except Exception``
                # below would not match (correct), but OTel's
                # ``start_as_current_span`` context-manager ``__exit__``
                # still flips span status to ERROR on ANY exception
                # propagating through it — polluting 5xx error rate
                # dashboards with cancellations.
                #
                # Round-8 P2 fix: only suppress (set OK) when the
                # cancellation is genuinely the dominant signal —
                # i.e. either NO response started, OR the started
                # response was non-5xx. If a ``>=500`` response was
                # already emitted via the send wrapper, the ERROR
                # span status reflects a real server fault; we MUST
                # NOT overwrite it with OK just because a cancel
                # raced in afterwards (OTel SDK allows
                # ``ERROR → OK`` transitions, so the bare
                # ``set_status(OK)`` would silently undo it).
                started = response_started_status[0]
                if started is None or started < 500:
                    try:
                        root_span.set_status(Status(StatusCode.OK))
                    except Exception:
                        pass
                # else: a >=500 outcome was already recorded — leave
                # the ERROR status (and the existing
                # ``http.status_code`` attr) untouched.
                raise
            except Exception:
                # Real server fault (uncaught exception bubbled past
                # all FastAPI handlers, or ``ServerErrorMiddleware``
                # re-raised after rendering 500). Set status code +
                # ERROR BEFORE re-raise. Skip overwriting the attr if
                # the send wrapper already wrote one (e.g. 500
                # actually streamed before re-raise).
                try:
                    if "http.status_code" not in (
                        getattr(root_span, "attributes", {}) or {}
                    ):
                        root_span.set_attribute("http.status_code", 500)
                except Exception:
                    pass
                # Backfill ``http.route`` if routing matched before
                # the exception fired (handler-level raise).
                try:
                    template = _route_template_from_scope(scope)
                    if template is not None:
                        root_span.set_attribute("http.route", template)
                except Exception:
                    pass
                try:
                    root_span.set_status(Status(StatusCode.ERROR))
                except Exception:
                    pass
                raise
            finally:
                # ``finally`` runs after the response body is fully
                # drained (SSE generator completion, client
                # disconnect raising ``CancelledError``, or
                # downstream exception). ``reset_trace_context``
                # always reverts the contextvar so the next request
                # on this task starts from a clean slate. The OTel
                # root span ends when the surrounding ``with`` exits
                # (just after this finally runs).
                reset_trace_context(token)
