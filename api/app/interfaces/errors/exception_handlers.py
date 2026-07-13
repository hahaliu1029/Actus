import logging
import uuid

from app.application.errors.exceptions import AppException, TooManyRequestsError
from app.application.services.extension_install_service import (
    AcknowledgeRequiredError,
    BatchNotAllowedError,
    ForceRequiredError,
)
from app.domain.external.mailbox_publisher import MailboxPublishOversize
from app.domain.models.extension_governance import (
    InvalidStateTransitionError,
    ManagedByPluginError,
    MissingObservationError,
    OperationPendingError,
    RevisionConflictError,
)
from app.domain.services.permission.errors import (
    PEInfrastructureUnavailable,
    PolicyConflict,
    SessionModeViolation,
    UnsupportedSource,
    WriterIntegrityError,
)
from app.domain.services.permission.sources import (
    PE_SUPPORTED_SOURCES,
)
from app.infrastructure.observability.context import (
    reset_trace_context,
    set_trace_context,
)
from app.interfaces.schemas import Response
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

logger = logging.getLogger(__name__)

# B5 PR-S1-5: scope keys written by ``ObservabilityMiddleware`` so
# this module — which runs *outside* that middleware whenever the
# registered handler is for ``Exception`` / ``500`` (Starlette routes
# those keys to ``ServerErrorMiddleware``) — can still (a) attach the
# canonical ``X-Request-ID`` header on every JSON response, and
# (b) re-bind the ``TraceContext`` for the catch-all handler's
# ``logger.error`` call so the LogRecord factory injects the same
# ``trace_id`` / ``request_id`` the response header carries.
# The scope mapping is durable across the contextvar reset that fires
# in ``ObservabilityMiddleware``'s ``finally`` block, so reading
# from ``request.scope`` is safe even from the catch-all handler.
_REQUEST_ID_SCOPE_KEY = "actus_request_id"
_TRACE_CONTEXT_SCOPE_KEY = "actus_trace_context"


def _request_id_headers(request: Request) -> dict[str, str]:
    """Return ``{"X-Request-ID": <id>}`` if the middleware bound one.

    Empty dict when the request never traversed
    ``ObservabilityMiddleware`` (e.g., test scaffolds that mount the
    handler on a bare app). The empty case is silent — the canonical
    contract is "if a request_id exists, propagate it"; missing keys
    are not an error.
    """
    request_id = request.scope.get(_REQUEST_ID_SCOPE_KEY)
    if not request_id:
        return {}
    return {"X-Request-ID": str(request_id)}


def _bind_scope_trace_context(request: Request):
    """Temporarily re-bind the request's stashed ``TraceContext``.

    Used by the catch-all 500 handler — which runs in
    ``ServerErrorMiddleware`` (outside ``ObservabilityMiddleware``)
    after the contextvar has been reset by the middleware's
    ``finally`` block. Without re-binding, ``logger.error`` would
    go through ``_actus_log_record_factory`` with no bound context
    and emit ``trace_id="-"`` / ``request_id="-"``, breaking
    ``trace_id``-keyed log join for every 500.

    Returns the contextvar token to pass to
    ``reset_trace_context`` (in a ``finally`` block), or ``None``
    when no context was on scope (test scaffolds without the
    middleware) — callers should skip the reset in that case.
    """
    ctx = request.scope.get(_TRACE_CONTEXT_SCOPE_KEY)
    if ctx is None:
        return None
    return set_trace_context(ctx)


def register_exception_handlers(app: FastAPI) -> None:
    """处理Actus项目中所有的异常并进行统一处理，涵盖：自定义业务状态异常、HTTP异常、通用异常"""

    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request, exc: AppException
    ) -> JSONResponse:
        """自定义应用异常处理器，捕获AppException并返回标准化响应"""

        logger.error(f"App exception: {exc.msg}")

        headers: dict[str, str] = _request_id_headers(request)
        if isinstance(exc, TooManyRequestsError):
            retry_after = (exc.data or {}).get("retry_after")
            if retry_after is not None:
                headers["Retry-After"] = str(retry_after)

        return JSONResponse(
            status_code=exc.status_code,
            content=Response(code=exc.code, msg=exc.msg, data=exc.data or {}).model_dump(),
            headers=headers or None,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """HTTP异常处理器，捕获HTTPException并返回标准化响应。

        合并 ``exc.headers`` 到响应头：``HTTPException`` 携带的 ``headers``
        承载协议级语义（``WWW-Authenticate`` for 401 / ``Retry-After``
        for 503 / etc.），在标准化响应时丢掉就会让 ``auth.py`` 与
        ``metrics_routes.py`` 等显式声明 ``headers={"WWW-Authenticate":
        "Bearer"}`` 的 401 路径变成裸 401，破坏 RFC 7235 §3.1 契约
        且让 Prometheus / curl / 浏览器无法识别认证方式。

        合并顺序故意把 ``_request_id_headers`` 放最后，让系统级的
        ``X-Request-ID`` 胜过 caller 同 key —— ``X-Request-ID`` 是
        中间件维护的不变量，不允许被业务异常覆写。
        """

        logger.error(f"HTTP exception: {exc.detail}")

        headers: dict[str, str] = {
            **(exc.headers or {}),
            **_request_id_headers(request),
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=Response(code=exc.status_code, msg=exc.detail, data={}).model_dump(),
            headers=headers or None,
        )

    @app.exception_handler(PolicyConflict)
    async def policy_conflict_handler(
        request: Request, exc: PolicyConflict
    ) -> JSONResponse:
        """PE race / dedup / arg_digest mismatch → 409 Conflict."""
        return JSONResponse(
            status_code=409,
            content={"error": str(exc)},
            headers=_request_id_headers(request) or None,
        )

    @app.exception_handler(SessionModeViolation)
    async def session_mode_violation_handler(
        request: Request, exc: SessionModeViolation
    ) -> JSONResponse:
        """PE session lifecycle terminal state → 410 Gone."""
        return JSONResponse(
            status_code=410,
            content={"error": "session_state_invalid", "detail": str(exc)},
            headers=_request_id_headers(request) or None,
        )

    @app.exception_handler(MailboxPublishOversize)
    async def mailbox_oversize_handler(
        request: Request, exc: MailboxPublishOversize
    ) -> JSONResponse:
        """C3 PR-2 — mailbox envelope serialized size exceeded
        ``APPROVAL_PAYLOAD_MAX_BYTES``. 413 Payload Too Large.

        Defensive surface: ``RedisMailboxPublisher`` only raises when an
        internal caller (handlers / services) violates the cap. There is
        no external request path that constructs an oversized envelope
        in PR-2 — this handler exists so that future endpoints which
        synthesize envelopes from request bodies (PR-4+) surface a
        canonical 413 with a stable JSON shape instead of falling
        through to the catch-all 500.
        """
        return JSONResponse(
            status_code=413,
            content={"error": "mailbox_envelope_oversize", "detail": str(exc)},
            headers=_request_id_headers(request) or None,
        )

    @app.exception_handler(WriterIntegrityError)
    async def writer_integrity_handler(
        request: Request, exc: WriterIntegrityError
    ) -> JSONResponse:
        """PE writer downstream IntegrityError → 500 with correlation_id for ops tracing."""
        correlation_id = str(uuid.uuid4())[:16]
        logger.error(
            "WriterIntegrityError correlation_id=%s: %s", correlation_id, exc
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal", "correlation_id": correlation_id},
            headers=_request_id_headers(request) or None,
        )

    @app.exception_handler(UnsupportedSource)
    async def unsupported_source_handler(
        request: Request, exc: UnsupportedSource
    ) -> JSONResponse:
        """PE-1 §5.2: PE-internal contract violation surfaced via HTTP
        preflight (chat resume with unknown tool_source). 422 because the
        client supplied a value the server cannot satisfy."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "unsupported_tool_source",
                "source": exc.source,
                "supported_sources": sorted(PE_SUPPORTED_SOURCES),
            },
            headers=_request_id_headers(request) or None,
        )

    @app.exception_handler(PEInfrastructureUnavailable)
    async def pe_infra_unavailable_handler(
        request: Request, exc: PEInfrastructureUnavailable
    ) -> JSONResponse:
        """PE-1 §5.2: Redis / queue / writer unavailable mid-evaluate or
        mid-resume. 503 so the client (or proxy) can retry after a short
        backoff. Carries correlation_id so ops can join the request to
        backend logs."""
        correlation_id = str(uuid.uuid4())[:16]
        logger.error(
            "PEInfrastructureUnavailable correlation_id=%s reason=%s",
            correlation_id, exc.reason,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": "pe_infrastructure_unavailable",
                "correlation_id": correlation_id,
            },
            headers=_request_id_headers(request) or None,
        )

    # ---- D1a 治理异常 → HTTP 映射（§9.2；T19 注册全量 8 条，T20 只消费不再注册）----
    # 返回体形状照抄本文件既有 PE handler 惯例（JSONResponse + {"code": ...}）——
    # 不走 ``Response`` schema（其 ``msg: str`` 无法承载 dict/findings）。
    def _governance_error(request: Request, status_code: int, code: str,
                          exc: Exception) -> JSONResponse:
        content: dict = {"code": code, "detail": str(exc)}
        summary = getattr(exc, "summary", None)
        if summary is not None:                    # Acknowledge/Force 携带 scan → +findings
            dumped = summary.model_dump()
            content["scan_report"] = dumped
            content["findings"] = dumped.get("findings", [])
        return JSONResponse(
            status_code=status_code,
            content=content,
            headers=_request_id_headers(request) or None,
        )

    @app.exception_handler(RevisionConflictError)
    async def revision_conflict_handler(
        request: Request, exc: RevisionConflictError
    ) -> JSONResponse:
        return _governance_error(request, 409, "revision_conflict", exc)

    @app.exception_handler(InvalidStateTransitionError)
    async def invalid_state_handler(
        request: Request, exc: InvalidStateTransitionError
    ) -> JSONResponse:
        return _governance_error(request, 409, "invalid_state", exc)

    @app.exception_handler(MissingObservationError)
    async def missing_observation_handler(
        request: Request, exc: MissingObservationError
    ) -> JSONResponse:
        return _governance_error(request, 409, "missing_observation", exc)

    @app.exception_handler(OperationPendingError)
    async def operation_pending_handler(
        request: Request, exc: OperationPendingError
    ) -> JSONResponse:
        return _governance_error(request, 409, "operation_pending", exc)

    @app.exception_handler(ManagedByPluginError)
    async def managed_by_plugin_handler(
        request: Request, exc: ManagedByPluginError
    ) -> JSONResponse:
        return _governance_error(request, 409, "managed_by_plugin", exc)

    @app.exception_handler(AcknowledgeRequiredError)
    async def acknowledge_required_handler(
        request: Request, exc: AcknowledgeRequiredError
    ) -> JSONResponse:
        return _governance_error(request, 409, "acknowledge_required", exc)

    @app.exception_handler(ForceRequiredError)
    async def force_required_handler(
        request: Request, exc: ForceRequiredError
    ) -> JSONResponse:
        return _governance_error(request, 422, "force_required", exc)

    @app.exception_handler(BatchNotAllowedError)
    async def batch_not_allowed_handler(
        request: Request, exc: BatchNotAllowedError
    ) -> JSONResponse:
        return _governance_error(request, 422, "single_server_required", exc)

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """通用异常处理器，捕获所有未处理的异常并返回标准化响应, 状态码500"""
        # B5 PR-S1-5 (review-found P2): re-bind the request's
        # TraceContext for the duration of the log emission so the
        # crash line carries the same trace_id / request_id the
        # response header carries. Without this, the LogRecord
        # factory sees ``get_trace_context() is None`` (the outer
        # ``ObservabilityMiddleware`` already reset the contextvar
        # before this handler runs in ``ServerErrorMiddleware``)
        # and pins both fields to ``"-"``, breaking trace-keyed log
        # join on every 500.
        token = _bind_scope_trace_context(request)
        try:
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
        finally:
            if token is not None:
                reset_trace_context(token)

        headers = _request_id_headers(request)
        return JSONResponse(
            status_code=500,
            content=Response(code=500, msg="Internal Server Error", data={}).model_dump(),
            headers=headers or None,
        )
