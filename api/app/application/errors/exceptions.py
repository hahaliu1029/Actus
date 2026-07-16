from typing import Any


class AppException(RuntimeError):
    """基础应用异常类，继承RuntimeError"""

    def __init__(
        self,
        code: int = 400,
        status_code: int = 400,
        msg: str = "应用程序异常",
        data: Any = None,
    ):
        """构造函数，完成错误数据初始化"""
        self.code = code
        self.status_code = status_code
        self.msg = msg
        self.data = data
        super().__init__(msg)


class BadRequestError(AppException):
    """客户端请求错误异常"""

    def __init__(self, msg: str = "错误的请求"):
        super().__init__(code=400, status_code=400, msg=msg)


class NotFoundError(AppException):
    """资源未找到异常"""

    def __init__(self, msg: str = "资源未找到"):
        super().__init__(code=404, status_code=404, msg=msg)


class ForbiddenError(AppException):
    """权限不足异常"""

    def __init__(self, msg: str = "无权访问"):
        super().__init__(code=403, status_code=403, msg=msg)


class ConflictError(AppException):
    """资源冲突异常"""

    def __init__(self, msg: str = "资源状态冲突"):
        super().__init__(code=409, status_code=409, msg=msg)


class SandboxDisabledError(AppException):
    """SPM PR-3 Task 28 — ``sandbox_provision_mode == "off"`` 下访问需要沙箱平面的
    端点（file/download/shell/takeover-start/reopen/retry-from-suspend/skills-create）。

    off 部署没有沙箱面（INV-SPM-3 zero-touch），这些操作无法服务。合同（spec §5.6
    / §5.2c-8，r27/codex R27 冻结）：``code=409, status_code=409,
    msg="SANDBOX_DISABLED"``——``AppException.code`` 与统一 ``Response.code`` 协议
    都是 ``int``（字符串 code 会在异常处理器构造响应模型时炸），所以稳定机读哨兵放
    ``msg`` 字段。FE/测试判定 = HTTP 409 且 ``msg == "SANDBOX_DISABLED"``。

    WS（vnc / takeover-shell）不走本异常：它们在 accept 后发自有 payload
    ``{"type":"status","code":"SANDBOX_DISABLED"}`` 再 ``close(code=4409)``。
    """

    def __init__(self, msg: str = "SANDBOX_DISABLED"):
        super().__init__(code=409, status_code=409, msg=msg)


class ValidationError(AppException):
    """数据验证错误异常"""

    def __init__(self, msg: str = "数据验证失败", data: Any = None):
        super().__init__(code=422, status_code=422, msg=msg, data=data)


class TooManyRequestsError(AppException):
    """请求过多异常"""

    def __init__(
        self,
        msg: str = "请求过多，请稍后重试",
        retry_after: int | None = None,
        limit: int | None = None,
        window_seconds: int | None = None,
        bucket: str | None = None,
    ):
        data: dict[str, int | str] = {}
        if retry_after is not None:
            data["retry_after"] = retry_after
        if limit is not None:
            data["limit"] = limit
        if window_seconds is not None:
            data["window_seconds"] = window_seconds
        if bucket is not None:
            data["bucket"] = bucket
        super().__init__(code=429, status_code=429, msg=msg, data=data or None)


class SecurityError(ForbiddenError):
    """安全校验失败（路径穿越 / user_id 不匹配 / sandbox 越权等）。

    继承 ``ForbiddenError`` → HTTP 403。独立命名便于日志审计和 metrics 区分，
    domain / application 层统一抛该异常，interfaces 层由既有 exception handler
    走 ``status_code`` 映射。
    """

    def __init__(self, msg: str = "安全校验失败"):
        super().__init__(msg=msg)


class QuotaExceededError(TooManyRequestsError):
    """Memory / Gate 日配额超限。

    继承 ``TooManyRequestsError`` → HTTP 429。保留 retry_after / limit / bucket
    等字段以供前端展示；``bucket`` 推荐使用 ``"memory_user_daily"`` /
    ``"memory_gate_daily"`` 等值。
    """

    def __init__(
        self,
        msg: str = "配额已用尽",
        retry_after: int | None = None,
        limit: int | None = None,
        window_seconds: int | None = None,
        bucket: str | None = None,
    ):
        super().__init__(
            msg=msg,
            retry_after=retry_after,
            limit=limit,
            window_seconds=window_seconds,
            bucket=bucket,
        )


class ServiceUnavailableError(AppException):
    """服务不可用异常"""

    def __init__(self, msg: str = "服务暂不可用，请稍后重试"):
        super().__init__(code=503, status_code=503, msg=msg)


class ServerRequestsError(AppException):
    """服务器请求错误异常"""

    def __init__(self, msg: str = "服务器请求错误"):
        super().__init__(code=500, status_code=500, msg=msg)


class InternalError(AppException):
    """A7 / rewrite-layer invariant violation — HTTP 500.

    Raised when downstream code receives data that should have been transformed
    upstream (e.g. HTTPS image_url reaching _rewrites.py under a profile with
    accepts_image_url=False). Represents a **developer bug** in the caller, not
    a runtime condition to recover from. Adapter catches and logs .error, then
    re-raises so LangGraph RetryPolicy / upper layers can surface the bug.

    spec §4.3 item 6 / §4.6 "InternalError catch & log" / §9 R10.
    """

    def __init__(self, msg: str = "内部错误") -> None:
        super().__init__(code=500, status_code=500, msg=msg)


class ConfigError(AppException):
    """A7 provider registry — unknown provider id or invalid profile config.

    Raised by ``get_profile`` when a caller asks for a provider that is not
    registered. Indicates either a programming error (hardcoded bad id) or
    a misconfigured ``config.yaml``. HTTP 500 since the system cannot
    service the request until the configuration is corrected.
    """

    def __init__(self, msg: str = "配置错误") -> None:
        super().__init__(code=500, status_code=500, msg=msg)


# ---- B5 prompt assembly errors (HTTP wrappers) -------------------------- #
# The domain layer defines the canonical exceptions in
# ``app.domain.services.prompts.errors``. These wrappers exist only so the
# interfaces layer's HTTP handler can return structured error responses.
# Domain code should raise the domain exceptions directly; the interfaces
# exception handler catches and converts them to HTTP 500.


class PromptAssemblyHTTPError(AppException):
    """HTTP wrapper for any ``PromptAssemblyError`` raised at runtime.

    The interfaces exception handler catches ``PromptAssemblyError`` from
    the domain layer and re-raises as this AppException subclass to surface
    a structured 500 to the client.
    """

    def __init__(self, msg: str):
        super().__init__(code=500, status_code=500, msg=msg)


# ---- C2 coordinator errors (domain-internal signals) -------------------- #
#
# Intentionally NOT subclasses of AppException: these are domain-level
# control-flow signals that the orchestrator catches and routes via
# ``step_result_candidate`` / SSE events. They never reach the
# interfaces exception handler — surfacing them as HTTP errors would
# leak coordinator internals to the client.
#
# A future Phase 2 may promote some of these to user-visible errors with
# proper AppException wrappers (e.g. CoordinatorBudgetExhausted → 429),
# but PR-5 keeps them domain-internal.


class CoordinatorError(Exception):
    """Base for C2 coordinator errors (PR-5+).

    Subclasses signal specific failure modes that the orchestrator
    handles distinctly. Production code should catch the specific
    subclass; ``CoordinatorError`` itself is the umbrella for tests
    and broad except clauses.
    """


class PatchConflict(CoordinatorError):
    """[C2 PR-5 §9.3] Cross-worker same-path write detected by the reducer.

    Raised when two coordinator children's PatchManifests both touch
    the same path — the reducer routes via ``GroupOutcome.CONFLICT``
    instead and the orchestrator surfaces this exception type when an
    application-layer caller (rather than the reducer itself) needs to
    signal the same condition.
    """


class PatchApplyError(CoordinatorError):
    """[C2 PR-5 §10.2] PatchApplier step failure.

    Raised by application-layer callers that wrap the applier and
    need to translate a non-SUCCESS ``ApplyOutcome`` into a control-
    flow exception (e.g. when the orchestrator's outer task needs to
    fail-fast rather than continue with the partial result text).
    """


class CoordinatorBudgetExhausted(CoordinatorError):
    """[C2 PR-6 §14.3] Per-child or per-run budget exhausted.

    Reserved for the PR-6 budget watchdogs (token / wallclock). PR-5
    declares the type so the PR-5 → PR-6 boundary is wire-stable: the
    applier + orchestrator don't yet raise this, but downstream test
    fixtures can import it.
    """
