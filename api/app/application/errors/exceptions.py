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
