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


class ServiceUnavailableError(AppException):
    """服务不可用异常"""

    def __init__(self, msg: str = "服务暂不可用，请稍后重试"):
        super().__init__(code=503, status_code=503, msg=msg)


class ServerRequestsError(AppException):
    """服务器请求错误异常"""

    def __init__(self, msg: str = "服务器请求错误"):
        super().__init__(code=500, status_code=500, msg=msg)
