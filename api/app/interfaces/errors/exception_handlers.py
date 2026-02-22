import logging

from app.application.errors.exceptions import AppException, TooManyRequestsError
from app.interfaces.schemas import Response
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """处理Actus项目中所有的异常并进行统一处理，涵盖：自定义业务状态异常、HTTP异常、通用异常"""

    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request, exc: AppException
    ) -> JSONResponse:
        """自定义应用异常处理器，捕获AppException并返回标准化响应"""

        logger.error(f"App exception: {exc.msg}")

        headers: dict[str, str] = {}
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
        """HTTP异常处理器，捕获HTTPException并返回标准化响应"""

        logger.error(f"HTTP exception: {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content=Response(code=exc.status_code, msg=exc.detail, data={}).model_dump(),
        )

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """通用异常处理器，捕获所有未处理的异常并返回标准化响应, 状态码500"""
        # 这里可以添加日志记录等操作
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content=Response(code=500, msg="Internal Server Error", data={}).model_dump(),
        )
