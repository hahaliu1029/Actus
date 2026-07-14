import logging

from fastapi import Request

from app.core.config import get_settings
from app.interfaces.service_dependencies import get_supervisor_service

logger = logging.getLogger(__name__)


async def auto_extend_timeout_middleware(request: Request, call_next):
    """普通API活动把sandbox cleanup lease重置为默认窗口。"""
    # 1.获取系统配置与supervisor服务
    settings = get_settings()
    supervisor_service = get_supervisor_service()

    # 2.普通API活动只重置默认cleanup window；控制端点自身不二次续租
    ignore_paths = (
        "/api/supervisor/activate-timeout",
        "/api/supervisor/extend-timeout",
        "/api/supervisor/reset-timeout",
        "/api/supervisor/cancel-timeout",
        "/api/supervisor/timeout-status",
    )
    if (
        settings.server_timeout_minutes is not None
        and supervisor_service.timeout_active
        and request.url.path.startswith("/api/")
        and not request.url.path.startswith(ignore_paths)
        and supervisor_service.expand_enabled
    ):
        try:
            await supervisor_service.reset_timeout()
            logger.debug("调用API请求而重置超时销毁时长: %s", request.url.path)
        except Exception as e:
            logger.warning("自动重置超时失败: %s", str(e))

    response = await call_next(request)
    return response
