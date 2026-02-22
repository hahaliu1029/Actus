from fastapi import APIRouter

from . import (
    admin_routes,
    app_config_routes,
    auth_routes,
    file_routes,
    session_routes,
    status_routes,
    user_routes,
)


def create_api_routes() -> APIRouter:
    """创建API路由，涵盖整个项目的所有路由管理"""

    api_router = APIRouter()

    # 认证相关路由 (无需认证)
    api_router.include_router(auth_routes.router)

    # 业务路由 (需要认证)
    api_router.include_router(status_routes.router)
    api_router.include_router(app_config_routes.router)
    api_router.include_router(file_routes.router)

    # 用户路由
    api_router.include_router(user_routes.router)

    # 管理员路由
    api_router.include_router(admin_routes.router)

    api_router.include_router(session_routes.router)

    return api_router


router = create_api_routes()
