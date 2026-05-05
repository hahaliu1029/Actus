from fastapi import APIRouter

from . import (
    admin_routes,
    app_config_routes,
    auth_routes,
    cost_routes,
    file_routes,
    memory_routes,
    metrics_routes,
    notification_routes,
    session_compaction_routes,
    skill_routes,
    skill_v2_routes,
    session_routes,
    status_routes,
    user_routes,
    user_tool_policies_routes,
    user_tools_v2_routes,
)


def create_api_routes() -> APIRouter:
    """创建API路由，涵盖整个项目的所有路由管理"""

    api_router = APIRouter()

    # 认证相关路由 (无需认证)
    api_router.include_router(auth_routes.router)

    # 业务路由 (需要认证)
    api_router.include_router(status_routes.router)
    api_router.include_router(app_config_routes.router)
    api_router.include_router(skill_routes.router)
    api_router.include_router(skill_v2_routes.router)
    api_router.include_router(file_routes.router)

    # 用户路由
    api_router.include_router(user_routes.router)
    api_router.include_router(user_tools_v2_routes.router)
    api_router.include_router(user_tool_policies_routes.router)
    api_router.include_router(memory_routes.router)
    api_router.include_router(notification_routes.router)

    # 管理员路由
    api_router.include_router(admin_routes.router)

    api_router.include_router(session_routes.router)
    api_router.include_router(session_compaction_routes.router)
    api_router.include_router(cost_routes.router)

    # B5 PR-S3-3: Prometheus scrape (内部使用)。无 auth dependency
    # —— 鉴权在 handler 内做 constant-time bearer compare，禁用时整体
    # 404，对 OpenAPI 隐藏。
    api_router.include_router(metrics_routes.router)

    return api_router


router = create_api_routes()
