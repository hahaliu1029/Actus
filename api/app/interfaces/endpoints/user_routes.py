"""用户路由模块 - 用户工具偏好管理"""

import asyncio
import logging

from app.application.services.app_config_service import AppConfigService
from app.application.services.skill_service import SkillService
from app.application.services.user_tool_preference_service import (
    UserToolPreferenceService,
)
from app.domain.models.user_tool_preference import ToolType
from app.infrastructure.repositories.db_skill_repository import DBSkillRepository
from app.infrastructure.repositories.db_user_tool_preference_repository import (
    DBUserToolPreferenceRepository,
)
from app.infrastructure.storage.postgres import get_postgres
from app.interfaces.dependencies import CurrentUser
from app.interfaces.schemas import Response
from app.interfaces.schemas.user import ToolPreferenceRequest, ToolWithPreference
from app.interfaces.service_dependencies import get_app_config_service
from fastapi import APIRouter, Depends

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/user/tools", tags=["用户工具偏好"])


@router.get(
    "/mcp",
    response_model=Response,
    summary="获取 MCP 工具列表（带用户偏好）",
    description="获取所有 MCP 工具列表，包含用户的个人启用状态",
)
async def get_mcp_tools(
    current_user: CurrentUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response:
    """获取 MCP 工具列表"""
    # 1. 先查询用户偏好（短暂持有 DB 连接）
    pref_map: dict[str, bool] = {}
    postgres = get_postgres()
    try:
        async with postgres.session_factory() as session:
            pref_repo = DBUserToolPreferenceRepository(session)
            pref_service = UserToolPreferenceService(pref_repo)

            user_prefs = await pref_service.get_user_preferences(
                current_user.id, ToolType.MCP
            )
            pref_map = {pref.tool_id: pref.enabled for pref in user_prefs}
    except asyncio.CancelledError:
        logger.warning("get_mcp_tools 请求被取消，返回上游取消")
        raise
    except Exception as e:
        logger.exception(
            f"查询用户MCP工具偏好失败，降级为默认启用(user_id={current_user.id}): {e}"
        )

    # 2. 再做 MCP 探测（不持有 DB 连接，避免取消态影响数据库连接）
    mcp_servers = await app_config_service.get_mcp_servers()

    # 3. 组装响应
    tools = []
    for server in mcp_servers:
        tools.append(
            ToolWithPreference(
                tool_id=server.server_name,
                tool_name=server.server_name,
                description=None,  # MCP 服务器可能没有描述字段
                enabled_global=server.enabled,
                enabled_user=pref_map.get(server.server_name, True),  # 默认启用
            )
        )

    return Response.success(data={"tools": tools})


@router.post(
    "/mcp/{server_name}/enabled",
    response_model=Response,
    summary="设置 MCP 工具个人启用状态",
    description="设置用户对某个 MCP 工具的个人启用/禁用状态",
)
async def set_mcp_tool_enabled(
    server_name: str,
    request: ToolPreferenceRequest,
    current_user: CurrentUser,
) -> Response:
    """设置 MCP 工具个人启用状态"""
    postgres = get_postgres()

    async with postgres.session_factory() as session:
        pref_repo = DBUserToolPreferenceRepository(session)
        pref_service = UserToolPreferenceService(pref_repo)

        await pref_service.set_tool_enabled(
            current_user.id,
            ToolType.MCP,
            server_name,
            request.enabled,
        )
        await session.commit()

        return Response.success(msg="设置成功")


@router.get(
    "/a2a",
    response_model=Response,
    summary="获取 A2A 工具列表（带用户偏好）",
    description="获取所有 A2A 工具列表，包含用户的个人启用状态",
)
async def get_a2a_tools(
    current_user: CurrentUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response:
    """获取 A2A 工具列表"""
    # 1. 先查询用户偏好（短暂持有 DB 连接）
    pref_map: dict[str, bool] = {}
    postgres = get_postgres()
    try:
        async with postgres.session_factory() as session:
            pref_repo = DBUserToolPreferenceRepository(session)
            pref_service = UserToolPreferenceService(pref_repo)

            user_prefs = await pref_service.get_user_preferences(
                current_user.id, ToolType.A2A
            )
            pref_map = {pref.tool_id: pref.enabled for pref in user_prefs}
    except asyncio.CancelledError:
        logger.warning("get_a2a_tools 请求被取消，返回上游取消")
        raise
    except Exception as e:
        logger.exception(
            f"查询用户A2A工具偏好失败，降级为默认启用(user_id={current_user.id}): {e}"
        )

    # 2. 再做 A2A 探测（不持有 DB 连接）
    a2a_servers = await app_config_service.get_a2a_servers()

    # 3. 组装响应
    tools = []
    for server in a2a_servers:
        tools.append(
            ToolWithPreference(
                tool_id=server.id,
                tool_name=server.name,
                description=server.description,
                enabled_global=server.enabled,
                enabled_user=pref_map.get(server.id, True),  # 默认启用
            )
        )

    return Response.success(data={"tools": tools})


@router.post(
    "/a2a/{a2a_id}/enabled",
    response_model=Response,
    summary="设置 A2A 工具个人启用状态",
    description="设置用户对某个 A2A 工具的个人启用/禁用状态",
)
async def set_a2a_tool_enabled(
    a2a_id: str,
    request: ToolPreferenceRequest,
    current_user: CurrentUser,
) -> Response:
    """设置 A2A 工具个人启用状态"""
    postgres = get_postgres()

    async with postgres.session_factory() as session:
        pref_repo = DBUserToolPreferenceRepository(session)
        pref_service = UserToolPreferenceService(pref_repo)

        await pref_service.set_tool_enabled(
            current_user.id,
            ToolType.A2A,
            a2a_id,
            request.enabled,
        )
        await session.commit()

        return Response.success(msg="设置成功")


@router.get(
    "/skills",
    response_model=Response,
    summary="获取 Skill 工具列表（带用户偏好）",
    description="获取所有 Skill 列表，包含用户的个人启用状态",
)
async def get_skill_tools(
    current_user: CurrentUser,
) -> Response:
    """获取 Skill 工具列表"""
    pref_map: dict[str, bool] = {}
    tools: list[ToolWithPreference] = []
    postgres = get_postgres()

    try:
        async with postgres.session_factory() as session:
            pref_service = UserToolPreferenceService(
                DBUserToolPreferenceRepository(session)
            )
            skill_service = SkillService(DBSkillRepository(session))

            user_prefs = await pref_service.get_user_preferences(
                current_user.id, ToolType.SKILL
            )
            pref_map = {pref.tool_id: pref.enabled for pref in user_prefs}

            skills = await skill_service.list_skills()
            for skill in skills:
                tools.append(
                    ToolWithPreference(
                        tool_id=skill.id,
                        tool_name=skill.name,
                        description=skill.description,
                        enabled_global=skill.enabled,
                        enabled_user=pref_map.get(skill.id, True),
                    )
                )
    except asyncio.CancelledError:
        logger.warning("get_skill_tools 请求被取消，返回上游取消")
        raise
    except Exception as e:
        logger.exception(
            f"查询用户Skill工具偏好失败，降级为空列表(user_id={current_user.id}): {e}"
        )

    return Response.success(data={"tools": tools})


@router.post(
    "/skills/{skill_id}/enabled",
    response_model=Response,
    summary="设置 Skill 工具个人启用状态",
    description="设置用户对某个 Skill 工具的个人启用/禁用状态",
)
async def set_skill_tool_enabled(
    skill_id: str,
    request: ToolPreferenceRequest,
    current_user: CurrentUser,
) -> Response:
    """设置 Skill 工具个人启用状态"""
    postgres = get_postgres()

    async with postgres.session_factory() as session:
        pref_repo = DBUserToolPreferenceRepository(session)
        pref_service = UserToolPreferenceService(pref_repo)

        await pref_service.set_tool_enabled(
            current_user.id,
            ToolType.SKILL,
            skill_id,
            request.enabled,
        )
        await session.commit()

        return Response.success(msg="设置成功")
