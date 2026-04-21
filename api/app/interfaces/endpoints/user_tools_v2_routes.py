"""用户工具偏好 v2 路由。"""

from __future__ import annotations

from app.application.services.skill_service import SkillService
from app.application.services.user_tool_enablement_service import UserToolEnablementService
from app.domain.models.user_tool_enablement import ToolType
from app.infrastructure.repositories.db_user_tool_enablement_repository import (
    DBUserToolEnablementRepository,
)
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.storage.postgres import get_db_session
from app.interfaces.dependencies import CurrentUser
from app.interfaces.schemas import Response
from app.interfaces.schemas.user import ToolPreferenceRequest, ToolWithPreference
from core.config import get_settings
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

settings = get_settings()
router = APIRouter(prefix="/v2/user/tools", tags=["用户工具偏好v2"])


@router.get(
    "/skills",
    response_model=Response,
    summary="获取 Skill 工具列表（v2）",
)
async def get_skill_tools(
    current_user: CurrentUser,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response:
    pref_service = UserToolEnablementService(DBUserToolEnablementRepository(db_session))
    skill_service = SkillService(FileSkillRepository(settings.skills_root_dir))

    user_prefs = await pref_service.list_user_enablements(current_user.id, ToolType.SKILL)
    pref_map = {pref.tool_id: pref.enabled for pref in user_prefs}
    skills = await skill_service.list_skills()

    tools = [
        ToolWithPreference(
            tool_id=skill.id,
            tool_name=skill.name,
            description=skill.description,
            enabled_global=skill.enabled,
            enabled_user=pref_map.get(skill.id, True),
        )
        for skill in skills
    ]

    return Response.success(data={"tools": tools})


@router.post(
    "/skills/{skill_key}/enabled",
    response_model=Response,
    summary="设置 Skill 工具个人启用状态（v2）",
)
async def set_skill_tool_enabled(
    skill_key: str,
    request: ToolPreferenceRequest,
    current_user: CurrentUser,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response:
    pref_repo = DBUserToolEnablementRepository(db_session)
    pref_service = UserToolEnablementService(pref_repo)
    await pref_service.set_tool_enabled(
        current_user.id,
        ToolType.SKILL,
        skill_key,
        request.enabled,
    )
    await db_session.commit()
    return Response.success(msg="设置成功")
