"""Skill v2 路由（文件系统权威源）"""

from __future__ import annotations

from app.application.services.app_config_service import AppConfigService
from app.application.services.skill_service import SkillService
from app.application.services.user_tool_preference_service import UserToolPreferenceService
from app.domain.models.app_config import SkillRiskPolicy
from app.domain.models.user_tool_preference import ToolType
from app.infrastructure.repositories.db_user_tool_preference_repository import (
    DBUserToolPreferenceRepository,
)
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.storage.postgres import get_db_session
from app.interfaces.dependencies import AdminUser, CurrentUser
from app.interfaces.schemas import Response
from app.interfaces.schemas.skill import (
    SkillInstallRequest,
    SkillItem,
    SkillListResponse,
    SkillRiskPolicyItem,
)
from app.interfaces.service_dependencies import get_app_config_service
from core.config import get_settings
from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

settings = get_settings()
router = APIRouter(prefix="/v2/skills", tags=["Skill生态v2"])


def _build_skill_service() -> SkillService:
    return SkillService(FileSkillRepository(settings.skills_root_dir))


@router.get(
    path="",
    response_model=Response[SkillListResponse],
    summary="获取 Skill 列表（v2）",
)
async def list_skills(admin_user: AdminUser) -> Response[SkillListResponse]:
    service = _build_skill_service()
    skills = await service.list_skills()
    return Response.success(
        data=SkillListResponse(
            skills=[
                SkillItem(
                    id=skill.id,
                    slug=skill.slug,
                    name=skill.name,
                    description=skill.description,
                    version=skill.version,
                    source_type=skill.source_type,
                    source_ref=skill.source_ref,
                    runtime_type=skill.runtime_type,
                    enabled=skill.enabled,
                    installed_by=skill.installed_by,
                    created_at=skill.created_at.isoformat(),
                    updated_at=skill.updated_at.isoformat(),
                    bundle_file_count=int(
                        (skill.manifest or {}).get("bundle_file_count") or 0
                    ),
                    context_ref_count=int(
                        (skill.manifest or {}).get("context_ref_count") or 0
                    ),
                    last_sync_at=((skill.manifest or {}).get("last_sync_at") or None),
                )
                for skill in skills
            ]
        )
    )


@router.post(
    path="/install",
    response_model=Response[dict | None],
    summary="安装 Skill（v2）",
)
async def install_skill(
    request: SkillInstallRequest,
    admin_user: AdminUser,
) -> Response[dict | None]:
    service = _build_skill_service()
    await service.install_skill(
        source_type=request.source_type,
        source_ref=request.source_ref,
        manifest=request.manifest,
        skill_md=request.skill_md,
        installed_by=admin_user.id,
    )
    return Response.success(msg="Skill 安装成功")


@router.post(
    path="/{skill_key}/enabled",
    response_model=Response[dict | None],
    summary="更新 Skill 全局启用状态（v2）",
)
async def set_skill_enabled(
    skill_key: str,
    admin_user: AdminUser,
    enabled: bool = Body(..., embed=True),
) -> Response[dict | None]:
    service = _build_skill_service()
    await service.set_skill_enabled(skill_key, enabled)
    return Response.success(msg="Skill 状态更新成功")


@router.delete(
    path="/{skill_key}",
    response_model=Response[dict | None],
    summary="删除 Skill（v2）",
)
async def delete_skill(
    skill_key: str,
    admin_user: AdminUser,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response[dict | None]:
    skill_service = _build_skill_service()
    pref_service = UserToolPreferenceService(DBUserToolPreferenceRepository(db_session))

    await skill_service.delete_skill(skill_key)
    await pref_service.delete_tool_preferences(ToolType.SKILL, skill_key)
    await db_session.commit()
    return Response.success(msg="Skill 删除成功")


@router.get(
    path="/policy",
    response_model=Response[SkillRiskPolicyItem],
    summary="获取 Skill 风险策略（v2）",
)
async def get_skill_policy(
    current_user: CurrentUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[SkillRiskPolicyItem]:
    policy = await app_config_service.get_skill_risk_policy()
    return Response.success(data=SkillRiskPolicyItem(mode=policy.mode.value))


@router.post(
    path="/policy",
    response_model=Response[SkillRiskPolicyItem],
    summary="更新 Skill 风险策略（v2）",
)
async def update_skill_policy(
    request: SkillRiskPolicyItem,
    admin_user: AdminUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[SkillRiskPolicyItem]:
    policy = await app_config_service.update_skill_risk_policy(
        SkillRiskPolicy(mode=request.mode)
    )
    return Response.success(
        msg="Skill 风险策略已更新",
        data=SkillRiskPolicyItem(mode=policy.mode.value),
    )
