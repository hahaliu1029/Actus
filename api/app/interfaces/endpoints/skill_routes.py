"""Skill 生态管理路由"""

from typing import Optional

from app.application.services.skill_service import SkillService
from app.application.services.user_tool_preference_service import UserToolPreferenceService
from app.domain.models.user_tool_preference import ToolType
from app.infrastructure.repositories.db_skill_repository import DBSkillRepository
from app.infrastructure.repositories.db_user_tool_preference_repository import (
    DBUserToolPreferenceRepository,
)
from app.infrastructure.storage.postgres import get_db_session
from app.interfaces.dependencies import AdminUser
from app.interfaces.schemas import Response
from app.interfaces.schemas.skill import (
    SkillDiscoveryItem,
    SkillDiscoveryResponse,
    SkillInstallRequest,
    SkillItem,
    SkillListResponse,
)
from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/app-config/skills", tags=["Skill生态"])


@router.get(
    path="",
    response_model=Response[SkillListResponse],
    summary="获取 Skill 列表",
    description="获取系统中已安装的 Skill 列表（仅管理员）",
)
async def list_skills(
    admin_user: AdminUser,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response[SkillListResponse]:
    service = SkillService(DBSkillRepository(db_session))
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
                )
                for skill in skills
            ]
        )
    )


@router.post(
    path="/install",
    response_model=Response[Optional[dict]],
    summary="安装 Skill",
    description="根据来源标识和 Manifest 安装 Skill（仅管理员）",
)
async def install_skill(
    request: SkillInstallRequest,
    admin_user: AdminUser,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response[Optional[dict]]:
    service = SkillService(DBSkillRepository(db_session))
    await service.install_skill(
        source_type=request.source_type,
        source_ref=request.source_ref,
        manifest=request.manifest,
        skill_md=request.skill_md,
        installed_by=admin_user.id,
    )
    await db_session.commit()
    return Response.success(msg="Skill 安装成功")


@router.post(
    path="/{skill_id}/enabled",
    response_model=Response[Optional[dict]],
    summary="更新 Skill 全局启用状态",
    description="根据 Skill id 更新全局启用状态（仅管理员）",
)
async def set_skill_enabled(
    skill_id: str,
    admin_user: AdminUser,
    enabled: bool = Body(..., embed=True),
    db_session: AsyncSession = Depends(get_db_session),
) -> Response[Optional[dict]]:
    service = SkillService(DBSkillRepository(db_session))
    await service.set_skill_enabled(skill_id, enabled)
    await db_session.commit()
    return Response.success(msg="Skill 状态更新成功")


@router.post(
    path="/{skill_id}/delete",
    response_model=Response[Optional[dict]],
    summary="删除 Skill",
    description="根据 Skill id 删除指定 Skill（仅管理员）",
)
async def delete_skill(
    skill_id: str,
    admin_user: AdminUser,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response[Optional[dict]]:
    skill_service = SkillService(DBSkillRepository(db_session))
    pref_service = UserToolPreferenceService(DBUserToolPreferenceRepository(db_session))

    await skill_service.delete_skill(skill_id)
    await pref_service.delete_tool_preferences(ToolType.SKILL, skill_id)
    await db_session.commit()

    return Response.success(msg="Skill 删除成功")


@router.get(
    path="/discovery/mcp",
    response_model=Response[SkillDiscoveryResponse],
    summary="发现 MCP Skill",
    description="发现可安装的 MCP Skill（仅管理员）",
)
async def discover_mcp_skills(
    admin_user: AdminUser,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response[SkillDiscoveryResponse]:
    service = SkillService(DBSkillRepository(db_session))
    skills = await service.discover_mcp_skills()
    return Response.success(
        data=SkillDiscoveryResponse(
            skills=[
                SkillDiscoveryItem(
                    source_type=item.source_type,
                    source_ref=item.source_ref,
                    name=item.name,
                    description=item.description,
                    runtime_type=item.runtime_type,
                )
                for item in skills
            ]
        )
    )


@router.get(
    path="/discovery/github",
    response_model=Response[SkillDiscoveryResponse],
    summary="发现 GitHub Skill",
    description="发现可安装的 GitHub Skill（仅管理员）",
)
async def discover_github_skills(
    admin_user: AdminUser,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response[SkillDiscoveryResponse]:
    service = SkillService(DBSkillRepository(db_session))
    skills = await service.discover_github_skills()
    return Response.success(
        data=SkillDiscoveryResponse(
            skills=[
                SkillDiscoveryItem(
                    source_type=item.source_type,
                    source_ref=item.source_ref,
                    name=item.name,
                    description=item.description,
                    runtime_type=item.runtime_type,
                )
                for item in skills
            ]
        )
    )
