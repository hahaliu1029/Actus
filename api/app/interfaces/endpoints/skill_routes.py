"""Skill 旧接口（已迁移到 v2）。"""

from __future__ import annotations

from typing import Optional

from app.interfaces.dependencies import AdminUser
from app.interfaces.schemas import Response
from app.interfaces.schemas.skill import SkillInstallRequest
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/app-config/skills", tags=["Skill生态"])


def _moved_response(path: str) -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content=Response.fail(
            code=410,
            msg="SKILL_API_MOVED",
            data={"code": "SKILL_API_MOVED", "migrate_to": path},
        ).model_dump(),
    )


@router.get(
    path="",
    response_model=Response,
    summary="获取 Skill 列表（已迁移）",
)
async def list_skills(admin_user: AdminUser) -> Response:
    return _moved_response("/v2/skills")


@router.post(
    path="/install",
    response_model=Response[Optional[dict]],
    summary="安装 Skill（已迁移）",
)
async def install_skill(request: SkillInstallRequest, admin_user: AdminUser) -> Response:
    return _moved_response("/v2/skills/install")


@router.post(
    path="/{skill_id}/enabled",
    response_model=Response[Optional[dict]],
    summary="更新 Skill 全局启用状态（已迁移）",
)
async def set_skill_enabled(
    skill_id: str,
    admin_user: AdminUser,
    enabled: bool = Body(..., embed=True),
) -> Response:
    return _moved_response(f"/v2/skills/{skill_id}/enabled")


@router.post(
    path="/{skill_id}/delete",
    response_model=Response[Optional[dict]],
    summary="删除 Skill（已迁移）",
)
async def delete_skill(skill_id: str, admin_user: AdminUser) -> Response:
    return _moved_response(f"/v2/skills/{skill_id}")
