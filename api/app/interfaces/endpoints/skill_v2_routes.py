"""Skill v2 路由（文件系统权威源）"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from app.application.errors.exceptions import SandboxDisabledError
from app.application.services.app_config_service import AppConfigService
from app.application.services.skill_service import SkillService
from app.application.services.user_tool_enablement_service import UserToolEnablementService
from app.domain.models.app_config import SkillRiskPolicy
from app.domain.models.skill_creator import SkillCreationResult
from app.domain.models.user_tool_enablement import ToolType
from app.infrastructure.repositories.db_user_tool_enablement_repository import (
    DBUserToolEnablementRepository,
)
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.storage.postgres import get_db_session
from app.interfaces.dependencies import AdminUser, CurrentUser
from app.interfaces.schemas import Response
from app.application.services.skill_export_service import SkillExportService
from app.interfaces.schemas.skill import (
    BundleFileItem,
    SkillDetailResponse,
    SkillExportFormat,
    SkillInstallRequest,
    SkillItem,
    SkillListResponse,
    SkillRiskPolicyItem,
    SkillToolItem,
)
from app.interfaces.service_dependencies import (
    get_app_config_service,
    get_skill_creator_service,
    get_skill_export_service,
)
from core.config import get_settings
from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel
from sse_starlette import EventSourceResponse, ServerSentEvent
from sqlalchemy.ext.asyncio import AsyncSession

settings = get_settings()
router = APIRouter(prefix="/v2/skills", tags=["Skill生态v2"])
SSE_HEADERS = {"X-Accel-Buffering": "no"}


def _build_skill_service(request: Request | None = None) -> SkillService:
    """构造 SkillService；request 在场时注入 D1a 治理 ports（off=None 零行为变化）。

    治理触发路由（install/delete）透传 request 让 ports 生效；只读路由（detail/enabled）
    不透传 → ports 默认 None → off-safe byte-identical。
    """
    write_port = getattr(request.app.state, "extension_registry_write_port", None) if request else None
    read_port = getattr(request.app.state, "extension_registry_read_port", None) if request else None
    return SkillService(
        FileSkillRepository(settings.skills_root_dir),
        registry_write_port=write_port,
        registry_read_port=read_port,
    )


class SkillCreateRequest(BaseModel):
    description: str


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
    response_model=Response[dict],
    summary="安装 Skill（v2）",
)
async def install_skill(
    request: SkillInstallRequest,
    admin_user: AdminUser,
    http_request: Request,
    force: bool = Query(False, description="强制安装 dangerous skill"),
) -> Response[dict]:
    service = _build_skill_service(http_request)
    skill = await service.install_skill(
        source_type=request.source_type,
        source_ref=request.source_ref,
        manifest=request.manifest,
        skill_md=request.skill_md,
        installed_by=admin_user.id,
        trust_origin="user_installed",
        force=force,
        actor_id=admin_user.id,   # D1a §6.1-3：standalone 安装 actor 透传治理 hook
    )

    scan_report = skill.scan_report or {}
    from app.domain.services.trust_matrix import compute_base_floor, compute_final_risk
    _base = compute_base_floor(skill.runtime_type, skill.trust_origin)
    _manifest_risk = (
        (skill.manifest or {}).get("policy", {}).get("risk_level")
        if isinstance(skill.manifest, dict) else None
    )
    _final = compute_final_risk(
        _base, scan_report.get("verdict"), _manifest_risk
    )

    return Response.success(data={
        "installed": True,
        "skill_id": skill.id,
        "verdict": scan_report.get("verdict", "safe"),
        "findings": scan_report.get("findings", [])[:20],
        "forced": force and scan_report.get("verdict") == "dangerous",
        "final_risk": _final.name.lower(),
        "trust_origin": skill.trust_origin,
    })


@router.post(
    path="/create",
    summary="AI 创建 Skill（SSE）",
)
async def create_skill_ai(
    request: SkillCreateRequest,
    admin_user: AdminUser,
    creator_service=Depends(get_skill_creator_service),
) -> EventSourceResponse:
    # SPM PR-3 Task 28 (spec §5.6 matrix): AI skill creation runs generated code
    # in a sandbox — off has no sandbox plane. Raise 409 SANDBOX_DISABLED here,
    # BEFORE the EventSourceResponse is constructed (raising inside the generator
    # would surface as a mid-stream 200 SSE error, not a 409). INV-SPM-7.
    if get_settings().sandbox_provision_mode == "off":
        raise SandboxDisabledError()

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        try:
            async for event in creator_service.create(
                description=request.description,
                sandbox=None,
                installed_by=admin_user.id,
            ):
                if isinstance(event, SkillCreationResult):
                    yield ServerSentEvent(
                        event="complete",
                        data=event.model_dump_json(),
                    )
                else:
                    yield ServerSentEvent(
                        event="progress",
                        data=event.model_dump_json(),
                    )
        except Exception as exc:
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"error": str(exc)}, ensure_ascii=False),
            )

    return EventSourceResponse(event_generator(), headers=SSE_HEADERS)


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
    http_request: Request,
    db_session: AsyncSession = Depends(get_db_session),
) -> Response[dict | None]:
    skill_service = _build_skill_service(http_request)
    pref_service = UserToolEnablementService(DBUserToolEnablementRepository(db_session))

    await skill_service.delete_skill(skill_key, actor_id=admin_user.id)
    await pref_service.delete_enablements_by_tool(ToolType.SKILL, skill_key)
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


@router.get(
    path="/{skill_key}/export",
    summary="导出 Skill（v2）",
)
async def export_skill(
    skill_key: str,
    admin_user: AdminUser,
    format: SkillExportFormat = Query(..., description="导出格式"),
    skill_export_service: SkillExportService = Depends(get_skill_export_service),
) -> FastAPIResponse:
    zip_bytes, filename = await skill_export_service.export_skill(skill_key, format)
    return FastAPIResponse(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    path="/{skill_key}",
    response_model=Response[SkillDetailResponse],
    summary="获取 Skill 详情（v2）",
)
async def get_skill_detail(
    skill_key: str,
    admin_user: AdminUser,
) -> Response[SkillDetailResponse]:
    service = _build_skill_service()
    skill = await service.get_skill(skill_key)

    manifest = skill.manifest or {}
    raw_tools = manifest.get("tools") or []
    tools = [
        SkillToolItem(
            name=t.get("name", ""),
            description=t.get("description", ""),
            parameters=t.get("parameters") or {},
            required=t.get("required") or [],
            entry=t.get("entry"),
        )
        for t in raw_tools
        if isinstance(t, dict)
    ]

    raw_bundle_files: list = []
    from pathlib import Path as _Path

    bundle_index_path = (
        _Path(settings.skills_root_dir) / skill_key / "bundle_index.json"
    )
    if bundle_index_path.exists():
        try:
            raw_bundle_files = json.loads(
                bundle_index_path.read_text(encoding="utf-8")
            )
        except Exception:
            raw_bundle_files = []

    bundle_files = [
        BundleFileItem(
            path=bf.get("path", ""),
            size=bf.get("size", 0),
            sha256=bf.get("sha256", ""),
            is_text=bf.get("is_text", False),
        )
        for bf in raw_bundle_files
        if isinstance(bf, dict)
    ]

    return Response.success(
        data=SkillDetailResponse(
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
            bundle_file_count=int(manifest.get("bundle_file_count") or 0),
            context_ref_count=int(manifest.get("context_ref_count") or 0),
            last_sync_at=manifest.get("last_sync_at"),
            tools=tools,
            skill_md=str(manifest.get("skill_md") or ""),
            bundle_files=bundle_files,
            activation=manifest.get("activation") or {},
            policy=manifest.get("policy") or {},
            security=manifest.get("security") or {},
        )
    )
