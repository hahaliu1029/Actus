"""D1a §9.2 治理路由族（行政动作 + 观测刷新 + 审计翻页）——全 AdminUser。

off 门（``mode=off`` → provider 返回 None）：除 ``GET /governance`` 返回字面量零外，
一律 409 ``governance_disabled``（零 registry 读，R4-11）。异常→HTTP 映射由项目
``exception_handlers`` 统一注册（T19 已落全量 8 条治理映射），本路由只 raise domain
异常不自映射。``kind`` path 参数手动校验 ∈ ``GovernedExtensionKind`` 四值（不 Literal，
从 ``get_args`` 派生避免镜像第二词表），非法 → 422。

plugin 三条路由（install / delete / enabled）不在本文件——留 Task 24（§9.2 尾三条）。
"""
from __future__ import annotations

from typing import Any, get_args

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.domain.external.extension_admission import AuditPage
from app.domain.models.extension_governance import GovernedExtensionKind
from app.interfaces.dependencies.auth import AdminUser
from app.interfaces.schemas.extension_governance import (
    ApprovePinsBody,
    GovernanceCASBody,
    GovernanceQuarantineBody,
    RefreshBatchBody,
)
from app.interfaces.service_dependencies import get_extension_governance_service

router = APIRouter(prefix="/v2/extensions", tags=["扩展治理"])

# 词表单权威：从 domain Literal 派生四值集合，不镜像第二词表（INV-D1-1）
_VALID_KINDS = frozenset(get_args(GovernedExtensionKind))

_OFF_SUMMARY = {
    "mode": "off",
    "unpinned_count": 0,
    "missing_observation_count": 0,
    "quarantined_count": 0,
}


def _governance_disabled() -> JSONResponse:
    """off 门：治理关闭时非 GET /governance 端点统一 409（形状对齐 _governance_error 家族）。"""
    return JSONResponse(
        status_code=409,
        content={"code": "governance_disabled",
                 "detail": "extension governance is disabled (mode=off)"})


def _validate_kind(kind: str) -> None:
    if kind not in _VALID_KINDS:
        raise HTTPException(status_code=422, detail=f"invalid extension kind: {kind}")


def _item_dict(outcome: Any) -> dict[str, Any]:
    return {
        "kind": outcome.kind,
        "ext_id": outcome.ext_id,
        "outcome": outcome.outcome,
        "row_revision": outcome.row_revision,
    }


def _serialize_audit(page: AuditPage) -> dict[str, Any]:
    return {
        "entries": [
            {
                "id": str(e.id),
                "kind": e.kind,
                "ext_id": e.ext_id,
                "actor_user_id": e.actor_user_id,
                "event": e.event,
                "before": e.before,
                "after": e.after,
                "details": e.details,
                "correlation_id": str(e.correlation_id) if e.correlation_id else None,
                "created_at": e.created_at.isoformat(),
            }
            for e in page.entries
        ],
        "next_cursor": page.next_cursor,
    }


# ---- 静态字面量路由先声明（FastAPI 按声明顺序匹配）------------------------


@router.get("/governance", summary="治理摘要（GET，Admin）")
async def get_governance(
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
) -> dict[str, Any]:
    """off → 字面量零（零 registry 读，R4-11）；on → service.summary()。"""
    if service is None:
        return dict(_OFF_SUMMARY)
    return await service.summary()


@router.post("/refresh-observations", summary="批量观测刷新（POST，Admin）")
async def refresh_observations(
    body: RefreshBatchBody,
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
):
    if service is None:
        return _governance_disabled()
    outcomes = await service.refresh_observations_batch(all=body.all, items=body.items)
    return {"items": [_item_dict(o) for o in outcomes]}


@router.post("/approve-pins", summary="批量 pin 转正（POST，Admin）")
async def approve_pins(
    body: ApprovePinsBody,
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
):
    if service is None:
        return _governance_disabled()
    outcomes = await service.approve_pins(
        all=body.all, items=body.items, actor_id=admin_user.id)
    return {"items": [_item_dict(o) for o in outcomes]}


@router.get("/audit", summary="治理审计翻页（GET，Admin）")
async def list_audit(
    admin_user: AdminUser,
    kind: str | None = Query(default=None),
    ext_id: str | None = Query(default=None),
    event: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    service=Depends(get_extension_governance_service),
):
    if service is None:
        return _governance_disabled()
    page = await service.list_audit(
        kind=kind, ext_id=ext_id, event=event, cursor=cursor, limit=limit)
    return _serialize_audit(page)


# ---- 参数化路由（/{kind}/{ext_id}/...；kind 手动校验四值）------------------


@router.post("/{kind}/{ext_id}/refresh-observation", summary="单项观测刷新（POST，Admin）")
async def refresh_observation(
    kind: str,
    ext_id: str,
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
):
    if service is None:
        return _governance_disabled()
    _validate_kind(kind)
    result = await service.refresh_observation(kind, ext_id)
    return {
        "outcome": result.outcome,
        "row_revision": result.row_revision,
        "surface_summary": result.surface_summary,
    }


@router.post("/{kind}/{ext_id}/quarantine", summary="隔离（POST，Admin）")
async def quarantine(
    kind: str,
    ext_id: str,
    body: GovernanceQuarantineBody,
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
):
    if service is None:
        return _governance_disabled()
    _validate_kind(kind)
    new_rev = await service.quarantine(
        kind, ext_id, expected_row_revision=body.expected_row_revision,
        actor_id=admin_user.id, note=body.note)
    return {"row_revision": new_rev}


@router.post("/{kind}/{ext_id}/reapprove", summary="解除隔离并重新 pin（POST，Admin）")
async def reapprove(
    kind: str,
    ext_id: str,
    body: GovernanceCASBody,
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
):
    if service is None:
        return _governance_disabled()
    _validate_kind(kind)
    new_rev = await service.reapprove(
        kind, ext_id, expected_row_revision=body.expected_row_revision,
        actor_id=admin_user.id)
    return {"row_revision": new_rev}


@router.post("/{kind}/{ext_id}/governance-disable", summary="治理停用（POST，Admin）")
async def governance_disable(
    kind: str,
    ext_id: str,
    body: GovernanceCASBody,
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
):
    if service is None:
        return _governance_disabled()
    _validate_kind(kind)
    new_rev = await service.set_enabled(
        kind, ext_id, enabled=False,
        expected_row_revision=body.expected_row_revision, actor_id=admin_user.id)
    return {"row_revision": new_rev}


@router.post("/{kind}/{ext_id}/governance-enable", summary="治理启用（POST，Admin）")
async def governance_enable(
    kind: str,
    ext_id: str,
    body: GovernanceCASBody,
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
):
    if service is None:
        return _governance_disabled()
    _validate_kind(kind)
    new_rev = await service.set_enabled(
        kind, ext_id, enabled=True,
        expected_row_revision=body.expected_row_revision, actor_id=admin_user.id)
    return {"row_revision": new_rev}
