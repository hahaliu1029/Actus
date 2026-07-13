"""D1a §9.2 Plugin 路由族（§9.2 尾三条 + install + list）——全 AdminUser（PR-6 收口）。

四条路由：
- ``POST /v2/plugins/install``——``PluginInstallService.install``（T21-T23）。``dry_run`` →
  ``PluginInstallPreview`` 零写 200；否则 ``InstallResult`` 三态映射（R4#3）：
  completed→200（PluginDetail 摘要）/ compensated→422 ``plugin_install_failed_compensated``
  （body 含 collided_targets）/ failed→500 ``plugin_install_failed_requires_admin``。
- ``DELETE /v2/plugins/{plugin_ext_id}``——``PluginInstallService.uninstall``（body
  ``{expected_row_revision}`` 必填，缺→422）。
- ``POST /v2/plugins/{plugin_ext_id}/enabled``——**复用 T20 ``ExtensionGovernanceService
  .set_enabled``**（enabled=true/false 同一迁移服务 + 前置：非终态 operation → 409
  operation_pending，防专用端点绕过，R47#1）。
- ``GET /v2/plugins``——``PluginSagaStore.list_plugin_details``（membership/operations
  声明读者，PluginDetail[]）。

off 门：三 provider（install service / governance service / saga store）``mode=off`` 均
返回 None → 四条一律 409 ``governance_disabled``（无 GET 字面量豁免——与 §9.2 治理 GET
不同，plugin 列表无零态语义）。install preflight 的 plain-Exception 拒绝
（``UnsupportedManifestVersionError`` / ``BundleContainmentError`` / pydantic
``ValidationError``——非 ``AppException``，全局 handler 会落 500）由本路由就地翻 422
（``UnsupportedManifestVersionError`` docstring 显式指派 interfaces/T24 映射；作用域限
plugin install 路，不加全局 handler blast radius）；``AppException`` 族
（``ValidationError`` / ``ConflictError`` / ``Acknowledge/ForceRequiredError`` /
``OperationPendingError`` 等）照旧走既有全局映射（422/409）。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.application.services.plugin_install_service import (
    BundleContainmentError,
    InstallResult,
    PluginInstallPreview,
)
from app.domain.models.plugin_manifest import UnsupportedManifestVersionError
from app.domain.models.skill import SkillSourceType
from app.interfaces.dependencies.auth import AdminUser
from app.interfaces.schemas.extension_governance import (
    PluginEnabledBody,
    PluginInstallRequest,
    PluginUninstallBody,
)
from app.interfaces.service_dependencies import get_extension_governance_service

router = APIRouter(prefix="/v2/plugins", tags=["扩展治理"])

# install preflight 的 plain-Exception 拒绝族（非 AppException——就地翻 422 拒装）
_INSTALL_REJECT_422 = (
    UnsupportedManifestVersionError,
    BundleContainmentError,
    PydanticValidationError,
)


# ---- provider（app.state 句柄穿透；off → None，与 T19/T20 provider 同款）------------
def get_plugin_install_service(request: Request):
    """T24：Plugin 安装管道单例句柄穿透（``mode≠off`` → ``PluginInstallService``；off → None）。

    句柄由 lifespan 治理段挂 ``app.state.plugin_install_service``。off → None → install/delete
    409 ``governance_disabled``。挂 app.state 本身不满足 ``Depends``——本 provider 是承载面。"""
    return getattr(request.app.state, "plugin_install_service", None)


def get_plugin_saga_store(request: Request):
    """T24：Plugin saga store 句柄穿透（GET /v2/plugins list 读者；off → None → 409）。"""
    return getattr(request.app.state, "plugin_saga_store", None)


# ---- helpers --------------------------------------------------------------
def _governance_disabled() -> JSONResponse:
    """off 门：plugin 路由治理关闭统一 409（形状对齐 _governance_error 家族）。"""
    return JSONResponse(
        status_code=409,
        content={"code": "governance_disabled",
                 "detail": "extension governance is disabled (mode=off)"})


def _parse_source_type(source_type: str) -> SkillSourceType:
    try:
        return SkillSourceType(source_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid source_type: {source_type}") from exc


def _install_result_response(result: InstallResult) -> JSONResponse:
    """§8.3 安装终态 API 合同（R4#3——补偿/失败不得伪装 2xx）。"""
    if result.status == "completed":
        return JSONResponse(status_code=200, content={
            "plugin_ext_id": result.plugin_ext_id,
            "operation_id": str(result.operation_id),
            "status": "completed"})
    if result.status == "compensated":
        return JSONResponse(status_code=422, content={
            "code": "plugin_install_failed_compensated",
            "operation_id": str(result.operation_id),
            "error": result.error,
            "collided_targets": result.collided_targets})
    # failed：补偿自身失败/config 不可读——父行阻断态，需 Admin 处置
    return JSONResponse(status_code=500, content={
        "code": "plugin_install_failed_requires_admin",
        "operation_id": str(result.operation_id)})


def _serialize_plugin_detail(row: Any) -> dict[str, Any]:
    """``PluginDetailRow`` → PluginDetail wire 形状（last_operation + members 逐字投影）。"""
    op = row.last_operation
    return {
        "ext_id": row.ext_id,
        "name": row.name,
        "version": row.version,
        "status": row.status,
        "artifact_hash": row.artifact_hash,
        "row_revision": row.row_revision,
        "last_operation": (
            {"type": op.type, "state": op.state,
             "error": op.error, "updated_at": op.updated_at}
            if op is not None else None),
        "members": [
            {"declared_component_id": m.declared_component_id, "kind": m.kind,
             "ext_id": m.ext_id, "expected_hash": m.expected_hash,
             "installed_version": m.installed_version,
             "managed_by_plugin": m.managed_by_plugin, "status": m.status,
             "scan_verdict": m.scan_verdict, "scan_report": m.scan_report}
            for m in row.members],
    }


# ---- 静态字面量路由先声明（FastAPI 按声明顺序匹配）--------------------------


@router.get("", summary="Plugin 列表（GET，Admin）")
async def list_plugins(
    admin_user: AdminUser,
    store=Depends(get_plugin_saga_store),
):
    """membership/operations 声明读者——PluginDetail[]（off → 409）。"""
    if store is None:
        return _governance_disabled()
    rows = await store.list_plugin_details()
    return [_serialize_plugin_detail(r) for r in rows]


@router.post("/install", summary="Plugin 安装（POST，Admin）")
async def install_plugin(
    body: PluginInstallRequest,
    admin_user: AdminUser,
    service=Depends(get_plugin_install_service),
):
    """dry_run → 200 preview（零写）；否则 InstallResult 三态映射（R4#3）。"""
    if service is None:
        return _governance_disabled()
    source_type = _parse_source_type(body.source_type)
    try:
        result = await service.install(
            source_type, body.source_ref, actor_id=admin_user.id,
            dry_run=body.dry_run, force=body.force, acknowledge=body.acknowledge)
    except _INSTALL_REJECT_422 as exc:
        # plain-Exception preflight 拒绝（非 AppException）→ 就地翻 422（否则全局 500）
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(result, PluginInstallPreview):
        return result.model_dump(mode="json")
    return _install_result_response(result)


# ---- 参数化路由（/{plugin_ext_id}/...；{ext_id}=业务 id，R1#16）--------------


@router.post("/{plugin_ext_id}/enabled", summary="Plugin 启停（POST，Admin）")
async def set_plugin_enabled(
    plugin_ext_id: str,
    body: PluginEnabledBody,
    admin_user: AdminUser,
    service=Depends(get_extension_governance_service),
):
    """**复用 T20 set_enabled** 同一迁移服务 + 前置（R47#1：非终态 operation → 409
    operation_pending，防专用端点绕过）。off → 409（治理服务 None）。"""
    if service is None:
        return _governance_disabled()
    new_rev = await service.set_enabled(
        "plugin", plugin_ext_id, enabled=body.enabled,
        expected_row_revision=body.expected_row_revision, actor_id=admin_user.id)
    return {"row_revision": new_rev}


@router.delete("/{plugin_ext_id}", summary="Plugin 卸载（DELETE，Admin）")
async def uninstall_plugin(
    plugin_ext_id: str,
    body: PluginUninstallBody,
    admin_user: AdminUser,
    service=Depends(get_plugin_install_service),
):
    """§8.3-6 卸载 saga（expected_row_revision 必填——缺 → 422 由 pydantic 保证）。off → 409。"""
    if service is None:
        return _governance_disabled()
    await service.uninstall(
        plugin_ext_id, actor_id=admin_user.id,
        expected_row_revision=body.expected_row_revision)
    return {"plugin_ext_id": plugin_ext_id, "status": "uninstalled"}
