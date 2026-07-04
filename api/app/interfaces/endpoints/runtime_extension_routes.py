"""B9 运行时扩展路由（interfaces 层）——GET 聚合清单。

`GET /api/v1/runtime/extensions`：把 `RuntimeExtensionService.get_extensions()`
返回的 domain snapshot 平移为 wire schema（Task 5 的 `to_wire_response`）。
本端点**纯读、零探测/网络 I/O**（INV-B9-4）——探测数据来自注入 view 的内存快照。

角色投影：`current_user.is_admin()` 透传给 service，Admin 得全量、非 Admin 得
最小投影（裁剪逻辑在 service 层，端点只做透传 + 序列化）。

Task 15/16 在本 router 追加 `POST .../{kind}/{id}/probe` 与 enabled 端点；
Task 16 追加 catalog GET。
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.application.errors.exceptions import (
    AppException,
    NotFoundError,
    TooManyRequestsError,
)
from app.application.services.app_config_service import AppConfigService
from app.application.services.extension_probe_service import (
    MANUAL_PROBE_COOLDOWN_SECONDS,
    ExtensionProbeService,
    ProbeBusyError,
    ProbeDisabledError,
    ProbeGoneError,
)
from app.application.services.runtime_extension_service import (
    RuntimeExtensionService,
)
from app.application.services.skill_service import SkillService
from app.domain.models.app_config import MCPServerConfig
from app.domain.models.runtime_extension import ExtensionItemInfo
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.interfaces.dependencies.auth import AdminUser, CurrentUser
from app.interfaces.dependencies.rate_limit import rate_limit_write
from app.interfaces.schemas.runtime_extensions import (
    CatalogItem,
    ExtensionItem,
    RuntimeExtensionCatalogResponse,
    RuntimeExtensionsResponse,
    to_wire_item,
    to_wire_response,
)
from app.interfaces.service_dependencies import (
    get_app_config_service,
    get_extension_probe_service,
    get_runtime_extension_service,
)
from core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

router = APIRouter(prefix="/v1/runtime", tags=["运行时扩展"])


# —— 手动探测冷却窗 + 审计（Task 16，R4#10/R16#5 冻结）——
#
# `_PROBE_COOLDOWNS`：进程内 (kind, ext_id) → 上次实际执行探测的事件循环时间戳
# （`asyncio.get_running_loop().time()`，单调）。5s 内重复 → 429（P-10）。仅**实际
# 执行**的探测写时间戳；被 flag/enabled/404/冷却/rate-limit 拒绝的不写、不进审计。
_probe_audit_logger = logging.getLogger("actus.extension_probe")
_PROBE_COOLDOWNS: dict[tuple[str, str], float] = {}


def _reset_probe_cooldowns() -> None:
    """测试专用：清空模块级冷却 map（模块级可变状态跨用例隔离）。"""
    _PROBE_COOLDOWNS.clear()


def _audit_probe(
    admin_user_id: str,
    kind: str,
    ext_id: str,
    *,
    outcome: str,
    latency_ms: int | None,
    error_code: str | None,
) -> None:
    """三 kind 共用（R9#1 后签名改显式字段——skill 分支无 ProbeRecord）：
    mcp/a2a 调用方从 record 取（outcome="success" if record.state=="reachable"
    else "failure"）；skill 调用方从重组装条目的 health 取（outcome="success" if
    state=="ok" else "failure"，latency_ms=health.latency_ms（integrity 无测量则
    None），error_code=health.error_code）。

    ext_id 绝不以原文进 LogRecord：``id_display`` 仅保留可打印字符并截断（防日志
    注入），``id_hash`` 为稳定关联键（SHA1 前 16 位）。
    """
    _probe_audit_logger.info(
        "extension_probe",
        extra={
            "admin_user_id": admin_user_id,
            "kind": kind,
            "id_display": "".join(ch for ch in ext_id if ch.isprintable())[:64],
            "id_hash": hashlib.sha1(ext_id.encode("utf-8")).hexdigest()[:16],
            "outcome": outcome,
            "latency_ms": latency_ms,
            "error_code": error_code,
        },
    )


def _build_skill_service() -> SkillService:
    """对齐 skill_v2_routes.py:49-50 的既有构造惯例（B9 不新发明 DI 通道）。

    独立模块级函数，与 handler 无嵌套关系；测试以 monkeypatch 替换本属性注入 fake。
    """
    return SkillService(FileSkillRepository(settings.skills_root_dir))

# CATALOG_PATH：从本模块（app/interfaces/endpoints/）上溯两级到 app/interfaces/，
# 再定位 data/mcp_catalog.json（Task 8 新增静态资源包）。
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "mcp_catalog.json"


@functools.lru_cache(maxsize=1)
def load_catalog() -> list[CatalogItem]:
    """加载失败降级空列表 + warning（不能因目录文件损坏挂端点，spec §7）。"""
    try:
        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        items = [CatalogItem.model_validate(entry) for entry in raw]
        for it in items:
            MCPServerConfig.model_validate(it.config_template)
            if it.config_template.get("transport") != it.transport:
                raise ValueError(f"catalog[{it.id}] transport 不一致")
        return items
    except Exception:
        logger.warning("mcp_catalog.json 加载失败，降级空目录", exc_info=True)
        return []


@router.get(
    "/extensions",
    response_model=RuntimeExtensionsResponse,
    summary="运行时扩展聚合清单（GET）",
)
async def get_runtime_extensions(
    current_user: CurrentUser,
    service: Annotated[
        RuntimeExtensionService, Depends(get_runtime_extension_service)
    ],
) -> RuntimeExtensionsResponse:
    snap = await service.get_extensions(
        user_id=current_user.id, is_admin=current_user.is_admin()
    )
    return to_wire_response(snap)


# 路由顺序硬约束（spec §6）：catalog 必须先于 /extensions/{kind}/... 注册
# （本 task 时点参数路由未创建；Task 15/16 追加 /{kind}/{id}/probe|enabled 时，
#  静态字面量路由 /extensions/catalog 必须在参数路由之前声明，否则会被
#  /{kind}/{id} 吞掉——FastAPI 按声明顺序匹配）。
@router.get(
    "/extensions/catalog",
    response_model=RuntimeExtensionCatalogResponse,
    summary="MCP 推荐目录（静态，GET）",
)
async def get_extension_catalog(
    current_user: CurrentUser,
) -> RuntimeExtensionCatalogResponse:
    return RuntimeExtensionCatalogResponse(items=load_catalog())


class SetExtensionEnabledRequest(BaseModel):
    enabled: bool


# 路由顺序：本参数路由 /extensions/{kind}/{ext_id}/enabled 声明于
# /extensions/catalog 之后（catalog 静态字面量优先匹配，spec §6）。
@router.post(
    "/extensions/{kind}/{ext_id}/enabled",
    response_model=ExtensionItem,
    summary="运行时扩展统一启停 façade（POST，Admin）",
)
async def set_extension_enabled(
    kind: Literal["mcp", "a2a", "skill"],
    ext_id: str,
    payload: SetExtensionEnabledRequest,
    admin_user: AdminUser,
    request: Request,
    app_config_service: Annotated[AppConfigService, Depends(get_app_config_service)],
    runtime_service: Annotated[
        RuntimeExtensionService, Depends(get_runtime_extension_service)
    ],
    _rate: Annotated[None, Depends(rate_limit_write)],
) -> ExtensionItem:
    """B9 统一启停 façade（INV-B9-7）：委托 F9 既有 service，不新增写路径。

    四段（R3#1，缺一不可）：写委托 → probe 快照 invalidate（fail-open）→
    重组装取更新后条目 → 返回 wire item。
    """
    # 1) 写委托——单一写路径（INV-B9-7），既有 service 对不存在 id 抛
    #    NotFoundError → 经 AppException handler 透传为 404。
    if kind == "mcp":
        await app_config_service.set_mcp_server_enabled(ext_id, payload.enabled)
    elif kind == "a2a":
        await app_config_service.set_a2a_server_enabled(ext_id, payload.enabled)
    else:
        skill_service = _build_skill_service()
        await skill_service.set_skill_enabled(ext_id, payload.enabled)

    # 2) probe 快照 invalidate（fail-open：单例未启动/启动失败时为 None）。
    #    写已提交成功——invalidate 仅是内存快照簿记；即便半坏的 probe 单例抛错，
    #    也绝不能把已成功的 config 写翻成 500（INV-B9-7 fail-open bypass 合同）。
    probe_service = getattr(request.app.state, "extension_probe_service", None)
    if probe_service is not None:
        try:
            probe_service.invalidate(
                kind, ext_id, "enable" if payload.enabled else "disable"
            )
        except Exception:  # noqa: BLE001 - fail-open：invalidate 失败不翻转已成功的写
            logger.warning(
                "probe 快照 invalidate 失败（fail-open，写已成功）", exc_info=True
            )

    # 3) 重组装取更新后条目（Admin 视角全量）。
    snap = await runtime_service.get_extensions(user_id=admin_user.id, is_admin=True)
    item = next(
        (i for i in snap.items if (i.kind, i.id) == (kind, ext_id)), None
    )
    if item is None:
        raise NotFoundError(msg=f"扩展不存在: {kind}/{ext_id}")

    # 4) 返回更新后 wire item。
    return to_wire_item(item)


async def _reassemble_item(
    runtime_service: RuntimeExtensionService,
    *,
    admin_user_id: str,
    kind: str,
    ext_id: str,
) -> ExtensionItemInfo | None:
    """重组装取该 (kind, ext_id) 的最新条目（Admin 视角全量）。"""
    snap = await runtime_service.get_extensions(user_id=admin_user_id, is_admin=True)
    return next(
        (i for i in snap.items if (i.kind, i.id) == (kind, ext_id)), None
    )


# 路由顺序：本参数路由 /extensions/{kind}/{ext_id}/probe 声明于
# /extensions/catalog 之后（catalog 静态字面量优先匹配，spec §6）。
@router.post(
    "/extensions/{kind}/{ext_id}/probe",
    response_model=ExtensionItem,
    summary="运行时扩展手动探测（POST，Admin）",
)
async def probe_extension(
    kind: Literal["mcp", "a2a", "skill"],
    ext_id: str,
    admin_user: AdminUser,
    runtime_service: Annotated[
        RuntimeExtensionService, Depends(get_runtime_extension_service)
    ],
    probe_service: Annotated[
        ExtensionProbeService | None, Depends(get_extension_probe_service)
    ],
    _rate: Annotated[None, Depends(rate_limit_write)],
) -> ExtensionItem:
    """B9 手动探测端点（spec §3.1/§6）。

    端点序列（冻结）：rate limit（Depends）→ flag 检查（有效 probe 能力 provider
    为 False → 409 probe_disabled）→ 目标存在性/enabled 预检（404 / 409
    extension_disabled）→ 冷却窗检查（进程内 (kind,id) map，5s → 429 + retry_after）
    → kind 分派（R9#1）：
    - mcp/a2a → ``probe_one_manual``（预算内化在 service，端点不包 wait_for/shield；
      ``ProbeBusyError`` → 503 probe_busy；``ProbeDisabledError`` → 按 reason 409；
      ``ProbeGoneError`` → 404）
    - skill → 端点层短路，不调 ``probe_one_manual``（skill integrity 无网络无退避；
      存在性预检已在上一步做，此处直接重组装返回——GET 权威=repo 直读，天然等价
      "立即重扫该 skill"）

    审计日志：仅**实际执行**的探测进审计（每个 outcome 都记）；被冷却/rate-limit
    拒绝的不进（冷却拒绝仅 debug 日志）。
    """
    # 1) 重组装权威快照（pre-check + 最终返回同源）。
    snap = await runtime_service.get_extensions(user_id=admin_user.id, is_admin=True)

    # 2) flag 检查：有效 probe 能力关闭 → 409 probe_disabled（P-10）。
    if not snap.probe_enabled:
        raise AppException(
            code=409,
            status_code=409,
            msg="探测未启用",
            data={"reason": "probe_disabled"},
        )

    # 3) 目标存在性/enabled 预检。
    item = next(
        (i for i in snap.items if (i.kind, i.id) == (kind, ext_id)), None
    )
    if item is None:
        raise NotFoundError(msg=f"扩展不存在: {kind}/{ext_id}")
    if not item.config.enabled_global:
        raise AppException(
            code=409,
            status_code=409,
            msg="扩展已禁用",
            data={"reason": "extension_disabled"},
        )

    # 4) 冷却窗检查（进程内 (kind, ext_id) map，事件循环单调时间；5s 内 → 429）。
    #    冷却拒绝仅 debug 日志、不进审计（仅实际执行的探测进审计）。
    now = asyncio.get_running_loop().time()
    key = (kind, ext_id)
    last = _PROBE_COOLDOWNS.get(key)
    if last is not None:
        elapsed = now - last
        if elapsed < MANUAL_PROBE_COOLDOWN_SECONDS:
            retry_after = max(1, math.ceil(MANUAL_PROBE_COOLDOWN_SECONDS - elapsed))
            logger.debug(
                "manual probe 冷却拒绝 kind=%s elapsed=%.2fs", kind, elapsed
            )
            raise TooManyRequestsError(
                msg="探测过于频繁", retry_after=retry_after
            )

    # 5) kind 分派。
    if kind == "skill":
        # skill 短路：不调 probe_one_manual（服务层对 kind="skill" 抛 ValueError 防误用）。
        # 存在性预检已在步骤 3 做；此处直接以重组装条目返回（GET 权威=repo 直读，
        # 语义="立即重扫该 skill"）。审计 outcome 按 health.state。
        _PROBE_COOLDOWNS[key] = now
        outcome = "success" if item.health.state == "ok" else "failure"
        _audit_probe(
            admin_user.id,
            kind,
            ext_id,
            outcome=outcome,
            latency_ms=item.health.latency_ms,
            error_code=item.health.error_code,
        )
        return to_wire_item(item)

    # mcp / a2a：走 service 预算内化探测。probe_service 为 None（fail-open 未启动）
    # 与 flag off 等价——步骤 2 的 snap.probe_enabled 已把该情形拦成 409 probe_disabled。
    if probe_service is None:  # pragma: no cover - snap.probe_enabled 已拦
        raise AppException(
            code=409,
            status_code=409,
            msg="探测未启用",
            data={"reason": "probe_disabled"},
        )

    _PROBE_COOLDOWNS[key] = now
    try:
        record = await probe_service.probe_one_manual(kind, ext_id)
    except ProbeBusyError as exc:
        raise AppException(
            code=503,
            status_code=503,
            msg="探测繁忙，请稍后重试",
            data={"reason": "probe_busy"},
        ) from exc
    except ProbeDisabledError as exc:
        raise AppException(
            code=409,
            status_code=409,
            msg="探测未启用" if exc.reason == "probe_disabled" else "扩展已禁用",
            data={"reason": exc.reason},
        ) from exc
    except ProbeGoneError as exc:
        raise NotFoundError(msg=f"扩展不存在: {kind}/{ext_id}") from exc

    # 审计：outcome 从 record.state（reachable→success，其余→failure）。
    _audit_probe(
        admin_user.id,
        kind,
        ext_id,
        outcome="success" if record.state == "reachable" else "failure",
        latency_ms=record.latency_ms,
        error_code=record.error_code,
    )

    # 重组装取更新后条目返回（probe 已回写 record → 聚合器读到最新 health）。
    updated = await _reassemble_item(
        runtime_service, admin_user_id=admin_user.id, kind=kind, ext_id=ext_id
    )
    if updated is None:
        raise NotFoundError(msg=f"扩展不存在: {kind}/{ext_id}")
    return to_wire_item(updated)
