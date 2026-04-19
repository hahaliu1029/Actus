"""/v2/memories —— 用户级长期记忆管理 API。"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query
from pydantic import AwareDatetime

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.interfaces.dependencies import (
    CurrentUser,
    rate_limit_read,
    rate_limit_write,
)
from app.interfaces.schemas import Response
from app.interfaces.schemas.memory_schemas import (
    BulkDeleteRequest,
    CreateMemoryRequest,
    DeleteCountResponse,
    LegacyCleanupConfigResponse,
    MemoryCategory,
    MemoryDetail,
    MemoryItem,
    MemoryListResponse,
    ReindexResponse,
    UpdateMemoryRequest,
)
from app.interfaces.service_dependencies import get_memory_management_service
from core.config import get_settings

if TYPE_CHECKING:
    from app.application.services.memory_management_service import (
        MemoryManagementService,
    )
    from app.domain.models.memory_chunk import MemoryChunk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/memories", tags=["记忆管理"])


@router.get(
    path="",
    response_model=Response[MemoryListResponse],
    summary="获取用户长期记忆列表",
    description="分页 + 搜索 + 过滤。page_size 上限 50，query 最短 2 字符。",
    dependencies=[Depends(rate_limit_read)],
)
async def list_memories(
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
    query: str | None = Query(None, min_length=2, max_length=500),
    source: str | None = Query(None, max_length=64),
    category: MemoryCategory | None = Query(
        None,
        description="按 memory 分类过滤：user / rule / fact；不传返回全部（含 legacy NULL）。非法值返回 422（与 POST 路径对齐）",
    ),
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    updated_from: datetime | None = None,
    updated_to: datetime | None = None,
    # AwareDatetime: 拒绝 naive datetime 输入，避免不同时区部署节点对同一
    # cutoff 字符串解释不一致（codex round-3 [P2]）。客户端必须传 ISO 8601
    # 带时区，例如 ``2026-04-12T00:00:00Z`` 或 ``2026-04-12T08:00:00+08:00``。
    # 已知遗留：created_from / updated_to 等同类参数仍是 naive-tolerant
    # ``datetime``，独立 cleanup（不属于本 PR scope）。
    auto_promoted_after: AwareDatetime | None = Query(
        None,
        description=(
            "审阅最近 N 天 LLM gate 自动收录的 memory（design doc §777 入口）："
            "只返回 ``auto_promoted_at >= auto_promoted_after`` 的行。manual / "
            "memory_save 入口的行 ``auto_promoted_at IS NULL``，自动排除。"
            "通常配合 ``source=session_flush`` 使用。**必须 timezone-aware** "
            "（例如 ``2026-04-12T00:00:00Z``），naive datetime 返回 422。"
        ),
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
) -> Response[MemoryListResponse]:
    items, total = await service.list_memories(
        current_user.id,
        query=query,
        source=source,
        category=category,
        created_from=created_from,
        created_to=created_to,
        updated_from=updated_from,
        updated_to=updated_to,
        auto_promoted_after=auto_promoted_after,
        page=page,
        page_size=page_size,
    )
    return Response.success(
        data=MemoryListResponse(
            items=[MemoryItem(**_to_item_dict(c)) for c in items],
            total=total,
            page=page,
            page_size=page_size,
            has_next=(page * page_size < total),
        )
    )


@router.post(
    path="",
    response_model=Response[MemoryDetail],
    summary="手动创建长期记忆",
    description=(
        "Manual 写入入口。source 强制为 'manual'；pinned=True 仅允许 category='user'。"
        "受每日 user quota 限制，超限返回 429。"
    ),
    status_code=201,
    dependencies=[Depends(rate_limit_write)],
)
async def create_memory(
    body: CreateMemoryRequest,
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[MemoryDetail]:
    try:
        chunk = await service.create_memory(
            current_user.id,
            content=body.content,
            category=body.category,
            pinned=body.pinned,
            source="manual",
            tags=body.tags,
        )
    except ValueError as exc:
        # service 对空内容 / 非法分类抛 ValueError → 400
        raise BadRequestError(str(exc)) from exc
    # ConflictError / QuotaExceededError 由全局 exception handler 自动转 409 / 429
    return Response.success(data=MemoryDetail(**_to_detail_dict(chunk)))


@router.get(
    path="/cleanup-config",
    response_model=Response[LegacyCleanupConfigResponse],
    summary="获取 legacy 清理的时间边界配置",
    description=(
        "返回当前 deployment 的 ``memory_gate_rollout_at`` 配置。前端在显示"
        "\"清理旧记忆\"对话框前调用，根据返回值展示具体 cutoff 或警告语。"
        "注册在 ``/{chunk_id}`` 之前避免被 chunk_id 路径吞掉。"
    ),
    dependencies=[Depends(rate_limit_read)],
)
async def get_legacy_cleanup_config(
    current_user: CurrentUser,
) -> Response[LegacyCleanupConfigResponse]:
    # current_user 只用于 AuthN——配置本身是 deployment 级，不含 PII。
    # 读 settings 单例即可，不需要 service；避免为一个 scalar 起整条 DI 链。
    _ = current_user
    settings = get_settings()
    return Response.success(
        data=LegacyCleanupConfigResponse(
            rollout_at=settings.memory_gate_rollout_at,
        )
    )


@router.delete(
    path="/legacy",
    response_model=Response[DeleteCountResponse],
    summary="一键清理旧 session_flush 遗留记忆",
    description=(
        "清除未分类也未被 gate 收录的 session_flush 遗留："
        "``source='session_flush' AND category IS NULL AND auto_promoted_at IS NULL``。"
        "三个条件 AND 合取——**任何一个非 NULL / 不匹配的行都不会被删**："
        "新 flush 路径（已分类 或 已 auto-promoted）、manual / memory_save 入口都不受影响。\n\n"
        "**时间边界（codex fix P1）：** 若 settings 设 ``memory_gate_rollout_at``，"
        "SQL 额外 ``AND created_at < rollout_at``，仅清掉上线前的遗留；未设时"
        "沿用旧谓词（清除所有未分类 session_flush），由前端 dialog 显式警告。\n\n"
        "低频运维操作；返回实际删除的行数，空时返回 0（非 404）。"
    ),
    dependencies=[Depends(rate_limit_write)],
)
async def delete_legacy(
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[DeleteCountResponse]:
    # 必须注册在 ``/{chunk_id}`` 之前——FastAPI 按注册顺序匹配，
    # 否则 'legacy' 会被当成 chunk_id 参数走 delete_memory 路径。
    count = await service.delete_legacy_memories(current_user.id)
    return Response.success(data=DeleteCountResponse(deleted_count=count))


@router.get(
    path="/{chunk_id}",
    response_model=Response[MemoryDetail],
    summary="获取单条长期记忆详情",
    dependencies=[Depends(rate_limit_read)],
)
async def get_memory(
    chunk_id: str,
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[MemoryDetail]:
    chunk = await service.get_memory(current_user.id, chunk_id)
    if chunk is None:
        raise NotFoundError("记忆不存在")
    return Response.success(data=MemoryDetail(**_to_detail_dict(chunk)))


@router.patch(
    path="/{chunk_id}",
    response_model=Response[MemoryDetail],
    summary="编辑长期记忆（content 或 pinned 互斥二选一）",
    description=(
        "PATCH 支持 content 或 pinned 二选一（Schema 互斥 validator 强制）。"
        "一次请求只改一个字段——combo 改动请分两次调用，避免 partial failure。"
        "\n\n"
        "- ``content``: 非空字符串，走 update_memory_content（重算 embedding + "
        "fs_synced → false 等 reconciler 回写文件）\n"
        "- ``pinned``: bool，走 update_memory_pinned 单 SQL UPDATE，**保留"
        "现有 fs_synced**（避免吞 pre-existing pending backlog）；当前 PATCH"
        "**不回写文件**——文件 frontmatter 的 pinned 可能长期漂移，只在未来"
        "显式 rewrite/rebuild 路径下才可能带上新 pinned。``pinned=True`` 需"
        "``category='user'``（DB CHECK + service 验证），否则 400。"
        "**幂等语义**：对已是目标状态的行重复调用返 200 但不写 audit / "
        "不刷 updated_at。"
    ),
    dependencies=[Depends(rate_limit_write)],
)
async def update_memory(
    chunk_id: str,
    body: UpdateMemoryRequest,
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[MemoryDetail]:
    try:
        if body.content is not None:
            # content 路径：走既有 update_memory_content，fs_synced=False
            # 过渡等 reconciler 回写盘。
            updated = await service.update_memory_content(
                current_user.id, chunk_id, body.content
            )
        else:
            # pinned 路径：schema validator 保证 body.pinned is not None
            assert body.pinned is not None
            updated = await service.update_memory_pinned(
                current_user.id, chunk_id, body.pinned
            )
    except ValueError as exc:
        # service 对空内容 / 非法输入抛 ValueError，转为 400
        raise BadRequestError(str(exc)) from exc
    # ConflictError 由全局异常处理器自动转为 409
    if updated is None:
        raise NotFoundError("记忆不存在")
    return Response.success(data=MemoryDetail(**_to_detail_dict(updated)))


@router.delete(
    path="/{chunk_id}",
    response_model=Response[DeleteCountResponse],
    summary="删除单条长期记忆",
    dependencies=[Depends(rate_limit_write)],
)
async def delete_memory(
    chunk_id: str,
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[DeleteCountResponse]:
    deleted = await service.delete_memory(current_user.id, chunk_id)
    if not deleted:
        raise NotFoundError("记忆不存在")
    return Response.success(data=DeleteCountResponse(deleted_count=1))


@router.post(
    path="/{chunk_id}/reindex",
    response_model=Response[ReindexResponse],
    summary="从磁盘重建索引（hand-edit 闭环）",
    description=(
        "Power-user 工作流：用户 hand-edit "
        "``${MEMORY_ROOT_HOST}/{user_id}/{category}/{id}.md`` 的 body "
        "后调此 endpoint，服务端读盘 → 重算 embedding → UPDATE DB，"
        "``memory_search`` / ``memory_recall`` 立即能查到新内容，不必重启"
        " session 或跑 CLI reconciler。\n\n"
        "**Option A 权威契约**：\n"
        "- ``body`` → apply 到 DB\n"
        "- ``id`` → mismatch 直接 409（不允许 hand-edit 改 id）\n"
        "- ``source`` / ``created_at`` / ``auto_promoted_at`` → 系统字段，"
        "改动列入 warnings 忽略\n"
        "- ``title`` / ``category`` / ``pinned`` / ``tags`` → **file-only**："
        "改动留在文件层（File LIVE view 可见），但**不**进 DB / search / "
        "prompt；当前无受支持的同步路径\n\n"
        "**错误状态**: 404 chunk 不存在 / 409 file 不存在或 id 不匹配 / "
        "400 frontmatter parse 失败或空 body / 403 path 穿越或 symlink / "
        "503 deployment 未配置 file_store。"
    ),
    dependencies=[Depends(rate_limit_write)],
)
async def reindex_memory(
    chunk_id: str,
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[ReindexResponse]:
    result = await service.reindex_memory(current_user.id, chunk_id)
    return Response.success(
        data=ReindexResponse(
            reindexed_fields=result.reindexed_fields,
            warnings=result.warnings,
            fs_synced=result.fs_synced,
        )
    )


@router.post(
    path="/bulk-delete",
    response_model=Response[DeleteCountResponse],
    summary="批量删除长期记忆",
    dependencies=[Depends(rate_limit_write)],
)
async def bulk_delete(
    body: BulkDeleteRequest,
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[DeleteCountResponse]:
    count = await service.bulk_delete_memories(current_user.id, body.ids)
    return Response.success(data=DeleteCountResponse(deleted_count=count))


@router.post(
    path="/delete-all",
    response_model=Response[DeleteCountResponse],
    summary="清空当前用户全部长期记忆",
    dependencies=[Depends(rate_limit_write)],
)
async def delete_all(
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[DeleteCountResponse]:
    count = await service.delete_all_memories(current_user.id)
    return Response.success(data=DeleteCountResponse(deleted_count=count))


def _to_item_dict(chunk: "MemoryChunk") -> dict:
    return {
        "id": chunk.id,
        "content": chunk.content,
        "source": chunk.source,
        "created_at": chunk.created_at,
        "updated_at": chunk.updated_at,
        "session_id": chunk.session_id,
        # M1 PR-1 扩展字段——列表用来显示 badge / filter
        "category": chunk.category,
        "pinned": chunk.pinned,
        "auto_promoted_at": chunk.auto_promoted_at,
    }


def _to_detail_dict(chunk: "MemoryChunk") -> dict:
    d = _to_item_dict(chunk)
    d["content_hash"] = chunk.content_hash
    d["metadata"] = chunk.metadata
    d["fs_synced"] = chunk.fs_synced
    return d
