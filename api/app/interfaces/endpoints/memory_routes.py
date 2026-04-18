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
    MemoryCategory,
    MemoryDetail,
    MemoryItem,
    MemoryListResponse,
    UpdateMemoryRequest,
)
from app.interfaces.service_dependencies import get_memory_management_service

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
    summary="编辑长期记忆内容",
    dependencies=[Depends(rate_limit_write)],
)
async def update_memory(
    chunk_id: str,
    body: UpdateMemoryRequest,
    current_user: CurrentUser,
    service: "MemoryManagementService" = Depends(get_memory_management_service),
) -> Response[MemoryDetail]:
    try:
        updated = await service.update_memory_content(
            current_user.id, chunk_id, body.content
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
