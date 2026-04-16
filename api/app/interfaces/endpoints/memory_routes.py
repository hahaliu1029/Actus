"""/v2/memories —— 用户级长期记忆管理 API。"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.interfaces.dependencies import (
    CurrentUser,
    rate_limit_read,
    rate_limit_write,
)
from app.interfaces.schemas import Response
from app.interfaces.schemas.memory_schemas import (
    BulkDeleteRequest,
    DeleteCountResponse,
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
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    updated_from: datetime | None = None,
    updated_to: datetime | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
) -> Response[MemoryListResponse]:
    items, total = await service.list_memories(
        current_user.id,
        query=query,
        source=source,
        created_from=created_from,
        created_to=created_to,
        updated_from=updated_from,
        updated_to=updated_to,
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
    }


def _to_detail_dict(chunk: "MemoryChunk") -> dict:
    d = _to_item_dict(chunk)
    d["content_hash"] = chunk.content_hash
    d["metadata"] = chunk.metadata
    return d
