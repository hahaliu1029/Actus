"""Request/Response schemas for /v2/memories endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """列表/详情共享的基础字段。"""

    id: str
    content: str
    source: str
    created_at: datetime
    updated_at: datetime
    session_id: str | None = None


class MemoryDetail(MemoryItem):
    """详情视图：在列表字段之上附加 hash 与 metadata。"""

    content_hash: str
    metadata: dict[str, Any]


class MemoryListResponse(BaseModel):
    items: list[MemoryItem]
    total: int
    page: int
    page_size: int
    has_next: bool


class UpdateMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=50000)


class BulkDeleteRequest(BaseModel):
    # 元素级 max_length 对齐 memory_chunks.id 的 String(255) 上限，
    # 避免提交数 MB 的超长字符串打爆 IN (...) 查询内存
    ids: list[Annotated[str, Field(max_length=255)]] = Field(
        ..., min_length=1, max_length=500
    )


class DeleteCountResponse(BaseModel):
    deleted_count: int
