"""Request/Response schemas for /v2/memories endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

# M1 PR-1 category enum — DB CHECK 兜底，schema 用 Literal 在入口层也拦一道。
MemoryCategory = Literal["user", "rule", "fact"]


class MemoryItem(BaseModel):
    """列表/详情共享的基础字段。"""

    id: str
    content: str
    source: str
    created_at: datetime
    updated_at: datetime
    session_id: str | None = None
    # M1 PR-1 新增字段——列表也需要展示，让前端按 category 分 tab / 显示 pinned badge
    category: str | None = None
    pinned: bool = False
    auto_promoted_at: datetime | None = None


class MemoryDetail(MemoryItem):
    """详情视图：在列表字段之上附加 hash 与 metadata。"""

    content_hash: str
    metadata: dict[str, Any]
    # fs_synced 只在详情里暴露：list 视图无需关心同步状态，详情 / 编辑时有用
    fs_synced: bool = False


class MemoryListResponse(BaseModel):
    items: list[MemoryItem]
    total: int
    page: int
    page_size: int
    has_next: bool


class CreateMemoryRequest(BaseModel):
    """手动创建 memory 的入参。source 固定为 ``manual``，服务端不允许客户端覆写。

    ``pinned=True`` 只有在 ``category='user'`` 时合法；DB 层也有 CHECK 兜底，
    这里提前校验 → 返回 400 而非 500。
    """

    content: str = Field(..., min_length=1, max_length=50000)
    category: MemoryCategory
    pinned: bool = False

    @model_validator(mode="after")
    def _pinned_only_for_user_category(self) -> "CreateMemoryRequest":
        if self.pinned and self.category != "user":
            raise ValueError("pinned=True 仅允许在 category='user' 时使用")
        return self


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
