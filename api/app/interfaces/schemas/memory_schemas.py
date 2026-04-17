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

    ``tags`` 可选（设计 L71）：落到 ``metadata.tags`` + frontmatter ``tags:`` 字段。
    服务端侧做 strip + 去空 + 去重，但**不**强制小写化——尊重用户写 "Go" / "TypeScript"
    这类大小写敏感的标签。上限 20 条 × 每条 64 字符，避免一行超长 tag 撑爆 YAML。
    """

    content: str = Field(..., min_length=1, max_length=50000)
    category: MemoryCategory
    pinned: bool = False
    # 外层 max_length=20 是 tags 数量上限（DoS 保护）；单 tag 的长度校验放到
    # _normalize_tags 里做——item-level min_length 会在 "" 空字符串上直接抛 422，
    # 和我们"宽容清洗前端传来的空白 / 空串"的目标冲突；放 normalizer 里就可以先
    # strip + 丢空 再校验剩余 tag 的 64 字符上限。
    tags: list[str] | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def _pinned_only_for_user_category(self) -> "CreateMemoryRequest":
        if self.pinned and self.category != "user":
            raise ValueError("pinned=True 仅允许在 category='user' 时使用")
        return self

    @model_validator(mode="after")
    def _normalize_tags(self) -> "CreateMemoryRequest":
        """strip + 丢空 + 去重（保序、大小写敏感）+ 每 tag ≤64 字符。
        None / 空列表 / 全空白 都归一为 None，避免 service 收到 [] 和 None
        两种"无 tags" 表达。剩余的 tag 如果超过 64 字符直接 422（用户错误可见）。
        """
        if self.tags is None:
            return self
        seen: set[str] = set()
        cleaned: list[str] = []
        for raw in self.tags:
            t = raw.strip()
            if not t or t in seen:
                continue
            if len(t) > 64:
                raise ValueError(
                    f"单个 tag 最长 64 字符（当前 {len(t)}）：{t[:20]}…"
                )
            seen.add(t)
            cleaned.append(t)
        self.tags = cleaned if cleaned else None
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
