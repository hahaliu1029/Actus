from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RawChunk:
    """对话分块，不含 embedding。由 flow 产出，传递给 flush service。"""

    content: str
    session_id: str
    user_id: str
    source: str
    metadata: dict[str, Any]
    content_hash: str


@dataclass(frozen=True)
class FlushBatch:
    """一次 flush 提交的完整上下文。"""

    session_id: str
    user_id: str
    from_cursor: int
    target_cursor: int
    chunks: tuple[RawChunk, ...]


@dataclass(frozen=True)
class MemoryChunk:
    """已持久化的记忆分块——Repository 的读取返回类型。

    与 RawChunk（flush 管道输入，无 id/embedding/timestamps）形成
    生命周期分界：RawChunk 是"写入前"，MemoryChunk 是"读取后"。
    """

    id: str
    user_id: str
    content: str
    content_hash: str
    source: str  # "session_flush" | "manual" | "file"
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    session_id: str | None = None
    embedding: tuple[float, ...] | None = None
