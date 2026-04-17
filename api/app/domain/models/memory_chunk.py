from __future__ import annotations

import hashlib
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

    M1 PR-1 扩展字段（见 migration m2_add_memory_category_and_audit）：
    - ``category``：Optional[str]，``user | rule | fact``；历史行（M1 前）
      为 None。新写入路径必须显式赋值；DB CHECK constraint 拦住越界枚举。
    - ``auto_promoted_at``：仅 session_flush + gate-approved 的 memory
      有值；其余 None。UI 用它区分"用户手动写"和"LLM gate 自动晋升"。
    - ``fs_synced``：文件落盘状态。DB 写完 → false，FsMemoryWriter 回写
      成功 → true。FsReconciler 启动扫 false 行补写。
    - ``pinned``：仅 category='user' 时允许 true；DB CHECK 拦住其他类。
      M2 user_profile section 依赖它永远保留在 prompt 里。
    """

    id: str
    user_id: str
    content: str
    content_hash: str
    source: str  # "session_flush" | "manual" | "memory_save"
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    session_id: str | None = None
    embedding: tuple[float, ...] | None = None
    category: str | None = None
    auto_promoted_at: datetime | None = None
    fs_synced: bool = False
    pinned: bool = False


def memory_content_hash(content: str) -> str:
    """memory_chunks 表的 content_hash 统一计算函数。

    使用全量 SHA256，与 planner_react.py 写入路径保持一致。
    新增 memory_chunks.content_hash 读写路径时请复用此函数，避免算法漂移。
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
