"""B8 query-time 记忆召回 —— DTO dataclasses。

只放数据形状（stdlib + dataclasses）。纯 query 构建函数在
``app/domain/services/memory_recall.py``；缓存端口
``app/domain/external/recall_cache.py`` 引用本模块的 DTO——port 引用
方向对齐 memory_flusher / policy_snapshot_sink 先例（spec R6 P3-1）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RecallQueryMaterial:
    """recall provider 构建 query 的原始素材。

    ``entry`` 必填无默认（spec R9#5）：两个 planner 入口必须显式传
    "graph" / "detection"，防 telemetry 全落一个桶。``session_title``
    由调用方恒传 None，provider 内 best-effort 预取后经
    ``dataclasses.replace`` 填充。
    """

    message: str
    original_request: str | None
    session_title: str | None
    entry: str  # "graph" | "detection"


@dataclass(frozen=True)
class RecalledMemoryItem:
    chunk_id: str
    category: str | None  # user | rule | fact | None(legacy)
    content: str  # provider 阶段已折行 + 截断 ≤ RECALL_ITEM_RENDER_CAP
    created_at: datetime
    score: float  # decayed relevance（relevance × temporal decay；MMR 不改分值）


@dataclass(frozen=True)
class RecalledMemory:
    """runtime 视图，section 消费；不直接序列化进缓存（那是
    ``RecallCachePayload`` 的职责）。"""

    items: tuple[RecalledMemoryItem, ...]
    query_hash: str
    cache_hit: bool
    recall_id: str  # 每次 provider 调用新 uuid hex（cache 命中也新生成）；不进缓存、不进 prompt


@dataclass(frozen=True)
class RecallCachePayload:
    """缓存序列化契约：无 cache_hit、无 recall_id（spec R1#2/R4#4）。"""

    items: tuple[RecalledMemoryItem, ...]
    candidate_count: int  # SQL 候选数；cache hit 时 telemetry 取此值
