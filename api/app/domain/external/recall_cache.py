"""B8: session 级召回缓存端口。

typing.Protocol（结构化）——infrastructure 提供 ``RedisRecallCache``；
flow 侧 ``recall_cache=None`` 表示「不缓存、每次直检」（可用态而非
阻断依赖，spec R4#3）。DTO 引用方向：external → models（对齐
memory_flusher / policy_snapshot_sink 的 port 方向，spec R6 P3-1）。
"""
from __future__ import annotations

from typing import Protocol

from app.domain.models.memory_recall import RecallCachePayload


class RecallCache(Protocol):
    async def get(self, session_id: str, query_hash: str) -> RecallCachePayload | None: ...

    async def set(self, session_id: str, query_hash: str, payload: RecallCachePayload) -> None: ...
