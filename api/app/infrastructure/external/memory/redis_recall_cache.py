"""B8: Redis 召回缓存（逐字蓝本 = redis_embedding_cache）。

key = ``mem_recall:{session_id}:{query_hash}``；value = 单 JSON blob
``{"payload_version": 1, "candidate_count": int, "items": [...]}``。

明文暴露注记（spec §5.2 R3#6）：payload 含记忆内容**明文**（已截断
render cap）——与 RedisEmbeddingCache 只存 vector 的敏感度不同。前提 =
私有 Redis（compose 默认内网）；共享/暴露部署必须启用 Redis
auth/ACL/TLS，否则应保持 ``recall_mode: off``。TTL + 截断是缓解非加密。

陈旧性语义：session 内 memory 编辑/删除后缓存短期 stale——与 M1 prompt
snapshot 的 session-scoped stale 三视图语义一致，接受。

datetime 契约（spec R3#2）：``created_at.isoformat()`` 序列化 /
``datetime.fromisoformat()`` 反序列化（DB 列 tz-aware，roundtrip 保时刻）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from app.domain.models.memory_recall import RecallCachePayload, RecalledMemoryItem

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

PAYLOAD_VERSION = 1
"""JSON schema 结构版本——只管缓存 blob 的形状演进；检索参数的失效由
query_hash 内的 params_version 负责（两者职责不重叠，spec R1#6）。"""


class RedisRecallCache:
    """结构化满足 ``RecallCache`` Protocol（不继承）。"""

    PREFIX = "mem_recall:"

    def __init__(self, redis_client: "Redis", ttl: int) -> None:
        # ttl 无默认值：config（recall_cache_ttl_seconds）是唯一消费源，
        # runner 组装时显式传入（spec R10#1）。
        self._redis = redis_client
        self._ttl = ttl

    def _key(self, session_id: str, query_hash: str) -> str:
        return f"{self.PREFIX}{session_id}:{query_hash}"

    async def get(self, session_id: str, query_hash: str) -> RecallCachePayload | None:
        try:
            raw = await self._redis.get(self._key(session_id, query_hash))
            if raw is None:
                return None
            data = json.loads(raw)
            if data.get("payload_version") != PAYLOAD_VERSION:
                return None
            items = tuple(
                RecalledMemoryItem(
                    chunk_id=entry["chunk_id"],
                    category=entry.get("category"),
                    content=entry["content"],
                    created_at=datetime.fromisoformat(entry["created_at"]),
                    score=float(entry["score"]),
                )
                for entry in data["items"]
            )
            return RecallCachePayload(
                items=items, candidate_count=int(data["candidate_count"]),
            )
        except Exception:
            logger.debug(
                "recall cache get failed for session=%s (treated as miss)",
                session_id, exc_info=True,
            )
            return None

    async def set(
        self, session_id: str, query_hash: str, payload: RecallCachePayload,
    ) -> None:
        try:
            data = {
                "payload_version": PAYLOAD_VERSION,
                "candidate_count": payload.candidate_count,
                "items": [
                    {
                        "chunk_id": item.chunk_id,
                        "category": item.category,
                        "content": item.content,
                        "created_at": item.created_at.isoformat(),
                        "score": item.score,
                    }
                    for item in payload.items
                ],
            }
            await self._redis.set(
                self._key(session_id, query_hash),
                json.dumps(data, ensure_ascii=False),
                ex=self._ttl,
            )
        except Exception:
            logger.debug(
                "recall cache set failed for session=%s (skipped)",
                session_id, exc_info=True,
            )
