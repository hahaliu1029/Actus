from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.domain.models.memory_chunk import MemoryChunk


class MemoryChunkRepository(Protocol):
    """记忆分块仓库协议。

    选用 Protocol（而非 ABC）与 SessionRepository / FileRepository /
    AppConfigRepository 保持一致。项目中 SkillRepository 等使用 ABC，
    两种风格共存；新增 repo 统一选 Protocol 以利于 structural subtyping。
    """

    async def batch_insert_ignore(self, chunks: Sequence[MemoryChunk]) -> int:
        """幂等批量写入。已存在（user_id + content_hash）则跳过。
        返回实际新插入的数量。

        接受 embedding 为 None 的 chunk（provider 故障降级场景）。
        无向量的 chunk 仍会持久化文本内容，但不会被 search_by_vector 召回
        （不可召回的冷数据，直到 C8 文本检索或后续新增 backfill 方法）。
        """
        ...

    async def search_by_vector(
        self,
        user_id: str,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.35,
    ) -> list[MemoryChunk]:
        """向量相似度搜索。返回按 cosine 相似度降序排列的结果。

        threshold 语义：cosine 相似度下限（0~1），只返回 similarity >= threshold 的结果。
        C3 实现注意：pgvector <=> 算子返回的是 cosine distance（= 1 - similarity），
        因此过滤条件应为 ``WHERE embedding <=> $vec <= (1 - threshold)``，
        排序应为 ``ORDER BY embedding <=> $vec ASC``。

        必须过滤 embedding IS NOT NULL 的行——embedding 为空的 chunk 是不可召回
        的冷数据，不应出现在结果中（即使 top_k 大于非空候选数）。
        """
        ...

    async def delete_by_session(self, session_id: str) -> int:
        """删除指定会话的所有记忆分块。返回删除数量。"""
        ...

    async def get_by_id(self, chunk_id: str, user_id: str) -> MemoryChunk | None:
        """按 ID + user_id 精确读取。不存在或越权均返回 None。"""
        ...
