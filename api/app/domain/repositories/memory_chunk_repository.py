from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
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

    async def list_by_user(
        self,
        user_id: str,
        *,
        query: str | None = None,
        source: str | None = None,
        category: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> list[MemoryChunk]:
        """按过滤条件返回当前用户记忆的分页列表，按 updated_at DESC 排序。

        ``category`` 为 None 时不过滤（等价 PR-1 前行为）；传入 ``user/rule/fact``
        只返回对应类。legacy 行 ``category IS NULL`` 在任何非空 filter 下都不命中。
        """
        ...

    async def count_by_user(
        self,
        user_id: str,
        *,
        query: str | None = None,
        source: str | None = None,
        category: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
    ) -> int:
        """相同过滤条件的总条数，用于分页。"""
        ...

    async def find_pending_fs_sync(
        self,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryChunk]:
        """返回 ``fs_synced = false`` 的行，供 FsReconciler 拾回补写。

        - 不传 ``user_id`` 返回全局待同步（启动扫描路径）
        - 传 ``user_id`` 返回该用户（lazy per-user walk 路径）
        - 按 ``updated_at ASC`` 排序，先处理旧的 pending 行
        - 用 ``ix_memory_chunks_fs_synced_pending`` partial index 保证 O(pending)
        """
        ...

    async def mark_fs_synced(
        self, *, chunk_id: str, user_id: str, synced: bool = True
    ) -> bool:
        """将 chunk 的 fs_synced 标记为 ``synced``（双向）。

        - ``synced=True``：FsMemoryWriter 写盘成功后调用，false → true
        - ``synced=False``：update_memory_content / move_category 开启同步
          窗口时调用，true → false，再由 FsReconciler 或下一次写盘收尾

        返回是否真的命中一行（id + user_id 匹配）；不存在或越权返回 False。
        """
        ...

    async def update_content(
        self,
        *,
        chunk_id: str,
        user_id: str,
        content: str,
        content_hash: str,
        embedding: tuple[float, ...] | None,
    ) -> MemoryChunk | None:
        """编辑记忆内容。写入 content + content_hash + embedding，更新 updated_at。

        chunk_id 不存在或非该用户 → 返回 None。
        content_hash 冲突 (uq_memory_user_hash) → 抛 sqlalchemy.exc.IntegrityError。
        """
        ...

    async def delete_by_ids(
        self, *, user_id: str, ids: list[str]
    ) -> list[MemoryChunk]:
        """按 id 批量删除当前用户的记忆，返回实际被删除的行（DELETE ... RETURNING）。

        审计路径依赖"实际删除集"而非"请求集 ∩ 所有权集"——把删除与审计快照合并到
        同一条语句可避免 TOCTOU（READ COMMITTED 下两步之间的并发删除会让审计失真）。
        调用方如只需数量，取 ``len(result)``。
        """
        ...

    async def delete_all_by_user(self, *, user_id: str) -> list[MemoryChunk]:
        """删除当前用户全部记忆，返回实际被删除的行（``DELETE ... RETURNING``）。

        PR-5A 起返回完整 row 列表——调用方同时需要 source 分布（审计）和
        (id, category) 对（FsMemoryWriter.delete 清盘）。用单条 RETURNING
        保证这两视图出自同一条语句，避免 READ COMMITTED 下多次查询的竞态。

        调用方侧：``source 分布 = Counter(c.source for c in result)``；
        ``总数 = len(result)``。
        """
        ...
