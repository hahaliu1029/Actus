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
        pinned: bool | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
        auto_promoted_after: datetime | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> list[MemoryChunk]:
        """按过滤条件返回当前用户记忆的分页列表，按 updated_at DESC 排序。

        ``category`` 为 None 时不过滤（等价 PR-1 前行为）；传入 ``user/rule/fact``
        只返回对应类。legacy 行 ``category IS NULL`` 在任何非空 filter 下都不命中。

        ``pinned`` 为 None 时不过滤；True 只返回 pinned 行，False 只返回未 pinned
        行。M2 user_profile prompt section 用 ``pinned=True`` 先拉全部 pinned（由
        ``ix_memory_chunks_user_pinned`` partial index 支撑，独立于 recency 窗口），
        再拉 top-N unpinned，保证 "pinned 永远先浮现" 不被最近 N 条的滑窗吞掉。

        ``auto_promoted_after`` 为 None 时不过滤；传入 ``datetime`` 只返回
        ``auto_promoted_at >= auto_promoted_after`` 的行。配合 ``source='session_flush'``
        即 design doc §777 的"最近自动收录的 memory 审阅"路径——用户/运维想验证
        最近 N 天 LLM gate 收录质量。``auto_promoted_at IS NULL`` 的旧行（包括
        legacy 历史 + manual / memory_save 入口写的）在任何非空 filter 下都不命中。
        """
        ...

    async def count_by_user(
        self,
        user_id: str,
        *,
        query: str | None = None,
        source: str | None = None,
        category: str | None = None,
        pinned: bool | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
        auto_promoted_after: datetime | None = None,
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

    async def distinct_user_ids(self) -> list[str]:
        """返回所有写过 memory_chunks 的 user_id，用于 FsReconciler CLI 全量扫描。

        不过滤 fs_synced —— reconcile_all_users 要兼顾孤儿 DB 行和孤儿 fs 行，
        两侧都要走一遍。结果量级为 platform 总用户数，对单实例 <10k 用户可接受；
        更大规模场景下 CLI 自己分页即可（post-M3）。
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

    async def delete_legacy_by_user(
        self,
        *,
        user_id: str,
        rollout_at: datetime | None = None,
    ) -> list[MemoryChunk]:
        """删除"旧 session_flush"遗留行，返回被删除的完整行列表。

        "一键清理旧 session_flush" 入口的底层 SQL。

        条件（**AND** 合取，缺一不可）：
        - ``source = 'session_flush'`` — 只清 flush 管线遗留
        - ``category IS NULL`` — 未分类的老行；新 flush 行若经 LLM gate 收录
          会有 category='user/rule/fact'，不在清理范围内
        - ``auto_promoted_at IS NULL`` — LLM gate 没收录的；经 gate 收录的
          行会有 auto_promoted_at 非空，属于"系统已背书"的数据，不能乱删
        - ``rollout_at`` 非空 → 额外 AND ``created_at < rollout_at``（codex fix
          P1）：gate 关闭 deployment 里 post-launch 新写入的 session_flush
          也是 (NULL, NULL)，若不加时间边界会被误删。rollout_at 为空时沿用
          旧谓词（未经 gate 全清）——由 service 层在 UI 上显式警告。

        用 ``DELETE ... RETURNING *`` 单语句出，调用方一次拿到 "被删行数 +
        chunk_ids"（审计所需）+ (id, category) 对（fs 清盘需要，但 legacy
        本来 category IS NULL → 从未落盘，跳过）。
        """
        ...
