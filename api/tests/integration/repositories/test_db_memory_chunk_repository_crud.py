"""Repository CRUD 扩展测试。

需要 PostgreSQL + Redis 运行环境。
注意：memory_chunks 有 FK 到 users.id，测试中必须先创建父记录。
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.memory_chunk import MemoryChunk
from app.infrastructure.repositories.db_memory_chunk_repository import (
    DBMemoryChunkRepository,
)

pytestmark = pytest.mark.anyio


def _make_chunk(user_id: str, content: str, source: str = "session_flush") -> MemoryChunk:
    """测试辅助：构建 MemoryChunk domain 对象。"""
    from app.domain.models.memory_chunk import memory_content_hash

    now = datetime.now(timezone.utc)
    return MemoryChunk(
        id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        content_hash=memory_content_hash(content),
        source=source,
        metadata={},
        created_at=now,
        updated_at=now,
        session_id=None,
        embedding=None,
    )


@pytest.fixture
async def user_id(db_session: AsyncSession) -> str:
    """创建测试用户，返回 user_id。"""
    from app.infrastructure.models.user import UserModel

    uid = str(uuid.uuid4())
    db_session.add(UserModel(id=uid, username=f"test_{uid[:8]}", password_hash="x"))
    await db_session.flush()
    return uid


@pytest.fixture
async def repo(db_session: AsyncSession) -> DBMemoryChunkRepository:
    return DBMemoryChunkRepository(db_session)


class TestListByUser:
    async def test_returns_user_chunks_only(self, repo, user_id, db_session):
        c1 = _make_chunk(user_id, "记忆 A")
        c2 = _make_chunk(user_id, "记忆 B")
        await repo.batch_insert_ignore([c1, c2])
        await db_session.flush()

        result = await repo.list_by_user(user_id)
        assert len(result) == 2

    async def test_query_filters_by_content(self, repo, user_id, db_session):
        await repo.batch_insert_ignore([
            _make_chunk(user_id, "Python 编程技巧"),
            _make_chunk(user_id, "烹饪食谱"),
        ])
        await db_session.flush()

        result = await repo.list_by_user(user_id, query="Python")
        assert len(result) == 1
        assert "Python" in result[0].content

    async def test_source_filter(self, repo, user_id, db_session):
        await repo.batch_insert_ignore([
            _make_chunk(user_id, "a", source="session_flush"),
            _make_chunk(user_id, "b", source="memory_save"),
        ])
        await db_session.flush()

        result = await repo.list_by_user(user_id, source="memory_save")
        assert len(result) == 1

    async def test_pagination(self, repo, user_id, db_session):
        chunks = [_make_chunk(user_id, f"chunk {i}") for i in range(5)]
        await repo.batch_insert_ignore(chunks)
        await db_session.flush()

        page1 = await repo.list_by_user(user_id, offset=0, limit=2)
        page2 = await repo.list_by_user(user_id, offset=2, limit=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    async def test_pagination_stable_under_equal_updated_at(
        self, repo, user_id, db_session
    ):
        """当多行共享相同 updated_at 时（MemoryFlushService 一批复用同一 now 的典型
        情况），分页必须走确定性 tie-breaker——否则跨页会出现重复或漏项。"""
        shared_time = datetime.now(timezone.utc)
        chunks = [
            MemoryChunk(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=f"c{i}",
                content_hash=f"shared-pagination-hash-{i}",
                source="session_flush",
                metadata={},
                created_at=shared_time,
                updated_at=shared_time,  # 所有行 updated_at 完全相同
                session_id=None,
                embedding=None,
            )
            for i in range(6)
        ]
        await repo.batch_insert_ignore(chunks)
        await db_session.flush()

        # 两页拼回去必须刚好是全集，无重复、无遗漏
        page1 = await repo.list_by_user(user_id, offset=0, limit=3)
        page2 = await repo.list_by_user(user_id, offset=3, limit=3)
        combined = {row.id for row in page1} | {row.id for row in page2}
        assert len(page1) == 3
        assert len(page2) == 3
        assert combined == {c.id for c in chunks}


class TestCountByUser:
    async def test_count_matches_list(self, repo, user_id, db_session):
        await repo.batch_insert_ignore([
            _make_chunk(user_id, "a"),
            _make_chunk(user_id, "b"),
        ])
        await db_session.flush()

        count = await repo.count_by_user(user_id)
        assert count == 2

    async def test_count_with_query(self, repo, user_id, db_session):
        await repo.batch_insert_ignore([
            _make_chunk(user_id, "Python tips"),
            _make_chunk(user_id, "cooking"),
        ])
        await db_session.flush()

        count = await repo.count_by_user(user_id, query="Python")
        assert count == 1


class TestUpdateContent:
    async def test_updates_content_and_hash(self, repo, user_id, db_session):
        from app.domain.models.memory_chunk import memory_content_hash

        chunk = _make_chunk(user_id, "old content")
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        updated = await repo.update_content(
            chunk_id=chunk.id,
            user_id=user_id,
            content="new content",
            content_hash=memory_content_hash("new content"),
            embedding=None,
        )
        assert updated is not None
        assert updated.content == "new content"
        assert updated.content_hash == memory_content_hash("new content")

    async def test_returns_none_for_wrong_user(self, repo, user_id, db_session):
        chunk = _make_chunk(user_id, "some content")
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        result = await repo.update_content(
            chunk_id=chunk.id,
            user_id="nonexistent-user",
            content="hacked",
            content_hash="x",
            embedding=None,
        )
        assert result is None


class TestUpdatePinned:
    """``DBMemoryChunkRepository.update_pinned``：pin/unpin 单字段 UPDATE。

    关键不变式（codex round-11 P1）：**不动 fs_synced**。如果这条 chunk 原本
    fs_synced=false（先前 create/update 写盘失败等 reconciler 回写），
    pin/unpin 后 fs_synced 必须仍为 false；否则 reconciler scan_pending
    扫不到这条，磁盘继续 stale。
    """

    async def test_pin_user_chunk_preserves_pending_fs_synced_false(
        self, repo, user_id, db_session
    ):
        """核心回归：fs_synced=false 的行 pin 后仍为 false。"""
        import dataclasses as _dc

        chunk = _dc.replace(
            _make_chunk(user_id, "profile", source="manual"),
            category="user",
            pinned=False,
            fs_synced=False,  # 模拟先前写盘失败的 pending backlog
        )
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        updated = await repo.update_pinned(
            chunk_id=chunk.id, user_id=user_id, pinned=True
        )
        assert updated is not None
        assert updated.pinned is True
        # 关键：fs_synced 保留 false，未被强制翻 true
        assert updated.fs_synced is False

    async def test_unpin_preserves_pending_fs_synced_false(
        self, repo, user_id, db_session
    ):
        """对 pinned=true + fs_synced=false 的行 unpin 后 fs_synced 仍 false。"""
        import dataclasses as _dc

        chunk = _dc.replace(
            _make_chunk(user_id, "pinned profile", source="manual"),
            category="user",
            pinned=True,
            fs_synced=False,
        )
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        updated = await repo.update_pinned(
            chunk_id=chunk.id, user_id=user_id, pinned=False
        )
        assert updated is not None
        assert updated.pinned is False
        assert updated.fs_synced is False

    async def test_pin_with_fs_synced_true_stays_true(
        self, repo, user_id, db_session
    ):
        """fs_synced=true 的 happy-path 行也保留 true（不是被我们强制，
        是"保留原值"的另一方向）。"""
        import dataclasses as _dc

        chunk = _dc.replace(
            _make_chunk(user_id, "x", source="manual"),
            category="user",
            pinned=False,
            fs_synced=True,
        )
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        updated = await repo.update_pinned(
            chunk_id=chunk.id, user_id=user_id, pinned=True
        )
        assert updated is not None
        assert updated.fs_synced is True

    async def test_returns_none_for_wrong_user(
        self, repo, user_id, db_session
    ):
        import dataclasses as _dc

        chunk = _dc.replace(
            _make_chunk(user_id, "x"), category="user", pinned=False,
        )
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        result = await repo.update_pinned(
            chunk_id=chunk.id, user_id=str(uuid.uuid4()), pinned=True
        )
        assert result is None


class TestDeleteByIds:
    async def test_deletes_matching_ids_returning_rows(
        self, repo, user_id, db_session
    ):
        """delete_by_ids 返回 DELETE ... RETURNING 的实际删除行列表。"""
        c1 = _make_chunk(user_id, "a")
        c2 = _make_chunk(user_id, "b")
        c3 = _make_chunk(user_id, "c")
        await repo.batch_insert_ignore([c1, c2, c3])
        await db_session.flush()

        deleted = await repo.delete_by_ids(user_id=user_id, ids=[c1.id, c2.id])
        assert len(deleted) == 2
        deleted_ids = {row.id for row in deleted}
        assert deleted_ids == {c1.id, c2.id}
        # 返回的是完整 domain 对象（审计需要 content/source 字段）
        for row in deleted:
            assert row.content in {"a", "b"}

        remaining = await repo.list_by_user(user_id)
        assert len(remaining) == 1

    async def test_ignores_other_user_ids(self, repo, user_id, db_session):
        chunk = _make_chunk(user_id, "mine")
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        # 使用一个随机 UUID 作为"其它用户"——delete 只做过滤匹配，不需要 FK 存在
        deleted = await repo.delete_by_ids(user_id=str(uuid.uuid4()), ids=[chunk.id])
        assert deleted == []

    async def test_empty_ids_returns_empty_list(self, repo, user_id):
        deleted = await repo.delete_by_ids(user_id=user_id, ids=[])
        assert deleted == []


class TestDeleteAllByUser:
    async def test_deletes_all_returning_rows(
        self, repo, user_id, db_session
    ):
        """PR-5A: delete_all_by_user 返回 DELETE ... RETURNING * 的完整行列表。
        调用方（MemoryManagementService）从 rows 聚合 source_dist + 遍历
        (id, category) 调 FsMemoryWriter.delete。"""
        from collections import Counter

        await repo.batch_insert_ignore([
            _make_chunk(user_id, "a", source="session_flush"),
            _make_chunk(user_id, "b", source="session_flush"),
            _make_chunk(user_id, "c", source="memory_save"),
        ])
        await db_session.flush()

        deleted = await repo.delete_all_by_user(user_id=user_id)
        assert len(deleted) == 3
        # source 聚合出自 list，等同旧 dict API
        dist = Counter(row.source for row in deleted)
        assert dict(dist) == {"session_flush": 2, "memory_save": 1}

        remaining = await repo.list_by_user(user_id)
        assert len(remaining) == 0

    async def test_empty_user_returns_empty_list(self, repo, user_id):
        """没有记忆时返回空 list，调用方取 len == 0。"""
        deleted = await repo.delete_all_by_user(user_id=user_id)
        assert deleted == []


class TestAutoPromotedAfterFilter:
    """design doc §777 audit query: GET /v2/memories?source=session_flush
    &auto_promoted_after=<ts> 让用户/运维审阅最近 N 天 LLM gate 自动收录。
    NULL 行（manual / memory_save 入口）不命中。"""

    async def test_only_returns_rows_promoted_after_cutoff(
        self, repo, user_id, db_session
    ):
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        old = MemoryChunk(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content="old auto-promoted",
            content_hash="hash-old-auto-promote",
            source="session_flush",
            metadata={},
            created_at=now - timedelta(days=14),
            updated_at=now - timedelta(days=14),
            session_id=None,
            embedding=None,
            auto_promoted_at=now - timedelta(days=14),
        )
        recent = MemoryChunk(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content="recent auto-promoted",
            content_hash="hash-recent-auto-promote",
            source="session_flush",
            metadata={},
            created_at=now - timedelta(days=2),
            updated_at=now - timedelta(days=2),
            session_id=None,
            embedding=None,
            auto_promoted_at=now - timedelta(days=2),
        )
        await repo.batch_insert_ignore([old, recent])
        await db_session.flush()

        cutoff = now - timedelta(days=7)
        result = await repo.list_by_user(user_id, auto_promoted_after=cutoff)
        assert len(result) == 1
        assert result[0].id == recent.id

        count = await repo.count_by_user(user_id, auto_promoted_after=cutoff)
        assert count == 1

    async def test_null_auto_promoted_at_excluded(
        self, repo, user_id, db_session
    ):
        """manual / memory_save 入口的行 auto_promoted_at IS NULL，
        必须不被任何非空 auto_promoted_after filter 命中。"""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        manual = _make_chunk(user_id, "manual entry", source="manual")
        await repo.batch_insert_ignore([manual])
        await db_session.flush()

        # manual 行没设 auto_promoted_at（默认 None）
        cutoff = now - timedelta(days=365)  # 极宽 cutoff，仍应排除 NULL
        result = await repo.list_by_user(user_id, auto_promoted_after=cutoff)
        assert len(result) == 0

        count = await repo.count_by_user(user_id, auto_promoted_after=cutoff)
        assert count == 0

    async def test_inclusive_cutoff_boundary(
        self, repo, user_id, db_session
    ):
        """codex round-3 fence-post check: 参数名叫 *_after* 但语义是 inclusive
        (>=)。文档明确 inclusive，钉死边界以防未来意外改成 strict (>)。"""
        cutoff = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
        on_boundary = MemoryChunk(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content="exactly at cutoff",
            content_hash="hash-on-boundary",
            source="session_flush",
            metadata={},
            created_at=cutoff,
            updated_at=cutoff,
            session_id=None,
            embedding=None,
            auto_promoted_at=cutoff,  # 与 cutoff 完全相等
        )
        await repo.batch_insert_ignore([on_boundary])
        await db_session.flush()

        result = await repo.list_by_user(user_id, auto_promoted_after=cutoff)
        assert len(result) == 1, "inclusive cutoff (>=) must include row at boundary"
        assert result[0].id == on_boundary.id

    async def test_combines_with_source_filter(
        self, repo, user_id, db_session
    ):
        """典型用法：source='session_flush' AND auto_promoted_after=<ts>"""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        # 同时间段但 source 不同——只 session_flush 应命中
        flush_recent = MemoryChunk(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content="flush recent",
            content_hash="hash-flush-recent",
            source="session_flush",
            metadata={},
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(hours=1),
            session_id=None,
            embedding=None,
            auto_promoted_at=now - timedelta(hours=1),
        )
        # 边界场景：memory_save 也可以理论上有 auto_promoted_at（虽然实际 service
        # 路径不写），用此构造确认 source filter 真在用
        save_recent = MemoryChunk(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content="save recent with timestamp",
            content_hash="hash-save-recent",
            source="memory_save",
            metadata={},
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(hours=1),
            session_id=None,
            embedding=None,
            auto_promoted_at=now - timedelta(hours=1),
        )
        await repo.batch_insert_ignore([flush_recent, save_recent])
        await db_session.flush()

        cutoff = now - timedelta(days=1)
        result = await repo.list_by_user(
            user_id,
            source="session_flush",
            auto_promoted_after=cutoff,
        )
        assert len(result) == 1
        assert result[0].id == flush_recent.id


class TestDeleteLegacyByUser:
    """M3-A: DELETE /v2/memories/legacy 底层 repo 方法。

    语义：真删 ``source='session_flush' AND category IS NULL AND
    auto_promoted_at IS NULL`` 的行。这对应 design doc §689 P6 的
    "NULL-first 历史行" ——那是 M1 migration 前遗留的、从没走过 LLM gate
    也没分类的旧 flush 块，信噪比差，用户通过"一键清理"按钮丢掉。

    不命中的行（必须保留）：
    - ``category`` 已填值（user/rule/fact）→ 新路径写入，已分类
    - ``auto_promoted_at`` 非 NULL → LLM gate 后自动收录，已定性
    - ``source='manual'`` 或 ``'memory_save'`` → 用户显式写入或 Agent 帮记
    """

    async def test_deletes_only_legacy_rows(
        self, repo, user_id, db_session
    ):
        """多种不应被删的行（categorized / auto-promoted 近期 / auto-promoted
        epoch / manual / memory_save）+ 一种 legacy 行。只 legacy 被删。

        **epoch 边界**：``auto_promoted_at=1970-01-01T00:00:00Z`` 也**不**被
        删——SQL 条件是 ``IS NULL``，任何非 NULL 的 timestamp（即便 epoch）
        都保留。永久钉死这个设计语义，避免未来有人把条件改成
        ``auto_promoted_at < <某 cutoff>`` 把 epoch 误当 "legacy"。
        """
        now = datetime.now(timezone.utc)

        # 1) legacy (要删)：session_flush + category=None + auto_promoted_at=None
        legacy = _make_chunk(user_id, "legacy flush", source="session_flush")
        # 2) categorized (保留)：session_flush + category='fact'
        import dataclasses as _dc

        categorized = _dc.replace(
            _make_chunk(user_id, "categorized flush", source="session_flush"),
            category="fact",
        )
        # 3) auto-promoted 近期 (保留)：session_flush + auto_promoted_at=now
        promoted = _dc.replace(
            _make_chunk(user_id, "auto-promoted flush", source="session_flush"),
            auto_promoted_at=now,
        )
        # 4) auto-promoted epoch (保留)：IS NULL 边界——非 NULL 就不删，
        # 即便是 1970-01-01。钉死 "IS NULL ≠ 任何时间戳" 的设计语义。
        epoch_promoted = _dc.replace(
            _make_chunk(user_id, "epoch promoted", source="session_flush"),
            auto_promoted_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
        )
        # 5) manual (保留)：source='manual'
        manual = _make_chunk(user_id, "manual entry", source="manual")
        # 6) memory_save (保留)：source='memory_save'
        save = _make_chunk(user_id, "save entry", source="memory_save")

        await repo.batch_insert_ignore(
            [legacy, categorized, promoted, epoch_promoted, manual, save]
        )
        await db_session.flush()

        deleted = await repo.delete_legacy_by_user(user_id=user_id)
        assert len(deleted) == 1
        assert deleted[0].id == legacy.id
        assert deleted[0].source == "session_flush"
        assert deleted[0].category is None
        assert deleted[0].auto_promoted_at is None

        remaining = await repo.list_by_user(user_id)
        remaining_ids = {row.id for row in remaining}
        assert legacy.id not in remaining_ids
        assert {
            categorized.id,
            promoted.id,
            epoch_promoted.id,
            manual.id,
            save.id,
        } <= remaining_ids

    async def test_empty_user_returns_empty_list(self, repo, user_id):
        """无任何 legacy 时返回空 list（调用方取 len == 0）。"""
        deleted = await repo.delete_legacy_by_user(user_id=user_id)
        assert deleted == []

    async def test_does_not_touch_other_users(
        self, repo, user_id, db_session
    ):
        """跨 user_id 边界：别的 user 的 legacy 行不在本次清理范围内。"""
        from app.infrastructure.models.user import UserModel

        other_uid = str(uuid.uuid4())
        db_session.add(
            UserModel(id=other_uid, username=f"other_{other_uid[:8]}", password_hash="x")
        )
        await db_session.flush()

        mine = _make_chunk(user_id, "mine legacy", source="session_flush")
        theirs = _make_chunk(other_uid, "their legacy", source="session_flush")
        await repo.batch_insert_ignore([mine, theirs])
        await db_session.flush()

        deleted = await repo.delete_legacy_by_user(user_id=user_id)
        assert len(deleted) == 1
        assert deleted[0].id == mine.id

        # 对方的 legacy 行完好
        other_remaining = await repo.list_by_user(other_uid)
        assert len(other_remaining) == 1
        assert other_remaining[0].id == theirs.id


class TestListByUserTimeRange:
    async def test_created_from_filter(self, repo, user_id, db_session):
        from datetime import timedelta

        c1 = _make_chunk(user_id, "old")
        c2 = _make_chunk(user_id, "new")
        await repo.batch_insert_ignore([c1, c2])
        await db_session.flush()

        future = datetime.now(timezone.utc) + timedelta(hours=1)
        result = await repo.list_by_user(user_id, created_from=future)
        assert len(result) == 0  # 两条都在 future 之前

    async def test_updated_to_filter(self, repo, user_id, db_session):
        from datetime import timedelta

        await repo.batch_insert_ignore([_make_chunk(user_id, "x")])
        await db_session.flush()

        past = datetime.now(timezone.utc) - timedelta(hours=1)
        result = await repo.list_by_user(user_id, updated_to=past)
        assert len(result) == 0  # 记忆比 past 新


class TestIlikeEscape:
    async def test_percent_in_query(self, repo, user_id, db_session):
        await repo.batch_insert_ignore([
            _make_chunk(user_id, "100% success rate"),
            _make_chunk(user_id, "hello world"),
        ])
        await db_session.flush()

        result = await repo.list_by_user(user_id, query="100%")
        assert len(result) == 1
        assert "100%" in result[0].content

    async def test_underscore_in_query(self, repo, user_id, db_session):
        await repo.batch_insert_ignore([
            _make_chunk(user_id, "file_name.txt"),
            _make_chunk(user_id, "filename.txt"),
        ])
        await db_session.flush()

        result = await repo.list_by_user(user_id, query="file_name")
        assert len(result) == 1


class TestUpdateContentConflict:
    async def test_unique_violation_on_duplicate_hash(self, repo, user_id, db_session):
        """编辑后 content_hash 与已有记忆冲突时应抛 IntegrityError。

        SQLAlchemy 将 asyncpg.UniqueViolationError 包装为 sqlalchemy.exc.IntegrityError。
        """
        import sqlalchemy.exc
        from app.domain.models.memory_chunk import memory_content_hash

        await repo.batch_insert_ignore([
            _make_chunk(user_id, "content A"),
            _make_chunk(user_id, "content B"),
        ])
        await db_session.flush()

        chunks = await repo.list_by_user(user_id)
        chunk_b = next(c for c in chunks if c.content == "content B")

        # 把 B 改成和 A 一样的内容 → hash 冲突 → IntegrityError
        # asyncpg 违反唯一约束可能在 execute() 立即抛出，也可能推迟到 flush()；
        # 两行都放在 with 块内，确保无论哪种时序都能被 pytest.raises 捕获。
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            await repo.update_content(
                chunk_id=chunk_b.id,
                user_id=user_id,
                content="content A",
                content_hash=memory_content_hash("content A"),
                embedding=None,
            )
            await db_session.flush()


# 历史说明：TestGetByIds / TestCountBySource 随 bulk_delete / delete_all
# 迁移到 DELETE ... RETURNING 实现后已无调用方，测试连同协议方法一起删除。
# 对"跨用户 id 隔离"和"实际删除分布"的保护由 TestDeleteByIds 和
# TestDeleteAllByUser 的 RETURNING 测试直接覆盖。
