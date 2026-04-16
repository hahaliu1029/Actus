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
            _make_chunk(user_id, "b", source="file"),
        ])
        await db_session.flush()

        result = await repo.list_by_user(user_id, source="file")
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

        deleted = await repo.delete_by_ids(user_id="other-user", ids=[chunk.id])
        assert deleted == []

    async def test_empty_ids_returns_empty_list(self, repo, user_id):
        deleted = await repo.delete_by_ids(user_id=user_id, ids=[])
        assert deleted == []


class TestDeleteAllByUser:
    async def test_deletes_all_returning_source_distribution(
        self, repo, user_id, db_session
    ):
        """delete_all_by_user 返回 DELETE ... RETURNING source 聚合后的分布。"""
        await repo.batch_insert_ignore([
            _make_chunk(user_id, "a", source="session_flush"),
            _make_chunk(user_id, "b", source="session_flush"),
            _make_chunk(user_id, "c", source="file"),
        ])
        await db_session.flush()

        deleted = await repo.delete_all_by_user(user_id=user_id)
        assert deleted == {"session_flush": 2, "file": 1}
        # 总数即 sum(values)
        assert sum(deleted.values()) == 3

        remaining = await repo.list_by_user(user_id)
        assert len(remaining) == 0

    async def test_empty_user_returns_empty_dict(self, repo, user_id):
        """没有记忆时返回空 dict，调用方取 sum(values) == 0。"""
        deleted = await repo.delete_all_by_user(user_id=user_id)
        assert deleted == {}


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
