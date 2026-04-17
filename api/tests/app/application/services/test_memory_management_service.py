"""Tests for MemoryManagementService.

覆盖：
- list_memories 正常分页 + page_size 钳位
- update_memory_content 空值/越权/embedding 降级/Conflict
- delete_memory / bulk_delete_memories / delete_all_memories
- 审计写入：edit / bulk_delete / delete_all 分支
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.memory_management_service import MemoryManagementService
from app.domain.models.memory_chunk import MemoryChunk

from tests.conftest import TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio


def _fake_quota_redis(
    *, incr_result: int = 1, pipeline_raises: Exception | None = None
) -> AsyncMock:
    """造一个兼容 memory_quota pipeline 使用方式的 Redis AsyncMock。

    memory_quota 使用 ``async with redis.pipeline(transaction=True) as pipe:
        pipe.incr(...); pipe.expire(...); await pipe.execute()``。
    同时保留 ``redis.decr`` 以供 refund 路径使用。
    """
    redis = AsyncMock()
    pipe = MagicMock()
    pipe.incr = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    if pipeline_raises is not None:
        pipe.execute = AsyncMock(side_effect=pipeline_raises)
    else:
        pipe.execute = AsyncMock(return_value=[incr_result, True])

    @asynccontextmanager
    async def pipeline_cm(transaction: bool = True):
        yield pipe

    redis.pipeline = pipeline_cm
    redis.decr = AsyncMock(return_value=max(incr_result - 1, 0))
    redis._pipe = pipe  # 暴露给测试做调用断言
    return redis


def _chunk(user_id: str = TEST_USER_ID_FIXED, content: str = "test") -> MemoryChunk:
    now = datetime.now(timezone.utc)
    return MemoryChunk(
        id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        content_hash="h",
        source="session_flush",
        metadata={},
        created_at=now,
        updated_at=now,
        session_id=None,
        embedding=None,
    )


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.list_by_user = AsyncMock(return_value=[])
    repo.count_by_user = AsyncMock(return_value=0)
    repo.get_by_id = AsyncMock(return_value=None)
    repo.update_content = AsyncMock(return_value=None)
    # delete_by_ids 现在返回 DELETE ... RETURNING 的实际删除行列表
    repo.delete_by_ids = AsyncMock(return_value=[])
    # PR-5A 起 delete_all_by_user 返回完整 row 列表（RETURNING *），供服务
    # 层同时聚合 source_dist（审计）+ 提取 (id, category)（fs 清盘）
    repo.delete_all_by_user = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_embed():
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[[0.1] * 512])
    return provider


@pytest.fixture
def mock_session():
    """构造支持 async with 的 mock session。"""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()  # add is sync
    return session


@pytest.fixture
def service(mock_repo, mock_embed, mock_session):
    @asynccontextmanager
    async def fake_session_factory():
        yield mock_session

    return MemoryManagementService(
        repo_factory=lambda sess: mock_repo,
        embedding_provider=mock_embed,
        session_factory=fake_session_factory,
    )


class TestListMemories:
    async def test_returns_items_and_total(self, service, mock_repo):
        chunk = _chunk()
        mock_repo.list_by_user.return_value = [chunk]
        mock_repo.count_by_user.return_value = 1

        items, total = await service.list_memories(TEST_USER_ID_FIXED)
        assert len(items) == 1
        assert total == 1

    async def test_page_size_clamped_to_50(self, service, mock_repo):
        mock_repo.list_by_user.return_value = []
        mock_repo.count_by_user.return_value = 0

        await service.list_memories(TEST_USER_ID_FIXED, page_size=100)
        call_kwargs = mock_repo.list_by_user.call_args
        assert call_kwargs.kwargs["limit"] == 50


class TestUpdateMemoryContent:
    async def test_empty_content_raises(self, service):
        with pytest.raises(ValueError, match="empty"):
            await service.update_memory_content(TEST_USER_ID_FIXED, "chunk-1", "   ")

    async def test_not_found_returns_none(self, service, mock_repo):
        mock_repo.get_by_id.return_value = None
        result = await service.update_memory_content(TEST_USER_ID_FIXED, "missing", "new text")
        assert result is None

    async def test_embedding_failure_degrades(self, service, mock_repo, mock_embed):
        from app.domain.external.embedding_provider import EmbeddingUnavailableError

        old = _chunk()
        updated = _chunk(content="new")
        mock_repo.get_by_id.return_value = old
        mock_repo.update_content.return_value = updated
        mock_embed.embed.side_effect = EmbeddingUnavailableError("provider down")

        result = await service.update_memory_content(TEST_USER_ID_FIXED, old.id, "new")
        assert result is not None
        # embedding=None should be passed to repo
        call_kwargs = mock_repo.update_content.call_args.kwargs
        assert call_kwargs["embedding"] is None

    async def test_conflict_raises_conflict_error(self, service, mock_repo):
        from sqlalchemy.exc import IntegrityError

        old = _chunk()
        mock_repo.get_by_id.return_value = old

        # 构造带 sqlstate="23505" 的 orig，模拟 asyncpg.UniqueViolationError
        class _FakeUnique(Exception):
            sqlstate = "23505"

        mock_repo.update_content.side_effect = IntegrityError(
            "duplicate", params=None, orig=_FakeUnique("unique violation")
        )

        from app.application.errors.exceptions import ConflictError

        with pytest.raises(ConflictError):
            await service.update_memory_content(TEST_USER_ID_FIXED, old.id, "dup content")

    async def test_non_unique_integrity_error_propagates(self, service, mock_repo):
        """FK 违反等其他完整性错误不应被误包装为 ConflictError。"""
        from sqlalchemy.exc import IntegrityError

        old = _chunk()
        mock_repo.get_by_id.return_value = old

        class _FakeFK(Exception):
            sqlstate = "23503"  # foreign_key_violation

        mock_repo.update_content.side_effect = IntegrityError(
            "fk violation", params=None, orig=_FakeFK("fk")
        )

        with pytest.raises(IntegrityError):
            await service.update_memory_content(TEST_USER_ID_FIXED, old.id, "content")


class TestDeleteMemory:
    async def test_not_found_returns_false(self, service, mock_repo):
        mock_repo.get_by_id.return_value = None
        assert await service.delete_memory(TEST_USER_ID_FIXED, "missing") is False

    async def test_success_returns_true(self, service, mock_repo):
        chunk = _chunk()
        mock_repo.get_by_id.return_value = chunk
        mock_repo.delete_by_ids.return_value = [chunk]
        assert await service.delete_memory(TEST_USER_ID_FIXED, chunk.id) is True

    async def test_concurrent_delete_returns_false(self, service, mock_repo):
        """get_by_id 看到了 chunk，但 DELETE ... RETURNING 返回空（被并发删了）。"""
        mock_repo.get_by_id.return_value = _chunk()
        mock_repo.delete_by_ids.return_value = []
        assert await service.delete_memory(TEST_USER_ID_FIXED, "id") is False


class TestBulkDelete:
    async def test_returns_count(self, service, mock_repo):
        mock_repo.delete_by_ids.return_value = [
            _chunk(content="a"),
            _chunk(content="b"),
            _chunk(content="c"),
        ]
        count = await service.bulk_delete_memories(TEST_USER_ID_FIXED, ["a", "b", "c"])
        assert count == 3


class TestDeleteAll:
    async def test_returns_count(self, service, mock_repo):
        # PR-5A: delete_all_by_user 返回完整 row 列表；总数 = len(list)
        mock_repo.delete_all_by_user.return_value = [
            _chunk(content="a"),
            _chunk(content="b"),
            _chunk(content="c"),
        ]
        count = await service.delete_all_memories(TEST_USER_ID_FIXED)
        assert count == 3

    async def test_empty_returns_zero_and_no_audit(
        self, service, mock_repo, mock_session
    ):
        mock_repo.delete_all_by_user.return_value = []
        count = await service.delete_all_memories(TEST_USER_ID_FIXED)
        assert count == 0
        assert not mock_session.add.called


class TestAuditWritten:
    async def test_edit_writes_audit(self, service, mock_repo, mock_session):
        old = _chunk()
        updated = _chunk(content="new")
        mock_repo.get_by_id.return_value = old
        mock_repo.update_content.return_value = updated

        await service.update_memory_content(TEST_USER_ID_FIXED, old.id, "new")
        # _write_audit calls session.add with MemoryAuditLogModel
        assert mock_session.add.called
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.action == "edit"
        assert audit_obj.old_snapshot["content"] == old.content
        # 审计与业务写入必须同事务 commit
        mock_session.commit.assert_called_once()

    async def test_delete_all_writes_audit_with_returning_distribution(
        self, service, mock_repo, mock_session
    ):
        """delete_all 审计的 source_distribution 与 affected_count 必须
        来自同一条 DELETE ... RETURNING * 语句（PR-5A）。Service 在 rows
        上做 Counter 聚合，保证审计 + fs 清盘共享同一真实集。"""
        import dataclasses
        flush_rows = [_chunk(content=f"flush_{i}") for i in range(42)]
        file_rows = [
            dataclasses.replace(_chunk(content=f"file_{i}"), source="file")
            for i in range(8)
        ]
        rows = flush_rows + file_rows
        mock_repo.delete_all_by_user.return_value = rows

        await service.delete_all_memories(TEST_USER_ID_FIXED)
        assert mock_session.add.called
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.action == "delete_all"
        assert audit_obj.affected_count == 50
        # 不再冗余写 total_before_delete（== affected_count）
        assert "total_before_delete" not in audit_obj.old_snapshot
        assert audit_obj.old_snapshot["source_distribution"] == {
            "session_flush": 42,
            "file": 8,
        }
        mock_session.commit.assert_called_once()

    async def test_bulk_delete_audit_records_only_actually_deleted(
        self, service, mock_repo, mock_session
    ):
        """审计只记录 DELETE ... RETURNING 实际删除的行。

        即便请求里混入了越权或并发已被删除的 id，审计也只会看到真正删掉的那些。
        """
        owned = _chunk(content="mine")
        # repo.delete_by_ids 返回 DELETE ... RETURNING 的结果——只有 owned 真的被删
        mock_repo.delete_by_ids.return_value = [owned]

        await service.bulk_delete_memories(
            TEST_USER_ID_FIXED, [owned.id, "not-mine-id", "already-gone-by-concurrent-delete"]
        )
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.action == "bulk_delete"
        assert audit_obj.chunk_ids == [owned.id]
        assert "not-mine-id" not in audit_obj.chunk_ids
        assert "already-gone-by-concurrent-delete" not in audit_obj.chunk_ids
        assert audit_obj.affected_count == 1
        assert len(audit_obj.old_snapshot["deleted_summaries"]) == 1
        mock_session.commit.assert_called_once()

    async def test_delete_writes_audit(self, service, mock_repo, mock_session):
        """单条删除同样必须写审计，action='delete'。"""
        chunk = _chunk(content="to-delete")
        mock_repo.get_by_id.return_value = chunk
        mock_repo.delete_by_ids.return_value = [chunk]

        result = await service.delete_memory(TEST_USER_ID_FIXED, chunk.id)

        assert result is True
        assert mock_session.add.called
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.action == "delete"
        assert audit_obj.chunk_id == chunk.id
        # 快照字段存在（截断到 200 字符；短内容保持原样）
        assert audit_obj.old_snapshot["content"] == chunk.content
        assert audit_obj.old_snapshot["content_hash"] == chunk.content_hash
        mock_session.commit.assert_called_once()

    async def test_edit_audit_truncates_long_content(
        self, service, mock_repo, mock_session
    ):
        """长内容的审计快照应被截断到 200 字符，避免 PII 流入日志聚合。"""
        long_text = "A" * 5000
        old = _chunk(content=long_text)
        updated = _chunk(content="B" * 300)
        mock_repo.get_by_id.return_value = old
        mock_repo.update_content.return_value = updated

        await service.update_memory_content(TEST_USER_ID_FIXED, old.id, "B" * 300)

        audit_obj = mock_session.add.call_args[0][0]
        assert len(audit_obj.old_snapshot["content"]) == 200
        assert len(audit_obj.new_snapshot["content"]) == 200
        mock_session.commit.assert_called_once()


# ─── M1 PR-2: create_memory ──────────────────────────────────────────────────


class TestCreateMemory:
    """MemoryManagementService.create_memory —— manual/memory_save 写入入口。"""

    async def test_happy_path_db_only_mode(
        self, service, mock_repo, mock_session
    ):
        """file_store=None（PR-0 默认 / PR-5A 前）→ 只落 DB，fs_synced 保持 False。"""
        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)

        chunk = await service.create_memory(
            TEST_USER_ID_FIXED,
            content="user prefers dark mode",
            category="user",
        )

        assert chunk.user_id == TEST_USER_ID_FIXED
        assert chunk.category == "user"
        assert chunk.source == "manual"
        assert chunk.pinned is False
        assert chunk.fs_synced is False  # DB-only 模式不翻 true
        mock_repo.batch_insert_ignore.assert_awaited_once()
        mock_session.commit.assert_called_once()

    async def test_happy_path_with_noop_file_store_sets_fs_synced(
        self, mock_repo, mock_embed, mock_session
    ):
        """注入 NoopFileMemoryStore（测试默认）→ write no-op 成功 → fs_synced=True。"""
        from app.domain.external.file_memory_store import NoopFileMemoryStore

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=NoopFileMemoryStore(),
        )

        chunk = await svc.create_memory(
            TEST_USER_ID_FIXED, content="hello", category="rule"
        )
        assert chunk.fs_synced is True
        mock_repo.mark_fs_synced.assert_awaited_once()

    async def test_empty_content_rejected(self, service):
        with pytest.raises(ValueError, match="empty"):
            await service.create_memory(TEST_USER_ID_FIXED, "  ", "user")

    async def test_invalid_category_rejected(self, service):
        with pytest.raises(ValueError, match="category"):
            await service.create_memory(TEST_USER_ID_FIXED, "x", "nope")

    async def test_invalid_source_rejected(self, service):
        with pytest.raises(ValueError, match="source"):
            await service.create_memory(
                TEST_USER_ID_FIXED, "x", "user", source="legacy"
            )

    async def test_pinned_requires_user_category(self, service):
        with pytest.raises(ValueError, match="pinned"):
            await service.create_memory(
                TEST_USER_ID_FIXED, "x", "rule", pinned=True
            )

    async def test_duplicate_content_hash_raises_conflict(
        self, service, mock_repo
    ):
        """batch_insert_ignore 返回 0（ON CONFLICT）→ ConflictError。"""
        from app.application.errors.exceptions import ConflictError

        mock_repo.batch_insert_ignore = AsyncMock(return_value=0)
        with pytest.raises(ConflictError):
            await service.create_memory(
                TEST_USER_ID_FIXED, "duplicate", "user"
            )

    async def test_embedding_failure_degrades_to_cold_write(
        self, service, mock_repo, mock_embed
    ):
        """embedding provider 故障 → 写 None embedding + 继续 INSERT。"""
        from app.domain.external.embedding_provider import EmbeddingUnavailableError

        mock_embed.embed = AsyncMock(side_effect=EmbeddingUnavailableError("circuit open"))
        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)

        chunk = await service.create_memory(
            TEST_USER_ID_FIXED, "no embedding", "fact"
        )
        assert chunk.embedding is None
        mock_repo.batch_insert_ignore.assert_awaited_once()

    async def test_fs_write_failure_keeps_fs_synced_false(
        self, mock_repo, mock_embed, mock_session
    ):
        """file_store.write 抛异常 → DB 不 rollback，fs_synced=False 留给 reconciler。"""

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        class _BoomStore:
            async def write(self, **kwargs):
                raise OSError("disk full")

            async def delete(self, **kwargs):
                pass

            async def move_category(self, **kwargs):
                pass

        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=_BoomStore(),
        )
        chunk = await svc.create_memory(TEST_USER_ID_FIXED, "x", "fact")
        assert chunk.fs_synced is False
        # mark_fs_synced 不应被调用（写盘失败）
        mock_repo.mark_fs_synced.assert_not_called()

    async def test_quota_exceeded_blocks_write(
        self, mock_repo, mock_embed, mock_session
    ):
        """Redis counter 超过 daily_cap → QuotaExceededError，DB 未落。"""
        from app.application.errors.exceptions import QuotaExceededError

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        fake_redis = _fake_quota_redis(incr_result=501)

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            redis=fake_redis,
            user_daily_quota=500,
        )

        with pytest.raises(QuotaExceededError) as exc_info:
            await svc.create_memory(TEST_USER_ID_FIXED, "blocked", "user")

        assert exc_info.value.status_code == 429
        # DB 不应 hit
        mock_repo.batch_insert_ignore.assert_not_called()

    async def test_quota_fail_open_on_redis_error(
        self, mock_repo, mock_embed, mock_session
    ):
        """pipeline 抛异常 → 配额检查 fail-open，写入继续。"""

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        fake_redis = _fake_quota_redis(pipeline_raises=Exception("redis down"))

        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)
        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            redis=fake_redis,
            user_daily_quota=500,
        )

        chunk = await svc.create_memory(TEST_USER_ID_FIXED, "ok", "rule")
        assert chunk.id is not None
        mock_repo.batch_insert_ignore.assert_awaited_once()

    async def test_duplicate_refunds_quota(
        self, mock_repo, mock_embed, mock_session
    ):
        """ConflictError 路径必须 DECR 已 INCR 的配额——防止 retry bomb 耗光配额。"""
        from app.application.errors.exceptions import ConflictError

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        fake_redis = _fake_quota_redis(incr_result=3)

        mock_repo.batch_insert_ignore = AsyncMock(return_value=0)  # ON CONFLICT
        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            redis=fake_redis,
            user_daily_quota=500,
        )

        with pytest.raises(ConflictError):
            await svc.create_memory(TEST_USER_ID_FIXED, "dup", "user")

        fake_redis._pipe.incr.assert_called_once()  # quota pipeline 正常 INCR
        fake_redis.decr.assert_awaited_once()  # 然后 refund

    async def test_fail_open_then_duplicate_does_not_refund(
        self, mock_repo, mock_embed, mock_session
    ):
        """fail-open pipeline（Redis 抛异常）+ 随后的 duplicate → **不**触发 DECR。

        否则会对不存在的 key 做 DECR，Redis 恢复后当天计数持久为负值，用户
        少算额度。本测试钉住 "quota_was_incremented 为 false 时跳过 refund"。
        """
        from app.application.errors.exceptions import ConflictError

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        fake_redis = _fake_quota_redis(pipeline_raises=Exception("redis down"))
        # 如果被调用会暴露 bug
        fake_redis.decr = AsyncMock(return_value=-1)

        mock_repo.batch_insert_ignore = AsyncMock(return_value=0)
        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            redis=fake_redis,
            user_daily_quota=500,
        )

        with pytest.raises(ConflictError):
            await svc.create_memory(TEST_USER_ID_FIXED, "dup", "user")

        fake_redis.decr.assert_not_called()

    async def test_ctor_rejects_redis_without_quota(
        self, mock_repo, mock_embed, mock_session
    ):
        """redis 和 user_daily_quota 必须同传或同不传——防 misconfig。"""

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        with pytest.raises(ValueError, match="redis"):
            MemoryManagementService(
                repo_factory=lambda s: mock_repo,
                embedding_provider=mock_embed,
                session_factory=fake_session_factory,
                redis=AsyncMock(),
                user_daily_quota=None,
            )
        with pytest.raises(ValueError, match="redis"):
            MemoryManagementService(
                repo_factory=lambda s: mock_repo,
                embedding_provider=mock_embed,
                session_factory=fake_session_factory,
                redis=None,
                user_daily_quota=500,
            )

    async def test_mark_fs_synced_failure_is_swallowed(
        self, mock_repo, mock_embed
    ):
        """mark_fs_synced 抛异常不应把请求炸成 500——DB+文件已写成，
        FsReconciler 后续会把 flag 翻对。"""
        from app.domain.external.file_memory_store import NoopFileMemoryStore

        # 两条独立的 mock_session——第一条（INSERT）成功，第二条（UPDATE）抛
        insert_session = AsyncMock()
        insert_session.commit = AsyncMock()
        insert_session.rollback = AsyncMock()
        mark_session = AsyncMock()
        mark_session.commit = AsyncMock(side_effect=Exception("pool exhausted"))
        mark_session.rollback = AsyncMock()
        call_count = {"n": 0}

        @asynccontextmanager
        async def fake_session_factory():
            call_count["n"] += 1
            yield insert_session if call_count["n"] == 1 else mark_session

        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=NoopFileMemoryStore(),
        )

        # 不应传播异常——吞掉 + warning
        chunk = await svc.create_memory(TEST_USER_ID_FIXED, "ok", "fact")
        assert chunk.id is not None
        # fs_synced 保持 False（DB 侧未翻）——reconciler 会补
        assert chunk.fs_synced is False


# ─── M1 PR-5A: fs sync 挂接（update / delete / bulk_delete / delete_all）──────


class _RecordingFileStore:
    """Simple ``FileMemoryStore`` impl that records every call for assertions.

    Subclass / configure via constructor to inject failures. Default is
    "all ops succeed, no-op on disk". ``write_fails`` triggers exception on
    every write; ``delete_fails`` triggers exception on every delete. Uses
    the same ``NoopFileMemoryStore``-style signatures (kwargs).
    """

    def __init__(
        self,
        *,
        write_fails: Exception | None = None,
        delete_fails: Exception | None = None,
    ):
        self.writes: list[dict] = []
        self.deletes: list[dict] = []
        self._write_fails = write_fails
        self._delete_fails = delete_fails

    async def write(self, **kwargs):
        self.writes.append(kwargs)
        if self._write_fails is not None:
            raise self._write_fails

    async def delete(self, **kwargs):
        self.deletes.append(kwargs)
        if self._delete_fails is not None:
            raise self._delete_fails

    async def move_category(self, **kwargs):
        pass


class TestDeriveTitle:
    """P2-1 回归：service 落盘的 frontmatter 必须含 canonical 声明的 title 字段。"""

    def test_derive_title_takes_first_line_stripped(self):
        from app.application.services.memory_management_service import _derive_title
        assert _derive_title("  Hello world  \ntrailing line") == "Hello world"

    def test_derive_title_empty_content_falls_back_to_untitled(self):
        from app.application.services.memory_management_service import _derive_title
        assert _derive_title("") == "untitled"
        assert _derive_title("   \n\n  ") == "untitled"

    def test_derive_title_truncates_long_first_line_with_ellipsis(self):
        from app.application.services.memory_management_service import (
            _TITLE_MAX_LENGTH,
            _derive_title,
        )
        long_text = "A" * (_TITLE_MAX_LENGTH + 20)
        out = _derive_title(long_text)
        assert len(out) == _TITLE_MAX_LENGTH + 1  # 80 chars + "…"
        assert out.endswith("…")

    def test_build_frontmatter_includes_title(self):
        """_build_frontmatter output 的 key 集必须覆盖 canonical—— PR-5A
        service 和 writer 协议不能各说各话。"""
        from datetime import datetime, timezone

        from app.application.services.memory_management_service import (
            MemoryManagementService,
        )
        from app.domain.models.memory_chunk import MemoryChunk

        chunk = MemoryChunk(
            id="01HXYZ",
            user_id=TEST_USER_ID_FIXED,
            content="pref: go 10 years\nmore details",
            content_hash="h",
            source="manual",
            metadata={"tags": ["go"]},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            session_id=None,
            embedding=None,
            category="user",
            pinned=True,
        )
        fm = MemoryManagementService._build_frontmatter(chunk)
        # Canonical 8 字段缺一不可
        expected = {"id", "title", "category", "source", "created_at",
                    "updated_at", "pinned", "tags"}
        assert expected <= set(fm.keys())
        assert fm["title"] == "pref: go 10 years"  # content 首行


class TestCreateFsSync:
    async def test_fs_write_includes_title_in_frontmatter(
        self, mock_repo, mock_embed, mock_session
    ):
        """Happy-path create → file_store.write 收到的 frontmatter 必须含 title。"""
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)
        store = _RecordingFileStore()

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        await svc.create_memory(
            TEST_USER_ID_FIXED, "user prefers dark mode\nother notes", "user"
        )
        assert len(store.writes) == 1
        fm = store.writes[0]["frontmatter"]
        assert "title" in fm
        assert fm["title"] == "user prefers dark mode"

    async def test_fs_write_failure_logs_audit(
        self, mock_repo, mock_embed, mock_session
    ):
        """FsMemoryWriter 重试耗尽后抛 OSError → service 写 fs_write_failed 审计。

        设计 L426：retries 耗尽 → memory_audit_log.action='fs_write_failed' +
        错误快照供 ops grep。MagicMock 会为每一条不同 session 记录 .add 调用，
        这里断言任一 add 调用的对象 action 是 fs_write_failed。
        """
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)
        store = _RecordingFileStore(write_fails=OSError("disk full"))

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        chunk = await svc.create_memory(TEST_USER_ID_FIXED, "x", "fact")
        assert chunk.fs_synced is False
        # add 至少被调用一次——fs_write_failed audit 写入路径
        audit_calls = [
            c for c in mock_session.add.call_args_list
            if getattr(c[0][0], "action", None) == "fs_write_failed"
        ]
        assert len(audit_calls) == 1
        audit = audit_calls[0][0][0]
        assert audit.new_snapshot["failed_op"] == "create"
        assert audit.new_snapshot["error_type"] == "OSError"
        assert "disk full" in audit.new_snapshot["error_msg"]


class TestUpdateFsSync:
    async def test_update_calls_file_store_write_with_overwrite(
        self, mock_repo, mock_embed, mock_session
    ):
        """PATCH → file_store.write(overwrite=True) + fs_synced flip True。"""
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        old = _chunk()
        import dataclasses
        updated = dataclasses.replace(old, content="new content", fs_synced=False, category="user")
        mock_repo.get_by_id.return_value = old
        mock_repo.update_content.return_value = updated
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)
        store = _RecordingFileStore()

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        result = await svc.update_memory_content(TEST_USER_ID_FIXED, old.id, "new content")

        assert result is not None
        assert result.fs_synced is True
        assert len(store.writes) == 1
        call = store.writes[0]
        assert call["overwrite"] is True
        assert call["memory_id"] == old.id
        assert call["content"] == "new content"
        # mark_fs_synced 翻 True
        mock_repo.mark_fs_synced.assert_awaited_once()

    async def test_update_fs_failure_keeps_fs_synced_false(
        self, mock_repo, mock_embed, mock_session
    ):
        """file_store.write 失败 → 不翻 fs_synced=true，审计写入 fs_write_failed。"""
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        old = _chunk()
        import dataclasses
        updated = dataclasses.replace(old, content="n", fs_synced=False, category="user")
        mock_repo.get_by_id.return_value = old
        mock_repo.update_content.return_value = updated
        mock_repo.mark_fs_synced = AsyncMock()
        store = _RecordingFileStore(write_fails=OSError("disk full"))

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        result = await svc.update_memory_content(TEST_USER_ID_FIXED, old.id, "n")
        assert result is not None
        assert result.fs_synced is False
        mock_repo.mark_fs_synced.assert_not_called()

    async def test_update_legacy_null_category_skips_fs(
        self, mock_repo, mock_embed, mock_session
    ):
        """Legacy row（category IS NULL）没有文件盘路径——update 跳过 fs sync。"""
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        import dataclasses
        old = _chunk()
        old = dataclasses.replace(old, category=None)
        updated = dataclasses.replace(old, content="n")
        mock_repo.get_by_id.return_value = old
        mock_repo.update_content.return_value = updated
        store = _RecordingFileStore()

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        await svc.update_memory_content(TEST_USER_ID_FIXED, old.id, "n")
        # Legacy 行没写盘，file_store 不应被 write 调到
        assert store.writes == []


class TestDeleteFsSync:
    async def test_delete_calls_file_store_delete(
        self, mock_repo, mock_embed, mock_session
    ):
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        import dataclasses
        chunk = dataclasses.replace(_chunk(), category="user")
        mock_repo.delete_by_ids.return_value = [chunk]
        store = _RecordingFileStore()

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        assert await svc.delete_memory(TEST_USER_ID_FIXED, chunk.id) is True
        assert len(store.deletes) == 1
        assert store.deletes[0]["memory_id"] == chunk.id
        assert store.deletes[0]["category"] == "user"

    async def test_delete_legacy_null_category_skips_fs(
        self, mock_repo, mock_embed, mock_session
    ):
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        import dataclasses
        chunk = dataclasses.replace(_chunk(), category=None)
        mock_repo.delete_by_ids.return_value = [chunk]
        store = _RecordingFileStore()

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        await svc.delete_memory(TEST_USER_ID_FIXED, chunk.id)
        assert store.deletes == []

    async def test_delete_fs_failure_is_swallowed(
        self, mock_repo, mock_embed, mock_session
    ):
        """file_store.delete 失败不应让 DELETE API 返回 500——
        留作孤儿等 reconciler 清。"""
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        import dataclasses
        chunk = dataclasses.replace(_chunk(), category="user")
        mock_repo.delete_by_ids.return_value = [chunk]
        store = _RecordingFileStore(delete_fails=OSError("disk gone"))

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        # Shouldn't raise
        assert await svc.delete_memory(TEST_USER_ID_FIXED, chunk.id) is True

    async def test_bulk_delete_calls_file_store_delete_per_id(
        self, mock_repo, mock_embed, mock_session
    ):
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        import dataclasses
        chunks = [
            dataclasses.replace(_chunk(content=f"c{i}"), category="user")
            for i in range(3)
        ]
        # 混一条 legacy 进去
        chunks.append(dataclasses.replace(_chunk(content="legacy"), category=None))
        mock_repo.delete_by_ids.return_value = chunks
        store = _RecordingFileStore()

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        count = await svc.bulk_delete_memories(
            TEST_USER_ID_FIXED, [c.id for c in chunks]
        )
        assert count == 4
        # 3 条 user + 1 条 legacy → 只 fs delete 3 条
        assert len(store.deletes) == 3
        assert all(d["category"] == "user" for d in store.deletes)

    async def test_delete_all_calls_file_store_delete_per_row(
        self, mock_repo, mock_embed, mock_session
    ):
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        import dataclasses
        rows = [
            dataclasses.replace(_chunk(content=f"r{i}"), category="user")
            for i in range(5)
        ]
        mock_repo.delete_all_by_user.return_value = rows
        store = _RecordingFileStore()

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
        )

        count = await svc.delete_all_memories(TEST_USER_ID_FIXED)
        assert count == 5
        assert len(store.deletes) == 5
