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

    async def test_forwards_auto_promoted_after_to_repo(self, service, mock_repo):
        """design doc §777 audit query 透传：service 必须把 auto_promoted_after
        透到 repo 的 list + count 两边，否则 total 与 items 跨页对不上。"""
        cutoff = datetime(2026, 4, 12, tzinfo=timezone.utc)
        await service.list_memories(
            TEST_USER_ID_FIXED,
            source="session_flush",
            auto_promoted_after=cutoff,
        )
        list_kwargs = mock_repo.list_by_user.call_args.kwargs
        count_kwargs = mock_repo.count_by_user.call_args.kwargs
        assert list_kwargs["auto_promoted_after"] == cutoff
        assert count_kwargs["auto_promoted_after"] == cutoff


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


class TestDeleteLegacy:
    """M3-A: 一键清理旧 session_flush 数据（design doc §689）。

    - 返回值 = DELETE ... RETURNING 实际行数
    - 空时不写审计（与 delete_all 一致）
    - 写审计 ``action='delete_legacy'`` + affected_count + chunk_ids
    - Legacy 行 ``category IS NULL`` 从未写过 fs → 跳过 ``file_store.delete``
    """

    async def test_returns_count_from_returning_rows(
        self, service, mock_repo
    ):
        mock_repo.delete_legacy_by_user = AsyncMock(
            return_value=[_chunk(content="l1"), _chunk(content="l2")]
        )
        count = await service.delete_legacy_memories(TEST_USER_ID_FIXED)
        assert count == 2

    async def test_empty_returns_zero_and_no_audit(
        self, service, mock_repo, mock_session
    ):
        mock_repo.delete_legacy_by_user = AsyncMock(return_value=[])
        count = await service.delete_legacy_memories(TEST_USER_ID_FIXED)
        assert count == 0
        assert not mock_session.add.called
        # 没 rows 也没必要 commit
        mock_session.commit.assert_not_called()

    async def test_writes_audit_with_chunk_ids_and_hashes(
        self, service, mock_repo, mock_session
    ):
        """审计记录 chunk_ids + content_hashes + rollout_at，不含 content preview。

        content_hash 无 PII 可长期保留，用于事后查 orphan bug 或误点追溯；
        content 不入 audit 以避开敏感数据流入 log 聚合系统。``rollout_at``
        记录本次清理的时间边界（codex fix P1）；默认 service fixture 未设 →
        None，表明沿用旧谓词。
        """
        import dataclasses
        rows = [
            dataclasses.replace(
                _chunk(content=f"legacy_{i}"),
                content_hash=f"hash-legacy-{i}",
            )
            for i in range(3)
        ]
        mock_repo.delete_legacy_by_user = AsyncMock(return_value=rows)

        await service.delete_legacy_memories(TEST_USER_ID_FIXED)

        assert mock_session.add.called
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.action == "delete_legacy"
        assert audit_obj.affected_count == 3
        assert audit_obj.chunk_ids == [r.id for r in rows]
        # 关键：hash 列表与 chunk_ids 顺序对齐 + rollout_at None，**不**含 content preview
        assert audit_obj.old_snapshot == {
            "content_hashes": [f"hash-legacy-{i}" for i in range(3)],
            "rollout_at": None,
        }
        assert "content" not in audit_obj.old_snapshot
        mock_session.commit.assert_called_once()

    async def test_rollout_at_forwarded_to_repo_and_audit(
        self, mock_repo, mock_embed, mock_session
    ):
        """rollout_at 非空时：(a) 透传到 repo.delete_legacy_by_user；
        (b) 审计 old_snapshot["rollout_at"] 记录 ISO 串。
        codex fix P1：legacy 清理时间边界必须端到端可追溯。"""
        from datetime import datetime, timezone

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        cutoff = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
        rows = [_chunk(content="legacy-cut")]
        mock_repo.delete_legacy_by_user = AsyncMock(return_value=rows)

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            memory_gate_rollout_at=cutoff,
        )
        await svc.delete_legacy_memories(TEST_USER_ID_FIXED)

        # (a) 透传到 repo
        mock_repo.delete_legacy_by_user.assert_awaited_once_with(
            user_id=TEST_USER_ID_FIXED, rollout_at=cutoff
        )
        # (b) 审计记录 rollout_at（ISO 串，非 datetime 对象，便于 JSON 序列化）
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.old_snapshot["rollout_at"] == cutoff.isoformat()

    async def test_warns_and_skips_fs_when_non_null_category_slips_through(
        self, mock_repo, mock_embed, mock_session, caplog
    ):
        """防御契约：repo 若意外返 category 非 None 的行 → warn + 绝不 fs_delete。

        两个断言缺一不可：
        - logger.warning 被触发（运维可发现 repo bug）
        - fs_store.delete **绝不**被调用（legacy 行从未落盘，误调可能把正常
          文件当 orphan 删）。需要注入真的 fs_store 才能钉这个契约，默认
          service fixture 的 file_store=None 覆盖不到这层。
        """
        import dataclasses
        import logging

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        fs_store = AsyncMock()
        fs_store.delete = AsyncMock()

        # 造一条 category='user' 的漏网行（repo bug simulation）
        slipped = dataclasses.replace(
            _chunk(content="shouldn't be here"), category="user"
        )
        mock_repo.delete_legacy_by_user = AsyncMock(return_value=[slipped])

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=fs_store,
        )

        with caplog.at_level(logging.WARNING):
            await svc.delete_legacy_memories(TEST_USER_ID_FIXED)

        # 契约 1：warn 被触发
        assert any(
            "delete_legacy 遇到 category=user" in record.message
            for record in caplog.records
        ), f"expected defensive warn, got {[r.message for r in caplog.records]}"
        # 契约 2：fs_store.delete 绝不被调用，即便 slipped row 有非 None category
        fs_store.delete.assert_not_called()

    async def test_skips_fs_delete_for_null_category_rows(
        self, mock_repo, mock_embed, mock_session
    ):
        """Legacy 行 category IS NULL，从没写过文件——不应触发 fs_store.delete。"""
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        fs_store = AsyncMock()
        fs_store.delete = AsyncMock()

        rows = [_chunk(content="legacy")]
        # 明确 category=None（dataclass 默认已是 None，显式 override 留给 reader）
        import dataclasses
        rows = [dataclasses.replace(r, category=None) for r in rows]
        mock_repo.delete_legacy_by_user = AsyncMock(return_value=rows)

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=fs_store,
        )

        await svc.delete_legacy_memories(TEST_USER_ID_FIXED)

        # 关键断言：file_store.delete 不被调用（legacy 从未写过盘）
        fs_store.delete.assert_not_called()


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
        from app.infrastructure.external.memory.frontmatter import derive_title
        assert derive_title("  Hello world  \ntrailing line") == "Hello world"

    def test_derive_title_empty_content_falls_back_to_untitled(self):
        from app.infrastructure.external.memory.frontmatter import derive_title
        assert derive_title("") == "untitled"
        assert derive_title("   \n\n  ") == "untitled"

    def test_derive_title_truncates_long_first_line_with_ellipsis(self):
        from app.infrastructure.external.memory.frontmatter import (
            TITLE_MAX_LENGTH,
            derive_title,
        )
        long_text = "A" * (TITLE_MAX_LENGTH + 20)
        out = derive_title(long_text)
        assert len(out) == TITLE_MAX_LENGTH + 1  # 80 chars + "…"
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


class TestCreateTags:
    """PR-7: 手动创建支持 tags → metadata["tags"] → frontmatter。"""

    async def test_clean_tags_helper_dedupes_strips_and_caps(self):
        from app.application.services.memory_management_service import _clean_tags

        # 空 / None → []
        assert _clean_tags(None) == []
        assert _clean_tags([]) == []
        # strip + 丢空 + 去重（大小写敏感、保序）
        assert _clean_tags([" Go ", "", "  ", "react", "go", "Go"]) == [
            "Go", "react", "go",
        ]
        # 超过 20 条截断
        many = [f"t{i}" for i in range(30)]
        out = _clean_tags(many)
        assert len(out) == 20
        assert out[0] == "t0" and out[19] == "t19"
        # 单 tag 超长截断到 64
        long_tag = "x" * 100
        assert _clean_tags([long_tag]) == ["x" * 64]

    async def test_create_memory_persists_tags_to_metadata(
        self, service, mock_repo, mock_session
    ):
        """service.create_memory(tags=[...]) → chunk.metadata["tags"] 被填充。"""
        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)

        chunk = await service.create_memory(
            TEST_USER_ID_FIXED,
            content="prefers Go",
            category="user",
            tags=["Go", "backend"],
        )
        assert chunk.metadata.get("tags") == ["Go", "backend"]

    async def test_create_memory_empty_tags_no_metadata_key(
        self, service, mock_repo, mock_session
    ):
        """tags=None 或 []  不写 metadata["tags"] key——避免 DB 里一堆空 list 噪声。"""
        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)

        chunk_none = await service.create_memory(
            TEST_USER_ID_FIXED, "x", "rule", tags=None
        )
        assert "tags" not in chunk_none.metadata

        chunk_empty = await service.create_memory(
            TEST_USER_ID_FIXED, "y", "rule", tags=[]
        )
        assert "tags" not in chunk_empty.metadata

    async def test_build_frontmatter_renders_tags_from_metadata(self):
        """_build_frontmatter 已读 metadata['tags']——这里回归 tags 从 service
        的 metadata 一路传到 frontmatter 的链路。"""
        from datetime import datetime, timezone

        from app.application.services.memory_management_service import (
            MemoryManagementService,
        )
        from app.domain.models.memory_chunk import MemoryChunk

        chunk = MemoryChunk(
            id="01HXYZ",
            user_id=TEST_USER_ID_FIXED,
            content="c",
            content_hash="h",
            source="manual",
            metadata={"tags": ["Go", "TypeScript"]},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            session_id=None,
            embedding=None,
            category="user",
        )
        fm = MemoryManagementService._build_frontmatter(chunk)
        assert fm["tags"] == ["Go", "TypeScript"]


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

    async def test_fs_write_failure_emits_fs_permanent_failure_notification(
        self, mock_repo, mock_embed, mock_session
    ):
        """FsMemoryWriter 重试耗尽 → MemoryNotificationEmitter.emit 被调用，
        event_type='fs_permanent_failure'。

        回归：schema / notification_schemas.py 把 fs_permanent_failure 列为
        M1 已知 event_type，但 PR-5B 合入前 service 从不发射它——对外契约是
        空承诺。这条测试锁住 emitter 被真正调用的路径。
        """
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)
        store = _RecordingFileStore(write_fails=OSError("disk full"))

        emitter = AsyncMock()
        emitter.emit = AsyncMock()

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
            notification_emitter=emitter,
        )

        await svc.create_memory(TEST_USER_ID_FIXED, "content", "fact")

        emitter.emit.assert_awaited_once()
        kwargs = emitter.emit.await_args.kwargs
        assert kwargs["user_id"] == TEST_USER_ID_FIXED
        assert kwargs["event_type"] == "fs_permanent_failure"
        assert kwargs["payload"]["category"] == "fact"
        assert kwargs["payload"]["action"] == "create"
        assert kwargs["payload"]["error_type"] == "OSError"
        assert "disk full" in kwargs["payload"]["error_msg"]
        # chunk_id 必须带——用户托盘 UI / 运维诊断都依赖它定位具体条目
        assert kwargs["payload"]["chunk_id"]

    async def test_fs_write_failure_without_emitter_does_not_crash(
        self, mock_repo, mock_embed, mock_session
    ):
        """Legacy/test 路径不传 emitter 时 fs 写失败应降级为只写 audit，
        不抛也不 emit——保持 DB-only 部署可用。"""
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
            # notification_emitter 显式不传
        )

        chunk = await svc.create_memory(TEST_USER_ID_FIXED, "content", "user")
        assert chunk.fs_synced is False  # audit 路径保留

    async def test_emitter_failure_does_not_break_create(
        self, mock_repo, mock_embed, mock_session
    ):
        """emitter.emit 抛异常（Redis 挂 / DB 挂）时 create_memory 仍然成功返回——
        通知是 advisory，不能 fail 整条写入路径。DBMemoryNotificationEmitter
        实现里 swallow 异常；这里用 AsyncMock 模拟一个会 raise 的 emitter
        验证 service 层也是 contract-safe 的（ future emitter 实现若不 swallow
        也不会把 create 拖垮）。"""
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session

        mock_repo.batch_insert_ignore = AsyncMock(return_value=1)
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)
        store = _RecordingFileStore(write_fails=OSError("disk full"))

        class _RaisingEmitter:
            async def emit(self, **kwargs):
                raise RuntimeError("emitter db down")

        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=store,
            notification_emitter=_RaisingEmitter(),
        )

        # service._try_emit_fs_failure_notification 内部加了 try/except 兜底，
        # 即使 emitter 实现违反"内部 swallow"契约，create 也能正常返回。
        # 通知丢了是 acceptable，写入路径不能被一条通知拖垮。
        chunk = await svc.create_memory(TEST_USER_ID_FIXED, "content", "fact")
        assert chunk.id
        assert chunk.fs_synced is False  # fs 仍然失败


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


# ─── update_memory_pinned（PATCH pin/unpin 扩展）──────────────────────────

class TestUpdateMemoryPinned:
    """``MemoryManagementService.update_memory_pinned``：单字段切 pinned。

    DB CHECK 约束 ``pinned=true OR category='user'``：非 user 类 + pin=True
    应被 service 前置校验拒绝（400），race 场景兜底的 IntegrityError 23514
    也 map 到同一错误。``repo.update_pinned`` **保留现有 fs_synced**（不强
    制置 True），避免吞掉 pre-existing pending backlog——pin 切换不改盘面
    正文，fs_synced 该是什么就是什么，让 reconciler / FsMemoryWriter 各走
    各的同步路径。
    """

    def _user_chunk(self, pinned: bool = False):
        import dataclasses
        return dataclasses.replace(
            _chunk(content="profile"),
            category="user",
            pinned=pinned,
            fs_synced=True,
        )

    async def test_returns_none_when_chunk_missing(
        self, service, mock_repo
    ):
        mock_repo.get_by_id.return_value = None
        result = await service.update_memory_pinned(
            TEST_USER_ID_FIXED, "missing", True
        )
        assert result is None

    async def test_pin_user_chunk_happy_path(
        self, service, mock_repo, mock_session
    ):
        import dataclasses
        chunk = self._user_chunk(pinned=False)
        updated_row = dataclasses.replace(chunk, pinned=True)
        mock_repo.get_by_id.return_value = chunk
        mock_repo.update_pinned = AsyncMock(return_value=updated_row)

        result = await service.update_memory_pinned(
            TEST_USER_ID_FIXED, chunk.id, True
        )

        assert result is not None
        assert result.pinned is True
        mock_repo.update_pinned.assert_awaited_once()
        call_kwargs = mock_repo.update_pinned.call_args.kwargs
        assert call_kwargs["pinned"] is True
        # audit action='pin'
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.action == "pin"
        assert audit_obj.old_snapshot == {"pinned": False}
        assert audit_obj.new_snapshot == {"pinned": True}

    async def test_unpin_happy_path_writes_unpin_action(
        self, service, mock_repo, mock_session
    ):
        import dataclasses
        chunk = self._user_chunk(pinned=True)
        mock_repo.get_by_id.return_value = chunk
        mock_repo.update_pinned = AsyncMock(
            return_value=dataclasses.replace(chunk, pinned=False)
        )

        await service.update_memory_pinned(TEST_USER_ID_FIXED, chunk.id, False)
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.action == "unpin"
        assert audit_obj.old_snapshot == {"pinned": True}
        assert audit_obj.new_snapshot == {"pinned": False}

    async def test_pin_rejected_for_non_user_category(
        self, service, mock_repo
    ):
        """core contract：category=rule/fact/None 时 pin=True → 400。"""
        from app.application.errors.exceptions import BadRequestError
        import dataclasses
        chunk = dataclasses.replace(
            _chunk(content="rule content"), category="rule", pinned=False
        )
        mock_repo.get_by_id.return_value = chunk
        mock_repo.update_pinned = AsyncMock()

        with pytest.raises(BadRequestError, match="category='user'"):
            await service.update_memory_pinned(
                TEST_USER_ID_FIXED, chunk.id, True
            )
        # 不应推进到 DB update
        mock_repo.update_pinned.assert_not_called()

    async def test_unpin_allowed_for_non_user_category_is_noop(
        self, service, mock_repo, mock_session
    ):
        """DB CHECK 保证 category!=user 的 chunk pinned 永远 False，所以对
        这类 chunk 调 unpin（target=False）是 no-op。

        这条测试的 spirit 是"非 user 类也不会被 BadRequestError 拦"（因为
        pinned=False 不触发 pinned/category 约束）。结合 codex round-11 P1
        的 no-op 短路：本 case 走 short-circuit path 返 existing，不调
        update_pinned、不写 audit——仍然不报错，语义一致（幂等）。
        """
        import dataclasses
        chunk = dataclasses.replace(
            _chunk(content="fact"), category="fact", pinned=False,
        )
        mock_repo.get_by_id.return_value = chunk
        mock_repo.update_pinned = AsyncMock()  # 不应被调（no-op）

        result = await service.update_memory_pinned(
            TEST_USER_ID_FIXED, chunk.id, False
        )
        # 幂等：返 existing chunk，非 None，非 BadRequest
        assert result is chunk
        mock_repo.update_pinned.assert_not_called()

    async def test_noop_pin_short_circuits_no_db_write(
        self, service, mock_repo, mock_session
    ):
        """codex round-11 P1：对已 pinned=True 的行再 pin → no-op 短路，
        不调 repo.update_pinned、不写 audit、不 commit、不刷 updated_at。"""
        chunk = self._user_chunk(pinned=True)
        mock_repo.get_by_id.return_value = chunk
        mock_repo.update_pinned = AsyncMock()  # 不应被调

        result = await service.update_memory_pinned(
            TEST_USER_ID_FIXED, chunk.id, True
        )
        # 返回现有 chunk（语义：幂等设目标状态成功）
        assert result is chunk
        mock_repo.update_pinned.assert_not_called()
        # 不写 audit
        assert not mock_session.add.called
        # 不 commit
        mock_session.commit.assert_not_called()

    async def test_noop_unpin_short_circuits_for_non_user(
        self, service, mock_repo, mock_session
    ):
        """对 category=rule 的行 unpin（pinned=False），DB invariant 下
        它 pinned 本就是 False → 也走 no-op 短路。"""
        import dataclasses
        chunk = dataclasses.replace(
            _chunk(content="rule"), category="rule", pinned=False
        )
        mock_repo.get_by_id.return_value = chunk
        mock_repo.update_pinned = AsyncMock()

        result = await service.update_memory_pinned(
            TEST_USER_ID_FIXED, chunk.id, False
        )
        assert result is chunk
        mock_repo.update_pinned.assert_not_called()
        assert not mock_session.add.called

    async def test_integrity_error_check_violation_maps_to_400(
        self, service, mock_repo, mock_embed, mock_session
    ):
        """race 场景兜底：前置 get_by_id 看到 category='user'，update 时
        category 被并发改成非 user → DB CHECK 抛 IntegrityError(23514) →
        service 映射 400（与前置校验同语义）。"""
        from sqlalchemy.exc import IntegrityError
        from app.application.errors.exceptions import BadRequestError

        chunk = self._user_chunk(pinned=False)
        mock_repo.get_by_id.return_value = chunk

        class _FakeCheck(Exception):
            sqlstate = "23514"

        mock_repo.update_pinned = AsyncMock(
            side_effect=IntegrityError(
                "check", params=None, orig=_FakeCheck("check violation")
            )
        )
        with pytest.raises(BadRequestError, match="CHECK"):
            await service.update_memory_pinned(
                TEST_USER_ID_FIXED, chunk.id, True
            )


# ─── reindex_memory（Option A：post-M3 hand-edit 闭环）─────────────────────

class TestReindexMemory:
    """``MemoryManagementService.reindex_memory`` Option A：只同步 body；
    其它 frontmatter 字段进 warnings 不 apply。"""

    def _make_chunk_with_fs(
        self, *, content: str = "original body", category: str = "user"
    ):
        """生成一个 fs_synced=True、category 非空的 chunk（reindex 前提条件）。"""
        import dataclasses
        return dataclasses.replace(
            _chunk(content=content),
            category=category,
            fs_synced=True,
            content_hash="hash-original",
        )

    def _make_file_store(self, *, read_return=None, read_exc=None):
        """Fake FileMemoryStore for reindex tests."""
        store = AsyncMock()
        if read_exc is not None:
            store.read = AsyncMock(side_effect=read_exc)
        else:
            store.read = AsyncMock(return_value=read_return)
        store.write = AsyncMock()
        store.delete = AsyncMock()
        return store

    def _build_svc(self, mock_repo, mock_embed, mock_session, file_store):
        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session
        return MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=file_store,
        )

    async def test_not_found_when_db_miss(
        self, mock_repo, mock_embed, mock_session
    ):
        from app.application.errors.exceptions import NotFoundError

        mock_repo.get_by_id.return_value = None
        store = self._make_file_store(read_return=({"id": "x"}, "body"))
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        with pytest.raises(NotFoundError):
            await svc.reindex_memory(TEST_USER_ID_FIXED, "missing")
        # DB miss 时不应读 fs（防止浪费 I/O + 信息泄露）
        store.read.assert_not_called()

    async def test_service_unavailable_when_file_store_is_none(
        self, mock_repo, mock_embed, mock_session
    ):
        """codex round-4 P2：file_store=None 是 deployment 配置问题（不是
        客户端请求错），应该映射到 503，不是 400。"""
        from app.application.errors.exceptions import ServiceUnavailableError

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session
        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=None,
        )
        with pytest.raises(ServiceUnavailableError, match="DB-only"):
            await svc.reindex_memory(TEST_USER_ID_FIXED, "any")

    async def test_service_unavailable_when_noop_store_raises(
        self, mock_repo, mock_embed, mock_session
    ):
        """NoopFileMemoryStore.read() 抛 NotImplementedError →
        service 映射到 503（与 file_store=None 同族语义）。"""
        from app.application.errors.exceptions import ServiceUnavailableError
        from app.domain.external.file_memory_store import NoopFileMemoryStore

        chunk = self._make_chunk_with_fs()
        mock_repo.get_by_id.return_value = chunk

        @asynccontextmanager
        async def fake_session_factory():
            yield mock_session
        svc = MemoryManagementService(
            repo_factory=lambda s: mock_repo,
            embedding_provider=mock_embed,
            session_factory=fake_session_factory,
            file_store=NoopFileMemoryStore(),
        )
        with pytest.raises(ServiceUnavailableError):
            await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

    async def test_conflict_when_legacy_null_category(
        self, mock_repo, mock_embed, mock_session
    ):
        from app.application.errors.exceptions import ConflictError

        import dataclasses
        legacy = dataclasses.replace(
            _chunk(content="legacy"), category=None, fs_synced=False
        )
        mock_repo.get_by_id.return_value = legacy
        store = self._make_file_store(read_return=({"id": "x"}, "body"))
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        with pytest.raises(ConflictError, match="legacy"):
            await svc.reindex_memory(TEST_USER_ID_FIXED, legacy.id)
        store.read.assert_not_called()

    async def test_conflict_when_file_missing(
        self, mock_repo, mock_embed, mock_session
    ):
        from app.application.errors.exceptions import ConflictError

        chunk = self._make_chunk_with_fs()
        mock_repo.get_by_id.return_value = chunk
        store = self._make_file_store(read_exc=FileNotFoundError("gone"))
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        with pytest.raises(ConflictError, match="磁盘上不存在"):
            await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

    async def test_bad_request_on_frontmatter_parse_fail(
        self, mock_repo, mock_embed, mock_session
    ):
        from app.application.errors.exceptions import BadRequestError

        chunk = self._make_chunk_with_fs()
        mock_repo.get_by_id.return_value = chunk
        store = self._make_file_store(read_exc=ValueError("bad YAML"))
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        with pytest.raises(BadRequestError, match="frontmatter 解析"):
            await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

    async def test_conflict_on_id_mismatch(
        self, mock_repo, mock_embed, mock_session
    ):
        from app.application.errors.exceptions import ConflictError

        chunk = self._make_chunk_with_fs()
        mock_repo.get_by_id.return_value = chunk
        # frontmatter.id 与 DB.id 不匹配 → hand-edit 改了 id
        store = self._make_file_store(
            read_return=({"id": "different-id-hand-edited", "source": chunk.source}, "new body")
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        with pytest.raises(ConflictError, match="id 不匹配"):
            await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)
        # id mismatch 不推进 update
        mock_repo.update_content.assert_not_called()

    async def test_noop_when_body_unchanged(
        self, mock_repo, mock_embed, mock_session
    ):
        """盘上 body 与 DB content 相同 → no-op 幂等返回，不写 DB / audit。"""
        chunk = self._make_chunk_with_fs(content="same body")
        mock_repo.get_by_id.return_value = chunk
        store = self._make_file_store(
            read_return=(
                {"id": chunk.id, "source": chunk.source, "category": chunk.category},
                "same body",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        result = await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)
        assert result.reindexed_fields == []
        assert result.fs_synced is True
        # no-op：update_content 不被调，audit 不写，commit 不发生
        mock_repo.update_content.assert_not_called()
        assert not mock_session.add.called

    async def test_happy_path_uses_reindex_content_not_update_content(
        self, mock_repo, mock_embed, mock_session
    ):
        """codex round-4 P0：body 改动必须走 ``repo.reindex_content`` 单语句
        UPDATE（fs_synced=True），**不**走 update_content（后者置 False
        会被 reconciler 当 pending 覆盖 hand-edit）。

        同时钉死：
        - update_content 绝不被调用（P0 race fix 的核心不变式）
        - 后置 _try_mark_fs_synced(True) 也不需要（reindex_content 原子置 True）
        """
        import dataclasses

        chunk = self._make_chunk_with_fs(content="original body")
        mock_repo.get_by_id.return_value = chunk
        updated_row = dataclasses.replace(
            chunk, content="edited body", content_hash="new-hash", fs_synced=True
        )
        mock_repo.reindex_content = AsyncMock(return_value=updated_row)
        mock_repo.update_content = AsyncMock()  # 不应被调
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)

        store = self._make_file_store(
            read_return=(
                {"id": chunk.id, "source": chunk.source, "category": chunk.category},
                "edited body",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        result = await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

        assert result.reindexed_fields == ["content"]
        assert result.warnings == []
        assert result.fs_synced is True

        # P0 核心不变式：reindex_content 被调，update_content **绝不**被调
        mock_repo.reindex_content.assert_awaited_once()
        mock_repo.update_content.assert_not_called()

        call_kwargs = mock_repo.reindex_content.call_args.kwargs
        assert call_kwargs["content"] == "edited body"
        assert call_kwargs["content_hash"] != chunk.content_hash
        # audit action='reindex'
        audit_obj = mock_session.add.call_args[0][0]
        assert audit_obj.action == "reindex"
        assert audit_obj.old_snapshot["content_hash"] == chunk.content_hash
        assert audit_obj.new_snapshot["content_hash"] == call_kwargs["content_hash"]
        # 核心 P0 race fix：reindex 完成后**不再**需要后置 mark_fs_synced(True)
        # 因为 reindex_content 已经原子置 True；不经过 False 窗口
        mock_repo.mark_fs_synced.assert_not_called()

    async def test_noop_with_stale_fs_synced_false_repairs_flag(
        self, mock_repo, mock_embed, mock_session
    ):
        """codex round-4 P0 no-op 分支 fix：existing.fs_synced=False 时，
        即使 body 没变（no-op）也要调 mark_fs_synced(True)——否则把 stale
        False 留给 reconciler 会引起"canonical frontmatter 覆盖 hand-edit"
        的 race 再次发生。"""
        import dataclasses

        # 盘与 DB body 相同，但 DB fs_synced=False（某次先前 update 后没翻回来）
        chunk = dataclasses.replace(
            self._make_chunk_with_fs(content="same body"), fs_synced=False
        )
        mock_repo.get_by_id.return_value = chunk
        mock_repo.mark_fs_synced = AsyncMock(return_value=True)
        store = self._make_file_store(
            read_return=(
                {"id": chunk.id, "source": chunk.source, "category": chunk.category},
                "same body",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        result = await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

        assert result.reindexed_fields == []  # body 同 → no-op
        # 但 stale fs_synced=False 必须被修：mark_fs_synced(True) 被调
        mock_repo.mark_fs_synced.assert_awaited_once()
        call_kwargs = mock_repo.mark_fs_synced.call_args.kwargs
        assert call_kwargs["synced"] is True

    async def test_noop_with_fresh_fs_synced_true_no_db_writes(
        self, mock_repo, mock_embed, mock_session
    ):
        """no-op + existing.fs_synced=True → 完全空操作，不碰 DB。"""
        chunk = self._make_chunk_with_fs(content="same body")  # fs_synced=True
        mock_repo.get_by_id.return_value = chunk
        mock_repo.mark_fs_synced = AsyncMock()
        store = self._make_file_store(
            read_return=(
                {"id": chunk.id, "source": chunk.source, "category": chunk.category},
                "same body",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        result = await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)
        assert result.reindexed_fields == []
        assert result.fs_synced is True
        # fresh True 不需要修
        mock_repo.mark_fs_synced.assert_not_called()

    async def test_warnings_for_ignored_frontmatter_fields(
        self, mock_repo, mock_embed, mock_session
    ):
        """codex round-4 P1 权威契约：warnings 覆盖所有 5 个 file-only /
        系统字段（source / created_at / auto_promoted_at / title /
        category / pinned / tags）。

        id 不在 warnings 里 —— mismatch 直接 409（见 test_conflict_on_id_mismatch）。
        """
        import dataclasses

        chunk = self._make_chunk_with_fs(content="orig body")  # source="session_flush"
        mock_repo.get_by_id.return_value = chunk
        updated_row = dataclasses.replace(
            chunk, content="edited body", content_hash="nh", fs_synced=True
        )
        mock_repo.reindex_content = AsyncMock(return_value=updated_row)

        # hand-edit 改了全套 file-only + 系统字段
        store = self._make_file_store(
            read_return=(
                {
                    "id": chunk.id,
                    "source": "manual",              # 系统字段：session_flush → manual
                    "created_at": "1970-01-01T00:00:00+00:00",  # 系统字段：改到 epoch
                    "category": "rule",              # file-only：user → rule
                    "pinned": True,                  # file-only
                    "tags": ["new-tag"],             # file-only
                    "title": "my custom title",      # file-only：vs derived
                },
                "edited body",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        result = await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

        assert result.reindexed_fields == ["content"]

        joined = " | ".join(result.warnings)
        # 系统字段
        assert "source" in joined and "manual" in joined
        assert "created_at" in joined
        # file-only 字段
        assert "category" in joined and "rule" in joined
        assert "pinned" in joined
        assert "tags" in joined
        # **title 必须被检查**（codex round-4 P1 漏检补）
        assert "title" in joined
        assert "my custom title" in joined

        # **诚实文案**（codex round-4 P1）：warnings 不能承诺虚假恢复路径
        assert "PATCH" not in joined, (
            f"warning 不该提 PATCH（PATCH 只收 content 不接受 category/pinned/tags）: {joined}"
        )
        assert "reconciler" not in joined, (
            f"warning 不该提 reconciler（walk 不写回 frontmatter 到 DB）: {joined}"
        )
        # 但要明示"留在文件侧，不进 DB/search/prompt"语义
        assert "文件" in joined or "file" in joined.lower()

        # P0 race fix 回归：走 reindex_content 不走 update_content
        mock_repo.reindex_content.assert_awaited_once()

    async def test_title_mismatch_warned_even_when_body_changed(
        self, mock_repo, mock_embed, mock_session
    ):
        """codex round-4 P1 补测：hand-edit title 改动 → warning 说 title
        仍由正文首行派生覆盖，不 apply 到 DB。"""
        import dataclasses

        chunk = self._make_chunk_with_fs(content="orig body")
        mock_repo.get_by_id.return_value = chunk
        mock_repo.reindex_content = AsyncMock(
            return_value=dataclasses.replace(
                chunk, content="new first line\nmore", fs_synced=True
            )
        )
        store = self._make_file_store(
            read_return=(
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "category": chunk.category,
                    "title": "用户自定义标题",
                },
                "new first line\nmore",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)
        result = await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

        joined = " | ".join(result.warnings)
        assert "title" in joined
        assert "用户自定义标题" in joined
        # 提示系统真实行为：title 来自 derive_title(body) 即 "new first line"
        assert "new first line" in joined

    async def test_empty_body_rejected_with_400(
        self, mock_repo, mock_embed, mock_session
    ):
        """codex round-4 P2：hand-edit 删光正文 → 400，与 create/update 契约一致。

        否则 hand-edit 成为唯一绕过 "content must not be empty" 不变式的
        入口，contract 漂移。
        """
        from app.application.errors.exceptions import BadRequestError

        chunk = self._make_chunk_with_fs(content="orig")
        mock_repo.get_by_id.return_value = chunk
        # 盘上 body 被清空（只留空白）
        store = self._make_file_store(
            read_return=(
                {"id": chunk.id, "source": chunk.source, "category": chunk.category},
                "   \n\n  ",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        with pytest.raises(BadRequestError, match="body 为空"):
            await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

    async def test_embedding_failure_degrades_to_none(
        self, mock_repo, mock_embed, mock_session
    ):
        """embedding provider 故障 → 写 None embedding 继续 UPDATE。"""
        import dataclasses
        from app.domain.external.embedding_provider import EmbeddingUnavailableError

        chunk = self._make_chunk_with_fs(content="orig body")
        mock_repo.get_by_id.return_value = chunk
        mock_repo.reindex_content = AsyncMock(
            return_value=dataclasses.replace(
                chunk, content="new body", content_hash="nh", fs_synced=True
            )
        )
        mock_embed.embed = AsyncMock(
            side_effect=EmbeddingUnavailableError("circuit open")
        )

        store = self._make_file_store(
            read_return=(
                {"id": chunk.id, "source": chunk.source, "category": chunk.category},
                "new body",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        result = await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)

        assert result.reindexed_fields == ["content"]
        # embedding=None 被传给 reindex_content（冷数据写入）
        assert mock_repo.reindex_content.call_args.kwargs["embedding"] is None

    async def test_trailing_newline_does_not_trigger_reindex(
        self, mock_repo, mock_embed, mock_session
    ):
        """fs 文件末尾 ``\\n``（POSIX 规范）vs DB content（无 trailing \\n）
        应当视作等价，避免每次 hand-edit 跑一下都以为改了。"""
        chunk = self._make_chunk_with_fs(content="body without newline")
        mock_repo.get_by_id.return_value = chunk
        mock_repo.reindex_content = AsyncMock()  # 不应被调
        # 盘上 body 多一个尾 \n
        store = self._make_file_store(
            read_return=(
                {"id": chunk.id, "source": chunk.source, "category": chunk.category},
                "body without newline\n",
            )
        )
        svc = self._build_svc(mock_repo, mock_embed, mock_session, store)

        result = await svc.reindex_memory(TEST_USER_ID_FIXED, chunk.id)
        assert result.reindexed_fields == []  # no-op
        mock_repo.reindex_content.assert_not_called()
