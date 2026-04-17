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
    # delete_all_by_user 返回 RETURNING source 聚合后的分布
    repo.delete_all_by_user = AsyncMock(return_value={})
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
        # delete_all_by_user 返回 RETURNING source 聚合后的分布；总数 = sum(values)
        mock_repo.delete_all_by_user.return_value = {
            "session_flush": 7,
            "file": 3,
        }
        count = await service.delete_all_memories(TEST_USER_ID_FIXED)
        assert count == 10

    async def test_empty_returns_zero_and_no_audit(
        self, service, mock_repo, mock_session
    ):
        mock_repo.delete_all_by_user.return_value = {}
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
        来自同一条 DELETE ... RETURNING source 语句，避免多次查询的竞态。"""
        mock_repo.delete_all_by_user.return_value = {
            "session_flush": 42,
            "file": 8,
        }

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
