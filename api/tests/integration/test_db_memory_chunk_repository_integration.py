"""Integration tests for DBMemoryChunkRepository — real PostgreSQL + pgvector.

Requires: SQLALCHEMY_DATABASE_URL env var pointing to pgvector-enabled PostgreSQL.
Run: cd api && uv run -- python -m pytest tests/integration/test_db_memory_chunk_repository_integration.py -v

FK constraints: memory_chunks.user_id → users.id (CASCADE),
memory_chunks.session_id → sessions.id (SET NULL). Both are IMMEDIATE.
每个用例必须先内联创建 user/session 父记录，再写 chunk。
用 text() 原始 SQL 避免依赖 User/Session ORM 模型。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from app.domain.models.memory_chunk import MemoryChunk
from app.infrastructure.models.memory_chunk_orm import MEMORY_EMBEDDING_DIM
from app.infrastructure.repositories.db_memory_chunk_repository import DBMemoryChunkRepository

pytestmark = pytest.mark.anyio

# ---- Helpers ----


async def _ensure_user(db_session, user_id: str) -> None:
    """Insert a minimal user row to satisfy FK. Idempotent (ON CONFLICT DO NOTHING)."""
    await db_session.execute(
        text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
        {"uid": user_id},
    )


async def _ensure_session(db_session, session_id: str) -> None:
    """Insert a minimal session row to satisfy FK. Idempotent."""
    await db_session.execute(
        text("INSERT INTO sessions (id) VALUES (:sid) ON CONFLICT DO NOTHING"),
        {"sid": session_id},
    )


def _make_chunk(
    *,
    content: str = "test content",
    user_id: str = "user-integ-1",
    session_id: str | None = "sess-integ-1",
    embedding: tuple[float, ...] | None = None,
    content_hash: str | None = None,
) -> MemoryChunk:
    return MemoryChunk(
        id=str(uuid.uuid4()),
        user_id=user_id,
        session_id=session_id,
        content=content,
        content_hash=content_hash or f"hash_{uuid.uuid4().hex[:16]}",
        source="session_flush",
        metadata={"test": True},
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
        embedding=embedding,
    )


# Pre-computed normalized vectors for deterministic distance tests.
# cosine_distance(A, A) = 0.0 (identical), cosine_distance(A, C) ≈ 2.0 (opposite)
_VEC_A = tuple([1.0 / (MEMORY_EMBEDDING_DIM ** 0.5)] * MEMORY_EMBEDDING_DIM)
_VEC_C = tuple([-v for v in _VEC_A])  # opposite to A → distance ≈ 2

_SEARCH_UID = "user-search-test"
_SEARCH_SID = "sess-search-test"


# ---- batch_insert_ignore tests ----

class TestBatchInsertIgnore:

    async def test_insert_returns_count(self, db_session) -> None:
        """Scenario 1: Normal insert returns correct count."""
        await _ensure_user(db_session, "user-integ-1")
        await _ensure_session(db_session, "sess-integ-1")
        repo = DBMemoryChunkRepository(db_session)
        chunks = [_make_chunk(content=f"chunk {i}") for i in range(3)]

        result = await repo.batch_insert_ignore(chunks)
        await db_session.flush()

        assert result == 3

    async def test_duplicate_is_idempotent(self, db_session) -> None:
        """Scenario 2: Re-inserting same (user_id, content_hash) returns 0."""
        await _ensure_user(db_session, "user-integ-1")
        await _ensure_session(db_session, "sess-integ-1")
        repo = DBMemoryChunkRepository(db_session)
        chunk = _make_chunk(content_hash="fixed_hash_for_dedup")

        first = await repo.batch_insert_ignore([chunk])
        await db_session.flush()
        assert first == 1

        # Same user_id + content_hash, different id
        dup = _make_chunk(content_hash="fixed_hash_for_dedup", content="different text")
        second = await repo.batch_insert_ignore([dup])
        await db_session.flush()
        assert second == 0

    async def test_insert_with_none_embedding(self, db_session) -> None:
        """Scenario 3: embedding=None chunk (cold data) can be inserted."""
        await _ensure_user(db_session, "user-integ-1")
        await _ensure_session(db_session, "sess-integ-1")
        repo = DBMemoryChunkRepository(db_session)
        chunk = _make_chunk(embedding=None)

        result = await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        assert result == 1


# ---- search_by_vector tests ----

class TestSearchByVector:

    async def _seed(self, db_session) -> DBMemoryChunkRepository:
        """Create parent records + insert 3 chunks: near, far, cold."""
        await _ensure_user(db_session, _SEARCH_UID)
        await _ensure_session(db_session, _SEARCH_SID)
        repo = DBMemoryChunkRepository(db_session)
        chunks = [
            _make_chunk(user_id=_SEARCH_UID, session_id=_SEARCH_SID,
                        content="near", embedding=_VEC_A, content_hash="near_hash"),
            _make_chunk(user_id=_SEARCH_UID, session_id=_SEARCH_SID,
                        content="far", embedding=_VEC_C, content_hash="far_hash"),
            _make_chunk(user_id=_SEARCH_UID, session_id=_SEARCH_SID,
                        content="cold", embedding=None, content_hash="cold_hash"),
        ]
        await repo.batch_insert_ignore(chunks)
        await db_session.flush()
        return repo

    async def test_basic_search_returns_results(self, db_session) -> None:
        """Scenario 4: Search returns matching results ordered by distance."""
        # Seed with near (identical) + mid (slightly different) + cold (no embedding)
        await _ensure_user(db_session, _SEARCH_UID)
        await _ensure_session(db_session, _SEARCH_SID)
        repo = DBMemoryChunkRepository(db_session)

        # _VEC_A and _VEC_MID both have positive cosine similarity with _VEC_A,
        # so both pass threshold=0.0 (max_distance=1.0).
        # _VEC_MID: shift first element to create a slightly different vector.
        vec_mid = list(_VEC_A)
        vec_mid[0] = 0.0  # reduce similarity but keep it > 0
        vec_mid_t = tuple(vec_mid)

        chunks = [
            _make_chunk(user_id=_SEARCH_UID, session_id=_SEARCH_SID,
                        content="near", embedding=_VEC_A, content_hash="basic_near"),
            _make_chunk(user_id=_SEARCH_UID, session_id=_SEARCH_SID,
                        content="mid", embedding=vec_mid_t, content_hash="basic_mid"),
            _make_chunk(user_id=_SEARCH_UID, session_id=_SEARCH_SID,
                        content="cold", embedding=None, content_hash="basic_cold"),
        ]
        await repo.batch_insert_ignore(chunks)
        await db_session.flush()

        results = await repo.search_by_vector(
            user_id=_SEARCH_UID,
            embedding=list(_VEC_A),
            top_k=10,
            threshold=0.0,
        )

        assert len(results) >= 2
        assert all(isinstance(r, MemoryChunk) for r in results)
        # "near" (distance ≈ 0) should come before "mid" (distance > 0)
        contents = [r.content for r in results]
        assert contents.index("near") < contents.index("mid")

    async def test_cold_data_excluded(self, db_session) -> None:
        """Scenario 5: embedding=None chunks don't appear in results."""
        repo = await self._seed(db_session)

        results = await repo.search_by_vector(
            user_id=_SEARCH_UID,
            embedding=list(_VEC_A),
            top_k=10,
            threshold=0.0,
        )

        contents = [r.content for r in results]
        assert "cold" not in contents

    async def test_threshold_filters(self, db_session) -> None:
        """Scenario 6: High threshold filters out distant vectors."""
        repo = await self._seed(db_session)

        # threshold=0.99 → max_distance=0.01, only near-identical vectors pass
        results = await repo.search_by_vector(
            user_id=_SEARCH_UID,
            embedding=list(_VEC_A),
            top_k=10,
            threshold=0.99,
        )

        contents = [r.content for r in results]
        assert "near" in contents
        assert "far" not in contents

    async def test_top_k_limits_results(self, db_session) -> None:
        """Scenario 7: top_k caps the number of returned results."""
        uid = f"user-topk-{uuid.uuid4().hex[:8]}"
        await _ensure_user(db_session, uid)
        repo = DBMemoryChunkRepository(db_session)
        chunks = [
            _make_chunk(user_id=uid, session_id=None,
                        content=f"item_{i}", embedding=_VEC_A, content_hash=f"topk_{i}")
            for i in range(5)
        ]
        await repo.batch_insert_ignore(chunks)
        await db_session.flush()

        results = await repo.search_by_vector(
            user_id=uid,
            embedding=list(_VEC_A),
            top_k=2,
            threshold=0.0,
        )

        assert len(results) <= 2

    async def test_user_id_isolation(self, db_session) -> None:
        """Scenario 8: Different users can't see each other's chunks."""
        other_user = f"user-other-{uuid.uuid4().hex[:8]}"
        await _ensure_user(db_session, other_user)
        await _ensure_user(db_session, _SEARCH_UID)
        repo = DBMemoryChunkRepository(db_session)

        chunk = _make_chunk(
            user_id=other_user, session_id=None,
            embedding=_VEC_A, content_hash="isolation_hash",
        )
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        results = await repo.search_by_vector(
            user_id=_SEARCH_UID,
            embedding=list(_VEC_A),
            top_k=10,
            threshold=0.0,
        )

        user_ids = {r.user_id for r in results}
        assert other_user not in user_ids

    async def test_nonexistent_user_returns_empty(self, db_session) -> None:
        """Scenario 11: Search for user with no chunks returns empty list."""
        repo = DBMemoryChunkRepository(db_session)

        results = await repo.search_by_vector(
            user_id="user-does-not-exist",
            embedding=list(_VEC_A),
            top_k=10,
            threshold=0.0,
        )

        assert results == []


# ---- delete_by_session tests ----

class TestDeleteBySession:

    async def test_delete_returns_count(self, db_session) -> None:
        """Scenario 9: Delete existing chunks returns correct count."""
        uid = f"user-del-{uuid.uuid4().hex[:8]}"
        target_session = f"sess-del-{uuid.uuid4().hex[:8]}"
        await _ensure_user(db_session, uid)
        await _ensure_session(db_session, target_session)
        repo = DBMemoryChunkRepository(db_session)

        chunks = [
            _make_chunk(user_id=uid, session_id=target_session,
                        content=f"del_{i}", content_hash=f"del_{i}_{uuid.uuid4().hex[:8]}")
            for i in range(3)
        ]
        await repo.batch_insert_ignore(chunks)
        await db_session.flush()

        result = await repo.delete_by_session(target_session)
        assert result == 3

    async def test_delete_nonexistent_returns_zero(self, db_session) -> None:
        """Scenario 10: Delete non-existent session returns 0."""
        repo = DBMemoryChunkRepository(db_session)

        result = await repo.delete_by_session("sess-does-not-exist")
        assert result == 0


# ---- get_by_id tests (C6) ----

class TestGetById:

    async def test_found_returns_chunk(self, db_session) -> None:
        """get_by_id with correct user_id returns the chunk."""
        uid = f"user-getid-{uuid.uuid4().hex[:8]}"
        await _ensure_user(db_session, uid)
        repo = DBMemoryChunkRepository(db_session)
        chunk = _make_chunk(user_id=uid, session_id=None, embedding=None)
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        result = await repo.get_by_id(chunk.id, user_id=uid)
        assert result is not None
        assert result.id == chunk.id
        assert result.user_id == uid

    async def test_wrong_user_returns_none(self, db_session) -> None:
        """get_by_id with different user_id returns None (tenant isolation)."""
        uid = f"user-owner-{uuid.uuid4().hex[:8]}"
        other = f"user-other-{uuid.uuid4().hex[:8]}"
        await _ensure_user(db_session, uid)
        await _ensure_user(db_session, other)
        repo = DBMemoryChunkRepository(db_session)
        chunk = _make_chunk(user_id=uid, session_id=None, embedding=None)
        await repo.batch_insert_ignore([chunk])
        await db_session.flush()

        result = await repo.get_by_id(chunk.id, user_id=other)
        assert result is None

    async def test_nonexistent_id_returns_none(self, db_session) -> None:
        repo = DBMemoryChunkRepository(db_session)
        result = await repo.get_by_id("does-not-exist", user_id="any")
        assert result is None
