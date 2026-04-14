"""Integration tests for memory_chunks table.

Requires: PostgreSQL with pgvector extension (pgvector/pgvector:pg17 image).
Run: cd api && uv run -- python -m pytest tests/integration/test_memory_chunks_migration.py -v
"""

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from app.infrastructure.models.memory_chunk_orm import MEMORY_EMBEDDING_DIM, MemoryChunkModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def sample_chunk() -> MemoryChunkModel:
    return MemoryChunkModel(
        id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        content="The user prefers dark mode.",
        content_hash="a1b2c3d4" * 8,
        source="session_flush",
        metadata_={"turn": 5},
    )


class TestMemoryChunksTable:
    async def test_vector_extension_exists(self, db_session):
        result = await db_session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )
        assert result.scalar() == 1

    async def test_insert_and_read(self, db_session, sample_chunk):
        db_session.add(sample_chunk)
        await db_session.flush()

        result = await db_session.get(MemoryChunkModel, sample_chunk.id)
        assert result is not None
        assert result.content == "The user prefers dark mode."
        assert result.source == "session_flush"
        assert result.metadata_ == {"turn": 5}

    async def test_insert_with_embedding(self, db_session):
        """Verify vector column accepts correct-dimension embeddings."""
        embedding = [0.1] * MEMORY_EMBEDDING_DIM  # 512-dim vector
        chunk = MemoryChunkModel(
            id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            content="embedding test",
            content_hash="e1e2e3e4" * 8,
            source="session_flush",
            embedding=embedding,
        )
        db_session.add(chunk)
        await db_session.flush()

        result = await db_session.get(MemoryChunkModel, chunk.id)
        assert result is not None
        assert result.embedding is not None
        assert len(result.embedding) == MEMORY_EMBEDDING_DIM

    async def test_embedding_wrong_dimension_rejected(self, db_session):
        """Vector column rejects embeddings with wrong dimension."""
        wrong_dim_embedding = [0.1] * 256  # wrong: 256 instead of 512
        chunk = MemoryChunkModel(
            id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            content="wrong dim test",
            content_hash="f1f2f3f4" * 8,
            source="session_flush",
            embedding=wrong_dim_embedding,
        )
        db_session.add(chunk)
        with pytest.raises(sa.exc.DBAPIError):
            await db_session.flush()

    async def test_cosine_distance_query(self, db_session):
        """Verify HNSW index supports cosine distance queries."""
        user_id = str(uuid.uuid4())
        # Insert two chunks with different embeddings
        for i, hash_char in enumerate(["a", "b"]):
            emb = [0.0] * MEMORY_EMBEDDING_DIM
            emb[i] = 1.0  # orthogonal vectors
            chunk = MemoryChunkModel(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=f"vector test {i}",
                content_hash=hash_char * 64,
                source="session_flush",
                embedding=emb,
            )
            db_session.add(chunk)
        await db_session.flush()

        # Query with cosine distance
        query_vec = [0.0] * MEMORY_EMBEDDING_DIM
        query_vec[0] = 1.0
        result = await db_session.execute(
            text(
                "SELECT content FROM memory_chunks "
                "WHERE user_id = :uid "
                "ORDER BY embedding <=> CAST(:qvec AS vector) "
                "LIMIT 1"
            ),
            {"uid": user_id, "qvec": str(query_vec)},
        )
        closest = result.scalar()
        assert closest == "vector test 0"  # first vector is closest

    async def test_insert_without_session_id(self, db_session):
        """Manual memories have no session."""
        chunk = MemoryChunkModel(
            id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            session_id=None,
            content="User lives in Shanghai.",
            content_hash="d4c3b2a1" * 8,
            source="manual",
        )
        db_session.add(chunk)
        await db_session.flush()

        result = await db_session.get(MemoryChunkModel, chunk.id)
        assert result is not None
        assert result.session_id is None

    async def test_dedup_constraint(self, db_session, sample_chunk):
        """Same user + content_hash should be rejected."""
        db_session.add(sample_chunk)
        await db_session.flush()

        duplicate = MemoryChunkModel(
            id=str(uuid.uuid4()),
            user_id=sample_chunk.user_id,
            content="different content, same hash",
            content_hash=sample_chunk.content_hash,
            source="session_flush",
        )
        db_session.add(duplicate)
        with pytest.raises(sa.exc.IntegrityError):
            await db_session.flush()

    async def test_different_user_same_hash_ok(self, db_session, sample_chunk):
        """Different user with same content_hash should succeed."""
        db_session.add(sample_chunk)
        await db_session.flush()

        other_user_chunk = MemoryChunkModel(
            id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),  # different user
            content=sample_chunk.content,
            content_hash=sample_chunk.content_hash,
            source="session_flush",
        )
        db_session.add(other_user_chunk)
        await db_session.flush()  # should succeed

    async def test_hnsw_index_exists(self, db_session):
        result = await db_session.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'memory_chunks' AND indexname = 'idx_memory_embedding_hnsw'"
            )
        )
        assert result.scalar() == "idx_memory_embedding_hnsw"
