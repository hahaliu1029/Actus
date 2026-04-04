"""MemoryChunkModel unit tests (no DB required)."""

import uuid

from app.infrastructure.models.memory_chunk_orm import MEMORY_EMBEDDING_DIM, MemoryChunkModel


class TestMemoryChunkModel:
    def test_tablename(self):
        assert MemoryChunkModel.__tablename__ == "memory_chunks"

    def test_embedding_dim_constant(self):
        assert MEMORY_EMBEDDING_DIM == 512

    def test_create_instance(self):
        chunk_id = str(uuid.uuid4())
        model = MemoryChunkModel(
            id=chunk_id,
            user_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            content="test content",
            content_hash="a" * 64,
            source="session_flush",
        )
        assert model.id == chunk_id
        assert model.source == "session_flush"

    def test_nullable_session_id(self):
        """Manual memories have no session."""
        model = MemoryChunkModel(
            id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            session_id=None,
            content="manual memory",
            content_hash="b" * 64,
            source="manual",
        )
        assert model.session_id is None

    def test_metadata_explicit(self):
        """Python attribute is metadata_, DB column is metadata."""
        model = MemoryChunkModel(
            id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            content="test",
            content_hash="c" * 64,
            metadata_={"key": "value"},
        )
        assert model.metadata_ == {"key": "value"}

    def test_metadata_python_default(self):
        """metadata_ defaults to empty dict on Python side (before flush)."""
        model = MemoryChunkModel(
            id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            content="test",
            content_hash="d" * 64,
        )
        assert model.metadata_ == {}
