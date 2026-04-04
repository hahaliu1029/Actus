"""MemoryConfig embedding fields tests."""

import pytest
from pydantic import ValidationError

from app.domain.models.app_config import MemoryConfig


class TestMemoryConfigEmbeddingFields:
    def test_defaults_match_schema(self):
        """Default embedding_dim must match DB schema (MEMORY_EMBEDDING_DIM=512).
        Changing this default without a migration will cause dimension mismatch."""
        cfg = MemoryConfig()
        assert cfg.embedding_dim == 512
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_embedding_dim_must_be_positive(self):
        """embedding_dim must be >= 1."""
        with pytest.raises(ValidationError):
            MemoryConfig(embedding_dim=0)

    def test_embedding_model_overridable(self):
        cfg = MemoryConfig(embedding_model="custom-model")
        assert cfg.embedding_model == "custom-model"

    def test_existing_fields_unchanged(self):
        """Existing flush fields still work."""
        cfg = MemoryConfig(flush_enabled=True, flush_min_steps=3)
        assert cfg.flush_enabled is True
        assert cfg.flush_min_steps == 3
