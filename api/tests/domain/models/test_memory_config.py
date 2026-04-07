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


class TestMemoryConfigC4Fields:
    """C4 新增字段：embedding 连接 + circuit breaker 配置。"""

    def test_embedding_enabled_defaults_false(self) -> None:
        cfg = MemoryConfig()
        assert cfg.embedding_enabled is False

    def test_embedding_api_base_defaults_empty(self) -> None:
        cfg = MemoryConfig()
        assert cfg.embedding_api_base == ""

    def test_embedding_api_key_defaults_empty(self) -> None:
        cfg = MemoryConfig()
        assert cfg.embedding_api_key == ""

    def test_circuit_breaker_threshold_defaults_3(self) -> None:
        cfg = MemoryConfig()
        assert cfg.embedding_circuit_breaker_threshold == 3

    def test_circuit_breaker_threshold_min_1(self) -> None:
        with pytest.raises(ValidationError):
            MemoryConfig(embedding_circuit_breaker_threshold=0)

    def test_circuit_breaker_recovery_defaults_300(self) -> None:
        cfg = MemoryConfig()
        assert cfg.embedding_circuit_breaker_recovery_seconds == 300.0

    def test_circuit_breaker_recovery_min_10(self) -> None:
        with pytest.raises(ValidationError):
            MemoryConfig(embedding_circuit_breaker_recovery_seconds=5.0)

    def test_all_fields_overridable(self) -> None:
        cfg = MemoryConfig(
            embedding_enabled=True,
            embedding_api_base="https://api.example.com",
            embedding_api_key="sk-test",
            embedding_circuit_breaker_threshold=5,
            embedding_circuit_breaker_recovery_seconds=60.0,
        )
        assert cfg.embedding_enabled is True
        assert cfg.embedding_api_base == "https://api.example.com"
        assert cfg.embedding_api_key == "sk-test"
        assert cfg.embedding_circuit_breaker_threshold == 5
        assert cfg.embedding_circuit_breaker_recovery_seconds == 60.0
