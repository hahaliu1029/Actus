"""Verify memory embedding provider is constructed correctly during lifespan.

Drives real lifespan(app), patches external deps (DB, Redis, MinIO, Alembic, AgentService).
Pattern reference: test_lifespan_checkpointer_pool.py.
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.external.embedding_provider import (
    DisabledEmbeddingProvider,
    EmbeddingUnavailableError,
)
from app.infrastructure.external.embedding.circuit_breaker_embedding_provider import (
    CircuitBreakerEmbeddingProvider,
)
from app.infrastructure.models.memory_chunk_orm import MEMORY_EMBEDDING_DIM
from app.main import app, lifespan


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _enter_base_patches(stack: ExitStack) -> None:
    """Enter all external dep patches into the given ExitStack."""
    stack.enter_context(
        patch("app.main.get_redis", return_value=MagicMock(init=AsyncMock(), shutdown=AsyncMock()))
    )
    stack.enter_context(
        patch("app.main.get_postgres", return_value=MagicMock(
            init=AsyncMock(), shutdown=AsyncMock(), session_factory=MagicMock(),
        ))
    )
    stack.enter_context(
        patch("app.main.get_minio", return_value=MagicMock(init=AsyncMock(), shutdown=AsyncMock()))
    )
    stack.enter_context(patch("app.main.command"))  # skip Alembic migrations
    stack.enter_context(
        patch("app.main.get_agent_service", return_value=MagicMock(shutdown=AsyncMock()))
    )
    stack.enter_context(
        patch("app.infrastructure.checkpointer_pool.CheckpointerPool", return_value=MagicMock(
            open=AsyncMock(), close=AsyncMock(), pool=MagicMock(),
        ))
    )


def _enter_config_patch(
    stack: ExitStack,
    embedding_enabled: bool = False,
    api_base: str = "",
    api_key: str = "",
    dim: int = MEMORY_EMBEDDING_DIM,
) -> None:
    """Patch _load_app_config with specified memory embedding fields."""
    mock_cfg = MagicMock()
    mock_cfg.agent_config.memory.embedding_enabled = embedding_enabled
    mock_cfg.agent_config.memory.embedding_api_base = api_base
    mock_cfg.agent_config.memory.embedding_api_key = api_key
    mock_cfg.agent_config.memory.embedding_dim = dim
    mock_cfg.agent_config.memory.embedding_model = "text-embedding-3-small"
    mock_cfg.agent_config.memory.embedding_circuit_breaker_threshold = 3
    mock_cfg.agent_config.memory.embedding_circuit_breaker_recovery_seconds = 300.0
    mock_cfg.agent_config.memory.flush_enabled = True  # C5.1 不变量：flush_service 始终构建
    mock_cfg.agent_config.memory.flush_max_retries = 3
    mock_cfg.agent_config.memory.flush_circuit_breaker_threshold = 3
    stack.enter_context(
        patch("app.interfaces.service_dependencies._load_app_config", return_value=mock_cfg)
    )


class TestMemoryEmbeddingStartup:

    @pytest.mark.anyio
    async def test_disabled_sets_disabled_provider(self) -> None:
        with ExitStack() as stack:
            _enter_base_patches(stack)
            _enter_config_patch(stack, embedding_enabled=False)
            async with lifespan(app):
                provider = app.state.memory_embedding_provider
                assert isinstance(provider, DisabledEmbeddingProvider)

    @pytest.mark.anyio
    async def test_disabled_provider_raises_on_embed(self) -> None:
        with ExitStack() as stack:
            _enter_base_patches(stack)
            _enter_config_patch(stack, embedding_enabled=False)
            async with lifespan(app):
                provider = app.state.memory_embedding_provider
                with pytest.raises(EmbeddingUnavailableError, match="disabled"):
                    await provider.embed(["test"])

    @pytest.mark.anyio
    async def test_enabled_constructs_circuit_breaker_with_correct_dim(self) -> None:
        with ExitStack() as stack:
            _enter_base_patches(stack)
            _enter_config_patch(
                stack, embedding_enabled=True,
                api_base="https://api.openai.com/v1", api_key="sk-test",
                dim=MEMORY_EMBEDDING_DIM,
            )
            mock_oai = stack.enter_context(
                patch(
                    "app.infrastructure.external.embedding.openai_embedding_provider.OpenAIEmbeddingProvider",
                    return_value=MagicMock(dimensions=MEMORY_EMBEDDING_DIM),
                )
            )
            async with lifespan(app):
                provider = app.state.memory_embedding_provider
                assert isinstance(provider, CircuitBreakerEmbeddingProvider)
                # C4 核心约束：dimensions 必须显式传入
                mock_oai.assert_called_once_with(
                    api_base="https://api.openai.com/v1",
                    api_key="sk-test",
                    model="text-embedding-3-small",
                    dimensions=MEMORY_EMBEDDING_DIM,
                )

    @pytest.mark.anyio
    async def test_dim_mismatch_raises(self) -> None:
        with ExitStack() as stack:
            _enter_base_patches(stack)
            _enter_config_patch(
                stack, embedding_enabled=True,
                api_base="https://api.openai.com/v1", api_key="sk-test",
                dim=256,  # mismatch with MEMORY_EMBEDDING_DIM=512
            )
            with pytest.raises(RuntimeError, match="MEMORY_EMBEDDING_DIM"):
                async with lifespan(app):
                    pass

    @pytest.mark.anyio
    async def test_empty_api_base_raises(self) -> None:
        with ExitStack() as stack:
            _enter_base_patches(stack)
            _enter_config_patch(
                stack, embedding_enabled=True,
                api_base="", api_key="sk-test",
            )
            with pytest.raises(RuntimeError, match="embedding_api_base"):
                async with lifespan(app):
                    pass

    @pytest.mark.anyio
    async def test_empty_api_key_raises(self) -> None:
        with ExitStack() as stack:
            _enter_base_patches(stack)
            _enter_config_patch(
                stack, embedding_enabled=True,
                api_base="https://api.openai.com/v1", api_key="",
            )
            with pytest.raises(RuntimeError, match="embedding_api_key"):
                async with lifespan(app):
                    pass

    @pytest.mark.anyio
    async def test_flush_enabled_guarantees_flush_service(self) -> None:
        """System invariant: flush_enabled=True → app.state.flush_service is not None.

        Guards against cursor optimistic-advance causing silent batch loss
        when flush_service is accidentally absent.
        _enter_config_patch sets flush_enabled=True by default.
        """
        with ExitStack() as stack:
            _enter_base_patches(stack)
            _enter_config_patch(stack, embedding_enabled=False)
            async with lifespan(app):
                from app.application.services.memory_flush_service import MemoryFlushService
                flush_svc = app.state.flush_service
                assert flush_svc is not None
                assert isinstance(flush_svc, MemoryFlushService)
