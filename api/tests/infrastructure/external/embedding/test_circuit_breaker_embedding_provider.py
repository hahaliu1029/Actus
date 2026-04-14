"""Unit tests for CircuitBreakerEmbeddingProvider."""
from __future__ import annotations

from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from app.domain.external.embedding_provider import (
    EmbeddingUnavailableError,
)
from app.infrastructure.external.embedding.circuit_breaker_embedding_provider import (
    CircuitBreakerEmbeddingProvider,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_inner(embed_result: list[list[float]] | None = None) -> AsyncMock:
    inner = AsyncMock()
    inner.embed.return_value = embed_result or [[0.1, 0.2]]
    type(inner).dimensions = PropertyMock(return_value=512)
    type(inner).model_name = PropertyMock(return_value="test-model")
    return inner


class TestClosedState:
    async def test_success_passes_through(self) -> None:
        inner = _make_inner([[1.0, 2.0]])
        cb = CircuitBreakerEmbeddingProvider(inner, threshold=3)

        result = await cb.embed(["hello"])

        assert result == [[1.0, 2.0]]
        inner.embed.assert_awaited_once_with(["hello"])

    async def test_success_resets_failure_count(self) -> None:
        inner = _make_inner()
        cb = CircuitBreakerEmbeddingProvider(inner, threshold=3)

        # Fail twice (below threshold)
        inner.embed.side_effect = RuntimeError("api error")
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.embed(["x"])

        # Succeed → reset
        inner.embed.side_effect = None
        inner.embed.return_value = [[0.1]]
        await cb.embed(["x"])

        # Fail twice more → should NOT open (counter was reset)
        inner.embed.side_effect = RuntimeError("api error")
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.embed(["x"])

        # Still closed (2 < 3)
        inner.embed.side_effect = None
        inner.embed.return_value = [[0.2]]
        result = await cb.embed(["x"])
        assert result == [[0.2]]


class TestOpenState:
    async def test_opens_after_threshold_failures(self) -> None:
        inner = _make_inner()
        inner.embed.side_effect = RuntimeError("api error")
        cb = CircuitBreakerEmbeddingProvider(inner, threshold=3)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.embed(["x"])

        # 4th call → breaker open
        with pytest.raises(EmbeddingUnavailableError, match="circuit breaker open"):
            await cb.embed(["x"])

    async def test_embedding_unavailable_error_not_counted(self) -> None:
        """Inner raising EmbeddingUnavailableError should not increment counter."""
        inner = _make_inner()
        inner.embed.side_effect = EmbeddingUnavailableError("inner breaker")
        cb = CircuitBreakerEmbeddingProvider(inner, threshold=3)

        for _ in range(5):
            with pytest.raises(EmbeddingUnavailableError, match="inner breaker"):
                await cb.embed(["x"])

        # Counter should still be 0
        assert cb._consecutive_failures == 0


class TestRecovery:
    async def test_recovers_after_timeout(self) -> None:
        inner = _make_inner()
        inner.embed.side_effect = RuntimeError("api error")
        cb = CircuitBreakerEmbeddingProvider(inner, threshold=2, recovery_seconds=10.0)

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.embed(["x"])

        # Breaker is open
        with pytest.raises(EmbeddingUnavailableError):
            await cb.embed(["x"])

        # Fast-forward past recovery window
        with patch("app.infrastructure.external.embedding.circuit_breaker_embedding_provider.time") as mock_time:
            mock_time.monotonic.return_value = cb._last_failure_time + 11.0
            inner.embed.side_effect = None
            inner.embed.return_value = [[0.5]]

            result = await cb.embed(["x"])
            assert result == [[0.5]]
            assert cb._consecutive_failures == 0

    async def test_recovery_attempt_fails_reopens(self) -> None:
        inner = _make_inner()
        inner.embed.side_effect = RuntimeError("api error")
        cb = CircuitBreakerEmbeddingProvider(inner, threshold=2, recovery_seconds=10.0)

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.embed(["x"])

        # Fast-forward past recovery, but inner still fails
        with patch("app.infrastructure.external.embedding.circuit_breaker_embedding_provider.time") as mock_time:
            mock_time.monotonic.return_value = cb._last_failure_time + 11.0

            with pytest.raises(RuntimeError):
                await cb.embed(["x"])

            # Counter is now 1 (reset happened, then failed once)
            assert cb._consecutive_failures == 1


class TestDisabledProviderIntegration:
    async def test_inner_disabled_provider_propagates_unavailable(self) -> None:
        """Wrapping DisabledEmbeddingProvider should propagate EmbeddingUnavailableError
        without incrementing the failure counter."""
        from app.domain.external.embedding_provider import DisabledEmbeddingProvider

        cb = CircuitBreakerEmbeddingProvider(DisabledEmbeddingProvider(), threshold=3)
        with pytest.raises(EmbeddingUnavailableError, match="disabled"):
            await cb.embed(["x"])
        assert cb._consecutive_failures == 0


class TestPropertyDelegation:
    def test_dimensions_proxied(self) -> None:
        inner = _make_inner()
        cb = CircuitBreakerEmbeddingProvider(inner, threshold=3)
        assert cb.dimensions == 512

    def test_model_name_proxied(self) -> None:
        inner = _make_inner()
        cb = CircuitBreakerEmbeddingProvider(inner, threshold=3)
        assert cb.model_name == "test-model"
