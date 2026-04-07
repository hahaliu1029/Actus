from __future__ import annotations

import time

from app.domain.external.embedding_provider import (
    EmbeddingProvider,
    EmbeddingUnavailableError,
)


class CircuitBreakerEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider 容错 wrapper。

    连续失败 threshold 次后 breaker open，embed() 直接抛
    EmbeddingUnavailableError。recovery_seconds 后自动尝试恢复。
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        threshold: int = 3,
        recovery_seconds: float = 300.0,
    ) -> None:
        self._inner = inner
        self._threshold = threshold
        self._recovery_seconds = recovery_seconds
        self._consecutive_failures = 0
        self._last_failure_time: float | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._is_open():
            raise EmbeddingUnavailableError(
                f"circuit breaker open (failures={self._consecutive_failures})"
            )
        try:
            result = await self._inner.embed(texts)
            self._reset()
            return result
        except EmbeddingUnavailableError:
            raise  # 不重复计数内层 breaker 的异常
        except Exception:
            self._record_failure()
            raise

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def _is_open(self) -> bool:
        """Check if breaker is open. Side effect: resets state if recovery window elapsed."""
        if self._consecutive_failures < self._threshold:
            return False
        if (
            self._last_failure_time is not None
            and (time.monotonic() - self._last_failure_time) > self._recovery_seconds
        ):
            self._reset()  # side effect: half-open → closed transition
            return False
        return True

    def _record_failure(self) -> None:
        # asyncio 单线程事件循环中，并发 embed() 调用可能导致计数略有偏差
        # （两个并发调用都通过 _is_open 后都失败，计数 +2）。可接受的近似行为。
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()

    def _reset(self) -> None:
        self._consecutive_failures = 0
        self._last_failure_time = None
