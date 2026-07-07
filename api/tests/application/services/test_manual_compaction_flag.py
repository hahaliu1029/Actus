"""B11 §8: manual_compaction_flag helpers (fake redis, no infra)."""
from __future__ import annotations

import pytest

from app.application.services.manual_compaction_flag import (
    consume_manual_compact_pending,
    set_manual_compact_pending,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, int | None]] = []

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value
        self.set_calls.append((key, value, ex))

    async def getdel(self, key: str):  # reused by Task 6 consume tests
        return self.store.pop(key, None)


@pytest.mark.anyio
async def test_set_manual_compact_pending_stores_with_ttl():
    redis = _FakeRedis()
    await set_manual_compact_pending(redis, "sess-1")
    assert redis.store == {"manual_compact_pending:sess-1": "1"}
    assert redis.set_calls[0][2] == 86_400  # 24h TTL backstop


@pytest.mark.anyio
async def test_set_then_consume_returns_true_once():
    redis = _FakeRedis()
    await set_manual_compact_pending(redis, "sess-1")
    assert await consume_manual_compact_pending(redis, "sess-1") is True
    # GETDEL is atomic read-and-delete → second consume sees nothing.
    assert await consume_manual_compact_pending(redis, "sess-1") is False


@pytest.mark.anyio
async def test_consume_never_set_returns_false():
    redis = _FakeRedis()
    assert await consume_manual_compact_pending(redis, "sess-x") is False
