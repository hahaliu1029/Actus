"""B11 §8: pending-flag for manual /compact requests.

The POST endpoint SETs the flag; the next chat run at a compactable status
GETDELs it. Single source for the Redis key so set/consume never drift. This
module holds no infra import — the caller passes the raw Redis client
(``get_redis().client``), keeping layer boundaries clean.

``set_manual_compact_pending`` (the endpoint's SET) landed in Task 5;
``consume_manual_compact_pending`` (the chat run's GETDEL) landed test-first
in Task 6.
"""
from __future__ import annotations

_PENDING_TTL_SECONDS = 86_400  # 24h backstop for stale keys (at-most-once, spec §8)


def _pending_key(session_id: str) -> str:
    return f"manual_compact_pending:{session_id}"


async def set_manual_compact_pending(redis, session_id: str) -> None:
    await redis.set(_pending_key(session_id), "1", ex=_PENDING_TTL_SECONDS)


async def consume_manual_compact_pending(redis, session_id: str) -> bool:
    """Atomically read-and-delete the pending flag. Returns True if it was set."""
    value = await redis.getdel(_pending_key(session_id))
    return value is not None
