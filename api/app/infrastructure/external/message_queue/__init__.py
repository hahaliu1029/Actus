"""Message queue infrastructure package.

Re-exports public constants used across application/infrastructure boundaries.

B3-core PR-1 §3.2: ``STREAM_TTL_SECONDS`` is the shared 24h TTL applied to
``task:output:{task_id}`` streams (and the ``session:seq:{session_id}`` counter
that producers stamp via INCR). Application-layer callers (e.g.
``agent_service._emit_event``) import this constant rather than reaching into
``redis_stream_message_queue.py`` for the private ``_DEFAULT_STREAM_TTL_SECONDS``.
"""

from typing import Final

STREAM_TTL_SECONDS: Final[int] = 86400  # 24h, spec v3 §3.2

__all__ = ["STREAM_TTL_SECONDS"]
