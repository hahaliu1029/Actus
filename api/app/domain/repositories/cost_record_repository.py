"""B4 M0: CostRecordRepository protocol (domain-level contract).

Structural-typing protocol following the MemoryChunkRepository convention —
lets in-memory fakes satisfy the interface without subclassing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.domain.models.cost_record import CostRecord


class CostRecordRepository(Protocol):
    """Per-session cost row storage."""

    async def insert(self, record: "CostRecord") -> None:
        """Persist one cost record. Idempotent on ``run_id``.

        Implementations MUST use an ON CONFLICT DO NOTHING / unique-index
        idempotency path so handler retries cannot double-bill a single call.
        """
        ...

    async def find_by_session(self, session_id: str) -> list["CostRecord"]:
        """Return all cost rows for a session, ordered by (created_at, step_ix)."""
        ...
