"""C2 PR-3 §7.5 — CoordinatorRunRepository ABC skeleton.

PR-3 scope: ``bump_attempt`` ABC only — concrete impl is the JSONB UPDATE on
``sessions.coordinator_attempts`` already shipped on ``SessionRepository``
(``peek_coordinator_attempt`` / ``bump_coordinator_attempt``).

PR-7 expands with rehydrate methods:
  - ``find_children_by_coordinator_run(coordinator_run_id) -> list[ChildRow]``
  - ``find_terminal_envelopes(coordinator_run_id) -> list[ResultReadyEnvelope]``
  - reservation / lease bookkeeping helpers
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class CoordinatorRunRepository(ABC):
    """Skeleton — PR-7 fleshes out rehydrate methods."""

    @abstractmethod
    async def bump_attempt(self, *, session_id: str, step_id: str) -> int:
        """Atomic JSONB increment of ``sessions.coordinator_attempts[step_id]``.

        Returns the new attempt_ix (``>= 1``). Concrete impl delegates to
        ``SessionRepository.bump_coordinator_attempt``.
        """
        ...
