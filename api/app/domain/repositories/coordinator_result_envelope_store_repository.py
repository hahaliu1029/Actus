"""C2 v1 CoordinatorResultEnvelopeStoreRepository ABC (spec §13.x).

Domain-side contract used by the mailbox supervisor / coordinator
orchestrator to persist terminal RESULT_READY / CANCEL_ACK envelopes
for crash recovery, and by the PR-7 ``CoordinatorRehydrateService`` to
read them back on next-boot rehydrate.

Two methods:

- ``persist_terminal`` inserts one row carrying the minimum rehydrate
  subset of the envelope payload. The DB-level unique index on
  ``(coordinator_run_id, work_unit_id)`` enforces at-most-once
  semantics; callers that want idempotent retry should catch the
  ``IntegrityError`` from the impl.
- ``find_terminal_envelopes_by_run`` is the rehydrate read path —
  returns every terminal envelope row for a run, ordered by
  ``received_at`` so the consumer can reconstruct completion order.

The DB impl lives at ``infrastructure/repositories/
db_coordinator_result_envelope_store_repository``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CoordinatorResultEnvelopeStoreRepository(ABC):
    """Terminal-envelope persistence used by the mailbox supervisor."""

    @abstractmethod
    async def persist_terminal(
        self,
        *,
        coordinator_run_id: str,
        work_unit_id: str,
        child_session_id: str,
        envelope_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Insert minimum rehydrate fields. Re-insert with same (run_id,
        wu_id) raises UniqueViolation — callers must catch IntegrityError
        if idempotency retry is expected.

        ``envelope_type`` must be ``'RESULT_READY'`` or ``'CANCEL_ACK'``
        (caller-validated; the DB has no CHECK constraint to allow
        cheap future widening).

        ``payload`` is filtered by the impl to the minimum rehydrate
        subset (``outcome``, ``patch_manifest``, ``patch_manifest_ref``,
        ``cost_summary``, ``needs_authorization_details``, ``final_state``);
        free-text fields are stripped, oversized payloads truncated, and PII
        regex hits redacted before insert.
        """

    @abstractmethod
    async def find_terminal_envelopes_by_run(
        self, coordinator_run_id: str,
    ) -> list[Any]:
        """Return every terminal envelope row for ``coordinator_run_id``,
        ordered by ``received_at`` ascending.

        Used by PR-7 ``CoordinatorRehydrateService`` to rebuild the
        coordinator run's completed-work-unit map on next-boot
        rehydrate. Returns an empty list when no terminal envelopes
        have been persisted for the run.

        [codex R1 P2#3] Return element type is ``Any`` rather than
        forward-referencing the infra ORM ``CoordinatorResultEnvelopeStore``
        — domain ABCs should not name infrastructure types even under
        TYPE_CHECKING. Callers read attributes by name —
        ``envelope_type``, ``payload``, ``work_unit_id``,
        ``child_session_id``, ``received_at`` — which works for the
        live ORM impl, mocks, or any future domain DTO swap.
        """
