"""C2 v1 CoordinatorApplyAuditRepository ABC (spec §12.4).

Domain-side contract used by the PatchApplier (``application/services/
patch_applier``) to record apply lifecycle.

Three methods:

- ``insert_in_progress`` opens the audit row at the start of
  ``PatchApplier.apply`` (before any sandbox write). Returns the
  audit_id so the applier can update the same row on terminal status.
- ``update_terminal`` finalizes the row on any of the
  ``ApplyStatus`` values produced by the applier — see the migration
  docstring (``c2pr5_add_coordinator_apply_audit.py``) for the
  authoritative ``status`` taxonomy (``success`` /
  ``digest_drift`` / ``file_missing`` / ``file_exists`` /
  ``target_special_file`` /
  ``post_write_digest_mismatch`` / ``minio_fetch_failed`` /
  ``write_io_error`` / ``apply_aborted`` / ``rollback_partial``).
- ``find_latest_for_run`` is the rehydrate path (PR-7) + SSE replay
  hook (PR-8).

The DB impl lives at ``infrastructure/repositories/
db_coordinator_apply_audit_repository``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class CoordinatorApplyAuditRepository(ABC):
    """Audit-log writer used by the PatchApplier."""

    @abstractmethod
    async def insert_in_progress(
        self,
        *,
        coordinator_run_id: str,
        parent_session_id: str,
        plan_files_preview: list[str],
        plan_hash: Optional[str] = None,
    ) -> int:
        """Open the audit row; return the audit_id for later ``update_terminal``.

        ``plan_files_preview`` is the list of paths the apply plan will
        touch — stored as JSON so the operator can see "what was going
        to happen" if the apply later failed mid-way.

        ``plan_hash`` is the SHA-256 hex digest of the canonical
        PatchApplyPlan JSON, used by PR-7 rehydrate to detect when a
        retry is replaying the same plan as a prior partially-applied
        attempt (codex R4 P1). Optional for backward compat with
        callers that haven't migrated yet."""

    # [codex R7 P2] ``file_count`` is the *applied* dimension: the
    # number of FilePatchEntry that were successfully written before
    # the terminal status (0 on preflight aborts, equal to
    # ``plan.file_count`` on full success, partial on mid-apply
    # failures). The plan's intended file count is recoverable from
    # ``plan_files_preview`` JSONB. ``total_bytes`` is the *plan*
    # dimension (``plan.total_size_bytes``) since the applier may not
    # know per-entry byte deltas until each write completes. PR-7
    # rehydrate consumers should be aware of this asymmetry.
    @abstractmethod
    async def update_terminal(
        self,
        audit_id: int,
        *,
        status: str,
        file_count: int = 0,
        total_bytes: int = 0,
        failed_at_path: Optional[str] = None,
        failed_reason: Optional[str] = None,
        rollback_status: Optional[str] = None,
        rollback_failed_paths: Optional[list[str]] = None,
        applied_files: Optional[list[dict[str, Any]]] = None,
        diagnostics: Optional[dict[str, Any]] = None,
        duration_ms: int = 0,
    ) -> None:
        """Update the audit row to a terminal status.

        ``applied_files`` is the list of ``{"path": ..., "op": ...}``
        dicts the applier successfully wrote (used for replay on
        rehydrate).

        ``rollback_failed_paths`` is the list of parent-sandbox paths
        the rollback step could not restore (codex R2 P2) — operators
        consult this for manual recovery when
        ``status == 'rollback_partial'``. ``None`` when rollback was
        complete (or never attempted).

        ``diagnostics`` carries applier-side debug info (reducer
        diagnostics persistence is a PR-7 follow-up — see
        ``ReducerDiagnostics`` docstring).
        """

    @abstractmethod
    async def find_latest_for_run(
        self, coordinator_run_id: str,
    ) -> Optional[Any]:
        """Return the most recently-started audit row for ``coordinator_run_id``,
        or None if no apply attempt has been logged.

        Used by PR-7 rehydrate to decide whether an apply is in-flight,
        already-success (skip), or failed (retry / escalate).

        [codex R1 P2#3] Return type is ``Optional[Any]`` rather than
        forward-referencing the infra ORM ``CoordinatorApplyAudit`` —
        domain ABCs should not name infrastructure types even under
        TYPE_CHECKING. Callers in application/PR-7 read the row's
        attributes (``status``, ``coordinator_run_id``, etc.) by name,
        which works for the live ORM impl, mocks, or any future
        domain DTO swap."""
