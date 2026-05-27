"""C2 v1 PatchApplier — process-stable all-or-nothing apply (spec §10).

**Process-stable, NOT crash-safe.** Pod crash mid-apply needs manual
operator recovery (audit row stays at ``in_progress`` + snapshot files
linger on the local FS). PR-7 crash-recovery promotes this to crash-safe
with a persistent in-flight catalog.

Algorithm (spec §10.2):

1. Acquire Redis lock keyed on ``coordinator:apply:{run_id}`` —
   idempotency guard against re-entry (e.g. the orchestrator retries).
2. Cancel check — if the coordinator cancelled before we got to write,
   short-circuit to ``APPLY_ABORTED`` (no audit row needed; nothing
   happened on the sandbox).
3. Pre-apply audit insert — open an ``in_progress`` row with
   ``plan_files_preview`` so operators can see what was about to happen
   if a later step fails. Returns ``audit_id`` for the terminal update.
4. Preflight + snapshot — per modify/delete entry: check file exists,
   verify base_digest matches current, snapshot original bytes. Per add
   entry: check file does NOT exist. Any mismatch aborts cleanly with no
   snapshots persisted.
5. Actual apply — per entry: write atomically, verify post-write digest.
   Cancel checked between each entry. ANY failure triggers
   ``_rollback(reverse_order=True)`` and updates audit to terminal
   non-success status. Partial rollback emits ``HealthEvent`` for
   operator attention.
6. Success path — discard snapshots, update audit to ``success``.

**Why all-or-nothing per call**: a coordinator's reduced
PatchApplyPlan represents a single logical commit across N files; a
partial apply would leave the sandbox in an inconsistent state that
neither the reducer nor the planner know how to reason about. The
rollback path is the all-or-nothing guarantee.

**Known v1 limitations (PR-7 crash-recovery scope addresses):**

- *Redis lock TTL* (codex R3 P1#1): ``timeout=600`` ≈ 10 minutes; no
  renewal. Very long applies (huge plans, sandbox stall) could lose
  ownership; on ``__aexit__`` redis-py raises and the apply return
  value is masked. Mitigation today: plans are bounded by per-step
  budgets in PR-6, and 10 minutes covers typical PR-5 scope.
- *task.cancel() bypasses rollback* (codex R3 P1#3):
  ``asyncio.CancelledError`` does NOT subclass ``Exception``, so the
  preflight/apply/_rollback ``except Exception`` clauses don't fire
  on external ``task.cancel()``. The audit row stays at
  ``in_progress`` and snapshots leak — process-stable, not crash-safe.
  Use ``cancel_event`` for cooperative cancel; reserve ``task.cancel()``
  for forced terminate.
- *Finalize order: audit-before-discard* (codex R3 P2#10): we
  ``update_terminal`` first, then ``snapshot_store.discard``. If
  discard fails or the pod crashes between the two, the audit row is
  consistent but a few snapshot files linger in ``/tmp``. This is the
  correct trade-off — audit truth wins over disk hygiene. Operators
  can sweep ``/tmp/actus/coordinator-rollback`` post-incident.
- *Reducer trusts child manifest paths* (codex R3 P1#4): the reducer
  does NOT re-validate ``FilePatchEntry.path`` against the parent-
  issued ``WorkUnit.write_lease``. PR-5 relies on the schema-layer
  ``validate_relative_path_strict`` + child ChildScopeGate (PE Phase
  1) + reducer lineage cross-check (codex R5 P0:
  ``pm.coordinator_run_id`` + ``pm.work_unit_id`` must match parent
  values, mismatch → drop manifest + warn). Full per-path lease
  enforcement is Phase 2 PermissionEngine.
- *Cross-step apply contention* (codex R5 P1): the Redis lock key
  ``coordinator:apply:{run_id}`` is per-run; two coordinator steps
  on the SAME parent_session_id (different step_hash or attempt) can
  hold distinct locks and race on the same path. PR-5 mitigation:
  preflight digest/exists check at T0 + post-write digest at T1
  catches the visible races; a window remains between T0/T1 where
  the parent sandbox state can change. Phase 2 PermissionEngine + a
  sandbox CAS API (``write_if_digest`` / ``write_if_absent``)
  closes the window.
- *Unbounded list persistence* (codex R5 P1): ``PatchManifest.files``,
  ``plan_files_preview``, ``applied_files``, ``rollback_failed_paths``,
  ``HealthEvent.metrics.failed_paths`` all lack size caps. PR-6
  introduces per-step plan budgets (file_count, total_size_bytes,
  worker_count); until then a huge manifest amplifies into JSONB row
  size, SSE/HealthEvent payload, and audit update. Operators rely on
  per-coordinator-run resource quotas to bound this in practice.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional


_FAILED_REASON_MAX = 256
"""[codex R4 P1] Max length of ``failed_reason`` written to the audit
table. Mirror of ``coordinator_apply_audit.failed_reason String(256)``.
Truncate at the applier boundary so DB inserts never raise from an
oversized exception traceback masking the real apply outcome."""


def _compute_plan_hash(plan: "PatchApplyPlan") -> str:
    """[codex R4 P1] SHA-256 hex of the canonical PatchApplyPlan JSON.

    Used as the PR-7 rehydrate idempotency key — recomputing yields
    the same digest if and only if the plan being applied is exactly
    the one previously logged.

    Canonical form: ``json.dumps(..., sort_keys=True, ensure_ascii=False)``
    over the entry tuple. Field order within entries is deterministic
    via Pydantic's `model_dump` (frozen + extra=forbid in PatchManifest
    sub-models pins the shape).
    """
    payload = {
        "coordinator_run_id": plan.coordinator_run_id,
        "files": [
            {
                "path": e.path,
                "op": e.op,
                "base_digest": e.base_digest,
                "new_digest": e.new_digest,
                "content_ref": e.content_ref,
                "content_size": e.content_size,
            }
            for e in plan.files
        ],
        "total_size_bytes": plan.total_size_bytes,
        "file_count": plan.file_count,
        "source_work_unit_ids": list(plan.source_work_unit_ids),
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _truncate_reason(reason: Optional[str]) -> Optional[str]:
    """Cap ``failed_reason`` to fit the audit column's VARCHAR(256)
    and strip CR/LF so log lines + UI text aren't corrupted by
    multi-line exception traces.

    [codex R5 P2] Replaces newlines + tabs with single spaces so a
    ``str(exc)`` whose message contains a stacktrace fragment can't
    inject blank lines into the audit row + downstream log shipping.
    """
    if reason is None:
        return None
    # Sanitize whitespace: collapse all newline/CR/tab chars to a
    # single space so audit display and log line shipping don't
    # break on multi-line exception messages.
    sanitized = (
        reason.replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )
    if len(sanitized) <= _FAILED_REASON_MAX:
        return sanitized
    # Preserve a trailing marker so operators can see the truncation
    # happened rather than guess at the missing suffix.
    keep = _FAILED_REASON_MAX - len(" ...[truncated]")
    return sanitized[:keep] + " ...[truncated]"

if TYPE_CHECKING:
    from app.application.services.rollback_snapshot_store import (
        FileSnapshot,
        RollbackSnapshotStore,
    )
    from app.domain.external.artifact_storage import ArtifactStoragePort
    from app.domain.external.parent_sandbox import ParentSandboxPort
    from app.domain.models.patch_apply_plan import PatchApplyPlan
    from app.domain.repositories.coordinator_apply_audit_repository import (
        CoordinatorApplyAuditRepository,
    )


logger = logging.getLogger(__name__)


class ApplyStatus(StrEnum):
    """Terminal status the applier reports back to the orchestrator.

    Subset overlaps with the audit table ``status`` column — the audit
    table additionally has ``in_progress`` (set during apply, not
    surfaced via this enum).
    """

    SUCCESS = "success"
    DIGEST_DRIFT = "digest_drift"
    FILE_MISSING = "file_missing"
    FILE_EXISTS = "file_exists"
    POST_WRITE_DIGEST_MISMATCH = "post_write_digest_mismatch"
    MINIO_FETCH_FAILED = "minio_fetch_failed"
    WRITE_IO_ERROR = "write_io_error"
    ROLLBACK_PARTIAL = "rollback_partial"
    APPLY_ABORTED = "apply_aborted"


@dataclass(frozen=True)
class AppliedFileRecord:
    path: str
    op: str


@dataclass(frozen=True)
class AppliedFileFailure:
    path: str
    reason: str


@dataclass(frozen=True)
class ApplyDiagnostics:
    duration_ms: int


@dataclass(frozen=True)
class ApplyOutcome:
    """Returned to ``_run_parallel_backend`` for SSE/audit/orchestrator
    branching. ``applied_files`` is empty on cancel-before-write.
    """

    status: ApplyStatus
    applied_files: tuple[AppliedFileRecord, ...]
    failed_at: Optional[AppliedFileFailure]
    rollback_status: Optional[str]
    diagnostics: ApplyDiagnostics


# Type alias for the emit-event callable. Bound at composition root to
# the orchestrator's event queue. Typed loosely so HealthEvent stays out
# of the runtime import graph (it lives in domain/models/event.py and
# pulls in pydantic at import time).
_EmitEventCallable = Callable[[Any], Awaitable[None]]


class PatchApplier:
    """Process-stable all-or-nothing applier (spec §10).

    Constructor takes 4 ports:

    - ``snapshot_store`` — original-content store for rollback
    - ``audit_repo`` — coordinator_apply_audit writes
    - ``redis`` — distributed lock for idempotency
    - ``emit_event`` — async callable that publishes a HealthEvent on
      partial rollback (composition root binds to the orchestrator's
      event queue)

    Per-call ``parent_sandbox`` + ``minio_client`` + optional
    ``cancel_event`` come in via ``apply()`` — they vary per
    coordinator run while the applier itself is long-lived (one per
    orchestrator).
    """

    def __init__(
        self,
        *,
        snapshot_store: "RollbackSnapshotStore",
        audit_repo: "CoordinatorApplyAuditRepository",
        redis: Any,
        emit_event: _EmitEventCallable,
    ) -> None:
        self._snapshot_store = snapshot_store
        self._audit_repo = audit_repo
        self._redis = redis
        self._emit_event = emit_event

    async def apply(
        self,
        plan: "PatchApplyPlan",
        *,
        parent_sandbox: "ParentSandboxPort",
        minio_client: "ArtifactStoragePort",
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ApplyOutcome:
        """Apply ``plan`` to ``parent_sandbox``; return ``ApplyOutcome``.

        Holds a Redis lock for the duration. ``timeout=600`` is the
        lock's auto-expire TTL (Redis releases it after 10 minutes if
        our pod hangs); ``blocking=False`` means we fail-fast if another
        apply is in flight for the same run_id (idempotency guard).
        """
        started_at = time.time()
        lock_key = f"coordinator:apply:{plan.coordinator_run_id}"
        async with self._redis.lock(
            lock_key, blocking=False, timeout=600,
        ):
            return await self._apply_locked(
                plan, parent_sandbox, minio_client, cancel_event, started_at,
            )

    async def _apply_locked(
        self,
        plan: "PatchApplyPlan",
        parent_sandbox: "ParentSandboxPort",
        minio_client: "ArtifactStoragePort",
        cancel_event: Optional[asyncio.Event],
        started_at: float,
    ) -> ApplyOutcome:
        # ── Step 2: cancel-before-write fast-path ────────────────────────
        # Nothing has been written, no audit row exists — return cleanly
        # without dirtying the audit table. Orchestrator routes this to
        # the SSE "step cancelled" branch.
        if cancel_event is not None and cancel_event.is_set():
            return ApplyOutcome(
                status=ApplyStatus.APPLY_ABORTED,
                applied_files=(),
                failed_at=None,
                rollback_status=None,
                diagnostics=ApplyDiagnostics(
                    duration_ms=_elapsed_ms(started_at),
                ),
            )

        # ── Step 2.5: pre-apply audit insert ─────────────────────────────
        # ``parent_session_id`` parsed from coordinator_run_id per §4.7
        # shape ``f"{session_id}:{step_id_hash16}:a{attempt_ix}"`` —
        # session_id is everything before the first colon.
        parent_session_id = plan.coordinator_run_id.split(":", 1)[0]
        # [codex R4 P1] Capture the plan_hash NOW so PR-7 rehydrate can
        # detect "this retry is the same plan as a prior attempt"
        # without re-comparing FilePatchEntry tuples.
        plan_hash = _compute_plan_hash(plan)
        audit_id = await self._audit_repo.insert_in_progress(
            coordinator_run_id=plan.coordinator_run_id,
            parent_session_id=parent_session_id,
            plan_files_preview=[e.path for e in plan.files],
            plan_hash=plan_hash,
        )

        # ── Step 3: preflight + snapshot ─────────────────────────────────
        # Wrap the whole loop in try/except: ParentSandboxPort methods
        # may raise (e.g. ``exists()`` now raises ``OSError`` on RPC
        # failure per the adapter contract — codex R1 P1#3 fix). A
        # preflight raise must NOT escape the lock context — translate
        # to WRITE_IO_ERROR and finalize so the audit row is closed.
        snapshots: list["FileSnapshot"] = []
        try:
            for e in plan.files:
                if e.op in ("modify", "delete"):
                    if not await parent_sandbox.exists(e.path):
                        return await self._finalize(
                            audit_id, ApplyStatus.FILE_MISSING,
                            failed=AppliedFileFailure(
                                path=e.path, reason="missing",
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                        )
                    cur = await parent_sandbox.compute_digest(e.path)
                    if cur != e.base_digest:
                        return await self._finalize(
                            audit_id, ApplyStatus.DIGEST_DRIFT,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=f"drift cur={cur}",
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                        )
                    # Snapshot AFTER digest match — saves wasted
                    # snapshot write on the abort path.
                    orig = await parent_sandbox.read_file(e.path)
                    snap = await self._snapshot_store.save(
                        coordinator_run_id=plan.coordinator_run_id,
                        path=e.path,
                        content=orig,
                        original_digest=e.base_digest,
                    )
                    snapshots.append(snap)
                elif e.op == "add":
                    if await parent_sandbox.exists(e.path):
                        return await self._finalize(
                            audit_id, ApplyStatus.FILE_EXISTS,
                            failed=AppliedFileFailure(
                                path=e.path, reason="exists",
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                        )
        except Exception as preflight_exc:
            # No writes have happened yet (only snapshot reads), so no
            # rollback is needed — but we DO need to discard any
            # snapshots already saved in earlier iterations so the FS
            # doesn't accumulate orphans.
            return await self._finalize(
                audit_id, ApplyStatus.WRITE_IO_ERROR,
                failed=AppliedFileFailure(
                    path="<preflight>", reason=str(preflight_exc),
                ),
                applied=[],
                snapshots_to_discard=snapshots,
                started_at=started_at,
                plan=plan,
            )

        # ── Step 4: actual apply ─────────────────────────────────────────
        applied: list[AppliedFileRecord] = []
        for e in plan.files:
            # Per-entry cancel check (rollback any writes already in flight).
            if cancel_event is not None and cancel_event.is_set():
                rb, rb_failed = await self._rollback(
                    applied, snapshots, parent_sandbox, plan,
                )
                final_status = (
                    ApplyStatus.ROLLBACK_PARTIAL
                    if rb == "partial"
                    else ApplyStatus.APPLY_ABORTED
                )
                return await self._finalize(
                    audit_id, final_status,
                    failed=None,
                    applied=applied,
                    snapshots_to_discard=snapshots,
                    started_at=started_at,
                    plan=plan,
                    rollback_status=rb,
                    rollback_failed_paths=rb_failed,
                )
            try:
                if e.op == "delete":
                    await parent_sandbox.delete_file(e.path)
                else:  # add or modify
                    try:
                        content = await minio_client.get_bytes(e.content_ref)
                    except Exception as fetch_exc:
                        # Distinguish minio fetch from sandbox write
                        # failures so operators can debug the right
                        # subsystem.
                        rb, rb_failed = await self._rollback(
                            applied, snapshots, parent_sandbox, plan,
                        )
                        final_status = (
                            ApplyStatus.ROLLBACK_PARTIAL
                            if rb == "partial"
                            else ApplyStatus.MINIO_FETCH_FAILED
                        )
                        return await self._finalize(
                            audit_id, final_status,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=f"minio: {fetch_exc}",
                            ),
                            applied=applied,
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            rollback_status=rb,
                            rollback_failed_paths=rb_failed,
                        )
                    # [codex R5 P1] Verify the actual blob size matches
                    # the manifest's declared ``content_size``. The
                    # post-write digest check (read-from-sandbox)
                    # catches blob corruption, but doesn't catch a
                    # manifest that claims 0 bytes while the MinIO ref
                    # points at a huge blob with matching digest (a
                    # budget / audit escape). Verify pre-write.
                    if (
                        e.content_size is not None
                        and len(content) != e.content_size
                    ):
                        rb, rb_failed = await self._rollback(
                            applied, snapshots, parent_sandbox, plan,
                        )
                        final_status = (
                            ApplyStatus.ROLLBACK_PARTIAL
                            if rb == "partial"
                            else ApplyStatus.MINIO_FETCH_FAILED
                        )
                        return await self._finalize(
                            audit_id, final_status,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=(
                                    f"content_size mismatch "
                                    f"(declared={e.content_size} "
                                    f"actual={len(content)})"
                                ),
                            ),
                            applied=applied,
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            rollback_status=rb,
                            rollback_failed_paths=rb_failed,
                        )
                    # [codex R3 P1#2] Defensive cancel check between
                    # ``get_bytes`` and ``atomic_write_file``. Without
                    # this, a cancel that fired AFTER the minio fetch
                    # but BEFORE the sandbox write would still proceed
                    # to write that entry and only be caught at the top
                    # of the NEXT loop iteration — too late if this is
                    # the last entry.
                    if cancel_event is not None and cancel_event.is_set():
                        rb, rb_failed = await self._rollback(
                            applied, snapshots, parent_sandbox, plan,
                        )
                        final_status = (
                            ApplyStatus.ROLLBACK_PARTIAL
                            if rb == "partial"
                            else ApplyStatus.APPLY_ABORTED
                        )
                        return await self._finalize(
                            audit_id, final_status,
                            failed=None,
                            applied=applied,
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            rollback_status=rb,
                            rollback_failed_paths=rb_failed,
                        )
                    await parent_sandbox.atomic_write_file(e.path, content)
                    # Post-write digest verify: read the file back from the
                    # sandbox and compare. The previous version hashed the
                    # bytes we just sent, which only confirms our local
                    # copy — sandbox-side truncation or corruption would
                    # pass that check undetected. Re-reading is an extra
                    # RPC per entry but is the only correct way to catch
                    # sandbox-side write defects ([codex R1 P1#1]).
                    actual = await parent_sandbox.compute_digest(e.path)
                    if actual != e.new_digest:
                        # Record this entry as applied so _rollback can
                        # undo it alongside any prior successful entries.
                        applied.append(
                            AppliedFileRecord(path=e.path, op=e.op),
                        )
                        rb, rb_failed = await self._rollback(
                            applied, snapshots, parent_sandbox, plan,
                        )
                        final_status = (
                            ApplyStatus.ROLLBACK_PARTIAL
                            if rb == "partial"
                            else ApplyStatus.POST_WRITE_DIGEST_MISMATCH
                        )
                        return await self._finalize(
                            audit_id, final_status,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=(
                                    f"actual={actual} expected={e.new_digest}"
                                ),
                            ),
                            applied=applied,
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            rollback_status=rb,
                            rollback_failed_paths=rb_failed,
                        )
                applied.append(AppliedFileRecord(path=e.path, op=e.op))
            except Exception as exc:
                # Any other failure in the apply loop (sandbox RPC, OS
                # error, etc.). Rollback the already-completed entries
                # in ``applied``; the current entry is NOT in
                # ``applied`` because we only append after the full
                # write + post-write digest verify succeeds.
                #
                # [codex R7 P1 + R8 P1 + R10 P2] We do NOT add the
                # current entry to ``applied`` here. The reason: the
                # ``ParentSandboxPort`` contract treats
                # ``atomic_write_file`` / ``delete_file`` as "either
                # succeeds atomically OR raises with no observable
                # side effect" — that's the spec'd contract.
                #
                # **v1 gap**: the live sandbox HTTP service backing
                # ``SandboxHandle.upload_file`` does NOT yet fully
                # honor this — it writes via ``open(path, "wb")`` +
                # chunked write, so a mid-write exception can leave
                # a truncated file. See
                # ``ParentSandboxAdapter`` module docstring for the
                # full limitation table; closing the gap is a PR-7 /
                # sandbox-team follow-up (tmp+fsync+rename or
                # sandbox rename RPC). Until then, partial-file
                # leakage on a write that crashes mid-stream is a
                # documented v1 behavior, surfaced via the
                # ``failed_reason`` audit column.
                #
                # Including the current entry in rollback would cause
                # a false ROLLBACK_PARTIAL status when the rollback's
                # restore write fails on a file the apply write
                # never actually touched (typical case: 4xx-class
                # RPC failure before any byte hit the sandbox). The
                # tradeoff: post-PR-7 atomicity fixes the leak case;
                # PR-5 prioritizes audit clarity for the common case
                # over edge-case rollback for partial leaks.
                rb, rb_failed = await self._rollback(
                    applied, snapshots, parent_sandbox, plan,
                )
                final_status = (
                    ApplyStatus.ROLLBACK_PARTIAL
                    if rb == "partial"
                    else ApplyStatus.WRITE_IO_ERROR
                )
                return await self._finalize(
                    audit_id, final_status,
                    failed=AppliedFileFailure(
                        path=e.path, reason=str(exc),
                    ),
                    applied=applied,
                    snapshots_to_discard=snapshots,
                    started_at=started_at,
                    plan=plan,
                    rollback_status=rb,
                    rollback_failed_paths=rb_failed,
                )

        # ── Step 5: discard snapshots on success ─────────────────────────
        return await self._finalize(
            audit_id, ApplyStatus.SUCCESS,
            failed=None,
            applied=applied,
            snapshots_to_discard=snapshots,
            started_at=started_at,
            plan=plan,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    async def _rollback(
        self,
        applied: list[AppliedFileRecord],
        snapshots: list["FileSnapshot"],
        parent_sandbox: "ParentSandboxPort",
        plan: "PatchApplyPlan",
    ) -> "tuple[str, list[str]]":
        """Reverse-order undo for every entry in ``applied``.

        Returns ``"complete"`` or ``"partial"``.

        Per-op semantics:
        - ``add``    : ``delete_file(path)`` — the entry was new, so we
                       remove what we created.
        - ``modify`` : restore the snapshot via ``atomic_write_file``.
                       The snapshot was saved before the modify in §10.2
                       preflight; we look it up by ``original_path``.
        - ``delete`` : restore the snapshot via ``atomic_write_file``
                       (re-creates the file with original bytes).

        Reverse order matters when entries have inter-file invariants
        (e.g. modifying A then deleting B — un-deleting B before
        rolling back A preserves the temporary state ordering).

        [codex R1 P0#1 fix] The previous version walked only
        ``snapshots`` and never undid ``add`` ops, leaving newly-created
        files behind on rollback — a direct violation of the all-or-
        nothing contract. Now every entry in ``applied`` is undone.

        Partial rollback emits a HealthEvent so the operator sees the
        critical signal. The applier overrides the terminal status to
        ``ROLLBACK_PARTIAL`` when this returns ``"partial"``.
        """
        snap_by_path: dict[str, "FileSnapshot"] = {
            s.original_path: s for s in snapshots
        }
        failed_paths: list[str] = []
        for record in reversed(applied):
            try:
                if record.op == "add":
                    # We created this file. Undo by deleting it.
                    await parent_sandbox.delete_file(record.path)
                else:
                    # ``modify`` or ``delete``: a snapshot of the
                    # original bytes was saved in preflight.
                    snap = snap_by_path.get(record.path)
                    if snap is None:
                        # Programmer error — modify/delete must have a
                        # matching snapshot. Surface explicitly.
                        raise RuntimeError(
                            f"rollback: missing snapshot for {record.path!r} "
                            f"(op={record.op!r})",
                        )
                    content = await self._snapshot_store.load(snap)
                    await parent_sandbox.atomic_write_file(
                        snap.original_path, content,
                    )
            except Exception as exc:
                logger.error(
                    "PatchApplier rollback failed for %s (op=%s): %s",
                    record.path, record.op, exc,
                )
                failed_paths.append(record.path)

        if failed_paths:
            # Late import to avoid pulling the full event/pydantic graph
            # into the patch_applier module-load path.
            from app.domain.models.event import HealthEvent, HealthStatus
            try:
                await self._emit_event(HealthEvent(
                    status=HealthStatus.TERMINATING,
                    reason=(
                        f"PatchApplier rollback partial on run "
                        f"{plan.coordinator_run_id}: "
                        f"{len(failed_paths)} paths failed restore"
                    ),
                    action="hard_terminate",
                    metrics={
                        "code": "coordinator_apply_rollback_partial",
                        "coordinator_run_id": plan.coordinator_run_id,
                        "failed_paths": failed_paths,
                    },
                ))
            except Exception as emit_exc:
                # Emit failure must not mask the rollback result. Log
                # and continue — the audit row + return value still
                # carry the partial signal.
                logger.error(
                    "PatchApplier emit_event(rollback_partial) "
                    "raised: %s", emit_exc,
                )
            return "partial", failed_paths
        return "complete", []

    async def _finalize(
        self,
        audit_id: int,
        status: ApplyStatus,
        *,
        failed: Optional[AppliedFileFailure],
        applied: list[AppliedFileRecord],
        snapshots_to_discard: list["FileSnapshot"],
        started_at: float,
        plan: "PatchApplyPlan",
        rollback_status: Optional[str] = None,
        rollback_failed_paths: Optional[list[str]] = None,
    ) -> ApplyOutcome:
        """Write the terminal audit row + clean up snapshots + return.

        ``snapshots_to_discard`` is best-effort — the snapshot store's
        ``discard`` is itself idempotent (handles already-removed files
        without raising). We discard on EVERY terminal branch so the
        local FS doesn't accumulate orphaned snapshot files even on
        failure paths.

        [codex R4 P2] ``plan`` is threaded in so the audit row carries
        the original plan's ``total_size_bytes`` — previously this
        column always stored 0 because the applier didn't have a
        reference to the plan at finalize time. [codex R4 P1] also
        truncates ``failed.reason`` to fit the VARCHAR(256) column so
        a long traceback doesn't raise DBAPIError and mask the apply
        outcome.
        """
        duration_ms = _elapsed_ms(started_at)
        await self._audit_repo.update_terminal(
            audit_id,
            status=status.value,
            file_count=len(applied),
            total_bytes=plan.total_size_bytes,
            failed_at_path=failed.path if failed else None,
            failed_reason=_truncate_reason(
                failed.reason if failed else None,
            ),
            rollback_status=rollback_status,
            rollback_failed_paths=(
                list(rollback_failed_paths)
                if rollback_failed_paths
                else None
            ),
            applied_files=[
                {"path": r.path, "op": r.op} for r in applied
            ] if applied else None,
            duration_ms=duration_ms,
        )
        # Discard snapshots last so a discard failure doesn't lose the
        # audit terminal write. ``discard`` is idempotent.
        if snapshots_to_discard:
            try:
                # Identify the run_id from any snapshot (all share it).
                run_id = snapshots_to_discard[0].coordinator_run_id
                await self._snapshot_store.discard(
                    run_id, snapshots_to_discard,
                )
            except Exception as exc:
                # Don't mask the result on a cleanup failure.
                logger.error(
                    "PatchApplier snapshot discard failed: %s", exc,
                )
        return ApplyOutcome(
            status=status,
            applied_files=tuple(applied),
            failed_at=failed,
            rollback_status=rollback_status,
            diagnostics=ApplyDiagnostics(duration_ms=duration_ms),
        )


def _elapsed_ms(started_at: float) -> int:
    return int((time.time() - started_at) * 1000)
