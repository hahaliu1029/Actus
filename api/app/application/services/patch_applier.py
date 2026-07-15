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

- *Redis lock lease*: ``timeout=600`` is a crash-cleanup window, renewed
  by the exact owner every TTL/3. Ownership loss stops later writes and
  attempts rollback only after the same key is exclusively owned again;
  otherwise it fails closed to manual recovery. It is not an apply wallclock.
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
import math
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from redis.exceptions import LockError, LockNotOwnedError


_FAILED_REASON_MAX = 256
"""[codex R4 P1] Max length of ``failed_reason`` written to the audit
table. Mirror of ``coordinator_apply_audit.failed_reason String(256)``.
Truncate at the applier boundary so DB inserts never raise from an
oversized exception traceback masking the real apply outcome."""

COORDINATOR_APPLY_LOCK_TTL_SECONDS = 600.0
_APPLY_LOCK_TTL_SECONDS = COORDINATOR_APPLY_LOCK_TTL_SECONDS

_ACQUIRE_APPLY_LOCK_AND_RESET_MARKER_LUA = r"""
-- coordinator-apply-lock-acquire-reset-marker-v1
local acquired = redis.call(
  'SET', KEYS[1], ARGV[1], 'NX', 'PX', tonumber(ARGV[2])
)
if not acquired then
  return 0
end
redis.call('DEL', KEYS[2])
return 1
"""

_APPLY_LOCK_COMPARE_DELETE_LUA = r"""
-- coordinator-apply-lock-compare-delete-v1
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
return redis.call('DEL', KEYS[1])
"""


def coordinator_apply_lock_key(coordinator_run_id: str) -> str:
    """Canonical Redis key shared by apply and rehydrate lock probing."""
    return f"coordinator:apply:{coordinator_run_id}"


def coordinator_apply_reconcile_marker_key(coordinator_run_id: str) -> str:
    """Canonical run marker reset by acquire and consumed by rehydrate."""
    return f"coordinator:apply-reconcile:{coordinator_run_id}"


def bind_coordinator_apply_lock_token(lock: Any, owner_token: str) -> None:
    """Bind a Lua-acquired token to redis-py's Lock adapter."""
    local = getattr(lock, "local", None)
    if local is not None:
        redis = lock.redis
        try:
            encoder = redis.connection_pool.get_encoder()
        except AttributeError:
            encoder = redis.get_encoder()
        local.token = encoder.encode(owner_token)
        return
    if hasattr(lock, "local_token"):
        lock.local_token = owner_token
        return
    if hasattr(lock, "token"):
        lock.token = owner_token
        return
    raise TypeError("apply lock adapter cannot bind an owner token")


async def compare_delete_coordinator_apply_lock(
    redis: Any,
    lock_key: str,
    owner_token: str,
) -> bool:
    """Delete the canonical apply key only when ``owner_token`` still owns it."""
    deleted = await redis.eval(
        _APPLY_LOCK_COMPARE_DELETE_LUA,
        1,
        lock_key,
        owner_token,
    )
    return bool(int(deleted))


async def acquire_coordinator_apply_lock_and_reset_marker(
    *,
    redis: Any,
    lock: Any,
    lock_key: str,
    reconcile_marker_key: str,
    owner_token: str,
    ttl_seconds: float,
) -> bool:
    """Atomically acquire the canonical owner and reset its missing cycle."""
    ttl_ms = max(1, math.ceil(ttl_seconds * 1_000))
    try:
        result = await redis.eval(
            _ACQUIRE_APPLY_LOCK_AND_RESET_MARKER_LUA,
            2,
            lock_key,
            reconcile_marker_key,
            owner_token,
            str(ttl_ms),
        )
    except BaseException:
        try:
            await compare_delete_coordinator_apply_lock(
                redis,
                lock_key,
                owner_token,
            )
        except BaseException:
            logger.exception(
                "ambiguous coordinator apply acquire cleanup failed key=%s",
                lock_key,
            )
        raise
    if not bool(int(result)):
        return False
    try:
        bind_coordinator_apply_lock_token(lock, owner_token)
    except BaseException:
        try:
            await compare_delete_coordinator_apply_lock(
                redis,
                lock_key,
                owner_token,
            )
        except BaseException:
            logger.exception(
                "coordinator apply acquired-token cleanup failed key=%s",
                lock_key,
            )
        raise
    return True


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
    from app.application.services.group_lineage import GroupLineageFields
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
    APPLY_LOCK_LOST = "apply_lock_lost"
    TARGET_SPECIAL_FILE = "target_special_file"
    PARENT_NOT_REGULAR = "parent_not_regular"


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
_RollbackPhaseCallback = Callable[[], None]
_LeaseSleepCallable = Callable[[float], Awaitable[None]]
_OwnershipCheckCallable = Callable[[], Awaitable[bool]]


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
        apply_lock_ttl_seconds: float = _APPLY_LOCK_TTL_SECONDS,
        lease_sleep: _LeaseSleepCallable = asyncio.sleep,
    ) -> None:
        if (
            isinstance(apply_lock_ttl_seconds, bool)
            or not isinstance(apply_lock_ttl_seconds, (int, float))
            or not math.isfinite(float(apply_lock_ttl_seconds))
            or apply_lock_ttl_seconds <= 0
        ):
            raise ValueError(
                "apply_lock_ttl_seconds must be a finite number > 0",
            )
        self._snapshot_store = snapshot_store
        self._audit_repo = audit_repo
        self._redis = redis
        self._emit_event = emit_event
        self._apply_lock_ttl_seconds = float(apply_lock_ttl_seconds)
        self._lease_sleep = lease_sleep

    async def apply(
        self,
        plan: "PatchApplyPlan",
        *,
        parent_sandbox: "ParentSandboxPort",
        minio_client: "ArtifactStoragePort",
        cancel_event: Optional[asyncio.Event] = None,
        lineage: Optional["GroupLineageFields"] = None,
        on_rollback: Optional[_RollbackPhaseCallback] = None,
    ) -> ApplyOutcome:
        """Apply ``plan`` to ``parent_sandbox``; return ``ApplyOutcome``.

        Holds an owner-token Redis lease for the duration. ``timeout=600``
        is only the crash-cleanup window: a background task renews the same
        owner every TTL/3, so it is not an apply wallclock. ``blocking=False``
        still fails fast if another owner is applying the same run.

        [PR-9b-B INV-B4] ``lineage`` carries the group-level
        ``root_session_id`` / ``parent_session_id`` so the emitted
        ``CoordinatorApplyEvent`` carries full lineage parity with the
        dispatch/reduce/sibling-cancel events. ``None`` (legacy callers)
        leaves those event fields at their default ``None``; only
        ``coordinator_run_id`` is then populated. Per-child fields
        (``child_session_id`` / ``work_unit_id``) are never set on the
        apply event because apply is a group-level operation.
        """
        started_at = time.time()
        lock_key = coordinator_apply_lock_key(plan.coordinator_run_id)
        reconcile_marker_key = coordinator_apply_reconcile_marker_key(
            plan.coordinator_run_id,
        )
        lock = self._redis.lock(
            lock_key,
            blocking=False,
            timeout=self._apply_lock_ttl_seconds,
        )
        owner_token = uuid.uuid4().hex
        acquired = await self._acquire_apply_lock_and_reset_marker(
            lock=lock,
            lock_key=lock_key,
            reconcile_marker_key=reconcile_marker_key,
            owner_token=owner_token,
        )
        if not acquired:
            raise LockError(f"apply lock already held: {lock_key}")

        lock_lost = asyncio.Event()
        renew_task = asyncio.create_task(
            self._renew_apply_lock(
                lock,
                lock_lost,
                coordinator_run_id=plan.coordinator_run_id,
            ),
            name=f"apply-lock-renew:{plan.coordinator_run_id}",
        )
        try:
            return await self._apply_locked(
                plan, parent_sandbox, minio_client, cancel_event, started_at,
                apply_lock=lock,
                lock_lost=lock_lost,
                snapshot_attempt_token=owner_token,
                lineage=lineage,
                on_rollback=on_rollback,
            )
        finally:
            renew_task.cancel()
            try:
                await renew_task
            except asyncio.CancelledError:
                pass
            await self._release_apply_lock_if_owned(
                lock,
                coordinator_run_id=plan.coordinator_run_id,
            )

    async def _acquire_apply_lock_and_reset_marker(
        self,
        *,
        lock: Any,
        lock_key: str,
        reconcile_marker_key: str,
        owner_token: str,
    ) -> bool:
        """Atomically acquire the owner token and end any missing cycle."""
        return await acquire_coordinator_apply_lock_and_reset_marker(
            redis=self._redis,
            lock=lock,
            lock_key=lock_key,
            reconcile_marker_key=reconcile_marker_key,
            owner_token=owner_token,
            ttl_seconds=self._apply_lock_ttl_seconds,
        )

    @staticmethod
    def _bind_lock_local_token(lock: Any, owner_token: str) -> None:
        """Bind an atomically-acquired token to redis-py's Lock adapter."""
        bind_coordinator_apply_lock_token(lock, owner_token)

    async def _renew_apply_lock(
        self,
        lock: Any,
        lock_lost: asyncio.Event,
        *,
        coordinator_run_id: str,
    ) -> None:
        """Renew only the currently-owned token; never reacquire a lost lock."""
        renew_interval = self._apply_lock_ttl_seconds / 3
        while not lock_lost.is_set():
            try:
                await self._lease_sleep(renew_interval)
                if lock_lost.is_set():
                    return
                if not await lock.owned():
                    lock_lost.set()
                    logger.error(
                        "PatchApplier apply lock owner lost run=%s",
                        coordinator_run_id,
                    )
                    return
                extended = await lock.extend(
                    self._apply_lock_ttl_seconds,
                    replace_ttl=True,
                )
                if not extended:
                    lock_lost.set()
                    logger.error(
                        "PatchApplier apply lock renewal rejected run=%s",
                        coordinator_run_id,
                    )
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                lock_lost.set()
                logger.exception(
                    "PatchApplier apply lock renewal failed run=%s",
                    coordinator_run_id,
                )
                return

    async def _apply_lock_is_owned(
        self,
        lock: Any,
        lock_lost: asyncio.Event,
        *,
        coordinator_run_id: str,
    ) -> bool:
        """Fail closed when ownership cannot be verified at a write boundary."""
        if lock_lost.is_set():
            return False
        try:
            owned = bool(await lock.owned())
        except Exception:
            logger.exception(
                "PatchApplier apply lock ownership probe failed run=%s",
                coordinator_run_id,
            )
            owned = False
        if not owned:
            lock_lost.set()
        return owned

    async def _release_apply_lock_if_owned(
        self,
        lock: Any,
        *,
        coordinator_run_id: str,
    ) -> None:
        """Best-effort compare-owner release; never mask the apply outcome."""
        try:
            if not await lock.owned():
                return
            await lock.release()
        except LockNotOwnedError:
            logger.warning(
                "PatchApplier apply lock changed owner before release run=%s",
                coordinator_run_id,
            )
        except Exception:
            logger.exception(
                "PatchApplier apply lock release failed run=%s",
                coordinator_run_id,
            )

    async def _apply_locked(
        self,
        plan: "PatchApplyPlan",
        parent_sandbox: "ParentSandboxPort",
        minio_client: "ArtifactStoragePort",
        cancel_event: Optional[asyncio.Event],
        started_at: float,
        *,
        apply_lock: Any,
        lock_lost: asyncio.Event,
        snapshot_attempt_token: str,
        lineage: Optional["GroupLineageFields"] = None,
        on_rollback: Optional[_RollbackPhaseCallback] = None,
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

        async def apply_owner_is_owned() -> bool:
            return await self._apply_lock_is_owned(
                apply_lock,
                lock_lost,
                coordinator_run_id=plan.coordinator_run_id,
            )

        async def finalize_owned(*args: Any, **kwargs: Any) -> ApplyOutcome:
            kwargs["snapshot_discard_guard"] = apply_owner_is_owned
            return await self._finalize(*args, **kwargs)

        # ── Step 3: preflight + snapshot ─────────────────────────────────
        # Wrap the whole loop in try/except: ParentSandboxPort methods
        # may raise (e.g. ``exists()`` now raises ``OSError`` on RPC
        # failure per the adapter contract — codex R1 P1#3 fix). A
        # preflight raise must NOT escape the lock context — translate
        # to WRITE_IO_ERROR and finalize so the audit row is closed.
        snapshots: list["FileSnapshot"] = []
        preflight_terminal_started = False
        try:
            for e in plan.files:
                if e.op in ("modify", "delete"):
                    # [S1b 2a] Inode-typed preflight: probe the DIRECT lstat
                    # kind so a target that is a special file (FIFO / socket
                    # / block / char) is rejected BEFORE any read / snapshot /
                    # write. ``check_path`` replaces the bool-only ``exists()``
                    # on this branch (the ``add`` branch keeps ``exists()``).
                    check = await parent_sandbox.check_path(e.path)
                    if not check.exists:
                        preflight_terminal_started = True
                        return await finalize_owned(
                            audit_id, ApplyStatus.FILE_MISSING,
                            failed=AppliedFileFailure(
                                path=e.path, reason="missing",
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            lineage=lineage,
                        )
                    if check.kind in ("fifo", "socket", "block", "char"):
                        preflight_terminal_started = True
                        return await finalize_owned(
                            audit_id, ApplyStatus.TARGET_SPECIAL_FILE,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=(
                                    f"target is special file "
                                    f"(kind={check.kind})"
                                ),
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            lineage=lineage,
                        )
                    # [S2 PR-4 §3.4 R4-H] kind invariant: a modify/delete target
                    # MUST be a regular file. symlink/directory/other → reject
                    # (the symlink-following exists() backstop is replaced).
                    if check.kind != "regular":
                        preflight_terminal_started = True
                        return await finalize_owned(
                            audit_id, ApplyStatus.PARENT_NOT_REGULAR,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=f"target not regular (kind={check.kind})",
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            lineage=lineage,
                        )
                    cur = await parent_sandbox.compute_digest(e.path)
                    if cur != e.base_digest:
                        preflight_terminal_started = True
                        return await finalize_owned(
                            audit_id, ApplyStatus.DIGEST_DRIFT,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=f"drift cur={cur}",
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            lineage=lineage,
                        )
                    # Snapshot AFTER digest match — saves wasted
                    # snapshot write on the abort path.
                    orig = await parent_sandbox.read_file(e.path)
                    if not await self._apply_lock_is_owned(
                        apply_lock,
                        lock_lost,
                        coordinator_run_id=plan.coordinator_run_id,
                    ):
                        preflight_terminal_started = True
                        return await self._finish_apply_lock_lost(
                            audit_id=audit_id,
                            failed_path=e.path,
                            applied=[],
                            snapshots=snapshots,
                            parent_sandbox=parent_sandbox,
                            plan=plan,
                            started_at=started_at,
                            lineage=lineage,
                            on_rollback=on_rollback,
                            apply_lock=apply_lock,
                        )
                    snap = await self._snapshot_store.save(
                        coordinator_run_id=plan.coordinator_run_id,
                        path=e.path,
                        content=orig,
                        original_digest=e.base_digest,
                        attempt_token=snapshot_attempt_token,
                    )
                    snapshots.append(snap)
                    if not await self._apply_lock_is_owned(
                        apply_lock,
                        lock_lost,
                        coordinator_run_id=plan.coordinator_run_id,
                    ):
                        preflight_terminal_started = True
                        return await self._finish_apply_lock_lost(
                            audit_id=audit_id,
                            failed_path=e.path,
                            applied=[],
                            snapshots=snapshots,
                            parent_sandbox=parent_sandbox,
                            plan=plan,
                            started_at=started_at,
                            lineage=lineage,
                            on_rollback=on_rollback,
                            apply_lock=apply_lock,
                        )
                elif e.op == "add":
                    # [S2 PR-4 §3.4 R4-H] kind invariant: an add target MUST be
                    # missing. check_path (lstat, no symlink follow) replaces the
                    # symlink-following exists() — a parent-side symlink at the
                    # target path must NOT be written through. A regular file →
                    # FILE_EXISTS (existing semantics); any other inode →
                    # PARENT_NOT_REGULAR.
                    add_check = await parent_sandbox.check_path(e.path)
                    if add_check.kind == "regular":
                        preflight_terminal_started = True
                        return await finalize_owned(
                            audit_id, ApplyStatus.FILE_EXISTS,
                            failed=AppliedFileFailure(
                                path=e.path, reason="exists",
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            lineage=lineage,
                        )
                    if add_check.kind != "missing":
                        preflight_terminal_started = True
                        return await finalize_owned(
                            audit_id, ApplyStatus.PARENT_NOT_REGULAR,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=f"add target not missing (kind={add_check.kind})",
                            ),
                            applied=[],
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            lineage=lineage,
                        )
        except Exception as preflight_exc:
            if preflight_terminal_started:
                raise
            if not await self._apply_lock_is_owned(
                apply_lock,
                lock_lost,
                coordinator_run_id=plan.coordinator_run_id,
            ):
                return await self._finish_apply_lock_lost(
                    audit_id=audit_id,
                    failed_path=e.path,
                    applied=[],
                    snapshots=snapshots,
                    parent_sandbox=parent_sandbox,
                    plan=plan,
                    started_at=started_at,
                    lineage=lineage,
                    on_rollback=on_rollback,
                    apply_lock=apply_lock,
                )
            # No writes have happened yet (only snapshot reads), so no
            # rollback is needed — but we DO need to discard any
            # snapshots already saved in earlier iterations so the FS
            # doesn't accumulate orphans.
            return await finalize_owned(
                audit_id, ApplyStatus.WRITE_IO_ERROR,
                failed=AppliedFileFailure(
                    path="<preflight>", reason=str(preflight_exc),
                ),
                applied=[],
                snapshots_to_discard=snapshots,
                started_at=started_at,
                plan=plan,
                lineage=lineage,
            )

        # ── Step 4: actual apply ─────────────────────────────────────────
        applied: list[AppliedFileRecord] = []
        for e in plan.files:
            # Lease loss outranks a simultaneous cooperative cancel: a lost
            # distributed owner is an independent safety diagnosis and must
            # stop the next filesystem write.
            if not await self._apply_lock_is_owned(
                apply_lock,
                lock_lost,
                coordinator_run_id=plan.coordinator_run_id,
            ):
                return await self._finish_apply_lock_lost(
                    audit_id=audit_id,
                    failed_path=e.path,
                    applied=applied,
                    snapshots=snapshots,
                    parent_sandbox=parent_sandbox,
                    plan=plan,
                    started_at=started_at,
                    lineage=lineage,
                    on_rollback=on_rollback,
                    apply_lock=apply_lock,
                )
            # Per-entry cancel check (rollback any writes already in flight).
            if cancel_event is not None and cancel_event.is_set():
                rb, rb_failed = await self._rollback_with_phase(
                    applied,
                    snapshots,
                    parent_sandbox,
                    plan,
                    on_rollback,
                    ownership_check=apply_owner_is_owned,
                )
                final_status = (
                    ApplyStatus.ROLLBACK_PARTIAL
                    if rb == "partial"
                    else ApplyStatus.APPLY_ABORTED
                )
                return await finalize_owned(
                    audit_id, final_status,
                    failed=None,
                    applied=applied,
                    snapshots_to_discard=snapshots,
                    started_at=started_at,
                    plan=plan,
                    lineage=lineage,
                    rollback_status=rb,
                    rollback_failed_paths=rb_failed,
                )
            terminal_handling_started = False
            try:
                if e.op == "delete":
                    await parent_sandbox.delete_file(e.path)
                    applied.append(AppliedFileRecord(path=e.path, op=e.op))
                    if not await self._apply_lock_is_owned(
                        apply_lock,
                        lock_lost,
                        coordinator_run_id=plan.coordinator_run_id,
                    ):
                        terminal_handling_started = True
                        return await self._finish_apply_lock_lost(
                            audit_id=audit_id,
                            failed_path=e.path,
                            applied=applied,
                            snapshots=snapshots,
                            parent_sandbox=parent_sandbox,
                            plan=plan,
                            started_at=started_at,
                            lineage=lineage,
                            on_rollback=on_rollback,
                            apply_lock=apply_lock,
                        )
                else:  # add or modify
                    try:
                        content = await minio_client.get_bytes(e.content_ref)
                    except Exception as fetch_exc:
                        if not await self._apply_lock_is_owned(
                            apply_lock,
                            lock_lost,
                            coordinator_run_id=plan.coordinator_run_id,
                        ):
                            terminal_handling_started = True
                            return await self._finish_apply_lock_lost(
                                audit_id=audit_id,
                                failed_path=e.path,
                                applied=applied,
                                snapshots=snapshots,
                                parent_sandbox=parent_sandbox,
                                plan=plan,
                                started_at=started_at,
                                lineage=lineage,
                                on_rollback=on_rollback,
                                apply_lock=apply_lock,
                            )
                        # Distinguish minio fetch from sandbox write
                        # failures so operators can debug the right
                        # subsystem.
                        terminal_handling_started = True
                        rb, rb_failed = await self._rollback_with_phase(
                            applied,
                            snapshots,
                            parent_sandbox,
                            plan,
                            on_rollback,
                            ownership_check=apply_owner_is_owned,
                        )
                        final_status = (
                            ApplyStatus.ROLLBACK_PARTIAL
                            if rb == "partial"
                            else ApplyStatus.MINIO_FETCH_FAILED
                        )
                        return await finalize_owned(
                            audit_id, final_status,
                            failed=AppliedFileFailure(
                                path=e.path,
                                reason=f"minio: {fetch_exc}",
                            ),
                            applied=applied,
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            lineage=lineage,
                            rollback_status=rb,
                            rollback_failed_paths=rb_failed,
                        )
                    if not await self._apply_lock_is_owned(
                        apply_lock,
                        lock_lost,
                        coordinator_run_id=plan.coordinator_run_id,
                    ):
                        terminal_handling_started = True
                        return await self._finish_apply_lock_lost(
                            audit_id=audit_id,
                            failed_path=e.path,
                            applied=applied,
                            snapshots=snapshots,
                            parent_sandbox=parent_sandbox,
                            plan=plan,
                            started_at=started_at,
                            lineage=lineage,
                            on_rollback=on_rollback,
                            apply_lock=apply_lock,
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
                        terminal_handling_started = True
                        rb, rb_failed = await self._rollback_with_phase(
                            applied,
                            snapshots,
                            parent_sandbox,
                            plan,
                            on_rollback,
                            ownership_check=apply_owner_is_owned,
                        )
                        final_status = (
                            ApplyStatus.ROLLBACK_PARTIAL
                            if rb == "partial"
                            else ApplyStatus.MINIO_FETCH_FAILED
                        )
                        return await finalize_owned(
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
                            lineage=lineage,
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
                        terminal_handling_started = True
                        rb, rb_failed = await self._rollback_with_phase(
                            applied,
                            snapshots,
                            parent_sandbox,
                            plan,
                            on_rollback,
                            ownership_check=apply_owner_is_owned,
                        )
                        final_status = (
                            ApplyStatus.ROLLBACK_PARTIAL
                            if rb == "partial"
                            else ApplyStatus.APPLY_ABORTED
                        )
                        return await finalize_owned(
                            audit_id, final_status,
                            failed=None,
                            applied=applied,
                            snapshots_to_discard=snapshots,
                            started_at=started_at,
                            plan=plan,
                            lineage=lineage,
                            rollback_status=rb,
                            rollback_failed_paths=rb_failed,
                        )
                    await parent_sandbox.atomic_write_file(e.path, content)
                    # The write completed before the await returned. Record it
                    # before probing ownership so a loss at this boundary rolls
                    # back this file as well as earlier ones.
                    applied.append(AppliedFileRecord(path=e.path, op=e.op))
                    if not await self._apply_lock_is_owned(
                        apply_lock,
                        lock_lost,
                        coordinator_run_id=plan.coordinator_run_id,
                    ):
                        terminal_handling_started = True
                        return await self._finish_apply_lock_lost(
                            audit_id=audit_id,
                            failed_path=e.path,
                            applied=applied,
                            snapshots=snapshots,
                            parent_sandbox=parent_sandbox,
                            plan=plan,
                            started_at=started_at,
                            lineage=lineage,
                            on_rollback=on_rollback,
                            apply_lock=apply_lock,
                        )
                    # Post-write digest verify: read the file back from the
                    # sandbox and compare. The previous version hashed the
                    # bytes we just sent, which only confirms our local
                    # copy — sandbox-side truncation or corruption would
                    # pass that check undetected. Re-reading is an extra
                    # RPC per entry but is the only correct way to catch
                    # sandbox-side write defects ([codex R1 P1#1]).
                    actual = await parent_sandbox.compute_digest(e.path)
                    if not await self._apply_lock_is_owned(
                        apply_lock,
                        lock_lost,
                        coordinator_run_id=plan.coordinator_run_id,
                    ):
                        terminal_handling_started = True
                        return await self._finish_apply_lock_lost(
                            audit_id=audit_id,
                            failed_path=e.path,
                            applied=applied,
                            snapshots=snapshots,
                            parent_sandbox=parent_sandbox,
                            plan=plan,
                            started_at=started_at,
                            lineage=lineage,
                            on_rollback=on_rollback,
                            apply_lock=apply_lock,
                        )
                    if actual != e.new_digest:
                        terminal_handling_started = True
                        rb, rb_failed = await self._rollback_with_phase(
                            applied,
                            snapshots,
                            parent_sandbox,
                            plan,
                            on_rollback,
                            ownership_check=apply_owner_is_owned,
                        )
                        final_status = (
                            ApplyStatus.ROLLBACK_PARTIAL
                            if rb == "partial"
                            else ApplyStatus.POST_WRITE_DIGEST_MISMATCH
                        )
                        return await finalize_owned(
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
                            lineage=lineage,
                            rollback_status=rb,
                            rollback_failed_paths=rb_failed,
                        )
            except Exception as exc:
                if terminal_handling_started:
                    raise
                if not await self._apply_lock_is_owned(
                    apply_lock,
                    lock_lost,
                    coordinator_run_id=plan.coordinator_run_id,
                ):
                    return await self._finish_apply_lock_lost(
                        audit_id=audit_id,
                        failed_path=e.path,
                        applied=applied,
                        snapshots=snapshots,
                        parent_sandbox=parent_sandbox,
                        plan=plan,
                        started_at=started_at,
                        lineage=lineage,
                        on_rollback=on_rollback,
                        apply_lock=apply_lock,
                    )
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
                # As of **S1** (atomic_write_file design) the live sandbox
                # HTTP service backing ``SandboxHandle.upload_file`` writes
                # via ``mkstemp + fsync + os.replace`` (``_atomic_write_bytes``
                # in ``sandbox/app/services/file.py``), so per-file
                # final-path-content atomicity now holds for regular-file
                # targets (the content a patch entry carries in practice): a
                # write that raised left no truncated file, so NOT adding the
                # current entry to rollback is sound.
                # (As of **S1b** the apply path DOES stat-guard the target
                # inode type for the special set: the modify/delete preflight
                # rejects a direct FIFO/socket/block/char target with
                # ApplyStatus.TARGET_SPECIAL_FILE before the read (2a), and the
                # coordinator write refuses a special target with OSError
                # instead of the D12 write-through (2b, refuse_special=True),
                # so a TOCTOU special swap surfaces here as WRITE_IO_ERROR +
                # rollback rather than a FIFO hang. The agent's own write_file
                # keeps D12 unchanged. A leftover empty parent directory on a
                # pre-replace failure is still a benign side effect — S1 §1.
                # Path resolution / Gap B is closed by G2b — not an S1b concern.)
                #
                # Including the current entry in rollback would ALSO cause
                # a false ROLLBACK_PARTIAL status when the rollback's
                # restore write fails on a file the apply write never
                # actually touched (typical case: 4xx-class RPC failure
                # before any byte hit the sandbox).
                rb, rb_failed = await self._rollback_with_phase(
                    applied,
                    snapshots,
                    parent_sandbox,
                    plan,
                    on_rollback,
                    ownership_check=apply_owner_is_owned,
                )
                final_status = (
                    ApplyStatus.ROLLBACK_PARTIAL
                    if rb == "partial"
                    else ApplyStatus.WRITE_IO_ERROR
                )
                return await finalize_owned(
                    audit_id, final_status,
                    failed=AppliedFileFailure(
                        path=e.path, reason=str(exc),
                    ),
                    applied=applied,
                    snapshots_to_discard=snapshots,
                    started_at=started_at,
                    plan=plan,
                    lineage=lineage,
                    rollback_status=rb,
                    rollback_failed_paths=rb_failed,
                )

        # ── Step 5: discard snapshots on success ─────────────────────────
        return await finalize_owned(
            audit_id, ApplyStatus.SUCCESS,
            failed=None,
            applied=applied,
            snapshots_to_discard=snapshots,
            started_at=started_at,
            plan=plan,
            lineage=lineage,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    async def _finish_apply_lock_lost(
        self,
        *,
        audit_id: int,
        failed_path: str,
        applied: list[AppliedFileRecord],
        snapshots: list["FileSnapshot"],
        parent_sandbox: "ParentSandboxPort",
        plan: "PatchApplyPlan",
        started_at: float,
        lineage: Optional["GroupLineageFields"],
        on_rollback: Optional[_RollbackPhaseCallback],
        apply_lock: Any,
    ) -> ApplyOutcome:
        """Stop new writes; rollback only under verified exclusive ownership."""
        rollback_status: Optional[str] = None
        rollback_failed_paths: list[str] = []
        rollback_lock = None
        if applied or snapshots:
            rollback_lock = await self._acquire_rollback_ownership(
                plan.coordinator_run_id,
                previous_lock=apply_lock,
            )
        if rollback_lock is None:
            if applied:
                rollback_status = "partial"
                rollback_failed_paths = [record.path for record in applied]
                await self._emit_rollback_partial_health(
                    plan,
                    rollback_failed_paths,
                    detail="exclusive rollback ownership unavailable",
                )
            status = (
                ApplyStatus.ROLLBACK_PARTIAL
                if rollback_status == "partial"
                else ApplyStatus.APPLY_LOCK_LOST
            )
            return await self._finalize(
                audit_id,
                status,
                failed=AppliedFileFailure(
                    path=failed_path,
                    reason=ApplyStatus.APPLY_LOCK_LOST.value,
                ),
                applied=applied,
                # Snapshot names are shared by run/path. Without canonical
                # ownership, preserving old files is safer than deleting a
                # replacement owner's rollback data.
                snapshots_to_discard=[],
                started_at=started_at,
                plan=plan,
                rollback_status=rollback_status,
                rollback_failed_paths=rollback_failed_paths,
                lineage=lineage,
            )

        rollback_lost = asyncio.Event()
        renew_task = asyncio.create_task(
            self._renew_apply_lock(
                rollback_lock,
                rollback_lost,
                coordinator_run_id=plan.coordinator_run_id,
            ),
            name=f"apply-rollback-lock-renew:{plan.coordinator_run_id}",
        )

        async def rollback_owned() -> bool:
            return await self._apply_lock_is_owned(
                rollback_lock,
                rollback_lost,
                coordinator_run_id=plan.coordinator_run_id,
            )

        try:
            if applied:
                rollback_status, rollback_failed_paths = (
                    await self._rollback_with_phase(
                        applied,
                        snapshots,
                        parent_sandbox,
                        plan,
                        on_rollback,
                        ownership_check=rollback_owned,
                    )
                )
            status = (
                ApplyStatus.ROLLBACK_PARTIAL
                if rollback_status == "partial"
                else ApplyStatus.APPLY_LOCK_LOST
            )
            return await self._finalize(
                audit_id,
                status,
                failed=AppliedFileFailure(
                    path=failed_path,
                    reason=ApplyStatus.APPLY_LOCK_LOST.value,
                ),
                applied=applied,
                snapshots_to_discard=snapshots,
                started_at=started_at,
                plan=plan,
                rollback_status=rollback_status,
                rollback_failed_paths=rollback_failed_paths,
                lineage=lineage,
                snapshot_discard_guard=rollback_owned,
            )
        finally:
            renew_task.cancel()
            try:
                await renew_task
            except asyncio.CancelledError:
                pass
            await self._release_apply_lock_if_owned(
                rollback_lock,
                coordinator_run_id=plan.coordinator_run_id,
            )

    async def _acquire_rollback_ownership(
        self,
        coordinator_run_id: str,
        *,
        previous_lock: Any,
    ) -> Any | None:
        """Use the still-owned token or acquire a fresh token on the same key.

        A stale owner must never mutate the sandbox while a replacement apply
        owns the canonical lock. Redis errors therefore fail closed and route
        the run to manual recovery instead of performing an unsafe rollback.
        """
        try:
            if await previous_lock.owned():
                # A renew-loop error can set the loss signal while this token
                # is still present but close to expiry. Refresh synchronously
                # before the first rollback write; a rejected refresh is not
                # sufficient ownership evidence.
                extended = await previous_lock.extend(
                    self._apply_lock_ttl_seconds,
                    replace_ttl=True,
                )
                if extended:
                    return previous_lock
        except Exception:
            logger.exception(
                "PatchApplier previous apply owner probe failed before rollback "
                "run=%s",
                coordinator_run_id,
            )

        lock_key = coordinator_apply_lock_key(coordinator_run_id)
        rollback_lock = self._redis.lock(
            lock_key,
            blocking=False,
            timeout=self._apply_lock_ttl_seconds,
        )
        owner_token = uuid.uuid4().hex
        try:
            acquired = await self._acquire_apply_lock_and_reset_marker(
                lock=rollback_lock,
                lock_key=lock_key,
                reconcile_marker_key=coordinator_apply_reconcile_marker_key(
                    coordinator_run_id,
                ),
                owner_token=owner_token,
            )
        except Exception:
            logger.exception(
                "PatchApplier rollback ownership acquire failed run=%s",
                coordinator_run_id,
            )
            return None
        if not acquired:
            logger.error(
                "PatchApplier rollback skipped because replacement owner holds "
                "apply lock run=%s",
                coordinator_run_id,
            )
            return None
        return rollback_lock

    async def _rollback_with_phase(
        self,
        applied: list[AppliedFileRecord],
        snapshots: list["FileSnapshot"],
        parent_sandbox: "ParentSandboxPort",
        plan: "PatchApplyPlan",
        on_rollback: Optional[_RollbackPhaseCallback],
        ownership_check: Optional[_OwnershipCheckCallable] = None,
    ) -> "tuple[str, list[str]]":
        """Announce rollback before its first side effect, then always undo."""
        if on_rollback is not None:
            try:
                on_rollback()
            except asyncio.CancelledError:
                logger.warning(
                    "PatchApplier rollback phase callback failed run=%s; "
                    "rollback continues",
                    plan.coordinator_run_id,
                    exc_info=True,
                )
            except Exception:
                logger.warning(
                    "PatchApplier rollback phase callback failed run=%s; "
                    "rollback continues",
                    plan.coordinator_run_id,
                    exc_info=True,
                )
        if ownership_check is None:
            return await self._rollback(
                applied, snapshots, parent_sandbox, plan,
            )
        return await self._rollback(
            applied, snapshots, parent_sandbox, plan,
            ownership_check=ownership_check,
        )

    async def _rollback(
        self,
        applied: list[AppliedFileRecord],
        snapshots: list["FileSnapshot"],
        parent_sandbox: "ParentSandboxPort",
        plan: "PatchApplyPlan",
        *,
        ownership_check: Optional[_OwnershipCheckCallable] = None,
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
        pending = list(reversed(applied))
        for index, record in enumerate(pending):
            if ownership_check is not None and not await ownership_check():
                failed_paths.extend(item.path for item in pending[index:])
                break
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
                    if (
                        ownership_check is not None
                        and not await ownership_check()
                    ):
                        failed_paths.extend(
                            item.path for item in pending[index:]
                        )
                        break
                    await parent_sandbox.atomic_write_file(
                        snap.original_path, content,
                    )
                if ownership_check is not None and not await ownership_check():
                    failed_paths.extend(item.path for item in pending[index:])
                    break
            except Exception as exc:
                logger.error(
                    "PatchApplier rollback failed for %s (op=%s): %s",
                    record.path, record.op, exc,
                )
                failed_paths.append(record.path)

        if failed_paths:
            await self._emit_rollback_partial_health(plan, failed_paths)
            return "partial", failed_paths
        return "complete", []

    async def _emit_rollback_partial_health(
        self,
        plan: "PatchApplyPlan",
        failed_paths: list[str],
        *,
        detail: str = "paths failed restore",
    ) -> None:
        # Late import to avoid pulling the full event/pydantic graph into the
        # patch_applier module-load path.
        from app.domain.models.event import HealthEvent, HealthStatus
        try:
            await self._emit_event(HealthEvent(
                status=HealthStatus.TERMINATING,
                reason=(
                    f"PatchApplier rollback partial on run "
                    f"{plan.coordinator_run_id}: {len(failed_paths)} {detail}"
                ),
                action="hard_terminate",
                metrics={
                    "code": "coordinator_apply_rollback_partial",
                    "coordinator_run_id": plan.coordinator_run_id,
                    "failed_paths": failed_paths,
                },
            ))
        except Exception as emit_exc:
            # Emit failure must not mask the rollback result. Log and continue
            # — the audit row + return value still carry the partial signal.
            logger.error(
                "PatchApplier emit_event(rollback_partial) raised: %s",
                emit_exc,
            )

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
        lineage: Optional["GroupLineageFields"] = None,
        snapshot_discard_guard: Optional[_OwnershipCheckCallable] = None,
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
        # Check immediately before every cleanup side effect. Snapshot paths
        # are also attempt-token scoped, so a lease loss during the individual
        # filesystem await cannot alias a replacement owner's rollback file.
        for snapshot in snapshots_to_discard:
            if (
                snapshot_discard_guard is not None
                and not await snapshot_discard_guard()
            ):
                logger.error(
                    "PatchApplier snapshot discard skipped after ownership "
                    "loss run=%s",
                    plan.coordinator_run_id,
                )
                break
            try:
                await self._snapshot_store.discard(
                    snapshot.coordinator_run_id,
                    [snapshot],
                )
            except Exception as exc:
                # Don't mask the result on a cleanup failure.
                logger.error(
                    "PatchApplier snapshot discard failed: %s", exc,
                )

        # C2 PR-8 §13 Task 8.4 — emit CoordinatorApplyEvent. BEST-EFFORT:
        # try/except so a queue/serialization failure does not change the
        # ApplyOutcome we return. CoordinatorLineageMixin allows all 5
        # lineage fields to be None.
        #
        # [PR-9b-B INV-B4] When ``lineage`` is supplied by the caller
        # (main_graph._run_parallel_backend → B6), thread the group-level
        # ``root_session_id`` / ``parent_session_id`` onto the event for
        # full lineage parity with dispatch/reduce/sibling_cancel events.
        # Apply is a group-level operation, so the per-child fields
        # (``child_session_id`` / ``work_unit_id``) are deliberately NOT
        # set. Legacy callers (lineage=None) leave root/parent as the
        # mixin default None; frontend then correlates apply events via
        # ``coordinator_run_id`` to the parallel CoordinatorDispatchEvent.
        if self._emit_event is not None:
            try:
                from app.domain.models.event import CoordinatorApplyEvent
                event_kwargs: dict[str, Any] = {
                    "apply_status": status.value,
                    "file_count": len(applied),
                    "total_bytes": plan.total_size_bytes,
                    "failed_at_path": failed.path if failed else None,
                    "rollback_status": rollback_status,
                    "coordinator_run_id": plan.coordinator_run_id,
                }
                if lineage is not None:
                    if lineage.root_session_id is not None:
                        event_kwargs["root_session_id"] = lineage.root_session_id
                    if lineage.parent_session_id is not None:
                        event_kwargs["parent_session_id"] = lineage.parent_session_id
                await self._emit_event(CoordinatorApplyEvent(**event_kwargs))
            except Exception as emit_exc:
                logger.warning(
                    "PatchApplier emit_event(CoordinatorApplyEvent) "
                    "raised run=%s: %s",
                    plan.coordinator_run_id, emit_exc,
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
