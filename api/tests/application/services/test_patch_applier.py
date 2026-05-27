"""C2 PR-5 Task 5.6 — PatchApplier process-stable apply + rollback tests.

Spec ref: §10 (whole).

Coverage matrix:
- Happy path: SUCCESS with audit + discard
- Preflight aborts: FILE_MISSING / DIGEST_DRIFT / FILE_EXISTS (no snapshot, no rollback)
- Mid-apply aborts: WRITE_IO_ERROR / POST_WRITE_DIGEST_MISMATCH / MINIO_FETCH_FAILED (rollback)
- Cancel before write → APPLY_ABORTED (no audit row, no rollback)
- Cancel mid-apply → APPLY_ABORTED + rollback
- Rollback partial → emits HealthEvent with status=TERMINATING
- Audit row lifecycle: insert_in_progress → update_terminal
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.patch_applier import (
    ApplyStatus,
    PatchApplier,
)
from app.application.services.rollback_snapshot_store import FileSnapshot
from app.domain.models.event import HealthEvent, HealthStatus
from app.domain.models.patch_apply_plan import PatchApplyPlan
from app.domain.models.patch_manifest import FilePatchEntry

pytestmark = pytest.mark.anyio


_SHA_A = hashlib.sha256(b"a").hexdigest()
_SHA_B = hashlib.sha256(b"b").hexdigest()
_NEW_CONTENT = b"new content"
_NEW_DIGEST = hashlib.sha256(_NEW_CONTENT).hexdigest()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _NoopAsyncCM:
    """Minimal async-context-manager stand-in for Redis Lock.

    redis-py's ``Lock`` supports ``async with`` via ``__aenter__`` /
    ``__aexit__``. Tests don't need real lock semantics — the applier
    just needs the with-block to enter and exit without exception.
    """

    async def __aenter__(self) -> "_NoopAsyncCM":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


@pytest.fixture
def parent_sandbox() -> MagicMock:
    """Default fixture: preflight matches base_digest=_SHA_A; post-write
    verify returns _NEW_DIGEST. The applier now reads back from sandbox
    after each write (codex R1 P1#1 fix), so compute_digest is called
    twice per modify entry — once in preflight, once post-write.
    side_effect cycles: [_SHA_A (preflight), _NEW_DIGEST (post-write)].
    """
    s = MagicMock()
    s.exists = AsyncMock(return_value=True)
    s.compute_digest = AsyncMock(side_effect=[_SHA_A, _NEW_DIGEST])
    s.read_file = AsyncMock(return_value=b"original")
    s.atomic_write_file = AsyncMock()
    s.delete_file = AsyncMock()
    return s


@pytest.fixture
def minio() -> MagicMock:
    m = MagicMock()
    m.get_bytes = AsyncMock(return_value=_NEW_CONTENT)
    return m


@pytest.fixture
def snapshot_store() -> MagicMock:
    store = MagicMock()

    async def _save(*, coordinator_run_id, path, content, original_digest):
        return FileSnapshot(
            coordinator_run_id=coordinator_run_id,
            original_path=path,
            snapshot_path=f"/tmp/snap/{path}",
            original_digest=original_digest,
        )

    store.save = AsyncMock(side_effect=_save)
    store.load = AsyncMock(return_value=b"original")
    store.discard = AsyncMock()
    return store


@pytest.fixture
def audit_repo() -> MagicMock:
    r = MagicMock()
    r.insert_in_progress = AsyncMock(return_value=42)
    r.update_terminal = AsyncMock()
    return r


@pytest.fixture
def redis_mock() -> MagicMock:
    r = MagicMock()
    r.lock = MagicMock(return_value=_NoopAsyncCM())
    return r


@pytest.fixture
def emit_event() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def applier(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
    emit_event: AsyncMock,
) -> PatchApplier:
    return PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=redis_mock,
        emit_event=emit_event,
    )


def _modify_plan() -> PatchApplyPlan:
    return PatchApplyPlan(
        coordinator_run_id="r1",
        files=(
            FilePatchEntry(
                path="x.py", op="modify",
                base_digest=_SHA_A,
                new_digest=_NEW_DIGEST,
                content_ref="ref-r1",
                content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=len(_NEW_CONTENT),
        file_count=1,
        source_work_unit_ids=("wu1",),
    )


def _add_plan() -> PatchApplyPlan:
    return PatchApplyPlan(
        coordinator_run_id="r1",
        files=(
            FilePatchEntry(
                path="new.py", op="add",
                new_digest=_NEW_DIGEST,
                content_ref="ref-add",
                content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=len(_NEW_CONTENT),
        file_count=1,
        source_work_unit_ids=("wu1",),
    )


# ─── Happy path ──────────────────────────────────────────────────────────────


async def test_happy_path_success(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
) -> None:
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.SUCCESS
    assert len(out.applied_files) == 1
    assert out.applied_files[0].path == "x.py"
    assert out.rollback_status is None
    parent_sandbox.atomic_write_file.assert_awaited_once_with(
        "x.py", _NEW_CONTENT,
    )
    audit_repo.insert_in_progress.assert_awaited_once()
    audit_repo.update_terminal.assert_awaited_once()
    update_kwargs = audit_repo.update_terminal.await_args.kwargs
    assert update_kwargs["status"] == "success"
    snapshot_store.discard.assert_awaited()


async def test_add_happy_path(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """``add`` ops skip the modify/delete preflight branch so
    ``compute_digest`` is called exactly ONCE — the post-write verify.
    Override the fixture's two-item side_effect to a single
    post-write digest value that matches the manifest's new_digest."""
    parent_sandbox.exists = AsyncMock(return_value=False)
    parent_sandbox.compute_digest = AsyncMock(return_value=_NEW_DIGEST)
    plan = _add_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.SUCCESS
    parent_sandbox.atomic_write_file.assert_awaited_once_with(
        "new.py", _NEW_CONTENT,
    )


# ─── Preflight aborts ────────────────────────────────────────────────────────


async def test_digest_drift_aborts_without_rollback(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    snapshot_store: MagicMock,
) -> None:
    """Drift detected at preflight → DIGEST_DRIFT. No snapshots taken
    (drift caught before snapshot save), no rollback needed."""
    parent_sandbox.compute_digest = AsyncMock(return_value=_SHA_B)
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.DIGEST_DRIFT
    assert out.failed_at is not None
    assert out.failed_at.path == "x.py"
    parent_sandbox.atomic_write_file.assert_not_called()
    snapshot_store.save.assert_not_called()
    assert out.rollback_status is None


async def test_file_missing_aborts(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    parent_sandbox.exists = AsyncMock(return_value=False)
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.FILE_MISSING
    assert out.failed_at is not None
    assert out.failed_at.path == "x.py"


async def test_file_exists_aborts_add(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """``add`` op against an already-existing path → FILE_EXISTS."""
    parent_sandbox.exists = AsyncMock(return_value=True)
    plan = _add_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.FILE_EXISTS
    parent_sandbox.atomic_write_file.assert_not_called()


# ─── Mid-apply aborts ────────────────────────────────────────────────────────


async def test_minio_fetch_failed_triggers_rollback(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    snapshot_store: MagicMock,
) -> None:
    """MinIO fetch fails AFTER snapshot is taken → MINIO_FETCH_FAILED
    with rollback (status='complete' since no writes happened yet, but
    the rollback path still executes and the rollback_status='complete'
    pins the contract)."""
    minio.get_bytes = AsyncMock(
        side_effect=RuntimeError("minio: connection refused"),
    )
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.MINIO_FETCH_FAILED
    assert out.rollback_status == "complete"


async def test_post_write_digest_mismatch_rollback(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """Sandbox-side post-write digest doesn't match the manifest's
    ``new_digest`` (truncation / corruption on the sandbox side). The
    applier re-reads via ``parent_sandbox.compute_digest`` and detects
    the divergence → rollback.

    [codex R1 P1#1 fix] The previous version hashed the bytes the
    applier sent locally, which would never catch sandbox-side write
    defects. Now compute_digest is the authority."""
    # Preflight returns _SHA_A (base matches); post-write returns
    # _SHA_B (mismatch with _NEW_DIGEST).
    parent_sandbox.compute_digest = AsyncMock(side_effect=[_SHA_A, _SHA_B])
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.POST_WRITE_DIGEST_MISMATCH
    assert out.rollback_status == "complete"


async def test_write_io_error_triggers_rollback(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """The apply write fails on the only entry; the applier rolls
    back (no-op since ``applied`` is empty — the current entry isn't
    added to applied because the write didn't fully succeed; per
    ParentSandboxPort atomic_write_file contract a raise = no
    side-effect on sandbox). Pins the canonical 'apply failed → no
    partial state, rollback_status=complete' flow.

    [codex R8 P2 doc update] Previously the docstring said the
    rollback would call ``atomic_write_file`` a second time to
    restore; that's wrong — with a 1-entry plan and no successful
    writes, rollback walks an empty applied list and does nothing.
    The single ``raise OSError`` below is the only sandbox write the
    test triggers."""
    call_counter = {"n": 0}

    async def _write(path: str, content: bytes) -> None:
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            raise OSError("disk full")
        # No further writes expected; rollback walks empty applied list.

    parent_sandbox.atomic_write_file = AsyncMock(side_effect=_write)
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.WRITE_IO_ERROR
    assert out.failed_at is not None
    assert "disk full" in out.failed_at.reason
    assert out.rollback_status == "complete"


# ─── Cancel paths ────────────────────────────────────────────────────────────


async def test_cancel_before_audit_no_audit_row(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    audit_repo: MagicMock,
) -> None:
    """[spec §10.2 Step 2] cancel-before-audit: no audit row, no
    sandbox writes, no rollback. The orchestrator surfaces this via
    the SSE "step cancelled" path, not the apply audit."""
    cancel = asyncio.Event()
    cancel.set()
    plan = _modify_plan()
    out = await applier.apply(
        plan,
        parent_sandbox=parent_sandbox,
        minio_client=minio,
        cancel_event=cancel,
    )
    assert out.status == ApplyStatus.APPLY_ABORTED
    audit_repo.insert_in_progress.assert_not_called()
    parent_sandbox.atomic_write_file.assert_not_called()


async def test_cancel_mid_apply_triggers_rollback(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """Two-entry plan; cancel fires between entries. First entry was
    written (snapshot exists), second never starts. Rollback restores
    the first entry."""
    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(
            FilePatchEntry(
                path="a.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref-a", content_size=len(_NEW_CONTENT),
            ),
            FilePatchEntry(
                path="b.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref-b", content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=2 * len(_NEW_CONTENT),
        file_count=2,
        source_work_unit_ids=("wu1",),
    )

    cancel = asyncio.Event()
    call_count = {"n": 0}

    async def _write(path: str, content: bytes) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            cancel.set()

    parent_sandbox.atomic_write_file = AsyncMock(side_effect=_write)
    # 3 compute_digest calls: 2 preflight (a, b) + 1 post-write (a).
    parent_sandbox.compute_digest = AsyncMock(
        side_effect=[_SHA_A, _SHA_A, _NEW_DIGEST],
    )

    applier = PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=redis_mock,
        emit_event=emit_event,
    )
    out = await applier.apply(
        plan,
        parent_sandbox=parent_sandbox,
        minio_client=minio,
        cancel_event=cancel,
    )
    assert out.status == ApplyStatus.APPLY_ABORTED
    assert out.rollback_status == "complete"
    # Two snapshots taken (preflight saved both before apply started)
    assert snapshot_store.save.await_count == 2


# ─── Rollback partial → HealthEvent ──────────────────────────────────────────


async def test_rollback_partial_emits_health_event(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """Apply fails mid-way AND rollback's restore write itself fails →
    ROLLBACK_PARTIAL audit status + HealthEvent emitted to operator."""
    write_count = {"n": 0}

    async def _write(path: str, content: bytes) -> None:
        write_count["n"] += 1
        if write_count["n"] == 2:
            raise OSError("write 2 fails")
        if write_count["n"] == 3:
            raise OSError("rollback write fails")

    parent_sandbox.atomic_write_file = AsyncMock(side_effect=_write)
    # compute_digest sequence: 2 preflight + 1 post-write of entry 1
    # (entry 2 never gets to post-write because its apply raises).
    parent_sandbox.compute_digest = AsyncMock(
        side_effect=[_SHA_A, _SHA_A, _NEW_DIGEST],
    )

    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(
            FilePatchEntry(
                path="a.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref-a", content_size=len(_NEW_CONTENT),
            ),
            FilePatchEntry(
                path="b.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref-b", content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=2 * len(_NEW_CONTENT),
        file_count=2,
        source_work_unit_ids=("wu1",),
    )

    applier = PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=redis_mock,
        emit_event=emit_event,
    )
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )

    # [codex R1 P1#4] When rollback is partial, the terminal status is
    # overridden to ROLLBACK_PARTIAL — the original failure cause is
    # preserved in ``failed_at`` (the apply-time WRITE_IO_ERROR cause).
    assert out.status == ApplyStatus.ROLLBACK_PARTIAL
    assert out.rollback_status == "partial"
    assert out.failed_at is not None
    assert "write 2 fails" in out.failed_at.reason
    assert emit_event.await_count == 1
    emitted = emit_event.await_args.args[0]
    assert isinstance(emitted, HealthEvent)
    assert emitted.status == HealthStatus.TERMINATING
    assert emitted.metrics is not None
    assert (
        emitted.metrics["code"] == "coordinator_apply_rollback_partial"
    )


# ─── Audit lifecycle ────────────────────────────────────────────────────────


async def test_add_then_failure_rollback_deletes_added_file(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """[codex R1 P0#1 fix] Plan: add 'a.py' (succeeds) → modify 'b.py'
    (fails). Rollback MUST delete 'a.py' (the freshly-added file) AND
    restore 'b.py' (but b.py was never modified, so just no-op). The
    previous version walked only ``snapshots`` and never undid add ops,
    leaving 'a.py' behind on rollback — a direct violation of the
    all-or-nothing apply contract."""
    # add 'a.py' doesn't exist (preflight); modify 'b.py' exists.
    exists_seq = [False, True]  # a.py preflight, b.py preflight
    parent_sandbox.exists = AsyncMock(side_effect=exists_seq)
    # compute_digest: b.py preflight = _SHA_A; a.py post-write = _NEW_DIGEST
    parent_sandbox.compute_digest = AsyncMock(
        side_effect=[_SHA_A, _NEW_DIGEST],
    )
    # First write (a.py) succeeds; second (b.py) raises.
    write_count = {"n": 0}

    async def _write(path: str, content: bytes) -> None:
        write_count["n"] += 1
        if write_count["n"] == 2:
            raise OSError("write 2 fails")

    parent_sandbox.atomic_write_file = AsyncMock(side_effect=_write)

    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(
            FilePatchEntry(
                path="a.py", op="add",
                new_digest=_NEW_DIGEST,
                content_ref="ref-a", content_size=len(_NEW_CONTENT),
            ),
            FilePatchEntry(
                path="b.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref-b", content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=2 * len(_NEW_CONTENT),
        file_count=2,
        source_work_unit_ids=("wu1",),
    )

    applier = PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=redis_mock,
        emit_event=emit_event,
    )
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )

    assert out.status == ApplyStatus.WRITE_IO_ERROR
    assert out.rollback_status == "complete"
    # The rollback path called delete_file('a.py') in reverse order
    # (a.py was the only entry in ``applied``).
    parent_sandbox.delete_file.assert_awaited()
    delete_calls = [
        c.args for c in parent_sandbox.delete_file.await_args_list
    ]
    assert ("a.py",) in delete_calls


async def test_audit_insert_then_terminal(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    audit_repo: MagicMock,
) -> None:
    """Both audit calls happen in order: insert_in_progress before
    sandbox writes; update_terminal with audit_id from insert."""
    plan = _modify_plan()
    await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    audit_repo.insert_in_progress.assert_awaited_once()
    audit_repo.update_terminal.assert_awaited_once()
    update_args = audit_repo.update_terminal.await_args
    assert update_args.args[0] == 42


async def test_audit_carries_parent_session_id(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    audit_repo: MagicMock,
) -> None:
    """parent_session_id is parsed from coordinator_run_id per §4.7:
    everything before the first colon."""
    plan = PatchApplyPlan(
        coordinator_run_id="sess_abc:hash16:a0",
        files=(
            FilePatchEntry(
                path="x.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref", content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=len(_NEW_CONTENT),
        file_count=1,
        source_work_unit_ids=("wu1",),
    )
    await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    insert_kwargs = audit_repo.insert_in_progress.await_args.kwargs
    assert insert_kwargs["parent_session_id"] == "sess_abc"


async def test_audit_carries_plan_hash(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    audit_repo: MagicMock,
) -> None:
    """[codex R6 P2] insert_in_progress receives a deterministic
    plan_hash (SHA-256 hex over canonical PatchApplyPlan JSON). PR-7
    rehydrate uses this as the idempotency key when deciding whether
    a retry is replaying the same plan."""
    plan = _modify_plan()
    await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    insert_kwargs = audit_repo.insert_in_progress.await_args.kwargs
    plan_hash = insert_kwargs.get("plan_hash")
    assert plan_hash is not None
    assert len(plan_hash) == 64
    assert all(c in "0123456789abcdef" for c in plan_hash)


async def test_audit_carries_total_bytes(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    audit_repo: MagicMock,
) -> None:
    """[codex R6 P2] _finalize threads plan.total_size_bytes into
    update_terminal so the audit row captures the plan size for
    operator review + PR-6 budget reconciliation."""
    plan = _modify_plan()  # total_size_bytes = len(_NEW_CONTENT)
    await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    update_kwargs = audit_repo.update_terminal.await_args.kwargs
    assert update_kwargs["total_bytes"] == len(_NEW_CONTENT)


async def test_content_size_mismatch_triggers_fetch_failed(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """[codex R5 P1 + R6 P2] MinIO returns a blob whose length doesn't
    match the manifest's declared ``content_size``. Applier must
    reject pre-write — the post-write digest check would only catch
    blob-content drift, not a manifest claiming 0 bytes for a huge
    blob (budget / audit escape)."""
    # Fixture's _NEW_CONTENT is 11 bytes; manifest declares 11; the
    # mismatch comes from minio returning different-length bytes.
    minio.get_bytes = AsyncMock(return_value=b"differently sized content!!")
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.MINIO_FETCH_FAILED
    assert out.failed_at is not None
    assert "content_size mismatch" in out.failed_at.reason


async def test_cancel_between_get_bytes_and_write_aborts(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """[codex R3 P1 + R6 P2] Cancel fires between ``get_bytes`` and
    ``atomic_write_file``. The defensive mid-entry cancel check must
    catch it and abort BEFORE writing — without that check, the
    write would still happen and the cancel would only fire at the
    next loop iteration."""
    cancel = asyncio.Event()

    async def _get_bytes(ref: str) -> bytes:
        # Fire cancel AFTER get_bytes returns but BEFORE the applier
        # reaches atomic_write_file. The mid-entry check between
        # them is what we're pinning.
        cancel.set()
        return b"new content"

    minio.get_bytes = AsyncMock(side_effect=_get_bytes)
    plan = _modify_plan()

    applier = PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=redis_mock,
        emit_event=emit_event,
    )
    out = await applier.apply(
        plan,
        parent_sandbox=parent_sandbox,
        minio_client=minio,
        cancel_event=cancel,
    )
    assert out.status == ApplyStatus.APPLY_ABORTED
    # No write actually happened
    parent_sandbox.atomic_write_file.assert_not_called()


async def test_failed_reason_truncated_and_sanitized(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    audit_repo: MagicMock,
) -> None:
    """[codex R4 P1 + R5 P2 + R6 P2] failed_reason is capped at 256
    chars + multi-line exception traces are collapsed to single
    spaces so audit display + log line shipping don't break."""
    # Cause a WRITE_IO_ERROR with a long multi-line traceback message
    long_msg = "boom\nat line 1\n" + ("x" * 300)
    parent_sandbox.atomic_write_file = AsyncMock(side_effect=OSError(long_msg))
    plan = _modify_plan()
    await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    update_kwargs = audit_repo.update_terminal.await_args.kwargs
    sanitized = update_kwargs["failed_reason"]
    assert sanitized is not None
    assert "\n" not in sanitized
    assert "\r" not in sanitized
    assert len(sanitized) <= 256
    # Truncation marker preserved
    assert "...[truncated]" in sanitized
