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
from app.domain.external.parent_sandbox import SandboxPathCheck, WorkspaceScan
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

    def __init__(self) -> None:
        self.token: str | None = None
        self.release_calls = 0

    async def acquire(
        self, *, blocking: bool = False, token: str | None = None,
    ) -> bool:
        assert blocking is False
        self.token = token
        return True

    async def owned(self) -> bool:
        return self.token is not None

    async def extend(
        self, additional_time: float, *, replace_ttl: bool = False,
    ) -> bool:
        assert additional_time > 0
        assert replace_ttl is True
        return True

    async def release(self) -> None:
        self.release_calls += 1
        self.token = None

    async def __aenter__(self) -> "_NoopAsyncCM":
        self.token = "legacy-context-owner"
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.token = None
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
    s.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="regular")
    )
    s.compute_digest = AsyncMock(side_effect=[_SHA_A, _NEW_DIGEST])
    s.read_file = AsyncMock(return_value=b"original")
    s.atomic_write_file = AsyncMock()
    s.delete_file = AsyncMock()
    # [S2 PR-4] snapshot + quiesce surface (empty scan default; tests override)
    s.snapshot_workspace = AsyncMock(
        return_value=WorkspaceScan(entries={}, truncated=False)
    )
    s.kill_all_shell_sessions = AsyncMock(return_value=None)
    return s


def test_parent_sandbox_fixture_has_snapshot_and_quiesce(parent_sandbox) -> None:
    # [S2 PR-4 §9] every shared sandbox mock must EXPLICITLY wire the new surface
    # or downstream finalize/applier tests get a silent auto-child-mock (whose
    # return_value is NOT a WorkspaceScan) mid-run. NOTE: a bare `hasattr` is
    # useless here — `parent_sandbox` is a `MagicMock`, so `hasattr(m, "x")`
    # auto-creates a child mock and returns True for ANYTHING. Assert the attrs
    # were explicitly SET on the mock (present in `__dict__`) and that
    # snapshot_workspace is a real AsyncMock returning a real WorkspaceScan.
    assert "snapshot_workspace" in parent_sandbox.__dict__
    assert "kill_all_shell_sessions" in parent_sandbox.__dict__
    assert isinstance(parent_sandbox.snapshot_workspace, AsyncMock)
    scan = parent_sandbox.snapshot_workspace.return_value
    assert isinstance(scan, WorkspaceScan)


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
                path="d/x.py", op="modify",
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
                path="d/new.py", op="add",
                new_digest=_NEW_DIGEST,
                content_ref="ref-add",
                content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=len(_NEW_CONTENT),
        file_count=1,
        source_work_unit_ids=("wu1",),
    )


async def test_rollback_phase_callback_runs_before_rollback_side_effect(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
) -> None:
    events: list[str] = []

    def on_rollback() -> None:
        events.append("phase")

    async def rollback(*_args):
        events.append("rollback")
        return "complete", []

    applier._rollback = AsyncMock(side_effect=rollback)  # type: ignore[method-assign]
    await applier._rollback_with_phase(
        [], [], parent_sandbox, _add_plan(), on_rollback,
    )

    assert events == ["phase", "rollback"]


@pytest.mark.parametrize(
    "callback_error",
    [RuntimeError("phase failed"), asyncio.CancelledError()],
)
async def test_rollback_phase_callback_error_never_blocks_real_rollback(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    callback_error: BaseException,
    caplog,
) -> None:
    def on_rollback() -> None:
        raise callback_error

    applier._rollback = AsyncMock(return_value=("complete", []))  # type: ignore[method-assign]

    result = await applier._rollback_with_phase(
        [], [], parent_sandbox, _add_plan(), on_rollback,
    )

    assert result == ("complete", [])
    applier._rollback.assert_awaited_once()
    assert "rollback phase callback failed" in caplog.text


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
    assert out.applied_files[0].path == "d/x.py"
    assert out.rollback_status is None
    parent_sandbox.atomic_write_file.assert_awaited_once_with(
        "d/x.py", _NEW_CONTENT,
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
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=False, kind="missing")
    )
    parent_sandbox.compute_digest = AsyncMock(return_value=_NEW_DIGEST)
    plan = _add_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.SUCCESS
    parent_sandbox.atomic_write_file.assert_awaited_once_with(
        "d/new.py", _NEW_CONTENT,
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
    assert out.failed_at.path == "d/x.py"
    parent_sandbox.atomic_write_file.assert_not_called()
    snapshot_store.save.assert_not_called()
    assert out.rollback_status is None


async def test_file_missing_aborts(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    # modify-branch preflight now probes via check_path() (not exists());
    # a missing target still aborts FILE_MISSING.
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=False, kind="missing")
    )
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.FILE_MISSING
    assert out.failed_at is not None
    assert out.failed_at.path == "d/x.py"


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
                path="d/a.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref-a", content_size=len(_NEW_CONTENT),
            ),
            FilePatchEntry(
                path="d/b.py", op="modify",
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
                path="d/a.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref-a", content_size=len(_NEW_CONTENT),
            ),
            FilePatchEntry(
                path="d/b.py", op="modify",
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
    # [C2 PR-8 §13 Task 8.4] _finalize now also emits a CoordinatorApplyEvent,
    # so emit_event is awaited twice on rollback_partial: once for the
    # HealthEvent (TERMINATING) and once for the CoordinatorApplyEvent.
    from app.domain.models.event import CoordinatorApplyEvent
    assert emit_event.await_count == 2
    emitted_payloads = [c.args[0] for c in emit_event.await_args_list]
    health_evs = [e for e in emitted_payloads if isinstance(e, HealthEvent)]
    apply_evs = [
        e for e in emitted_payloads if isinstance(e, CoordinatorApplyEvent)
    ]
    assert len(health_evs) == 1
    assert health_evs[0].status == HealthStatus.TERMINATING
    assert health_evs[0].metrics is not None
    assert (
        health_evs[0].metrics["code"] == "coordinator_apply_rollback_partial"
    )
    assert len(apply_evs) == 1
    assert apply_evs[0].apply_status == "rollback_partial"
    assert apply_evs[0].coordinator_run_id == "r1"


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
    # add a.py ⇒ parent missing; modify b.py ⇒ parent regular. The add branch
    # now probes check_path too, so a single return_value would FILE_EXISTS the add.
    def _check(path: str) -> SandboxPathCheck:
        if path == "d/a.py":
            return SandboxPathCheck(exists=False, kind="missing")
        return SandboxPathCheck(exists=True, kind="regular")  # d/b.py modify
    parent_sandbox.check_path = AsyncMock(side_effect=_check)
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
                path="d/a.py", op="add",
                new_digest=_NEW_DIGEST,
                content_ref="ref-a", content_size=len(_NEW_CONTENT),
            ),
            FilePatchEntry(
                path="d/b.py", op="modify",
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
    assert ("d/a.py",) in delete_calls


async def test_toctou_refused_write_rolls_back_prior_entry(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """2b at the applier level (spec §4 test 8): a refused write raises
    OSError -> WRITE_IO_ERROR **+ rollback of prior entries**. Two-entry plan:
    'a.py' add succeeds, then 'b.py' modify's write raises the special-file
    refuse — the applier rolls back the already-applied 'a.py' (delete_file in
    reverse) and reports WRITE_IO_ERROR with rollback 'complete'. (That the flag
    is SET is proven separately by test_atomic_write_file_sets_refuse_special_true.)"""
    import errno

    # a.py add ⇒ parent missing; b.py modify ⇒ parent regular. The add branch
    # now probes check_path too, so a single return_value would FILE_EXISTS the add.
    def _check(path: str) -> SandboxPathCheck:
        if path == "d/a.py":
            return SandboxPathCheck(exists=False, kind="missing")  # a.py add
        return SandboxPathCheck(exists=True, kind="regular")       # d/b.py modify
    parent_sandbox.check_path = AsyncMock(side_effect=_check)
    # compute_digest order: b.py modify preflight (=_SHA_A, matches base_digest)
    # then a.py post-write verify (=_NEW_DIGEST). (Same as the fixture default.)
    parent_sandbox.compute_digest = AsyncMock(side_effect=[_SHA_A, _NEW_DIGEST])

    write_count = {"n": 0}

    async def _write(path: str, content: bytes) -> None:
        write_count["n"] += 1
        if write_count["n"] == 2:  # b.py: the refused special write
            raise OSError(errno.EINVAL, "refusing to write special file")

    parent_sandbox.atomic_write_file = AsyncMock(side_effect=_write)

    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(
            FilePatchEntry(
                path="d/a.py", op="add",
                new_digest=_NEW_DIGEST, content_ref="ref-a",
                content_size=len(_NEW_CONTENT),
            ),
            FilePatchEntry(
                path="d/b.py", op="modify",
                base_digest=_SHA_A, new_digest=_NEW_DIGEST,
                content_ref="ref-b", content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=2 * len(_NEW_CONTENT),
        file_count=2,
        source_work_unit_ids=("wu1",),
    )
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status is ApplyStatus.WRITE_IO_ERROR
    assert out.rollback_status == "complete"
    # rollback undid the already-applied 'a.py' add (delete_file in reverse order).
    parent_sandbox.delete_file.assert_awaited_once_with("d/a.py")


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
                path="d/x.py", op="modify",
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


# ─── S1b 2a: special-file preflight reject ───────────────────────────────────


@pytest.mark.parametrize("special_kind", ["fifo", "socket", "block", "char"])
async def test_modify_over_special_target_rejected_before_read(
    special_kind: str,
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    snapshot_store: MagicMock,
) -> None:
    """2a: modify over ANY direct special inode (fifo/socket/block/char) ->
    TARGET_SPECIAL_FILE before any read/snapshot/write. Parameterized over the
    FULL special set so dropping socket/block/char from the applier reject tuple
    goes RED. A reorder that reads before the kind-check also goes RED."""
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind=special_kind)
    )
    out = await applier.apply(
        _modify_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status is ApplyStatus.TARGET_SPECIAL_FILE
    assert out.failed_at.path == "d/x.py"
    assert f"kind={special_kind}" in out.failed_at.reason
    parent_sandbox.compute_digest.assert_not_called()
    parent_sandbox.read_file.assert_not_called()
    snapshot_store.save.assert_not_called()
    parent_sandbox.atomic_write_file.assert_not_called()


async def test_add_over_regular_target_is_file_exists_via_check_path(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """[S2 PR-4 R4-H] The add branch now keys on ``check_path()`` (lstat, no
    symlink follow), NOT the symlink-following ``exists()``. An existing REGULAR
    target → FILE_EXISTS + no write. (An existing symlink/special/dir target →
    PARENT_NOT_REGULAR — covered by ``test_add_target_is_symlink_in_parent_rejected``.)
    Pin that ``check_path`` IS consulted and the dead ``exists()`` path is no
    longer relied on."""
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="regular")
    )
    parent_sandbox.exists = AsyncMock(return_value=False)  # dead path — must be ignored
    out = await applier.apply(
        _add_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status is ApplyStatus.FILE_EXISTS
    parent_sandbox.check_path.assert_awaited_once_with("d/new.py")
    parent_sandbox.exists.assert_not_called()  # add no longer consults exists()
    parent_sandbox.atomic_write_file.assert_not_called()


# ─── Task 11: owner-scoped rolling apply-lock lease ──────────────────────────


class _LeaseState:
    def __init__(self) -> None:
        self.now = 0.0
        self.owner: str | None = None
        self.expires_at = 0.0


class _OwnerLeaseLock:
    def __init__(
        self,
        state: _LeaseState,
        *,
        timeout: float,
        fail_renew_owned: bool = False,
        fail_extend: str | None = None,
        renew_failed: asyncio.Event | None = None,
    ) -> None:
        self.state = state
        self.timeout = timeout
        self.local_token: str | None = None
        self.fail_renew_owned = fail_renew_owned
        self.fail_extend = fail_extend
        self.renew_failed = renew_failed
        self.owned_calls = 0
        self.extend_calls = 0
        self.release_calls = 0

    async def acquire(
        self, *, blocking: bool = False, token: str | None = None,
    ) -> bool:
        assert blocking is False
        assert token
        if self.state.owner is not None and self.state.now < self.state.expires_at:
            return False
        self.local_token = token
        self.state.owner = token
        self.state.expires_at = self.state.now + self.timeout
        return True

    async def owned(self) -> bool:
        self.owned_calls += 1
        # The blocked preflight in the failure tests makes the renew loop the
        # first caller of owned(). This models either a compare-owner miss or
        # a Redis read failure without relying on wall-clock sleeps.
        if self.fail_renew_owned and self.owned_calls == 1:
            if self.renew_failed is not None:
                self.renew_failed.set()
            return False
        return (
            self.local_token is not None
            and self.state.owner == self.local_token
            and self.state.now < self.state.expires_at
        )

    async def extend(
        self, additional_time: float, *, replace_ttl: bool = False,
    ) -> bool:
        self.extend_calls += 1
        assert replace_ttl is True
        assert additional_time == self.timeout
        if self.fail_extend is not None:
            if self.renew_failed is not None:
                self.renew_failed.set()
            if self.fail_extend == "raise":
                raise RuntimeError("redis unavailable during extend")
            return False
        if not await self.owned():
            return False
        self.state.expires_at = self.state.now + additional_time
        return True

    async def release(self) -> None:
        from redis.exceptions import LockNotOwnedError

        self.release_calls += 1
        if not await self.owned():
            raise LockNotOwnedError("replacement owner holds apply lock")
        self.state.owner = None
        self.local_token = None

    def replace_owner(self, token: str = "replacement-owner") -> None:
        self.state.owner = token
        self.state.expires_at = self.state.now + self.timeout


class _OwnerLeaseRedis:
    def __init__(self, lock: _OwnerLeaseLock) -> None:
        self._lock = lock
        self.keys: list[str] = []

    def lock(
        self, key: str, *, blocking: bool, timeout: float,
    ) -> _OwnerLeaseLock:
        assert blocking is False
        assert timeout == self._lock.timeout
        self.keys.append(key)
        return self._lock


def _two_add_plan() -> PatchApplyPlan:
    return PatchApplyPlan(
        coordinator_run_id="r1",
        files=(
            FilePatchEntry(
                path="d/a.py", op="add", new_digest=_NEW_DIGEST,
                content_ref="ref-a", content_size=len(_NEW_CONTENT),
            ),
            FilePatchEntry(
                path="d/b.py", op="add", new_digest=_NEW_DIGEST,
                content_ref="ref-b", content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=2 * len(_NEW_CONTENT),
        file_count=2,
        source_work_unit_ids=("wu1",),
    )


def _lease_applier(
    *,
    lock: _OwnerLeaseLock,
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    emit_event: AsyncMock,
    lease_sleep,
) -> PatchApplier:
    return PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=_OwnerLeaseRedis(lock),
        emit_event=emit_event,
        apply_lock_ttl_seconds=lock.timeout,
        lease_sleep=lease_sleep,
    )


async def test_apply_lock_renews_past_600_seconds_of_fake_time(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    state = _LeaseState()
    lock = _OwnerLeaseLock(state, timeout=600.0)
    crossed_600 = asyncio.Event()

    async def fake_sleep(delay: float) -> None:
        assert delay == 200.0
        state.now += delay
        if state.now > 600:
            crossed_600.set()
        await asyncio.sleep(0)

    async def delayed_check(_path: str) -> SandboxPathCheck:
        await crossed_600.wait()
        return SandboxPathCheck(exists=False, kind="missing")

    parent_sandbox.check_path = AsyncMock(side_effect=delayed_check)
    parent_sandbox.compute_digest = AsyncMock(return_value=_NEW_DIGEST)
    applier = _lease_applier(
        lock=lock,
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        emit_event=emit_event,
        lease_sleep=fake_sleep,
    )

    outcome = await applier.apply(
        _add_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )

    assert state.now > 600
    assert lock.extend_calls >= 4
    assert outcome.status is ApplyStatus.SUCCESS
    assert lock.release_calls == 1


@pytest.mark.parametrize("failure", ["owned", "extend_false", "extend_raise"])
async def test_apply_lock_renew_failure_stops_before_first_write(
    failure: str,
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    state = _LeaseState()
    renew_failed = asyncio.Event()
    lock = _OwnerLeaseLock(
        state,
        timeout=600.0,
        fail_renew_owned=failure == "owned",
        fail_extend=(
            "return_false" if failure == "extend_false"
            else "raise" if failure == "extend_raise"
            else None
        ),
        renew_failed=renew_failed,
    )

    async def immediate_sleep(_delay: float) -> None:
        await asyncio.sleep(0)

    async def delayed_check(_path: str) -> SandboxPathCheck:
        await renew_failed.wait()
        return SandboxPathCheck(exists=False, kind="missing")

    parent_sandbox.check_path = AsyncMock(side_effect=delayed_check)
    parent_sandbox.compute_digest = AsyncMock(return_value=_NEW_DIGEST)
    applier = _lease_applier(
        lock=lock,
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        emit_event=emit_event,
        lease_sleep=immediate_sleep,
    )

    outcome = await applier.apply(
        _add_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )

    assert outcome.status is ApplyStatus.APPLY_LOCK_LOST
    assert outcome.failed_at is not None
    assert outcome.failed_at.reason == "apply_lock_lost"
    parent_sandbox.atomic_write_file.assert_not_awaited()
    assert audit_repo.update_terminal.await_args.kwargs["status"] == "apply_lock_lost"


async def test_apply_lock_loss_after_fetch_rolls_back_prior_write_and_skips_next(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    state = _LeaseState()
    lock = _OwnerLeaseLock(state, timeout=600.0)
    never = asyncio.Event()

    async def dormant_sleep(_delay: float) -> None:
        await never.wait()

    fetch_count = 0

    async def fetch(_ref: str) -> bytes:
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count == 2:
            lock.replace_owner()
        return _NEW_CONTENT

    minio.get_bytes = AsyncMock(side_effect=fetch)
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=False, kind="missing"),
    )
    parent_sandbox.compute_digest = AsyncMock(return_value=_NEW_DIGEST)
    applier = _lease_applier(
        lock=lock,
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        emit_event=emit_event,
        lease_sleep=dormant_sleep,
    )

    outcome = await applier.apply(
        _two_add_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )

    assert outcome.status is ApplyStatus.APPLY_LOCK_LOST
    assert outcome.rollback_status == "complete"
    parent_sandbox.atomic_write_file.assert_awaited_once_with(
        "d/a.py", _NEW_CONTENT,
    )
    parent_sandbox.delete_file.assert_awaited_once_with("d/a.py")
    assert state.owner == "replacement-owner"
    assert lock.release_calls == 0


async def test_apply_lock_loss_after_write_rolls_back_that_write(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    state = _LeaseState()
    lock = _OwnerLeaseLock(state, timeout=600.0)
    never = asyncio.Event()

    async def dormant_sleep(_delay: float) -> None:
        await never.wait()

    async def write_then_replace(_path: str, _content: bytes) -> None:
        lock.replace_owner()

    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=False, kind="missing"),
    )
    parent_sandbox.atomic_write_file = AsyncMock(side_effect=write_then_replace)
    parent_sandbox.compute_digest = AsyncMock(return_value=_NEW_DIGEST)
    applier = _lease_applier(
        lock=lock,
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        emit_event=emit_event,
        lease_sleep=dormant_sleep,
    )

    outcome = await applier.apply(
        _add_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )

    assert outcome.status is ApplyStatus.APPLY_LOCK_LOST
    assert outcome.rollback_status == "complete"
    parent_sandbox.delete_file.assert_awaited_once_with("d/new.py")
    assert state.owner == "replacement-owner"
    assert lock.release_calls == 0


async def test_cancel_and_lock_loss_prefers_lock_lost_and_rolls_back_once(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    emit_event: AsyncMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    state = _LeaseState()
    lock = _OwnerLeaseLock(state, timeout=600.0)
    never = asyncio.Event()
    cancel = asyncio.Event()
    rollback_phases = 0

    async def dormant_sleep(_delay: float) -> None:
        await never.wait()

    async def write_then_race(_path: str, _content: bytes) -> None:
        cancel.set()
        lock.replace_owner()

    def on_rollback() -> None:
        nonlocal rollback_phases
        rollback_phases += 1

    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=False, kind="missing"),
    )
    parent_sandbox.atomic_write_file = AsyncMock(side_effect=write_then_race)
    parent_sandbox.compute_digest = AsyncMock(return_value=_NEW_DIGEST)
    applier = _lease_applier(
        lock=lock,
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        emit_event=emit_event,
        lease_sleep=dormant_sleep,
    )

    outcome = await applier.apply(
        _add_plan(),
        parent_sandbox=parent_sandbox,
        minio_client=minio,
        cancel_event=cancel,
        on_rollback=on_rollback,
    )

    assert outcome.status is ApplyStatus.APPLY_LOCK_LOST
    assert rollback_phases == 1
    parent_sandbox.delete_file.assert_awaited_once_with("d/new.py")


async def test_symlink_modify_target_is_parent_not_regular(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """[S2 PR-4 R4-H] 2a keys on the DIRECT lstat inode: a symlink modify
    target is kind="symlink" → rejected at the kind invariant BEFORE any
    digest read. No fall-open to compute_digest / WRITE_IO_ERROR; status is
    PARENT_NOT_REGULAR, explicitly NOT TARGET_SPECIAL_FILE."""
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="symlink")
    )
    parent_sandbox.compute_digest = AsyncMock(side_effect=OSError("download 500"))
    out = await applier.apply(
        _modify_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )
    parent_sandbox.check_path.assert_awaited_once_with("d/x.py")
    # [S2 PR-4 R4-H] modify target MUST be regular; a symlink is rejected at
    # the kind invariant BEFORE any digest read — no fall-open to WRITE_IO_ERROR.
    parent_sandbox.compute_digest.assert_not_awaited()
    parent_sandbox.atomic_write_file.assert_not_called()
    assert out.status is ApplyStatus.PARENT_NOT_REGULAR
    assert out.status is not ApplyStatus.TARGET_SPECIAL_FILE


async def test_modify_kind_other_is_parent_not_regular(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """[S2 PR-4 R4-H] the modify/delete branch no longer falls OPEN on a
    non-regular kind: kind="other" (old sandbox / soft lstat failure) is now a
    hard PARENT_NOT_REGULAR reject, NOT TARGET_SPECIAL_FILE, with no digest
    read."""
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="other")
    )
    out = await applier.apply(
        _modify_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )
    # [S2 PR-4 R4-H] the modify/delete branch no longer falls OPEN on a
    # non-regular kind — `other` is now a hard PARENT_NOT_REGULAR reject, no
    # digest read.
    assert out.status is ApplyStatus.PARENT_NOT_REGULAR
    assert out.status is not ApplyStatus.TARGET_SPECIAL_FILE
    parent_sandbox.compute_digest.assert_not_awaited()


async def test_delete_no_apply_step_inode_guard_d9_off(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """D9=off pin: a delete whose preflight saw kind=regular proceeds to
    delete_file(path) with no refuse_special kwarg at the apply step (the
    benign delete-TOCTOU is documented/accepted, not closed). FilePatchEntry
    op=delete REQUIRES base_digest and FORBIDS new_digest/content_ref/
    content_size (patch_manifest.py __post_init__)."""
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="regular")
    )
    parent_sandbox.compute_digest = AsyncMock(return_value=_SHA_A)
    delete_plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(FilePatchEntry(path="d/x.py", op="delete", base_digest=_SHA_A),),
        total_size_bytes=0,
        file_count=1,
        source_work_unit_ids=("wu1",),
    )
    out = await applier.apply(
        delete_plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status is ApplyStatus.SUCCESS
    parent_sandbox.delete_file.assert_awaited_once_with("d/x.py")


def test_target_special_file_status_value_and_length():
    """Pin the exact audit/SSE wire string + that it fits
    ``coordinator_apply_audit.status`` ``String(32)``. The other tests assert
    enum IDENTITY (``is ApplyStatus.TARGET_SPECIAL_FILE``), which would NOT
    catch a value typo or an over-32-char value that breaks the audit write /
    FE string pass-through."""
    assert ApplyStatus.TARGET_SPECIAL_FILE.value == "target_special_file"
    assert len(ApplyStatus.TARGET_SPECIAL_FILE.value) <= 32


def test_parent_not_regular_status_value_and_length():
    assert ApplyStatus.PARENT_NOT_REGULAR.value == "parent_not_regular"
    assert len(ApplyStatus.PARENT_NOT_REGULAR.value) <= 32


async def test_add_target_is_symlink_in_parent_rejected(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    # add target appears as a symlink in the parent → must NOT be written
    # through. check_path (lstat) sees kind="symlink"; the add branch now
    # probes check_path (not the symlink-following exists()) and rejects.
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="symlink")
    )
    out = await applier.apply(
        _add_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.PARENT_NOT_REGULAR
    parent_sandbox.atomic_write_file.assert_not_called()


async def test_add_target_missing_in_parent_ok(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=False, kind="missing")
    )
    parent_sandbox.compute_digest = AsyncMock(return_value=_NEW_DIGEST)  # post-write
    out = await applier.apply(
        _add_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.SUCCESS


async def test_modify_target_is_directory_in_parent_rejected(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="directory")
    )
    out = await applier.apply(
        _modify_plan(), parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.PARENT_NOT_REGULAR
    parent_sandbox.compute_digest.assert_not_awaited()
