import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.application.services import coordinator_child_runner as runner_module
from app.application.services.coordinator_child_runner import (
    ChildRunResult,
    CoordinatorChildRunner,
    _OutOfLeaseWriteError,
)
from app.domain.external.parent_sandbox import (
    SandboxPathCheck,
    WorkspaceScan,
    WorkspaceScanEntry,
)
from app.domain.models.mailbox_envelope import ResultReadyOutcome
from app.domain.models.work_unit import TreeLease, WorkUnit

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_SHA_NEW = hashlib.sha256(b"new").hexdigest()


class _Limits:
    max_snapshot_paths = 20000
    max_snapshot_files = 8000
    max_snapshot_total_bytes = 100 * 1024 * 1024
    max_snapshot_seconds = 30.0


def _entry(rel):
    return WorkspaceScanEntry(
        rel_path=rel, kind="regular", sha256=_SHA_NEW, size=3, mode=0o100644,
        link_target=None,
    )


_SHA_OLD = hashlib.sha256(b"old").hexdigest()


def _mod_entry(rel):
    # Sibling of _entry with a DIFFERENT sha256 so a PRE/POST tuple differs
    # → a `modify` diff under the ADD-only `_wu()` tree lease.
    return WorkspaceScanEntry(
        rel_path=rel, kind="regular", sha256=_SHA_OLD, size=3, mode=0o100644,
        link_target=None,
    )


def _wu():
    return WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write", shell_mode=True,
        allowed_tools=["file_write"],
        write_tree_lease=[TreeLease(prefix="workspace", ops=frozenset({"add"}))],
    )


def _runner(*, pre_scan, post_scans, parent_kind="missing"):
    # [S2 PR-4 §3.2] _capture_shell_snapshot_diff calls
    # self._child_sandbox.kill_all_shell_sessions() and
    # self._child_sandbox.snapshot_workspace(...) — the quiesce + POST scans run
    # against the CHILD sandbox (the runner distinguishes parent vs child;
    # coordinator_child_runner.py:227-228). Stub them on `child`, NOT `parent`.
    child = MagicMock()
    child.read_file = AsyncMock(return_value=b"new")
    child.snapshot_workspace = AsyncMock(side_effect=list(post_scans))
    child.kill_all_shell_sessions = AsyncMock(return_value=None)
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="ref")
    # The parent-side inode precheck (check_path) in the differ targets the
    # PARENT sandbox (self._parent_sandbox.check_path) — stub it on `parent`.
    parent = MagicMock()
    parent.check_path = AsyncMock(
        return_value=SandboxPathCheck(
            exists=parent_kind != "missing", kind=parent_kind
        )
    )
    r = CoordinatorChildRunner(
        cancel_event=asyncio.Event(), child_sandbox=child,
        parent_sandbox=parent, artifact_storage=artifact,
        budget=MagicMock(),
        coordinator_run_id="run-1",
    )
    r._pre_scan = pre_scan
    r._snapshot_limits = _Limits()
    return r, child


async def test_capture_success_two_identical_post_scans():
    post = WorkspaceScan(entries={"workspace/new.py": _entry("workspace/new.py")},
                         truncated=False)
    r, child = _runner(
        pre_scan=WorkspaceScan(entries={}, truncated=False),
        post_scans=[post, post],  # POST + stability re-scan identical
    )
    files = await r._capture_shell_snapshot_diff("run-1", _wu())
    assert len(files) == 1
    assert files[0].op == "add"
    # quiesce + POST scans run against the CHILD sandbox.
    child.kill_all_shell_sessions.assert_awaited_once()
    assert child.snapshot_workspace.await_count == 2


async def test_capture_non_quiescent_drift_raises_failed():
    post1 = WorkspaceScan(entries={"workspace/new.py": _entry("workspace/new.py")},
                          truncated=False)
    # second scan adds a file → live writer → tuple drift
    post2 = WorkspaceScan(
        entries={
            "workspace/new.py": _entry("workspace/new.py"),
            "workspace/late.py": _entry("workspace/late.py"),
        },
        truncated=False,
    )
    r, child = _runner(
        pre_scan=WorkspaceScan(entries={}, truncated=False),
        post_scans=[post1, post2],
    )
    with pytest.raises(RuntimeError, match="not quiescent"):
        await r._capture_shell_snapshot_diff("run-1", _wu())


async def test_capture_truncated_scan_raises_out_of_lease():
    post = WorkspaceScan(entries={}, truncated=True)
    r, child = _runner(
        pre_scan=WorkspaceScan(entries={}, truncated=False),
        post_scans=[post, post],
    )
    with pytest.raises(_OutOfLeaseWriteError, match="scan_truncated"):
        await r._capture_shell_snapshot_diff("run-1", _wu())


# --------------------------------------------------------------------------- #
# Task 4.7: run_work_unit PRE scan + shell-mode finalize routing
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _shell_mode_flag_on(monkeypatch):
    # [S2 §3.6] The runner gates shell-mode on `flag_on AND wu.shell_mode`. The
    # master flag defaults OFF, so the shell-capture run_work_unit tests below
    # (which exercise the shell finalize path) must turn it ON. The dedicated
    # flag-OFF test overrides this with its own monkeypatch. Patch the name in
    # the runner module's namespace (it imports the symbol, not the module).
    monkeypatch.setattr(
        runner_module, "is_coordinator_shell_mode_enabled", lambda: True
    )


def _fake_listener():
    listener = MagicMock()
    listener.start = AsyncMock()
    listener.shutdown = AsyncMock()
    ready = asyncio.Event(); ready.set()
    listener.ready_event = ready
    return listener


def _wire_shell_runner(*, post_scans, pre_scan_side_effect=None):
    publisher = MagicMock(); publisher.publish = AsyncMock()
    envf = MagicMock(); sentinel = object()
    envf.make_result_ready = MagicMock(return_value=sentinel)
    # CHILD sandbox: seed-install + PRE/POST snapshot scans + quiesce + read_file.
    child = MagicMock()
    child.read_file = AsyncMock(return_value=b"new")
    child.atomic_write_file = AsyncMock()
    child.compute_digest = AsyncMock(return_value=_SHA_NEW)
    child.kill_all_shell_sessions = AsyncMock(return_value=None)
    if pre_scan_side_effect is not None:
        child.snapshot_workspace = AsyncMock(side_effect=pre_scan_side_effect)
    else:
        # PRE scan (empty) + 2 POST scans
        child.snapshot_workspace = AsyncMock(
            side_effect=[WorkspaceScan(entries={}, truncated=False), *post_scans]
        )
    # PARENT sandbox: the differ's kind-invariant precheck calls
    # self._parent_sandbox.check_path(canon) (NOT the child's). The runner is
    # constructed WITH parent_sandbox=parent so check_path resolves; without it
    # self._parent_sandbox is None → AttributeError mid-finalize. Default
    # missing/regular per the add/modify scenario; the out-of-tree test overrides
    # runner._parent_sandbox.check_path.
    parent = MagicMock()
    parent.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=False, kind="missing")
    )
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="ref")
    cancel_event = asyncio.Event()
    runner = CoordinatorChildRunner(
        cancel_event=cancel_event, inner_runner=MagicMock(), publisher=publisher,
        child_sandbox=child, parent_sandbox=parent, artifact_storage=artifact,
        envelope_factory=envf,
        mailbox_subscriber=MagicMock(), budget=None,
        coordinator_run_id="run-1",
    )
    # [Task 4.7] budget=None skips the wallclock watchdog (mirrors
    # test_coordinator_child_runner_finalize_contract.py's `runner._budget = None`):
    # a MagicMock budget would trip ``max_wallclock_seconds > 0`` with a
    # TypeError inside the inner-invoke try, masking the shell-finalize path
    # under test as a generic FAILED. The budget enforcement path is exercised
    # by its own dedicated suite, not here.
    runner._snapshot_limits = _Limits()
    runner._inner_runner.invoke_until_done = AsyncMock(
        return_value=ChildRunResult(done_event=MagicMock(), tool_calls=())
    )
    return runner, cancel_event, publisher, envf


async def test_pre_scan_rpc_error_publishes_terminal_failed():
    runner, ce, pub, envf = _wire_shell_runner(
        post_scans=[],
        pre_scan_side_effect=[OSError("snapshot RPC down")],
    )
    fl = _fake_listener()
    with patch.object(runner_module, "CoordinatorChildCancelListener", return_value=fl):
        await runner.run_work_unit(
            coordinator_run_id="run-1", work_unit=_wu(), child_session_id="c1",
            spawn_manifest=MagicMock(), cancel_event=ce, root_session_id="r1",
        )
    pub.publish.assert_awaited()  # terminal envelope WAS published (no escape)
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.FAILED


async def test_pre_scan_timeout_publishes_terminal_failed():
    # [S2 §3.2 finalizer-budget invariant — HIGH] The PRE scan await MUST be
    # bounded by max_snapshot_seconds (asyncio.wait_for), EXACTLY like the POST
    # scan. A hung/stalled PRE snapshot RPC must trip the budget → asyncio.
    # TimeoutError → _SeedInstallError → _finalize_failed (terminal FAILED
    # published), NOT block on the api side until the DockerSandbox HTTP client
    # timeout (~600s). Two assertions in one test:
    #   (1) the bounded-await path raises asyncio.TimeoutError when the RPC
    #       outlasts the budget (a REAL hang vs a tiny budget — fails/hangs if the
    #       wait_for wrap is missing), and
    #   (2) that timeout routes to a FAILED terminal (never escapes / never hangs
    #       ~600s).
    # Pin the budget to a sub-second value and make the PRE snapshot_workspace
    # outlast it. _wire_shell_runner installs _Limits() (30s); shrink it here.
    class _TinyBudgetLimits:
        max_snapshot_paths = 20000
        max_snapshot_files = 8000
        max_snapshot_total_bytes = 100 * 1024 * 1024
        max_snapshot_seconds = 0.05  # PRE wait_for trips well before the RPC ends

    async def _hung_pre_scan(**_kwargs):
        # Outlasts the 0.05s budget by an order of magnitude; the wait_for guard
        # (NOT the ~600s client timeout) must cancel this.
        await asyncio.sleep(5.0)
        return WorkspaceScan(entries={}, truncated=False)

    runner, ce, pub, envf = _wire_shell_runner(post_scans=[])
    runner._snapshot_limits = _TinyBudgetLimits()
    runner._child_sandbox.snapshot_workspace = AsyncMock(side_effect=_hung_pre_scan)
    fl = _fake_listener()
    with patch.object(runner_module, "CoordinatorChildCancelListener", return_value=fl):
        # asyncio.wait_for(timeout=5.0) bounds the WHOLE run_work_unit so a missing
        # PRE-scan wait_for guard surfaces as this outer timeout (RED) instead of
        # silently hanging the suite for ~600s; with the guard the inner 0.05s
        # budget fires first and run_work_unit returns a FAILED terminal well
        # within 5s.
        await asyncio.wait_for(
            runner.run_work_unit(
                coordinator_run_id="run-1", work_unit=_wu(), child_session_id="c1",
                spawn_manifest=MagicMock(), cancel_event=ce, root_session_id="r1",
            ),
            timeout=5.0,
        )
    pub.publish.assert_awaited()  # terminal envelope WAS published (no escape/hang)
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.FAILED


async def test_shell_capture_success_publishes_success_with_manifest():
    post = WorkspaceScan(
        entries={"workspace/new.py": _entry("workspace/new.py")}, truncated=False,
    )
    runner, ce, pub, envf = _wire_shell_runner(post_scans=[post, post])
    fl = _fake_listener()
    with patch.object(runner_module, "CoordinatorChildCancelListener", return_value=fl):
        await runner.run_work_unit(
            coordinator_run_id="run-1", work_unit=_wu(), child_session_id="c1",
            spawn_manifest=MagicMock(), cancel_event=ce, root_session_id="r1",
        )
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.SUCCESS
    assert payload.patch_manifest is not None
    assert payload.patch_manifest.files[0].op == "add"


async def test_shell_capture_large_manifest_published_by_ref(monkeypatch):
    # [S2 §3.2 C1] The shell finalizer MUST route through the PR-2 build-only
    # by-ref producer (_build_manifest_payload_inline_or_ref), NOT an inline
    # ResultReadyPayload(patch_manifest=...). Force the by-ref branch by zeroing
    # the inline ceiling: a SUCCESS shell capture then publishes patch_manifest_ref
    # (and inline patch_manifest is None) — proving a large shell-diff manifest
    # survives the 64KB terminal-store truncation instead of silently zero-applying.
    monkeypatch.setattr(runner_module, "_MAX_INLINE_MANIFEST_BYTES", 0)
    post = WorkspaceScan(
        entries={"workspace/new.py": _entry("workspace/new.py")}, truncated=False,
    )
    runner, ce, pub, envf = _wire_shell_runner(post_scans=[post, post])
    fl = _fake_listener()
    with patch.object(runner_module, "CoordinatorChildCancelListener", return_value=fl):
        await runner.run_work_unit(
            coordinator_run_id="run-1", work_unit=_wu(), child_session_id="c1",
            spawn_manifest=MagicMock(), cancel_event=ce, root_session_id="r1",
        )
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.SUCCESS
    assert payload.patch_manifest is None
    assert payload.patch_manifest_ref is not None


async def test_shell_capture_byref_upload_failure_publishes_terminal_failed(monkeypatch):
    # [S2 §3.2 C1 / F2 P0] The by-ref MinIO upload is a NEW failure source. Because
    # _finalize_success_shell calls the BUILD-ONLY _build_manifest_payload_inline_or_ref
    # INSIDE its protected try (the upload lives there, not in the unprotected
    # publish step), a put_content_addressed_bytes failure must route to
    # _finalize_failed → a FAILED terminal IS published — it must NOT escape
    # run_work_unit's finally-only outer wrapper and strand the parent waiter.
    monkeypatch.setattr(runner_module, "_MAX_INLINE_MANIFEST_BYTES", 0)  # force by-ref
    post = WorkspaceScan(
        entries={"workspace/new.py": _entry("workspace/new.py")}, truncated=False,
    )
    runner, ce, pub, envf = _wire_shell_runner(post_scans=[post, post])
    # The manifest upload (by-ref branch) blows up.
    runner._artifact_storage.put_content_addressed_bytes = AsyncMock(
        side_effect=OSError("minio down")
    )
    fl = _fake_listener()
    with patch.object(runner_module, "CoordinatorChildCancelListener", return_value=fl):
        # No exception escapes — the upload failure is caught and finalized.
        await runner.run_work_unit(
            coordinator_run_id="run-1", work_unit=_wu(), child_session_id="c1",
            spawn_manifest=MagicMock(), cancel_event=ce, root_session_id="r1",
        )
    pub.publish.assert_awaited()  # terminal envelope WAS published (no escape)
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.FAILED


async def test_shell_capture_out_of_tree_publishes_specific_reason_on_wire():
    # [spec §9(i)] A tree-modify (modify under an ADD-only tree lease) must
    # surface the SPECIFIC §3.4 reason `out_of_tree_lease` on the
    # NEEDS_AUTHORIZATION envelope — NOT the hard-coded `out_of_path_lease`.
    # This proves the WIRE field (not just the exception message) carries the
    # code; a `pytest.raises(match=...)` on the differ only proves the message.
    pre = WorkspaceScan(
        entries={"workspace/a.py": _entry("workspace/a.py")}, truncated=False,
    )
    post = WorkspaceScan(
        entries={"workspace/a.py": _mod_entry("workspace/a.py")}, truncated=False,
    )
    # The rejection comes from the ADD-only tree lease vs a modify diff — the
    # lease check raises `out_of_tree_lease` BEFORE the parent precheck runs, so
    # the parent kind is not load-bearing here. Stub it on the PARENT (not the
    # child) anyway so a regular target would let the precheck pass if reached —
    # the differ calls self._parent_sandbox.check_path.
    runner, ce, pub, envf = _wire_shell_runner(post_scans=[post, post])
    runner._parent_sandbox.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="regular")
    )
    # override the PRE scan (default empty) with a seeded PRE so the diff is a
    # modify — PRE/POST scans run against the CHILD sandbox.
    runner._child_sandbox.snapshot_workspace = AsyncMock(
        side_effect=[pre, post, post]
    )
    fl = _fake_listener()
    with patch.object(runner_module, "CoordinatorChildCancelListener", return_value=fl):
        await runner.run_work_unit(
            coordinator_run_id="run-1", work_unit=_wu(), child_session_id="c1",
            spawn_manifest=MagicMock(), cancel_event=ce, root_session_id="r1",
        )
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
    assert payload.needs_authorization_details.reason == "out_of_tree_lease"
    # [§3.4] the bounded rejection_summary carries the offending path (proves the
    # populate path, not just the reason code).
    assert payload.needs_authorization_details.rejection_summary == (
        "workspace/a.py",
    )


async def test_shell_mode_unit_with_flag_off_takes_typed_path(monkeypatch):
    # [S2 §3.6] REGRESSION LOCK (no RED-first phase; per the PR-5 Task 5.4
    # convention). shell mode runs iff `flag_on AND wu.shell_mode`. With the
    # master flag OFF, a `shell_mode=True` unit MUST take the TYPED path: NO PRE
    # scan, NO snapshot finalize — _finalize_success (typed extractor), not
    # _finalize_success_shell. This is GREEN both BEFORE and AFTER the 4.7 impl:
    # pre-impl the runner unconditionally calls _finalize_success at :452 (no
    # shell branch exists), so the flag-off path is already the typed one; the
    # impl must preserve exactly that. There is no RED→GREEN here — a FAIL is the
    # regression this lock catches (the shell branch wrongly firing while the
    # master flag is OFF). Override the autouse flag-ON fixture to OFF.
    monkeypatch.setattr(
        runner_module, "is_coordinator_shell_mode_enabled", lambda: False
    )
    post = WorkspaceScan(
        entries={"workspace/new.py": _entry("workspace/new.py")}, truncated=False,
    )
    runner, ce, pub, envf = _wire_shell_runner(post_scans=[post, post])
    # Spy the typed vs shell finalizers so we can assert which one ran.
    runner._finalize_success = AsyncMock(return_value=MagicMock())
    runner._finalize_success_shell = AsyncMock(return_value=MagicMock())
    fl = _fake_listener()
    with patch.object(runner_module, "CoordinatorChildCancelListener", return_value=fl):
        await runner.run_work_unit(
            coordinator_run_id="run-1", work_unit=_wu(), child_session_id="c1",
            spawn_manifest=MagicMock(), cancel_event=ce, root_session_id="r1",
        )
    # Typed finalize ran; shell finalize did NOT.
    runner._finalize_success.assert_awaited_once()
    runner._finalize_success_shell.assert_not_awaited()
    # No PRE scan was taken (flag-off short-circuits the shell_mode_active gate).
    assert runner._pre_scan is None
    runner._child_sandbox.snapshot_workspace.assert_not_awaited()


async def test_shell_capture_byref_manifest_upload_stall_bounded_to_failed(
    monkeypatch,
):
    # [codex PR-4 R4 P0] A STALLED by-ref MANIFEST upload must be BOUNDED by the
    # snapshot budget so it cannot outlive the parent waiter (which would
    # synthesize TIMED_OUT) and then publish a contradictory LATE SUCCESS. The
    # build/upload wait_for trips → _finalize_failed → exactly one FAILED terminal,
    # well within the bound. The outer asyncio.wait_for(5s) makes a MISSING bound
    # surface as an outer timeout (RED) instead of a silent ~600s hang.
    monkeypatch.setattr(runner_module, "_MAX_INLINE_MANIFEST_BYTES", 0)  # force by-ref

    class _TinyBudgetLimits:
        max_snapshot_paths = 20000
        max_snapshot_files = 8000
        max_snapshot_total_bytes = 100 * 1024 * 1024
        max_snapshot_seconds = 0.05  # build/upload wait_for trips before the hang

    post = WorkspaceScan(
        entries={"workspace/new.py": _entry("workspace/new.py")}, truncated=False,
    )
    runner, ce, pub, envf = _wire_shell_runner(post_scans=[post, post])
    runner._snapshot_limits = _TinyBudgetLimits()

    # Hang ONLY the MANIFEST upload (manifest/ prefix); the capture's patch-content
    # upload (patch/ prefix) returns fast so the stall is isolated to the manifest
    # build step this P0 is about.
    async def _maybe_hung_upload(*, prefix, content, **_kw):
        if "manifest/" in prefix:
            await asyncio.sleep(5.0)
        return "ref"

    runner._artifact_storage.put_content_addressed_bytes = AsyncMock(
        side_effect=_maybe_hung_upload
    )
    fl = _fake_listener()
    with patch.object(
        runner_module, "CoordinatorChildCancelListener", return_value=fl
    ):
        await asyncio.wait_for(
            runner.run_work_unit(
                coordinator_run_id="run-1", work_unit=_wu(), child_session_id="c1",
                spawn_manifest=MagicMock(), cancel_event=ce, root_session_id="r1",
            ),
            timeout=5.0,
        )
    pub.publish.assert_awaited()  # one terminal published (no escape/hang)
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.FAILED
