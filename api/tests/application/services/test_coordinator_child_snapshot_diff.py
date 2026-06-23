# api/tests/application/services/test_coordinator_child_snapshot_diff.py
import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_child_runner import (
    _SNAPSHOT_REJECT_REASONS,
    CoordinatorChildRunner,
    _OutOfLeaseWriteError,
)
from app.domain.external.parent_sandbox import (
    SandboxPathCheck,
    WorkspaceScan,
    WorkspaceScanEntry,
)
from app.domain.models.work_unit import PathLease, TreeLease, WorkUnit

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_reject_reason_vocabulary_is_exhaustive() -> None:
    assert _SNAPSHOT_REJECT_REASONS == frozenset(
        {
            "out_of_path_lease",
            "out_of_tree_lease",
            "special_file",
            "symlink",
            "mode_only_change",
            "indeterminate_kind",
            "scan_truncated",
            "tree_add_target_exists",
            "parent_not_regular",
        }
    )


_SHA_OLD = hashlib.sha256(b"old").hexdigest()
_SHA_NEW = hashlib.sha256(b"new").hexdigest()


def _entry(rel, kind="regular", sha=None, size=3, mode=0o100644, link=None):
    return WorkspaceScanEntry(
        rel_path=rel, kind=kind, sha256=sha, size=size, mode=mode, link_target=link
    )


def _scan(entries):
    return WorkspaceScan(
        entries={e.rel_path: e for e in entries}, truncated=False
    )


def _runner(*, parent_check_missing=True):
    child = MagicMock()
    child.read_file = AsyncMock(return_value=b"new")
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="patchref")
    parent = MagicMock()
    parent.check_path = AsyncMock(
        return_value=SandboxPathCheck(
            exists=not parent_check_missing,
            kind="missing" if parent_check_missing else "regular",
        )
    )
    r = CoordinatorChildRunner(
        cancel_event=asyncio.Event(),
        child_sandbox=child,
        parent_sandbox=parent,
        artifact_storage=artifact,
        coordinator_run_id="run-1",
    )
    return r, parent


def _wu(*, leases=(), tree_leases=()):
    return WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write", shell_mode=True,
        allowed_tools=["file_write"],
        write_lease=list(leases), write_tree_lease=list(tree_leases),
    )


async def test_f1_tree_add_new_file():
    r, parent = _runner(parent_check_missing=True)
    wu = _wu(tree_leases=[TreeLease(prefix="workspace", ops=frozenset({"add"}))])
    pre = _scan([])
    post = _scan([_entry("workspace/new.py", sha=_SHA_NEW)])
    files = await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)
    assert len(files) == 1
    assert files[0].op == "add"
    assert files[0].base_digest is None
    assert files[0].path == "workspace/new.py"


async def test_f2_exact_add_file_lease():
    r, parent = _runner(parent_check_missing=True)
    wu = _wu(leases=[PathLease(path="workspace/new.py", op="add")])
    pre = _scan([])
    post = _scan([_entry("workspace/new.py", sha=_SHA_NEW)])
    files = await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)
    assert files[0].op == "add"


async def test_f3_modify_seeded_file():
    r, parent = _runner(parent_check_missing=False)
    wu = _wu(leases=[PathLease(path="pkg/a.py", op="modify", base_digest=_SHA_OLD,
                               seed_content_ref="s")])
    pre = _scan([_entry("pkg/a.py", sha=_SHA_OLD)])
    post = _scan([_entry("pkg/a.py", sha=_SHA_NEW)])
    files = await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)
    assert files[0].op == "modify"
    assert files[0].base_digest == _SHA_OLD


async def test_f4_delete_seeded_file():
    r, parent = _runner(parent_check_missing=False)
    wu = _wu(leases=[PathLease(path="pkg/a.py", op="delete", base_digest=_SHA_OLD,
                               seed_content_ref="s")])
    pre = _scan([_entry("pkg/a.py", sha=_SHA_OLD)])
    post = _scan([])
    files = await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)
    assert files[0].op == "delete"
    assert files[0].base_digest == _SHA_OLD


async def test_f5_modify_under_tree_only_zero_apply():
    r, parent = _runner(parent_check_missing=False)
    wu = _wu(tree_leases=[TreeLease(prefix="pkg", ops=frozenset({"add"}))])
    pre = _scan([_entry("pkg/a.py", sha=_SHA_OLD)])
    post = _scan([_entry("pkg/a.py", sha=_SHA_NEW)])
    with pytest.raises(_OutOfLeaseWriteError, match="out_of_tree_lease"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_f6_delete_under_tree_only_zero_apply():
    r, parent = _runner(parent_check_missing=False)
    wu = _wu(tree_leases=[TreeLease(prefix="pkg", ops=frozenset({"add"}))])
    pre = _scan([_entry("pkg/a.py", sha=_SHA_OLD)])
    post = _scan([])
    with pytest.raises(_OutOfLeaseWriteError, match="out_of_tree_lease"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_f7_outside_all_leases_zero_apply():
    r, parent = _runner(parent_check_missing=True)
    wu = _wu(tree_leases=[TreeLease(prefix="workspace", ops=frozenset({"add"}))])
    pre = _scan([])
    post = _scan([_entry("other/x.py", sha=_SHA_NEW)])
    with pytest.raises(_OutOfLeaseWriteError, match="out_of_path_lease"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_f9_special_file_zero_apply():
    r, parent = _runner(parent_check_missing=True)
    wu = _wu(tree_leases=[TreeLease(prefix="workspace", ops=frozenset({"add"}))])
    pre = _scan([])
    post = _scan([_entry("workspace/pipe", kind="fifo", sha=None)])
    with pytest.raises(_OutOfLeaseWriteError, match="special_file"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_f10_indeterminate_kind_zero_apply():
    r, parent = _runner(parent_check_missing=True)
    wu = _wu(tree_leases=[TreeLease(prefix="workspace", ops=frozenset({"add"}))])
    pre = _scan([])
    post = _scan([_entry("workspace/odd", kind="other", sha=None)])
    with pytest.raises(_OutOfLeaseWriteError, match="indeterminate_kind"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_f16_op_mismatch_delete_on_add_lease_zero_apply():
    r, parent = _runner(parent_check_missing=False)
    wu = _wu(leases=[PathLease(path="workspace/x.py", op="add")])
    pre = _scan([_entry("workspace/x.py", sha=_SHA_OLD)])  # present then deleted
    post = _scan([])
    with pytest.raises(_OutOfLeaseWriteError, match="out_of_path_lease"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_f21_exact_wins_over_tree_op_mismatch_zero_apply():
    r, parent = _runner(parent_check_missing=False)
    wu = _wu(
        leases=[PathLease(path="pkg/a.py", op="add")],
        tree_leases=[TreeLease(prefix="pkg", ops=frozenset({"add"}))],
    )
    pre = _scan([_entry("pkg/a.py", sha=_SHA_OLD)])
    post = _scan([_entry("pkg/a.py", sha=_SHA_NEW)])  # modify; exact lease is add-only
    with pytest.raises(_OutOfLeaseWriteError, match="out_of_path_lease"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_f12_regular_to_symlink_kind_change_reports_symlink():
    # [spec §3.4] A regular→symlink replacement is a kind change, but it must
    # zero-apply with reason `symlink` (the symlink-specific code), NOT the
    # generic `special_file`. The kind-change branch checks symlink FIRST.
    r, parent = _runner(parent_check_missing=False)
    wu = _wu(leases=[PathLease(path="pkg/a.py", op="modify", base_digest=_SHA_OLD,
                               seed_content_ref="s")])
    pre = _scan([_entry("pkg/a.py", sha=_SHA_OLD)])
    post = _scan([_entry("pkg/a.py", kind="symlink", sha=None, link="/etc/passwd")])
    with pytest.raises(_OutOfLeaseWriteError, match="symlink"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


# ----- Task 4.5: parent-side inode precheck fixtures (F22 / F25 / F26) -----
async def _run_with_parent_kind(*, op, parent_kind, lease_kind="tree"):
    child = MagicMock()
    child.read_file = AsyncMock(return_value=b"new")
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="ref")
    parent = MagicMock()
    parent.check_path = AsyncMock(
        return_value=SandboxPathCheck(
            exists=parent_kind != "missing", kind=parent_kind
        )
    )
    r = CoordinatorChildRunner(
        cancel_event=asyncio.Event(), child_sandbox=child,
        parent_sandbox=parent, artifact_storage=artifact,
        coordinator_run_id="run-1",
    )
    if lease_kind == "tree":
        wu = _wu(tree_leases=[TreeLease(prefix="workspace", ops=frozenset({"add"}))])
    else:
        wu = _wu(leases=[PathLease(path="workspace/x.py", op="modify",
                                   base_digest=_SHA_OLD, seed_content_ref="s")])
    if op == "add":
        pre = _scan([])
        post = _scan([_entry("workspace/x.py", sha=_SHA_NEW)])
    else:  # modify
        pre = _scan([_entry("workspace/x.py", sha=_SHA_OLD)])
        post = _scan([_entry("workspace/x.py", sha=_SHA_NEW)])
    return await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_f22_add_target_exists_as_regular_in_parent():
    with pytest.raises(_OutOfLeaseWriteError, match="tree_add_target_exists"):
        await _run_with_parent_kind(op="add", parent_kind="regular")


async def test_f25_add_target_is_symlink_in_parent():
    with pytest.raises(_OutOfLeaseWriteError, match="parent_not_regular"):
        await _run_with_parent_kind(op="add", parent_kind="symlink")


async def test_f26_modify_target_not_regular_in_parent():
    with pytest.raises(_OutOfLeaseWriteError, match="parent_not_regular"):
        await _run_with_parent_kind(op="modify", parent_kind="directory",
                                    lease_kind="file")


# ----- codex PR-4 R1 P1 fixes -----
async def test_kind_change_to_other_reports_indeterminate_kind():
    # [codex PR-4 R1 P1] A regular→other kind change must report the §3.4
    # `indeterminate_kind` code, NOT the `special_file` fall-through.
    r, parent = _runner(parent_check_missing=False)
    wu = _wu(leases=[PathLease(path="pkg/a.py", op="modify", base_digest=_SHA_OLD,
                               seed_content_ref="s")])
    pre = _scan([_entry("pkg/a.py", sha=_SHA_OLD)])
    post = _scan([_entry("pkg/a.py", kind="other", sha=None)])
    with pytest.raises(_OutOfLeaseWriteError, match="indeterminate_kind"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)


async def test_read_bytes_disagree_with_scan_raises_not_quiescent():
    # [codex PR-4 R1 P1] If the bytes read back disagree with the proven-stable
    # POST scan's sha256/size (a detached writer rewrote the file between the
    # stable scan and the read), the capture is NOT quiescent → RuntimeError
    # (→ FAILED), never a silent capture of unwitnessed bytes.
    r, parent = _runner(parent_check_missing=True)
    # read_file returns content whose sha256 != the post entry's _SHA_NEW.
    r._child_sandbox.read_file = AsyncMock(return_value=b"tampered-after-scan")
    wu = _wu(tree_leases=[TreeLease(prefix="workspace", ops=frozenset({"add"}))])
    pre = _scan([])
    post = _scan([_entry("workspace/new.py", sha=_SHA_NEW)])
    with pytest.raises(RuntimeError, match="not quiescent"):
        await r._extract_patch_files_from_snapshot("run-1", wu, pre, post)
