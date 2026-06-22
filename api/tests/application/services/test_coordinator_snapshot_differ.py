"""S2 PR-1 — pure snapshot differ, tree-INDEPENDENT §6 cases.

PR-4 owns lease/tree/parent-precheck logic + the remaining fixtures (F1-F7,
F9, F10, F16, F21, F22, F25, F26). PR-3 owns F24/F27.
"""
from __future__ import annotations

import hashlib

import pytest

from app.application.services.coordinator_snapshot_differ import (
    SnapshotChange,
    SnapshotDiffResult,
    diff_snapshots,
)
from app.domain.external.parent_sandbox import WorkspaceScan, WorkspaceScanEntry

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_SHA_X = hashlib.sha256(b"x").hexdigest()
_SHA_Y = hashlib.sha256(b"y").hexdigest()


def _reg(rel_path, sha, size=1, mode=0o100644) -> WorkspaceScanEntry:
    return WorkspaceScanEntry(
        rel_path=rel_path, kind="regular", sha256=sha, size=size,
        mode=mode, link_target=None,
    )


def _scan(entries) -> WorkspaceScan:
    return WorkspaceScan(
        entries={e.rel_path: e for e in entries}, truncated=False,
    )


def test_f20_empty_workspace_no_changes() -> None:
    # F20: empty workspace / no changes -> empty diff, no zero-apply.
    out = diff_snapshots(_scan([]), _scan([]))
    assert out.zero_apply is False
    assert out.changes == []


def test_f15_content_identical_rewrite_is_noop() -> None:
    # F15: same identity tuple PRE/POST -> no entry.
    pre = _scan([_reg("workspace/a.py", _SHA_X)])
    post = _scan([_reg("workspace/a.py", _SHA_X)])
    out = diff_snapshots(pre, post)
    assert out.zero_apply is False
    assert out.changes == []


def test_f14_temp_write_then_delete_absent_pre_and_post() -> None:
    # F14: file absent in both PRE and POST -> no entry.
    pre = _scan([_reg("workspace/keep.py", _SHA_X)])
    post = _scan([_reg("workspace/keep.py", _SHA_X)])
    out = diff_snapshots(pre, post)
    assert out.changes == []


def test_add_detected_via_tuple() -> None:
    pre = _scan([])
    post = _scan([_reg("workspace/new.py", _SHA_X)])
    out = diff_snapshots(pre, post)
    assert out.zero_apply is False
    assert out.changes == [
        SnapshotChange(
            rel_path="workspace/new.py", kind="regular", change="add",
            post_entry=post.entries["workspace/new.py"],
        )
    ]


def test_modify_detected_via_sha_change() -> None:
    pre = _scan([_reg("workspace/a.py", _SHA_X)])
    post = _scan([_reg("workspace/a.py", _SHA_Y)])
    out = diff_snapshots(pre, post)
    assert len(out.changes) == 1
    assert out.changes[0].change == "modify"
    assert out.changes[0].post_entry.sha256 == _SHA_Y


def test_f13_binary_modify_detected_via_size_only() -> None:
    # F13: binary file modify; same sha would be impossible, but size/mode are
    # part of the identity tuple — a size change alone flips to modify, no
    # text decode.
    pre = _scan([_reg("workspace/bin.dat", _SHA_X, size=10)])
    post = _scan([_reg("workspace/bin.dat", _SHA_X, size=20)])
    out = diff_snapshots(pre, post)
    assert len(out.changes) == 1
    assert out.changes[0].change == "modify"


def test_f19_directory_only_excluded_yields_no_change() -> None:
    # F19: directories are never in WorkspaceScan.entries (sandbox excludes
    # them), so a dir-only create/delete is structurally a no-op here.
    pre = _scan([])
    post = _scan([])
    out = diff_snapshots(pre, post)
    assert out.changes == []


def test_f17_truncated_scan_forces_zero_apply() -> None:
    # F17: scan.truncated=True on EITHER side -> group zero-apply.
    pre = WorkspaceScan(entries={}, truncated=False)
    post = WorkspaceScan(
        entries={"workspace/a.py": _reg("workspace/a.py", _SHA_X)},
        truncated=True,
    )
    out = diff_snapshots(pre, post)
    assert out.zero_apply is True
    assert out.zero_apply_reason == "scan_truncated"
    assert out.changes == []


def test_f17_truncated_pre_also_zero_applies() -> None:
    pre = WorkspaceScan(entries={}, truncated=True)
    post = WorkspaceScan(entries={}, truncated=False)
    out = diff_snapshots(pre, post)
    assert out.zero_apply is True
    assert out.zero_apply_reason == "scan_truncated"


def test_f18_bare_top_level_file_zero_applies() -> None:
    # F18: a changed path with no directory component is not dir-qualified ->
    # the canonical §3.4 reason ``out_of_path_lease`` (same as PR-4's capture).
    pre = WorkspaceScan(entries={}, truncated=False)
    post = _scan([_reg("foo.txt", _SHA_X)])
    out = diff_snapshots(pre, post)
    assert out.zero_apply is True
    assert out.zero_apply_reason == "out_of_path_lease"
    assert out.changes == []


def _sym(rel_path, target, mode=0o120777) -> WorkspaceScanEntry:
    return WorkspaceScanEntry(
        rel_path=rel_path, kind="symlink", sha256=None, size=len(target),
        mode=mode, link_target=target,
    )


def test_f8_symlink_creation_zero_applies() -> None:
    # F8: a symlink appears in POST -> group zero-apply (symlink not appliable).
    pre = WorkspaceScan(entries={}, truncated=False)
    post = _scan([_sym("workspace/link", "../t.py")])
    out = diff_snapshots(pre, post)
    assert out.zero_apply is True
    assert out.zero_apply_reason == "symlink"
    assert out.changes == []


def test_f12_kind_change_regular_to_symlink_zero_applies() -> None:
    # F12: same path flips regular -> symlink.
    pre = _scan([_reg("workspace/x", _SHA_X)])
    post = _scan([_sym("workspace/x", "y")])
    out = diff_snapshots(pre, post)
    assert out.zero_apply is True
    assert out.zero_apply_reason == "symlink"


def test_f11_mode_only_change_zero_applies() -> None:
    # F11: same sha256 + size, different mode -> mode-only change.
    pre = _scan([_reg("workspace/a.py", _SHA_X, size=5, mode=0o100644)])
    post = _scan([_reg("workspace/a.py", _SHA_X, size=5, mode=0o100755)])
    out = diff_snapshots(pre, post)
    assert out.zero_apply is True
    assert out.zero_apply_reason == "mode_only_change"


def test_f13_size_change_with_same_sha_is_still_modify_not_mode_only() -> None:
    # Guard: F11 only fires when sha AND size match (only mode differs).
    pre = _scan([_reg("workspace/bin.dat", _SHA_X, size=10, mode=0o100644)])
    post = _scan([_reg("workspace/bin.dat", _SHA_X, size=20, mode=0o100644)])
    out = diff_snapshots(pre, post)
    assert out.zero_apply is False
    assert out.changes[0].change == "modify"
