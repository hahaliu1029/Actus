"""S2 PR-1 — WorkspaceScan / WorkspaceScanEntry domain model contract."""
from __future__ import annotations

import dataclasses

import pytest

from app.domain.external.parent_sandbox import WorkspaceScan, WorkspaceScanEntry


def test_entry_is_frozen_and_carries_identity_tuple() -> None:
    e = WorkspaceScanEntry(
        rel_path="workspace/a.py",
        kind="regular",
        sha256="a" * 64,
        size=10,
        mode=0o100644,
        link_target=None,
    )
    assert e.rel_path == "workspace/a.py"
    assert e.kind == "regular"
    assert e.sha256 == "a" * 64
    assert e.size == 10
    assert e.mode == 0o100644
    assert e.link_target is None
    # frozen — mutation rejected
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.size = 11  # type: ignore[misc]


def test_symlink_entry_carries_link_target_and_no_sha() -> None:
    e = WorkspaceScanEntry(
        rel_path="workspace/link",
        kind="symlink",
        sha256=None,
        size=7,
        mode=0o120777,
        link_target="../target.py",
    )
    assert e.kind == "symlink"
    assert e.sha256 is None
    assert e.link_target == "../target.py"


def test_scan_is_frozen_and_keyed_by_rel_path() -> None:
    e = WorkspaceScanEntry(
        rel_path="workspace/a.py", kind="regular", sha256="b" * 64,
        size=3, mode=0o100644, link_target=None,
    )
    scan = WorkspaceScan(entries={"workspace/a.py": e}, truncated=False)
    assert scan.entries["workspace/a.py"] is e
    assert scan.truncated is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        scan.truncated = True  # type: ignore[misc]


def test_port_declares_snapshot_workspace_signature() -> None:
    import inspect

    from app.domain.external.parent_sandbox import ParentSandboxPort

    sig = inspect.signature(ParentSandboxPort.snapshot_workspace)
    params = sig.parameters
    assert "root" in params
    assert params["root"].default == "/home/ubuntu"
    # keyword-only caps
    for cap in ("max_paths", "max_files", "max_total_bytes", "max_seconds"):
        assert params[cap].kind == inspect.Parameter.KEYWORD_ONLY
