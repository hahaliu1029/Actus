"""S2 PR-1 — sandbox-side snapshot request/result schema parity.

CI-only convention: runs under the SANDBOX venv (`cd sandbox && uv run pytest`),
not the api venv (both expose a top-level ``app`` package).
"""
from __future__ import annotations


def test_request_defaults_and_fields() -> None:
    from app.interfaces.schemas.file import SnapshotWorkspaceRequest

    req = SnapshotWorkspaceRequest(
        max_paths=1000, max_files=500, max_total_bytes=1_000_000,
        max_seconds=5.0,
    )
    assert req.root == "/home/ubuntu"
    assert req.max_paths == 1000
    assert req.max_files == 500
    assert req.max_total_bytes == 1_000_000
    assert req.max_seconds == 5.0


def test_result_models_carry_identity_tuple() -> None:
    from app.models.file import WorkspaceScan, WorkspaceScanEntry

    entry = WorkspaceScanEntry(
        rel_path="workspace/a.py", kind="regular", sha256="a" * 64,
        size=3, mode=33188, link_target=None,
    )
    scan = WorkspaceScan(entries={"workspace/a.py": entry}, truncated=False)
    assert scan.entries["workspace/a.py"].kind == "regular"
    assert scan.truncated is False
