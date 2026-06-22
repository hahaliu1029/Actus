"""S2 PR-1 — snapshot_workspace is a forwarded sandbox RPC."""
from __future__ import annotations

from app.infrastructure.external.sandbox.sandbox_handle import (
    SANDBOX_FORWARDED_METHODS,
)


def test_snapshot_workspace_is_whitelisted() -> None:
    assert "snapshot_workspace" in SANDBOX_FORWARDED_METHODS


def test_protocols_declare_snapshot_workspace() -> None:
    from app.domain.external.sandbox import Sandbox, SandboxHandle

    assert hasattr(Sandbox, "snapshot_workspace")
    assert hasattr(SandboxHandle, "snapshot_workspace")
