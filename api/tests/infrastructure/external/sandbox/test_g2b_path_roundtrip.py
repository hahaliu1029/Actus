"""F3.2 G2b — path-transparency + bare-filename guard (INV-F2.3).

The agent's ``file_write``/``file_read`` pass the raw ``filepath`` straight to
the sandbox service, which (as of the Sandbox Workspace Isolation epic) anchors
a RELATIVE path ``P`` under ``workspace_root`` (``/home/ubuntu/P``) — identically
whether written by the agent, the coordinator seed-install, or the coordinator
apply. Round-trip identity is preserved; the anchor simply moved from the
process CWD ``/sandbox`` to ``/home/ubuntu`` (sandbox-side, in
``sandbox/app/core/workspace.py``). This module locks the HOST-side contract:

- (a) ``ParentSandboxAdapter.atomic_write_file`` is path-transparent: it does
  NOT prepend ``/workspace`` (or any root); the path passes through unchanged
  and ALL anchoring happens sandbox-side.
- (c) A bare filename is rejected by the adapter as coordinator manifest
  hygiene (manifest paths carry a directory component). NOTE: the sandbox
  service itself now ACCEPTS a bare name (anchors it to
  ``/home/ubuntu/<name>``); the adapter is the final hygiene guard.

The full round-trip against the real Docker sandbox is exercised by the F5 E2E.
"""
import io  # noqa: F401  (kept for parity with sibling adapter tests)

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.infrastructure.external.sandbox.parent_sandbox_adapter import (
    ParentSandboxAdapter,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ok_result(data=None):
    r = MagicMock()
    r.success = True
    r.data = data
    return r


async def test_adapter_passes_relative_path_unchanged_no_workspace_join():
    """[INV-F2.3] The adapter must NOT prepend /workspace (or any root) — it
    passes the manifest-relative path straight through. Sandbox-side this now
    anchors to workspace_root=/home/ubuntu (not the /sandbox CWD), matching
    where the agent's own file_write lands."""
    handle = MagicMock()
    handle.upload_file = AsyncMock(return_value=_ok_result())
    adapter = ParentSandboxAdapter(handle)
    await adapter.atomic_write_file("workspace/a.py", b"data")
    # upload_file called with the EXACT path (no /workspace prefix mutation)
    args, kwargs = handle.upload_file.call_args
    passed_path = args[1] if len(args) > 1 else kwargs.get("path")
    assert passed_path == "workspace/a.py"


async def test_adapter_rejects_bare_filename():
    """[INV-F2.3 (c)] The adapter rejects a bare filename as coordinator
    manifest hygiene (manifest paths carry a directory component). The sandbox
    service would otherwise anchor it to /home/ubuntu/<name>; the adapter guard
    keeps manifest paths explicit."""
    handle = MagicMock()
    handle.upload_file = AsyncMock(return_value=_ok_result())
    adapter = ParentSandboxAdapter(handle)
    with pytest.raises(ValueError, match="bare filename"):
        await adapter.atomic_write_file("foo.py", b"data")
    handle.upload_file.assert_not_called()
