"""F3.2 G2b — path-transparency + bare-filename guard (INV-F2.3).

The agent's ``file_write``/``file_read`` pass the raw ``filepath`` straight to
the sandbox service (WORKDIR ``/sandbox``, no workspace-root join), so a
manifest-relative path ``P`` resolves identically (``/sandbox/P``) whether
written by the agent, the coordinator seed-install, or the coordinator apply
— round-trip identity by construction. This module locks:

- (a) ``ParentSandboxAdapter.atomic_write_file`` is path-transparent: it does
  NOT prepend ``/workspace`` (or any root); the path passes through unchanged.
- (c) A bare filename (``os.path.dirname("foo.py") == ""``) would make the
  sandbox ``os.makedirs("")`` raise; the adapter rejects it loudly first.

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
    passes the manifest-relative path straight through, matching where the
    agent's own file_write lands (WORKDIR /sandbox)."""
    handle = MagicMock()
    handle.upload_file = AsyncMock(return_value=_ok_result())
    adapter = ParentSandboxAdapter(handle)
    await adapter.atomic_write_file("workspace/a.py", b"data")
    # upload_file called with the EXACT path (no /workspace prefix mutation)
    args, kwargs = handle.upload_file.call_args
    passed_path = args[1] if len(args) > 1 else kwargs.get("path")
    assert passed_path == "workspace/a.py"


async def test_adapter_rejects_bare_filename():
    """[INV-F2.3 (c)] A bare filename would make the sandbox os.makedirs('')
    raise; the adapter rejects it with a clear error instead."""
    handle = MagicMock()
    handle.upload_file = AsyncMock(return_value=_ok_result())
    adapter = ParentSandboxAdapter(handle)
    with pytest.raises(ValueError, match="bare filename"):
        await adapter.atomic_write_file("foo.py", b"data")
    handle.upload_file.assert_not_called()
