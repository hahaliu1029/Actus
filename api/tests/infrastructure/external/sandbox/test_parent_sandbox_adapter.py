"""C2 PR-5 Task 5.2 — ParentSandboxAdapter unit tests.

Spec ref: §10.5 (ParentSandboxPort contract).

The adapter is a thin translation shim — these tests pin:
- ``compute_digest`` returns SHA-256 hex on read success, ``None`` on
  FileNotFoundError (per §9.3 step 5 reducer contract).
- ``exists`` collapses ``ToolResult.success && .data`` into bool.
- ``read_file`` translates 404 → FileNotFoundError, other failures bubble.
- ``atomic_write_file`` / ``delete_file`` raise ``OSError`` on
  ToolResult.success=False so the applier routes to rollback.
"""
from __future__ import annotations

import hashlib
import io
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.tool_result import ToolResult
from app.infrastructure.external.sandbox.parent_sandbox_adapter import (
    ParentSandboxAdapter,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ok(data: object = True, message: str = "ok") -> ToolResult:
    return ToolResult(success=True, message=message, data=data)


def _fail(message: str = "boom") -> ToolResult:
    return ToolResult(success=False, message=message, data=None)


@pytest.fixture
def fake_sandbox() -> MagicMock:
    s = MagicMock()
    s.check_file_exists = AsyncMock(return_value=_ok(True))
    s.download_file = AsyncMock(return_value=io.BytesIO(b"hello"))
    s.upload_file = AsyncMock(return_value=_ok())
    s.delete_file = AsyncMock(return_value=_ok())
    return s


async def test_compute_digest_returns_sha256(fake_sandbox: MagicMock) -> None:
    adapter = ParentSandboxAdapter(fake_sandbox)
    expected = hashlib.sha256(b"hello").hexdigest()
    assert await adapter.compute_digest("x/y.py") == expected


async def test_compute_digest_missing_file_returns_none(
    fake_sandbox: MagicMock,
) -> None:
    """§9.3 step 5: missing file → no drift signal; reducer doesn't escalate."""
    fake_response = MagicMock()
    fake_response.status_code = 404
    err = Exception("not found")
    err.response = fake_response  # type: ignore[attr-defined]
    fake_sandbox.download_file = AsyncMock(side_effect=err)
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.compute_digest("missing.py") is None


async def test_compute_digest_5xx_bubbles(fake_sandbox: MagicMock) -> None:
    """Conservative translation: only 404 → FileNotFoundError; other
    HTTP failures must bubble so the orchestrator surfaces the real
    cause (network, auth, sandbox dead) instead of silently dropping
    them as 'file missing'."""
    fake_response = MagicMock()
    fake_response.status_code = 503
    err = RuntimeError("sandbox unreachable")
    err.response = fake_response  # type: ignore[attr-defined]
    fake_sandbox.download_file = AsyncMock(side_effect=err)
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(RuntimeError, match="sandbox unreachable"):
        await adapter.compute_digest("any.py")


async def test_exists_true_when_data_true(fake_sandbox: MagicMock) -> None:
    fake_sandbox.check_file_exists = AsyncMock(return_value=_ok(True))
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.exists("x.py") is True


async def test_exists_false_when_data_false(fake_sandbox: MagicMock) -> None:
    fake_sandbox.check_file_exists = AsyncMock(return_value=_ok(False))
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.exists("x.py") is False


async def test_exists_reads_dict_exists_key(fake_sandbox: MagicMock) -> None:
    """[codex R12 P1 fix] Live sandbox returns
    ``FileCheckResult({filepath, exists: bool})`` as ``result.data``
    — a dict that is ALWAYS truthy. Adapter must extract the
    ``exists`` key, not rely on truthiness of the dict itself.
    """
    fake_sandbox.check_file_exists = AsyncMock(return_value=_ok(
        {"filepath": "/abs/x.py", "exists": False},
    ))
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.exists("x.py") is False

    fake_sandbox.check_file_exists = AsyncMock(return_value=_ok(
        {"filepath": "/abs/x.py", "exists": True},
    ))
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.exists("x.py") is True


async def test_exists_reads_attr_exists(fake_sandbox: MagicMock) -> None:
    """If a future from_sandbox preserves the FileCheckResult Pydantic
    model, adapter must read ``.exists`` attribute correctly."""
    class _FakeResult:
        exists = False
        filepath = "/abs/x.py"

    fake_sandbox.check_file_exists = AsyncMock(
        return_value=_ok(_FakeResult()),
    )
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.exists("x.py") is False


async def test_exists_raises_on_rpc_failure(fake_sandbox: MagicMock) -> None:
    """[codex R1 P1#3 fix] RPC failure (e.g. sandbox down) → raise.

    The previous silent-False behavior caused the applier preflight to
    misreport FILE_MISSING for modify/delete (file may actually exist
    — we just couldn't reach the sandbox) or to silently proceed to
    write an ``add`` over an unverified path. Raising surfaces the
    real fault and lets the applier translate to WRITE_IO_ERROR with
    the sandbox error preserved in ``failed_reason``."""
    fake_sandbox.check_file_exists = AsyncMock(return_value=_fail("rpc dead"))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError, match="rpc dead"):
        await adapter.exists("x.py")


async def test_read_file_returns_bytes(fake_sandbox: MagicMock) -> None:
    fake_sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"world"))
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.read_file("x.py") == b"world"


async def test_read_file_404_translates_to_FileNotFoundError(
    fake_sandbox: MagicMock,
) -> None:
    fake_response = MagicMock()
    fake_response.status_code = 404
    err = Exception("not found")
    err.response = fake_response  # type: ignore[attr-defined]
    fake_sandbox.download_file = AsyncMock(side_effect=err)
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(FileNotFoundError, match="missing.py"):
        await adapter.read_file("missing.py")


async def test_atomic_write_file_uploads_bytes(fake_sandbox: MagicMock) -> None:
    adapter = ParentSandboxAdapter(fake_sandbox)
    await adapter.atomic_write_file("x.py", b"new content")
    assert fake_sandbox.upload_file.await_count == 1
    call_args = fake_sandbox.upload_file.await_args
    # First positional arg is the BinaryIO stream
    stream = call_args.args[0]
    assert isinstance(stream, io.BytesIO)
    assert stream.getvalue() == b"new content"
    assert call_args.args[1] == "x.py"


async def test_atomic_write_file_raises_OSError_on_failure(
    fake_sandbox: MagicMock,
) -> None:
    """Applier routes WRITE_IO_ERROR by catching Exception around per-entry
    write; OSError is the portable Exception family for IO failures, so
    this test pins the contract that adapter failures match that family."""
    fake_sandbox.upload_file = AsyncMock(return_value=_fail("disk full"))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError, match="disk full"):
        await adapter.atomic_write_file("x.py", b"x")


async def test_delete_file_raises_OSError_on_failure(
    fake_sandbox: MagicMock,
) -> None:
    fake_sandbox.delete_file = AsyncMock(return_value=_fail("perm denied"))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError, match="perm denied"):
        await adapter.delete_file("x.py")


async def test_delete_file_success_returns_none(fake_sandbox: MagicMock) -> None:
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.delete_file("x.py") is None
    fake_sandbox.delete_file.assert_awaited_once_with("x.py")


def test_adapter_has_no_destroy_or_suspend() -> None:
    """[spec §10.5 M1 invariant] The adapter MUST NOT expose destroy() or
    suspend(). Lifecycle stays exclusively with MailboxSupervisor.*Handler.
    """
    assert not hasattr(ParentSandboxAdapter, "destroy")
    assert not hasattr(ParentSandboxAdapter, "suspend")
