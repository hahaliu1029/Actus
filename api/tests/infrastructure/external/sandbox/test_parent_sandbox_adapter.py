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

from app.domain.external.parent_sandbox import SandboxPathCheck
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
    # Path carries a directory component (real coordinator manifest paths
    # always do) so it clears the F3.2 INV-F2.3 (c) bare-filename guard.
    await adapter.atomic_write_file("workspace/x.py", b"new content")
    assert fake_sandbox.upload_file.await_count == 1
    call_args = fake_sandbox.upload_file.await_args
    # First positional arg is the BinaryIO stream
    stream = call_args.args[0]
    assert isinstance(stream, io.BytesIO)
    assert stream.getvalue() == b"new content"
    # Path passes through unchanged — no /workspace-root join (path-transparency).
    assert call_args.args[1] == "workspace/x.py"


async def test_atomic_write_file_raises_OSError_on_failure(
    fake_sandbox: MagicMock,
) -> None:
    """Applier routes WRITE_IO_ERROR by catching Exception around per-entry
    write; OSError is the portable Exception family for IO failures, so
    this test pins the contract that adapter failures match that family."""
    fake_sandbox.upload_file = AsyncMock(return_value=_fail("disk full"))
    adapter = ParentSandboxAdapter(fake_sandbox)
    # Non-bare path so the OSError under test is the upload-failure branch,
    # not the F3.2 INV-F2.3 (c) bare-filename ValueError guard.
    with pytest.raises(OSError, match="disk full"):
        await adapter.atomic_write_file("workspace/x.py", b"x")


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


async def test_check_path_returns_exists_and_kind(fake_sandbox: MagicMock) -> None:
    fake_sandbox.check_file_exists = AsyncMock(
        return_value=_ok({"exists": True, "kind": "fifo"})
    )
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.check_path("workspace/p") == SandboxPathCheck(
        exists=True, kind="fifo",
    )


async def test_check_path_old_sandbox_no_kind_fails_open(fake_sandbox: MagicMock) -> None:
    # OLD sandbox image returns no "kind" -> fail-OPEN to a non-special value
    # so 2a proceeds (degrades to pre-S1b). exists=True -> "other"; absent -> "missing".
    fake_sandbox.check_file_exists = AsyncMock(return_value=_ok({"exists": True}))
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.check_path("p") == SandboxPathCheck(exists=True, kind="other")

    fake_sandbox.check_file_exists = AsyncMock(return_value=_ok({"exists": False}))
    assert await adapter.check_path("p") == SandboxPathCheck(exists=False, kind="missing")


async def test_check_path_raises_on_rpc_failure(fake_sandbox: MagicMock) -> None:
    fake_sandbox.check_file_exists = AsyncMock(return_value=_fail("boom"))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.check_path("p")


async def test_exists_unchanged_ignores_kind_field(fake_sandbox: MagicMock) -> None:
    # Backward-compat: exists() still reads only data["exists"].
    fake_sandbox.check_file_exists = AsyncMock(
        return_value=_ok({"exists": True, "kind": "fifo"})
    )
    adapter = ParentSandboxAdapter(fake_sandbox)
    assert await adapter.exists("p") is True


async def test_atomic_write_file_sets_refuse_special_true(fake_sandbox: MagicMock) -> None:
    """2b wiring: the coordinator adapter passes refuse_special=True to the
    sandbox upload (the anti-vacuity wiring test — proves the flag is SET,
    not just handled when raised). The path has a directory component so the
    bare-filename guard passes through."""
    adapter = ParentSandboxAdapter(fake_sandbox)
    await adapter.atomic_write_file("workspace/p", b"data")
    _, kwargs = fake_sandbox.upload_file.call_args
    assert kwargs.get("refuse_special") is True


async def test_snapshot_workspace_decodes_entries_and_truncated(
    fake_sandbox: MagicMock,
) -> None:
    from app.domain.external.parent_sandbox import (
        WorkspaceScan,
        WorkspaceScanEntry,
    )

    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "workspace/a.py": {
                "rel_path": "workspace/a.py", "kind": "regular",
                "sha256": "a" * 64, "size": 3, "mode": 33188,
                "link_target": None,
            },
            "workspace/link": {
                "rel_path": "workspace/link", "kind": "symlink",
                "sha256": None, "size": 7, "mode": 41471,
                "link_target": "../t.py",
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    scan = await adapter.snapshot_workspace(
        max_paths=1000, max_files=500, max_total_bytes=1_000_000,
        max_seconds=5.0,
    )
    assert isinstance(scan, WorkspaceScan)
    assert scan.truncated is False
    assert scan.entries["workspace/a.py"] == WorkspaceScanEntry(
        rel_path="workspace/a.py", kind="regular", sha256="a" * 64,
        size=3, mode=33188, link_target=None,
    )
    assert scan.entries["workspace/link"].link_target == "../t.py"


async def test_snapshot_workspace_truncated_true(fake_sandbox: MagicMock) -> None:
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {}, "truncated": True,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    scan = await adapter.snapshot_workspace(
        max_paths=1, max_files=1, max_total_bytes=1, max_seconds=0.001,
    )
    assert scan.truncated is True
    assert scan.entries == {}


async def test_snapshot_workspace_raises_on_rpc_failure(
    fake_sandbox: MagicMock,
) -> None:
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_fail("boom"))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1, max_files=1, max_total_bytes=1, max_seconds=1.0,
        )


async def test_snapshot_workspace_fails_closed_on_missing_truncated_flag(
    fake_sandbox: MagicMock,
) -> None:
    # Old image / malformed payload with no ``truncated`` key -> fail CLOSED
    # (treat as truncated, opposite of check_path's fail-open).
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {},
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    scan = await adapter.snapshot_workspace(
        max_paths=1, max_files=1, max_total_bytes=1, max_seconds=1.0,
    )
    assert scan.truncated is True


async def test_snapshot_workspace_malformed_entry_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # R2-B: a malformed-but-plausible payload — entries IS a mapping but an
    # entry dict is missing required keys (rel_path/size/mode) — must fail
    # CLOSED as OSError per the documented "Raises OSError on RPC failure"
    # contract, NOT raise a raw KeyError/AttributeError that escapes the
    # adapter's fail-closed envelope.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {"x": {"kind": "regular"}},
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


def test_non_coordinator_uploads_never_set_refuse_special():
    """Spec §4 test 15: ONLY the coordinator apply/seed path (this adapter,
    Step 5) sets refuse_special=True. The attachment caller (agent_task_runner)
    and skill-bundle caller (skill_bundle_sync) must NEVER pass refuse_special
    to upload_file — they keep the default False. Source-guard against a future
    edit accidentally hardening a non-coordinator upload."""
    import importlib.util
    import pathlib

    # `api/app` is a NAMESPACE package (no top-level __init__.py), so
    # `app.__file__` is None — locate each module's source via find_spec().origin.
    for mod in (
        "app.domain.services.agent_task_runner",
        "app.domain.services.tools.skill_bundle_sync",
    ):
        origin = importlib.util.find_spec(mod).origin
        text = pathlib.Path(origin).read_text(encoding="utf-8")
        assert "refuse_special" not in text, (
            f"{mod} unexpectedly references refuse_special — non-coordinator "
            f"uploads must keep the default False"
        )


async def test_snapshot_workspace_falsey_non_mapping_entries_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # Round-3 fail-OPEN fix: a malformed payload whose ``entries`` is a FALSEY
    # non-mapping (``[]``) with ``truncated=False`` previously decoded to a
    # "complete empty scan" because ``data.get("entries", {}) or {}`` coerced
    # ``[]`` -> ``{}``. That is a fail-OPEN: the differ sees no changes and
    # skips group-zero-apply. It must instead fail CLOSED as OSError (entries
    # is not a Mapping).
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": [], "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_absent_entries_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # ``entries`` absent entirely with ``truncated=False`` is also malformed —
    # an absent key is not a Mapping, so it must fail CLOSED as OSError rather
    # than coerce to an empty scan via the old ``or {}`` default.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_genuine_empty_scan_passes(
    fake_sandbox: MagicMock,
) -> None:
    # REGRESSION GUARD (must stay GREEN before AND after the round-3 fix): a
    # genuine empty scan ``{}`` is a Mapping, so it decodes to an empty
    # WorkspaceScan (NOT an OSError). The strict Mapping check must not reject
    # the legitimate empty-directory walk.
    from app.domain.external.parent_sandbox import WorkspaceScan

    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {}, "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    scan = await adapter.snapshot_workspace(
        max_paths=1000, max_files=500, max_total_bytes=1_000_000,
        max_seconds=5.0,
    )
    assert isinstance(scan, WorkspaceScan)
    assert scan.entries == {}
    assert scan.truncated is False


async def test_snapshot_workspace_falsey_int_truncated_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # Round-4 fail-OPEN fix (sibling of the round-3 ``entries`` fix): a
    # malformed ``truncated`` value that is a falsey NON-bool (``0``) must NOT
    # be coerced via ``bool(truncated)`` to ``False`` ("complete scan"). That
    # is a fail-OPEN: the differ trusts an incomplete/garbage walk and proceeds
    # to group-zero-apply. It must instead fail CLOSED as OSError (truncated is
    # present but not a real bool). ``isinstance(truncated, bool)`` correctly
    # rejects ``0`` even though ``bool`` is an ``int`` subclass.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {}, "truncated": 0,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_falsey_list_truncated_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # Same round-4 class as the ``0`` case: a falsey non-bool ``[]`` for
    # ``truncated`` must fail CLOSED as OSError, not coerce to a "complete
    # scan" via truthiness.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {}, "truncated": [],
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_truncated_real_bool_true_preserved(
    fake_sandbox: MagicMock,
) -> None:
    # REGRESSION GUARD (green before AND after the round-4 fix): a genuine
    # ``truncated=True`` bool must decode to ``scan.truncated is True``.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {}, "truncated": True,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    scan = await adapter.snapshot_workspace(
        max_paths=1000, max_files=500, max_total_bytes=1_000_000,
        max_seconds=5.0,
    )
    assert scan.truncated is True


async def test_snapshot_workspace_truncated_real_bool_false_preserved(
    fake_sandbox: MagicMock,
) -> None:
    # REGRESSION GUARD (green before AND after the round-4 fix): a genuine
    # ``truncated=False`` bool must decode to ``scan.truncated is False`` (a
    # complete scan — the differ proceeds). This is the value the round-4 fix
    # must keep working: a REAL bool ``False`` is honored, only a NON-bool
    # falsey ``truncated`` is rejected.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {}, "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    scan = await adapter.snapshot_workspace(
        max_paths=1000, max_files=500, max_total_bytes=1_000_000,
        max_seconds=5.0,
    )
    assert scan.truncated is False


# ---------------------------------------------------------------------------
# R7b — decode hardening (P1-h key==rel_path, P1-i per-kind invariants)
#
# Two remaining fail-OPEN gaps let GARBAGE entry data into the differ:
#  - P1-h: the dict KEY is trusted as the path even if it is non-``str`` or
#    disagrees with the entry's own ``rel_path`` — the differ would then key
#    a change by a path that contradicts the entry it carries.
#  - P1-i: per-kind structural invariants (regular⇒64hex sha + no link;
#    symlink⇒no sha + non-empty link; any other kind⇒no sha + no link) were
#    not enforced, so a payload could smuggle a sha onto a fifo or a regular
#    with a bogus/absent digest into the diff identity tuple.
# Both must fail CLOSED as OSError, consistent with the prior decode guards.
# ---------------------------------------------------------------------------


def _hex64(ch: str = "a") -> str:
    return ch * 64


async def test_snapshot_workspace_key_mismatch_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # P1-h: the dict KEY ("a.py") disagrees with the entry's own rel_path
    # ("b.py"). Trusting the key would key the diff by a path that contradicts
    # the entry — fail CLOSED as OSError.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "a.py": {
                "rel_path": "b.py", "kind": "regular", "sha256": _hex64(),
                "size": 1, "mode": 0o644, "link_target": None,
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_non_str_key_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # P1-h: JSON keys are always strings, but a defensive non-``str`` key
    # (here an int from a non-JSON / object-shaped payload) must still be
    # rejected — fail CLOSED as OSError.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            123: {
                "rel_path": "123", "kind": "regular", "sha256": _hex64(),
                "size": 1, "mode": 0o644, "link_target": None,
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_regular_with_none_sha_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # P1-i: a regular file MUST carry a 64-hex sha. ``sha256=None`` is a
    # contract violation (only regulars carry a sha) — fail CLOSED as OSError.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "x.py": {
                "rel_path": "x.py", "kind": "regular", "sha256": None,
                "size": 1, "mode": 0o644, "link_target": None,
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_regular_with_bad_sha_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # P1-i: a regular file's sha must be 64-char lowercase-hex. A short /
    # non-hex string ("abc") is malformed — fail CLOSED as OSError.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "x.py": {
                "rel_path": "x.py", "kind": "regular", "sha256": "abc",
                "size": 1, "mode": 0o644, "link_target": None,
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_symlink_with_none_link_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # P1-i: a symlink MUST carry a non-empty link_target. ``link_target=None``
    # is a contract violation — fail CLOSED as OSError.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "link": {
                "rel_path": "link", "kind": "symlink", "sha256": None,
                "size": 7, "mode": 41471, "link_target": None,
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_fifo_with_sha_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # P1-i: a non-regular, non-symlink kind (fifo) MUST have sha None AND
    # link_target None. A fifo carrying a non-None sha smuggles a digest into
    # the diff identity tuple — fail CLOSED as OSError.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "pipe": {
                "rel_path": "pipe", "kind": "fifo", "sha256": _hex64(),
                "size": 0, "mode": 4480, "link_target": None,
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_unknown_kind_fails_closed(
    fake_sandbox: MagicMock,
) -> None:
    # P1-i: ``kind`` must be one of the 9 FileKind values. A bogus kind
    # ("weird") that is not in the Literal set is malformed — fail CLOSED.
    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "x": {
                "rel_path": "x", "kind": "weird", "sha256": None,
                "size": 0, "mode": 0o644, "link_target": None,
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    with pytest.raises(OSError):
        await adapter.snapshot_workspace(
            max_paths=1000, max_files=500, max_total_bytes=1_000_000,
            max_seconds=5.0,
        )


async def test_snapshot_workspace_directory_kind_decodes(
    fake_sandbox: MagicMock,
) -> None:
    # DIRECTORY-KIND DECISION (chosen branch): a ``directory`` entry with
    # sha None + link None is STRUCTURALLY VALID per the P1-i rules
    # ("non-regular, non-symlink ⇒ sha None AND link None") and therefore
    # DECODES — we do NOT add an extra "reject directory/missing" rule.
    # Rationale: the four stated invariants are exhaustive; the sandbox walker
    # never EMITS a directory entry (directories are traversed, per the
    # WorkspaceScanEntry docstring), so an explicit reject rule would be
    # belt-and-suspenders beyond the spec's invariant list (over-engineering).
    # The structural decode here is the honest reflection of the rules.
    from app.domain.external.parent_sandbox import (
        WorkspaceScan,
        WorkspaceScanEntry,
    )

    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "subdir": {
                "rel_path": "subdir", "kind": "directory", "sha256": None,
                "size": 0, "mode": 16877, "link_target": None,
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    scan = await adapter.snapshot_workspace(
        max_paths=1000, max_files=500, max_total_bytes=1_000_000,
        max_seconds=5.0,
    )
    assert isinstance(scan, WorkspaceScan)
    assert scan.entries["subdir"] == WorkspaceScanEntry(
        rel_path="subdir", kind="directory", sha256=None,
        size=0, mode=16877, link_target=None,
    )


async def test_snapshot_workspace_valid_regular_and_symlink_decode(
    fake_sandbox: MagicMock,
) -> None:
    # REGRESSION GUARD (green before AND after the R7b fix): a fully valid
    # payload — a regular with a real 64-hex sha (key==rel_path, link None)
    # AND a symlink with a link_target set (sha None) — decodes to the correct
    # WorkspaceScan. Mirrors the existing happy-path test; pinned again here so
    # the R7b invariants are proven NOT to reject legitimate entries.
    from app.domain.external.parent_sandbox import (
        WorkspaceScan,
        WorkspaceScanEntry,
    )

    fake_sandbox.snapshot_workspace = AsyncMock(return_value=_ok({
        "entries": {
            "workspace/a.py": {
                "rel_path": "workspace/a.py", "kind": "regular",
                "sha256": _hex64(), "size": 3, "mode": 33188,
                "link_target": None,
            },
            "workspace/link": {
                "rel_path": "workspace/link", "kind": "symlink",
                "sha256": None, "size": 7, "mode": 41471,
                "link_target": "../t.py",
            },
        },
        "truncated": False,
    }))
    adapter = ParentSandboxAdapter(fake_sandbox)
    scan = await adapter.snapshot_workspace(
        max_paths=1000, max_files=500, max_total_bytes=1_000_000,
        max_seconds=5.0,
    )
    assert isinstance(scan, WorkspaceScan)
    assert scan.truncated is False
    assert scan.entries["workspace/a.py"] == WorkspaceScanEntry(
        rel_path="workspace/a.py", kind="regular", sha256=_hex64(),
        size=3, mode=33188, link_target=None,
    )
    assert scan.entries["workspace/link"] == WorkspaceScanEntry(
        rel_path="workspace/link", kind="symlink", sha256=None,
        size=7, mode=41471, link_target="../t.py",
    )
