"""F2.3 — patch extraction (G1e, INV-F1.6/F1.8).

[finish-core §5.1.5] After the child ReAct loop completes a write-phase work
unit, the parent turns the child's typed-write tool calls
(``ChildRunResult.tool_calls``) + the final bytes read from the child sandbox
into a list of ``FilePatchEntry``. Out-of-lease writes are rejected with
``_OutOfLeaseWriteError`` (defensive lease enforcement; the runtime PE gate is
deferred — §5.1.7).
"""
import asyncio
import hashlib

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.application.services.coordinator_child_runner import (
    CoordinatorChildRunner,
    ChildRunResult,
    _OutOfLeaseWriteError,
)
from app.domain.models.event import ToolEvent, ToolEventStatus
from app.domain.models.work_unit import PathLease, WorkUnit


@pytest.fixture
def anyio_backend():
    return "asyncio"


pytestmark = pytest.mark.anyio


def _tool_write(path, content="new"):
    return ToolEvent(
        tool_call_id="tc", tool_name="file", function_name="file_write",
        function_args={"filepath": path, "content": content},
        status=ToolEventStatus.CALLING,
    )


def _runner(child_sandbox, artifact):
    return CoordinatorChildRunner(
        cancel_event=asyncio.Event(), child_sandbox=child_sandbox,
        artifact_storage=artifact, coordinator_run_id="run-1",
    )


async def test_extraction_builds_modify_entry_from_tool_calls_and_child_bytes():
    final_bytes = b"new content"
    digest = hashlib.sha256(final_bytes).hexdigest()
    child_sandbox = MagicMock()
    child_sandbox.read_file = AsyncMock(return_value=final_bytes)
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="patchref-1")
    r = _runner(child_sandbox, artifact)
    wu = WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="a.py", op="modify", base_digest="b" * 64, seed_content_ref="s")],
    )
    result = ChildRunResult(done_event=MagicMock(), tool_calls=(_tool_write("a.py"),))
    files = await r._extract_patch_files_from_history("run-1", wu, result)
    assert len(files) == 1
    entry = files[0]
    assert entry.path == "a.py"
    assert entry.op == "modify"
    assert entry.base_digest == "b" * 64
    assert entry.new_digest == digest
    assert entry.content_ref == "patchref-1"
    child_sandbox.read_file.assert_awaited_once_with("a.py")  # final bytes from child sandbox


async def test_extraction_rejects_out_of_lease_write():
    child_sandbox = MagicMock()
    child_sandbox.read_file = AsyncMock(return_value=b"x")
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="r")
    r = _runner(child_sandbox, artifact)
    wu = WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="a.py", op="modify", base_digest="b" * 64, seed_content_ref="s")],
    )
    result = ChildRunResult(done_event=MagicMock(), tool_calls=(_tool_write("evil.py"),))
    with pytest.raises(_OutOfLeaseWriteError):
        await r._extract_patch_files_from_history("run-1", wu, result)


async def test_extraction_empty_tool_calls_yields_no_files():
    r = _runner(MagicMock(), MagicMock())
    wu = WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="a.py", op="modify", base_digest="b" * 64, seed_content_ref="s")],
    )
    result = ChildRunResult(done_event=MagicMock(), tool_calls=())
    files = await r._extract_patch_files_from_history("run-1", wu, result)
    assert files == []


def _tool_write_args(args):
    """A file_write ToolEvent with arbitrary function_args (for path-semantics)."""
    return ToolEvent(
        tool_call_id="tc", tool_name="file", function_name="file_write",
        function_args=args, status=ToolEventStatus.CALLING,
    )


# ---------------------------------------------------------------------------
# [F2 P1] Path extraction must match ChildScopeGate.extract_target_path:
# filepath-priority with NO truthy fallback. ``filepath=""`` stays "" (treated
# as no-target, skipped) and does NOT fall back to ``path`` — so a write the
# gate evaluated against "" is never misattributed to the ``path`` key here.
# ---------------------------------------------------------------------------

async def test_extraction_empty_filepath_does_not_fall_back_to_path():
    # filepath present but empty → no-target (skip), so the ``path`` key
    # ("a.py") must NOT produce an entry even though there's a matching lease.
    # This is the divergence the `args.get("filepath") or args.get("path")`
    # truthy fallback would have gotten WRONG (it would build an a.py entry).
    child_sandbox = MagicMock()
    child_sandbox.read_file = AsyncMock(return_value=b"x")
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="r")
    r = _runner(child_sandbox, artifact)
    wu = WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="a.py", op="modify", base_digest="b" * 64, seed_content_ref="s")],
    )
    result = ChildRunResult(
        done_event=MagicMock(),
        tool_calls=(_tool_write_args({"filepath": "", "path": "a.py", "content": "x"}),),
    )
    files = await r._extract_patch_files_from_history("run-1", wu, result)
    assert files == []  # empty filepath skipped; NO fallback to path="a.py"
    child_sandbox.read_file.assert_not_awaited()


async def test_extraction_filepath_priority_normal_case_still_builds_entry():
    # Sanity: a normal {"filepath": "a.py"} still produces the entry (the fix
    # only changes the empty-filepath edge, not the happy path).
    final_bytes = b"new content"
    child_sandbox = MagicMock()
    child_sandbox.read_file = AsyncMock(return_value=final_bytes)
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="patchref-1")
    r = _runner(child_sandbox, artifact)
    wu = WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="a.py", op="modify", base_digest="b" * 64, seed_content_ref="s")],
    )
    result = ChildRunResult(
        done_event=MagicMock(),
        tool_calls=(_tool_write_args({"filepath": "a.py", "content": "x"}),),
    )
    files = await r._extract_patch_files_from_history("run-1", wu, result)
    assert len(files) == 1
    assert files[0].path == "a.py"
    child_sandbox.read_file.assert_awaited_once_with("a.py")


async def test_extraction_falls_back_to_path_only_when_filepath_absent():
    # When the "filepath" KEY is absent entirely, the ``path`` key is used
    # (matches the gate's ``if path is None: path = args.get("path")``).
    final_bytes = b"data"
    child_sandbox = MagicMock()
    child_sandbox.read_file = AsyncMock(return_value=final_bytes)
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="patchref-2")
    r = _runner(child_sandbox, artifact)
    wu = WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="a.py", op="modify", base_digest="b" * 64, seed_content_ref="s")],
    )
    result = ChildRunResult(
        done_event=MagicMock(),
        tool_calls=(_tool_write_args({"path": "a.py", "content": "x"}),),
    )
    files = await r._extract_patch_files_from_history("run-1", wu, result)
    assert len(files) == 1
    assert files[0].path == "a.py"


async def test_extraction_filepath_none_falls_back_to_path():
    # Matches ChildScopeGate.extract_target_path: a filepath VALUE of None
    # falls back to "path" (value-based ``if path is None``, NOT key-presence).
    # The gate lease-checks this write against "a.py", so the manifest MUST
    # include it — a key-presence check ("filepath" in args) would WRONGLY skip
    # it (filepath key present → path=None → not a str → continue → []).
    final_bytes = b"x"
    child_sandbox = MagicMock()
    child_sandbox.read_file = AsyncMock(return_value=final_bytes)
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="ref-1")
    r = _runner(child_sandbox, artifact)
    wu = WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="a.py", op="modify", base_digest="b" * 64, seed_content_ref="s")],
    )
    result = ChildRunResult(
        done_event=MagicMock(),
        tool_calls=(_tool_write_args({"filepath": None, "path": "a.py", "content": "x"}),),
    )
    files = await r._extract_patch_files_from_history("run-1", wu, result)
    assert len(files) == 1
    assert files[0].path == "a.py"
    child_sandbox.read_file.assert_awaited_once_with("a.py")
