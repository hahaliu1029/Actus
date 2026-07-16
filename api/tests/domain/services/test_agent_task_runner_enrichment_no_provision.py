"""INV-SPM-12 — the post-outcome enrichment pump must NEVER provision.

Whole-epic fable audit R1 P1-1: PR-1b converted the runner's event-pump
post-processing from the raw eager handle to ``await self._sandbox_accessor.get()``;
PR-1c then made ``get()`` a provision-if-needed op. Because react_graph emits
``ToolEvent(status=CALLED)`` for Denied / AllowError outcomes too
(react_graph.py:671-717), a DENIED or FAILED sandbox tool call would still drive
``SandboxProvisioner`` → ``lifecycle.bind_new`` (container creation + attachment
flush + skill sync) from the observability pump — the exact side effect the
approval gate exists to prevent (spec §5.2f: "Denied/Ask 分支零 accessor/lifecycle
调用"; "供给不会先于/绕过审批").

Fix direction (i): every enrichment site switches from provisioning ``get()`` to
the ready-only ``peek()`` probe — ``None`` (no sandbox) skips the enrichment item,
non-``None`` uses the returned handle. always mode is byte-identical (Eager
``peek()`` ≡ ``get()``); only on_demand-without-a-sandbox changes: it now SKIPS
instead of provisioning.

These tests build a real ``OnDemandSandboxAccessor`` over a real
``SandboxProvisioner`` (fake lifecycle counting ``bind_new`` / ``acquire``) and
assert the pump performs ZERO provisioning on Denied / failed / attachment /
generated-file paths. The already-provisioned regression pins the always-parity
semantic (reading an existing sandbox's console is preserved).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_accessors import (
    OnDemandBrowserAccessor,
    OnDemandSandboxAccessor,
)
from app.application.services.sandbox_provisioner import SandboxProvisioner
from app.domain.errors.sandbox_lifecycle import SessionUnboundError
from app.domain.models.event import (
    BrowserToolContent,
    MessageEvent,
    ShellToolContent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.file import File
from app.domain.models.tool_result import ToolResult
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _FakeLifecycle:
    """Counts container-creating calls. ``acquire`` always misses (SessionUnbound)
    so ``SandboxProvisioner.get()`` falls through to ``bind_new`` — the physical
    container-creation edge whose call count is the provisioning proof."""

    def __init__(self) -> None:
        self.bind_new_calls = 0
        self.acquire_calls = 0
        self.handle = AsyncMock()

    async def acquire(self, session_id: str):
        self.acquire_calls += 1
        raise SessionUnboundError(session_id)

    async def bind_new(self, session_id: str, *, user_id=None):
        self.bind_new_calls += 1
        return self.handle


class _FakeUoW:
    """Async-CM stub for _sync_file_to_storage's session lookup (never reached
    on the no-sandbox path, present so the RED path doesn't AttributeError)."""

    def __init__(self) -> None:
        self.session = AsyncMock()

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


def _make_on_demand_runner() -> tuple[AgentTaskRunner, _FakeLifecycle, SandboxProvisioner]:
    """Real OnDemand accessor + real provisioner + fake lifecycle over a bypassed
    (``object.__new__``) runner. Mirrors the R1 audit repro but wires the full set
    of enrichment collaborators so every pump site is exercised for real."""
    lifecycle = _FakeLifecycle()
    provisioner = SandboxProvisioner(
        session_id="s1",
        user_id="u1",
        lifecycle=lifecycle,
        hooks=[],
        timeout_seconds=5,
    )
    runner = object.__new__(AgentTaskRunner)
    sandbox_accessor = OnDemandSandboxAccessor(provisioner)
    runner._sandbox_accessor = sandbox_accessor
    runner._browser_accessor = OnDemandBrowserAccessor(sandbox_accessor)
    runner._session_id = "s1"
    runner._uow = _FakeUoW()
    runner._file_storage = AsyncMock()
    return runner, lifecycle, provisioner


def _assert_no_provision(lifecycle: _FakeLifecycle, provisioner: SandboxProvisioner) -> None:
    assert lifecycle.bind_new_calls == 0, "enrichment pump provisioned a container"
    assert provisioner.state == "unprovisioned", "provisioner left non-unprovisioned"
    assert provisioner.peek() is None


# ============================================================
# No-provision contract — the five brief sites + skill (sixth
# structurally-identical read_shell_output enrichment).
# ============================================================


async def test_denied_shell_event_does_not_provision() -> None:
    """Brief site 1: shell CALLED (Denied shape) → read_shell_output enrichment
    must NOT provision (was bind_new==1)."""
    runner, lifecycle, provisioner = _make_on_demand_runner()
    evt = ToolEvent(
        tool_call_id="c1",
        tool_name="shell",
        function_name="shell_execute",
        function_args={"session_id": "sid1"},
        function_result=None,  # Denied path: no artifact/result
        status=ToolEventStatus.CALLED,
    )

    await runner._handle_tool_event(evt)

    _assert_no_provision(lifecycle, provisioner)
    # No sandbox → no console read → no ShellToolContent synthesized.
    assert evt.tool_content is None


async def test_file_event_does_not_provision() -> None:
    """Brief site 2: file CALLED with a filepath → read_file + _sync_file_to_storage
    enrichment must NOT provision."""
    runner, lifecycle, provisioner = _make_on_demand_runner()
    evt = ToolEvent(
        tool_call_id="c1",
        tool_name="file",
        function_name="file_view",
        function_args={"filepath": "/home/ubuntu/x.txt"},
        function_result=None,
        status=ToolEventStatus.CALLED,
    )

    await runner._handle_tool_event(evt)

    _assert_no_provision(lifecycle, provisioner)
    assert evt.tool_content is None


async def test_browser_event_does_not_provision() -> None:
    """Brief site 3: browser CALLED → _get_browser_screenshot → browser_accessor.get()
    (which chains sandbox_accessor.get()) must NOT provision."""
    runner, lifecycle, provisioner = _make_on_demand_runner()
    evt = ToolEvent(
        tool_call_id="c1",
        tool_name="browser",
        function_name="browser_view",
        function_args={"url": "https://x"},
        function_result=None,
        status=ToolEventStatus.CALLED,
    )

    await runner._handle_tool_event(evt)

    _assert_no_provision(lifecycle, provisioner)
    # No sandbox → screenshot skipped → no BrowserToolContent.
    assert evt.tool_content is None
    assert runner._browser_accessor.peek() is None


async def test_message_attachment_sync_does_not_provision() -> None:
    """Brief site 5: assistant MessageEvent carrying an LLM-authored /home/ubuntu
    attachment path → _sync_message_attachments_to_storage → _sync_file_to_storage
    must NOT provision at message-sync time."""
    runner, lifecycle, provisioner = _make_on_demand_runner()
    msg = MessageEvent(
        role="assistant",
        message="here is the file",
        attachments=[File(filename="x.pdf", filepath="/home/ubuntu/upload/x.pdf")],
    )

    await runner._sync_message_attachments_to_storage(msg)

    _assert_no_provision(lifecycle, provisioner)
    # Nothing synced (no sandbox to download from) → attachments cleared.
    assert msg.attachments == []


async def test_generated_files_sync_does_not_provision() -> None:
    """Brief site 4: _sync_generated_files(exec_dir) → list_files must NOT provision."""
    runner, lifecycle, provisioner = _make_on_demand_runner()

    await runner._sync_generated_files("/home/ubuntu/output")

    _assert_no_provision(lifecycle, provisioner)


async def test_shell_with_exec_dir_does_not_provision() -> None:
    """Brief site 1+4 through the real pump: shell CALLED carrying exec_dir →
    read_shell_output AND _sync_generated_files, both must NOT provision."""
    runner, lifecycle, provisioner = _make_on_demand_runner()
    evt = ToolEvent(
        tool_call_id="c1",
        tool_name="shell",
        function_name="shell_execute",
        function_args={"session_id": "sid1", "exec_dir": "/home/ubuntu/output"},
        function_result=None,
        status=ToolEventStatus.CALLED,
    )

    await runner._handle_tool_event(evt)

    _assert_no_provision(lifecycle, provisioner)


async def test_failed_skill_shell_session_does_not_provision() -> None:
    """Beyond the brief's 5 sites: the skill branch performs the SAME
    read_shell_output enrichment as the shell branch (agent_task_runner.py skill
    branch). A FAILED native skill still carries shell_session_id
    (skill.py:500), so this is a real enrichment provision site — closing it makes
    INV-SPM-12 a greppable "zero get() in the pump" invariant."""
    runner, lifecycle, provisioner = _make_on_demand_runner()
    evt = ToolEvent(
        tool_call_id="c1",
        tool_name="skill",
        function_name="skill_xyz",
        function_args={},
        # legacy result → projector surfaces fr.data with shell_session_id
        function_result=ToolResult(
            success=False,
            message="skill failed",
            data={"shell_session_id": "ssid42", "exec_dir": "/workspace"},
        ),
        status=ToolEventStatus.CALLED,
    )

    await runner._handle_tool_event(evt)

    _assert_no_provision(lifecycle, provisioner)


# ============================================================
# Always-parity regression — an ALREADY-provisioned on_demand
# sandbox is still read by the pump (reading an existing
# sandbox's console is harmless and preserved).
# ============================================================


async def test_denied_shell_reads_console_when_sandbox_already_provisioned() -> None:
    """on_demand WITH a ready sandbox + Denied shell → enrichment reads the console
    exactly as before, WITHOUT a fresh provision. This is the always-parity semantic
    the fix must preserve (peek returns the ready handle)."""
    runner, lifecycle, provisioner = _make_on_demand_runner()
    lifecycle.handle.read_shell_output = AsyncMock(
        return_value=MagicMock(data={"console_records": [{"cmd": "denied-but-read"}]})
    )

    # Provision once (the legitimate first sandbox tool call).
    await provisioner.get()
    assert provisioner.state == "ready"
    assert lifecycle.bind_new_calls == 1
    lifecycle.bind_new_calls = 0  # reset — any further bind_new is a NEW provision

    evt = ToolEvent(
        tool_call_id="c1",
        tool_name="shell",
        function_name="shell_execute",
        function_args={"session_id": "sid1"},
        function_result=None,
        status=ToolEventStatus.CALLED,
    )

    await runner._handle_tool_event(evt)

    assert lifecycle.bind_new_calls == 0, "enrichment re-provisioned an existing sandbox"
    assert isinstance(evt.tool_content, ShellToolContent)
    assert evt.tool_content.console == [{"cmd": "denied-but-read"}]
