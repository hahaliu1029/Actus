"""PE-0 Phase 9: tool_node dispatches native tool calls through pe.evaluate.

Tests use option (b) from the plan: call tool_node directly with AsyncMock
state/config rather than building the full graph. This avoids the need for
a checkpointer or LangGraph compilation while still exercising the PE branch.

All tests are async — we use pytest-anyio (same as other graph tests).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
    Passthrough,
)
from app.domain.services.permission.tool_call_spec import ToolCallSpec

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Fake PE implementations
# ---------------------------------------------------------------------------

class FakeRecordingPE:
    """Records every (call, ctx) pair passed to evaluate; returns AllowSuccess."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.next_outcome: str = "allow"  # "allow" | "deny" | "ask" | "error"

    async def evaluate(self, call: ToolCallSpec, ctx):
        self.calls.append((call, ctx))
        if self.next_outcome == "allow":
            return AllowSuccess(content="auto", data={})
        if self.next_outcome == "deny":
            return Denied(
                content="denied by fake PE",
                reason=DecisionReason(
                    type="approval_policy", code="fake_deny", message=""
                ),
            )
        if self.next_outcome == "ask":
            return Asked(
                content="waiting for user",
                reason=DecisionReason(
                    type="risk_enforce", code="medium", message="test"
                ),
            )
        if self.next_outcome == "error":
            return AllowError(
                content="error from fake PE",
                reason=DecisionReason(
                    type="exception", code="fake_error", message=""
                ),
                retryable=False,
            )
        return AllowSuccess(content="auto", data={})

    async def preflight_resume(self, *a, **kw):
        pass

    async def commit_resume(self, *a, **kw):
        pass


def _make_fake_ssm(mode=SessionStatus.RUNNING, revision=1):
    """Return an AsyncMock SSM that returns (mode, revision) from get_mode_with_revision."""
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(mode, revision))
    return ssm


def _make_state(tool_name: str, tool_args: dict, call_id: str = "tc1") -> dict:
    """Build a minimal ReactGraphState dict with one AIMessage tool call."""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": tool_name, "args": tool_args, "type": "tool_call"}],
            )
        ],
        "llm_input_messages": [],
        "step_description": "test",
        "original_request": "test",
        "language": "en",
        "attachments": [],
        "image_content_blocks": [],
        "events": [],
        "should_interrupt": False,
        "soft_hint_sent": False,
        "attempt_count": 0,
        "failure_count": 0,
        "completed_tool_call_prefix": [],
        "approved_tool_call_ids": [],
        "pending_ask_outcome": None,
        "pending_ask_tool_call_id": None,
        "pending_ask_artifact": None,
        "pending_ask_tool_args": None,
    }


def _make_config(fake_pe, fake_ssm, *, user_id="u", session_id="s", extra: dict | None = None):
    """Build a minimal RunnableConfig configurable dict."""
    configurable: dict = {
        "permission_engine": fake_pe,
        "session_state_machine": fake_ssm,
        "permission_engine_native_enabled": True,
        "user_id": user_id,
        "session_id": session_id,
        "thread_id": session_id,
    }
    if extra:
        configurable.update(extra)
    return {"configurable": configurable}


# ---------------------------------------------------------------------------
# Helper to build a minimal graph and extract the tool_node closure
# ---------------------------------------------------------------------------

def _build_tool_node_fn():
    """Build a minimal react_graph and return the internal tool_node function.

    We invoke build_react_graph with a stub LLM and a fake file_write tool,
    then retrieve tool_node from the compiled graph:
        graph.nodes["tool_node"].bound.afunc
    is the async closure created by build_react_graph.
    """
    from langchain_core.messages import AIMessage as _AIMessage
    from langchain_core.tools import tool as lc_tool
    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        return f"wrote {path}"

    @lc_tool
    async def shell_execute(command: str) -> str:
        """Run shell command."""
        return "ok"

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(
        return_value=_AIMessage(content='{"success":true,"result":"done","attachments":[]}')
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    graph = build_react_graph(stub_llm, [file_write, shell_execute])
    # Extract the tool_node closure: PregelNode.bound is a RunnableCallable
    # whose .afunc attribute holds the actual async tool_node function.
    tool_node_fn = graph.nodes["tool_node"].bound.afunc
    return tool_node_fn


# ---------------------------------------------------------------------------
# Task 9.1: PE.evaluate is called with correct ToolCallSpec
# ---------------------------------------------------------------------------

class TestToolNodeCallsPeEvaluate:
    async def test_tool_node_calls_pe_evaluate_with_tool_call_spec(self):
        """tool_node calls pe.evaluate once with a ToolCallSpec for a file_write call."""
        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_ssm = _make_fake_ssm()

        state = _make_state("file_write", {"path": "/x", "content": "hello"})
        config = _make_config(fake_pe, fake_ssm)

        await tool_node_fn(state, config)

        assert len(fake_pe.calls) == 1, "PE.evaluate should be called exactly once"
        spec, ctx = fake_pe.calls[0]
        assert isinstance(spec, ToolCallSpec)
        assert spec.tool_name == "file_write"
        assert spec.tool_source == "native"
        assert ctx.session_mode_revision is not None
        assert ctx.session_mode == SessionStatus.RUNNING

    async def test_tool_node_passes_user_id_and_session_id_to_spec(self):
        """ToolCallSpec receives user_id and session_id from configurable."""
        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_ssm = _make_fake_ssm()

        state = _make_state("file_write", {"path": "/y"})
        config = _make_config(fake_pe, fake_ssm, user_id="user-42", session_id="sess-99")

        await tool_node_fn(state, config)

        spec, _ = fake_pe.calls[0]
        assert spec.user_id == "user-42"
        assert spec.session_id == "sess-99"

    async def test_tool_node_pe_not_called_when_pe_absent(self):
        """When pe is not in configurable, PE branch is skipped (legacy path)."""
        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()

        state = _make_state("file_write", {"path": "/z"})
        # No permission_engine key
        config = {
            "configurable": {
                "user_id": "u",
                "session_id": "s",
            }
        }

        await tool_node_fn(state, config)

        assert len(fake_pe.calls) == 0, "PE should not be called when absent"

    async def test_tool_node_pe_not_called_when_flag_false(self):
        """When permission_engine_native_enabled=False, PE branch is skipped."""
        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_ssm = _make_fake_ssm()

        state = _make_state("file_write", {"path": "/a"})
        config = _make_config(fake_pe, fake_ssm, extra={"permission_engine_native_enabled": False})

        await tool_node_fn(state, config)

        assert len(fake_pe.calls) == 0, "PE should not be called when flag is off"


# ---------------------------------------------------------------------------
# Task 9.2: Outcome translation — AllowSuccess / Denied / Asked / AllowError
# ---------------------------------------------------------------------------

class TestPeOutcomeTranslation:
    async def test_allow_success_runs_wrapper_and_emits_tool_message(self):
        """AllowSuccess → wrapper runs → ToolMessage in result."""
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage

        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "allow"
        fake_ssm = _make_fake_ssm()

        state = _make_state("file_write", {"path": "/x", "content": "data"})
        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, "Should emit at least one ToolMessage"

    async def test_denied_emits_tool_message_without_running_wrapper(self):
        """Denied outcome → ToolMessage with error status, no wrapper execution."""
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage

        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "deny"
        fake_ssm = _make_fake_ssm()

        state = _make_state("file_write", {"path": "/x"})
        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        # The content should reflect the denial
        denial_msg = tool_messages[0]
        assert denial_msg.status == "error"

    async def test_asked_routes_to_interrupt_helper(self):
        """Asked outcome → Command(goto='interrupt_helper') with pending_ask_* fields.

        Uses file_write (not shell_execute) so the shell AST validator gate (P1#4)
        does not intercept before PE.evaluate is called.  The AST gate only applies
        to shell_execute; file_write goes directly to pe.evaluate which returns Asked.
        """
        from langgraph.types import Command

        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "ask"
        fake_ssm = _make_fake_ssm()

        # file_write is a native tool that bypasses the shell AST gate
        state = _make_state("file_write", {"path": "/workspace/test.txt"})
        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        assert isinstance(result, Command)
        assert result.goto == "interrupt_helper", (
            f"Expected goto='interrupt_helper', got {result.goto!r}"
        )
        assert result.update.get("pending_ask_tool_call_id") is not None
        assert result.update.get("pending_ask_outcome") is not None

    async def test_shell_execute_dangerous_blocked_by_ast_gate(self):
        """P1#4: shell_execute with dangerous command is blocked by AST gate, not PE.

        rm -rf / violates the cwd_boundary rule — AST gate returns Denied before
        pe.evaluate is ever called, so FakeRecordingPE.calls stays empty.
        """
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage

        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "ask"  # if PE were called, it would Ask; but it won't be
        fake_ssm = _make_fake_ssm()

        state = _make_state("shell_execute", {"command": "rm -rf /"})
        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        assert isinstance(result, Command)
        # The AST gate blocks this and returns a ToolMessage (not interrupt_helper)
        assert result.goto != "interrupt_helper", (
            "Dangerous shell_execute should be blocked by AST gate, not forwarded to interrupt_helper"
        )
        # PE.evaluate must NOT have been called (AST gate short-circuits before PE)
        assert len(fake_pe.calls) == 0, (
            "PE.evaluate must not be called when AST gate blocks the shell command"
        )

    async def test_allow_error_emits_tool_message_with_error_status(self):
        """AllowError outcome → ToolMessage with status='error', no crash."""
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage

        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "error"
        fake_ssm = _make_fake_ssm()

        state = _make_state("file_write", {"path": "/x"})
        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        assert tool_messages[0].status == "error"

    async def test_multiple_tool_calls_all_processed(self):
        """When AIMessage has 2 tool calls, PE.evaluate is called twice."""
        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "allow"
        fake_ssm = _make_fake_ssm()

        # Two tool calls in the same AIMessage
        state = {
            **_make_state("file_write", {"path": "/a"}),
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "tc1", "name": "file_write", "args": {"path": "/a"}, "type": "tool_call"},
                        {"id": "tc2", "name": "file_write", "args": {"path": "/b"}, "type": "tool_call"},
                    ],
                )
            ],
        }
        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        assert len(fake_pe.calls) == 2, "PE.evaluate should be called once per tool call"

    async def test_already_done_tool_calls_skipped(self):
        """Tool calls in completed_tool_call_prefix are skipped (I-4.1 invariant)."""
        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_ssm = _make_fake_ssm()

        state = {
            **_make_state("file_write", {"path": "/x"}),
            "completed_tool_call_prefix": ["tc1"],  # already done
        }
        config = _make_config(fake_pe, fake_ssm)

        await tool_node_fn(state, config)

        # tc1 was in already_done → skipped → PE not called
        assert len(fake_pe.calls) == 0


# ---------------------------------------------------------------------------
# Task 9.1: _build_tool_call_spec_from_tc module-level helper
# ---------------------------------------------------------------------------

class TestBuildToolCallSpecFromTc:
    def test_builds_correct_spec(self):
        """_build_tool_call_spec_from_tc populates all ToolCallSpec fields."""
        from app.domain.services.graphs.react_graph import _build_tool_call_spec_from_tc
        from app.domain.services.tools.tool_source_resolver import ToolSource

        tc = {"id": "tc99", "name": "file_write", "args": {"path": "/test"}, "type": "tool_call"}
        configurable = {"user_id": "alice", "session_id": "sess1"}
        tool_source = ToolSource(source="native", category="file", canonical_name="file_write")

        spec = _build_tool_call_spec_from_tc(tc, configurable, tool_source)

        assert spec.tool_name == "file_write"
        assert spec.tool_source == "native"
        assert spec.user_id == "alice"
        assert spec.session_id == "sess1"
        assert spec.tool_call_id == "tc99"
        assert spec.tool_args == {"path": "/test"}

    def test_builds_spec_with_risk_assessment(self):
        """_build_tool_call_spec_from_tc populates risk fields from assessment."""
        from app.domain.services.graphs.react_graph import _build_tool_call_spec_from_tc
        from app.domain.services.tools.tool_source_resolver import ToolSource
        from app.domain.services.risk_assessor import RiskAssessor

        tc = {"id": "tc1", "name": "shell_execute", "args": {"command": "rm -rf /"}, "type": "tool_call"}
        configurable = {"user_id": "bob", "session_id": "s2"}
        tool_source = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        assessor = RiskAssessor()
        assessment = assessor.assess("shell_execute", {"command": "rm -rf /"})

        spec = _build_tool_call_spec_from_tc(tc, configurable, tool_source, assessment)

        assert spec.risk_assessment is assessment
        assert spec.arg_digest is not None


# ---------------------------------------------------------------------------
# PE exceptions: SessionModeViolation + PolicyConflict are caught
# ---------------------------------------------------------------------------

class TestPeExceptionHandling:
    async def test_session_mode_violation_emits_error_tool_message(self):
        """SessionModeViolation from PE → AllowError ToolMessage, no crash."""
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage
        from app.domain.services.permission.errors import SessionModeViolation

        class FakeRaisingPE:
            async def evaluate(self, call, ctx):
                raise SessionModeViolation("session terminated")
            async def preflight_resume(self, *a, **kw): pass
            async def commit_resume(self, *a, **kw): pass

        tool_node_fn = _build_tool_node_fn()
        fake_ssm = _make_fake_ssm()
        state = _make_state("file_write", {"path": "/x"})
        config = _make_config(FakeRaisingPE(), fake_ssm)

        result = await tool_node_fn(state, config)

        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        assert tool_messages[0].status == "error"

    async def test_policy_conflict_emits_error_tool_message(self):
        """PolicyConflict from PE → AllowError ToolMessage, no crash."""
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage
        from app.domain.services.permission.errors import PolicyConflict

        class FakeConflictPE:
            async def evaluate(self, call, ctx):
                raise PolicyConflict("arg_digest_mismatch")
            async def preflight_resume(self, *a, **kw): pass
            async def commit_resume(self, *a, **kw): pass

        tool_node_fn = _build_tool_node_fn()
        fake_ssm = _make_fake_ssm()
        state = _make_state("file_write", {"path": "/x"})
        config = _make_config(FakeConflictPE(), fake_ssm)

        result = await tool_node_fn(state, config)

        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        assert tool_messages[0].status == "error"

    async def test_ssm_read_failure_is_fail_closed(self):
        """P1#1: SSM.get_mode_with_revision failure → error ToolMessage, _invoke_wrapper NOT called.

        When SSM raises transiently, tool_node must NOT execute the tool.
        It should surface an [SSM_UNAVAILABLE] error ToolMessage so the model
        can retry, rather than executing with no PE gate (fail-open).
        """
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage
        from unittest.mock import patch

        tool_node_fn = _build_tool_node_fn()

        # SSM that always raises
        failing_ssm = AsyncMock()
        failing_ssm.get_mode_with_revision = AsyncMock(
            side_effect=RuntimeError("DB connection lost")
        )

        fake_pe = FakeRecordingPE()  # should NOT be called
        state = _make_state("file_write", {"path": "/x"})
        config = _make_config(fake_pe, failing_ssm)

        # Patch _invoke_wrapper to detect if it was called (it must NOT be)
        invoke_wrapper_called = []
        original_module = __import__(
            "app.domain.services.graphs.react_graph",
            fromlist=["_invoke_wrapper"],
        )

        async def _spy_invoke_wrapper(*args, **kwargs):
            invoke_wrapper_called.append(True)
            return await original_module._invoke_wrapper(*args, **kwargs)

        with patch(
            "app.domain.services.graphs.react_graph._invoke_wrapper",
            side_effect=_spy_invoke_wrapper,
        ):
            result = await tool_node_fn(state, config)

        # _invoke_wrapper must NOT have been called
        assert not invoke_wrapper_called, (
            "P1#1 FAIL: _invoke_wrapper was called despite SSM read failure "
            "(fail-open behaviour detected — must be fail-closed)"
        )

        # PE.evaluate must NOT have been called either
        assert len(fake_pe.calls) == 0, (
            "P1#1 FAIL: PE.evaluate was called despite SSM read failure"
        )

        # An error ToolMessage must have been emitted
        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, (
            "P1#1 FAIL: no error ToolMessage was emitted on SSM read failure"
        )
        assert tool_messages[0].status == "error"
        assert "SSM_UNAVAILABLE" in tool_messages[0].content or "unavailable" in tool_messages[0].content.lower()


# ---------------------------------------------------------------------------
# No writer slot reads in _pe_dispatch (Phase 8.3 invariant preserved)
# ---------------------------------------------------------------------------

def test_pe_dispatch_does_not_read_writer_slot():
    """_pe_dispatch must not read configurable['approval_state_writer'].

    The writer slot was removed in Phase 8.3; PE owns writes internally.
    This is a string-search guard (Phase 11 adds the full AST scan).
    """
    from pathlib import Path
    src = (
        Path(__file__).parents[5]
        / "api/app/domain/services/graphs/react_graph.py"
    ).read_text()
    # Extract only the _pe_dispatch function body to check
    start = src.find("async def _pe_dispatch(")
    end = src.find("\n    # ======================================================================\n"
                   "    # End PE-0 Phase 9: _pe_dispatch", start)
    if start == -1:
        pytest.fail("_pe_dispatch function not found in react_graph.py")
    pe_dispatch_src = src[start:end] if end != -1 else src[start:]

    assert 'configurable.get("approval_state_writer")' not in pe_dispatch_src, (
        "_pe_dispatch must not read writer from configurable (PE owns writes)"
    )
    assert "configurable.get('approval_state_writer')" not in pe_dispatch_src, (
        "_pe_dispatch must not read writer from configurable (single-quote form)"
    )


# ---------------------------------------------------------------------------
# P1#3: Skill source tools bypass PE and go to direct execution
# ---------------------------------------------------------------------------

class TestPeSkillSourceRouting:
    """P1#3: non-native tool calls (skill/mcp/a2a) must NOT be evaluated by PE.

    They should be executed directly (equivalent to AllowSuccess from PE), so
    the R3 Skill Stage P and legacy mcp/a2a approval pipelines remain reachable.
    """

    async def test_skill_tool_bypasses_pe(self):
        """A skill tool call is executed directly; PE.evaluate is never called."""
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.types import Command
        from unittest.mock import AsyncMock, MagicMock

        @lc_tool
        async def my_skill_tool(x: str) -> str:
            """A skill-type tool."""
            return f"skill result: {x}"

        # Patch resolve_tool_source so my_skill_tool reports source=skill
        import app.domain.services.graphs.react_graph as rg_module
        from app.domain.services.tools.tool_source_resolver import ToolSource

        original_resolve = rg_module.resolve_tool_source

        def patched_resolve(name: str) -> ToolSource:
            if name == "my_skill_tool":
                return ToolSource(source="skill", category="skill", canonical_name=name)
            return original_resolve(name)

        stub_llm = AsyncMock()
        stub_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        stub_llm.bind_tools = MagicMock(return_value=stub_llm)

        graph = build_react_graph(stub_llm, [my_skill_tool])
        tool_node_fn = graph.nodes["tool_node"].bound.afunc

        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "ask"  # would interrupt if called, but must not be
        fake_ssm = _make_fake_ssm()

        state = _make_state("my_skill_tool", {"x": "hello"})
        config = _make_config(fake_pe, fake_ssm)

        import unittest.mock as _mock
        with _mock.patch.object(rg_module, "resolve_tool_source", patched_resolve):
            result = await tool_node_fn(state, config)

        # PE must NOT have been called for the skill tool
        assert len(fake_pe.calls) == 0, (
            "PE.evaluate must NOT be called for skill-source tools (P1#3)"
        )
        # The tool should have been executed directly → produces a ToolMessage
        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, "Skill tool must produce a ToolMessage"
        # Must go to pre_llm_node (not interrupt_helper) since directly executed
        assert result.goto == "pre_llm_node", (
            f"Expected pre_llm_node for directly-executed skill, got {result.goto!r}"
        )


# ---------------------------------------------------------------------------
# P1#5: build_permission_engine respects smart_approve_enabled flag
# ---------------------------------------------------------------------------

class TestBuildPermissionEngineSmartApproveGate:
    """P1#5: SmartApproveProvider only registered when smart_approve_enabled=True."""

    def test_smart_approve_disabled_means_empty_escalation_registry(self):
        """When smart_approve_enabled=False, escalation_registry is empty."""
        from app.application.composition.graph_assembly import build_permission_engine
        from unittest.mock import MagicMock, AsyncMock

        fake_writer = MagicMock()
        fake_reader = MagicMock()
        fake_queue = MagicMock()
        fake_ssm = MagicMock()
        fake_uow_factory = MagicMock()
        fake_llm = MagicMock()  # has a summary_llm but SA disabled

        pe = build_permission_engine(
            uow_factory=fake_uow_factory,
            writer=fake_writer,
            queue=fake_queue,
            session_machine=fake_ssm,
            reader=fake_reader,
            summary_llm=fake_llm,
            smart_approve_enabled=False,  # <-- gate
        )

        # DefaultPermissionEngine stores the registry
        from app.domain.services.permission.default_engine import DefaultPermissionEngine
        assert isinstance(pe, DefaultPermissionEngine)
        assert len(pe._escalation_registry) == 0, (
            "escalation_registry must be empty when smart_approve_enabled=False"
        )

    def test_smart_approve_enabled_with_llm_registers_provider(self):
        """When smart_approve_enabled=True and summary_llm is set, provider is registered."""
        from app.application.composition.graph_assembly import build_permission_engine
        from unittest.mock import MagicMock

        fake_llm = MagicMock()
        pe = build_permission_engine(
            uow_factory=MagicMock(),
            writer=MagicMock(),
            queue=MagicMock(),
            session_machine=MagicMock(),
            reader=MagicMock(),
            summary_llm=fake_llm,
            smart_approve_enabled=True,
        )

        from app.domain.services.permission.default_engine import DefaultPermissionEngine
        assert isinstance(pe, DefaultPermissionEngine)
        assert "smart_approve" in pe._escalation_registry, (
            "SmartApproveProvider must be registered when smart_approve_enabled=True with LLM"
        )

    def test_smart_approve_enabled_without_llm_stays_empty(self):
        """When smart_approve_enabled=True but summary_llm is None, registry is empty."""
        from app.application.composition.graph_assembly import build_permission_engine
        from unittest.mock import MagicMock

        pe = build_permission_engine(
            uow_factory=MagicMock(),
            writer=MagicMock(),
            queue=MagicMock(),
            session_machine=MagicMock(),
            reader=MagicMock(),
            summary_llm=None,  # no LLM
            smart_approve_enabled=True,
        )

        from app.domain.services.permission.default_engine import DefaultPermissionEngine
        assert isinstance(pe, DefaultPermissionEngine)
        assert len(pe._escalation_registry) == 0, (
            "escalation_registry must be empty when summary_llm is None"
        )


# ---------------------------------------------------------------------------
# P2#2: LOW-risk tools must have non-empty arg_digest in ToolCallSpec
# ---------------------------------------------------------------------------

class TestPeDispatchLowRiskArgDigest:
    """P2#2: When user policy is ASK for a LOW-metadata tool (e.g. file_read /
    browser_click), the ToolCallSpec built by _pe_dispatch must include a
    non-empty arg_digest so that session/always grants are correctly scoped
    to the specific arguments rather than covering the entire tool.
    """

    async def test_pe_dispatch_low_risk_tool_with_user_ask_policy_has_arg_digest(self):
        """file_read is LOW-risk metadata; ToolCallSpec must still have arg_digest.

        We intercept the ToolCallSpec passed to FakeRecordingPE.evaluate and
        assert that arg_digest is populated (not empty string / None).
        """
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph

        @lc_tool
        async def file_read(path: str) -> str:
            """Read file contents."""
            return "contents"

        stub_llm = MagicMock()
        stub_llm.ainvoke = AsyncMock(return_value=MagicMock(content="done"))
        stub_llm.bind_tools = MagicMock(return_value=stub_llm)

        graph = build_react_graph(stub_llm, [file_read])
        tool_node_fn = graph.nodes["tool_node"].bound.afunc

        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "allow"
        fake_ssm = _make_fake_ssm()

        state = _make_state("file_read", {"path": "/home/user/notes.txt"})
        config = _make_config(fake_pe, fake_ssm)

        await tool_node_fn(state, config)

        assert len(fake_pe.calls) >= 1, "PE.evaluate must be called for file_read"
        spec, _ctx = fake_pe.calls[0]
        assert spec.arg_digest is not None and spec.arg_digest != "", (
            "ToolCallSpec.arg_digest must be populated for LOW-risk tools (P2#2): "
            f"got {spec.arg_digest!r}"
        )


# ---------------------------------------------------------------------------
# Codex round-20 P2#1: approved_tool_call_ids bypasses pe.evaluate
# ---------------------------------------------------------------------------

class TestPeDispatchSkipsEvaluateForApprovedIds:
    """Codex round-20 P2#1: When a tool_call_id appears in approved_tool_call_ids
    (legacy interrupt_helper hot-switch fallback), _pe_dispatch must NOT call
    pe.evaluate.  Instead it should invoke the wrapper directly.

    Scenario: user policy = ASK (FakeRecordingPE.next_outcome="ask"). Without
    the fix, pe.evaluate would be called and return Asked, re-prompting the user
    even though they already approved via the legacy path.  With the fix,
    pe.evaluate is never called and the tool executes successfully.
    """

    async def test_pe_dispatch_skips_pe_evaluate_for_approved_tool_call_ids(self):
        """approved_tool_call_ids hit → pe.evaluate not called, wrapper executes."""
        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        # Policy set to ASK: without the fix, PE would re-prompt the user
        fake_pe.next_outcome = "ask"
        fake_ssm = _make_fake_ssm()

        call_id = "tc_approved_1"
        state = _make_state("file_write", {"path": "/approved.txt", "content": "ok"}, call_id=call_id)
        # Mark this tool call as already approved by legacy interrupt_helper
        state["approved_tool_call_ids"] = [call_id]

        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        # PE.evaluate must NOT have been called (user already approved via legacy path)
        assert len(fake_pe.calls) == 0, (
            f"Codex round-20 P2#1 FAIL: pe.evaluate was called {len(fake_pe.calls)} time(s) "
            f"even though call_id={call_id!r} is in approved_tool_call_ids. "
            "The fix should bypass pe.evaluate for pre-approved tool calls."
        )

        # The result should contain a ToolMessage (wrapper executed)
        from langchain_core.messages import ToolMessage
        cmd_update = result if isinstance(result, dict) else (result.update if hasattr(result, "update") else {})
        messages = (
            result.get("messages", [])
            if isinstance(result, dict)
            else getattr(result, "update", {}).get("messages", [])
        )
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, (
            "Codex round-20 P2#1 FAIL: expected at least one ToolMessage in result "
            f"(wrapper should have executed), got messages={messages!r}"
        )
        assert tool_messages[0].tool_call_id == call_id, (
            f"ToolMessage.tool_call_id should be {call_id!r}, "
            f"got {tool_messages[0].tool_call_id!r}"
        )

    async def test_pe_dispatch_non_approved_still_calls_pe_evaluate(self):
        """Non-approved tool calls still go through pe.evaluate (regression guard)."""
        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "allow"
        fake_ssm = _make_fake_ssm()

        call_id = "tc_not_approved"
        state = _make_state("file_write", {"path": "/normal.txt", "content": "hi"}, call_id=call_id)
        # Different ID in approved list — should NOT bypass PE for our call
        state["approved_tool_call_ids"] = ["tc_some_other_id"]

        config = _make_config(fake_pe, fake_ssm)

        await tool_node_fn(state, config)

        assert len(fake_pe.calls) == 1, (
            f"Codex round-20 P2#1 regression: pe.evaluate must still be called for "
            f"non-approved tool calls, got {len(fake_pe.calls)} calls"
        )


# ---------------------------------------------------------------------------
# P2#1 (Round-25): approved_tool_call_ids path must check session mode
# ---------------------------------------------------------------------------


class TestPeDispatchApprovedIdModeCheck:
    """P2#1 (round-25): legacy-approved replay must re-check session mode for parity
    with the pe_resume_outcomes replay path (round-23 P1#1).

    Scenario:
    - User approved a tool call while session was RUNNING.
    - Between approval and tool_node replay, session entered TAKEOVER/FINISHING.
    - approved_tool_call_ids contains the tool_call_id (legacy approval captured).
    - Without the fix, _pe_dispatch would invoke the wrapper in a non-live session.
    - With the fix, mode is re-checked and a Denied is surfaced instead.
    """

    async def test_pe_dispatch_approved_id_rejects_when_session_in_takeover(self):
        """approved_tool_call_ids + TAKEOVER mode → Denied, wrapper NOT invoked."""
        from langchain_core.messages import ToolMessage

        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        # SSM returns TAKEOVER — session left live mode after approval
        fake_ssm = _make_fake_ssm(mode=SessionStatus.TAKEOVER)

        call_id = "tc_approved_takeover"
        state = _make_state(
            "file_write",
            {"path": "/bad.txt", "content": "should not execute"},
            call_id=call_id,
        )
        state["approved_tool_call_ids"] = [call_id]

        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        # pe.evaluate must NOT be called (approved_tool_call_ids bypass skips it)
        assert len(fake_pe.calls) == 0, (
            "P2#1 round-25 FAIL: pe.evaluate should not be called for pre-approved IDs"
        )

        # The result must contain a ToolMessage with status="error" (Denied surfaced)
        messages = (
            result.get("messages", [])
            if isinstance(result, dict)
            else getattr(result, "update", {}).get("messages", [])
        )
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, (
            "P2#1 round-25 FAIL: expected a ToolMessage in result when mode is TAKEOVER"
        )
        # Denied variant → status must be 'error'
        assert tool_messages[0].status == "error", (
            f"P2#1 round-25 FAIL: ToolMessage status must be 'error' for Denied, "
            f"got {tool_messages[0].status!r}"
        )
        # The denial reason must reference mode change
        content = tool_messages[0].content or ""
        assert "session_mode_changed_before_replay" in content or "LEGACY_REPLAY_DENIED" in content or "TAKEOVER" in content, (
            f"P2#1 round-25 FAIL: ToolMessage content must mention mode change, got {content!r}"
        )

    async def test_pe_dispatch_approved_id_executes_when_session_running(self):
        """approved_tool_call_ids + RUNNING mode → wrapper invoked (regression guard)."""
        from langchain_core.messages import ToolMessage

        tool_node_fn = _build_tool_node_fn()
        fake_pe = FakeRecordingPE()
        fake_ssm = _make_fake_ssm(mode=SessionStatus.RUNNING)  # live mode — OK

        call_id = "tc_approved_running"
        state = _make_state(
            "file_write",
            {"path": "/ok.txt", "content": "execute me"},
            call_id=call_id,
        )
        state["approved_tool_call_ids"] = [call_id]

        config = _make_config(fake_pe, fake_ssm)

        result = await tool_node_fn(state, config)

        # pe.evaluate must NOT be called (approved_tool_call_ids bypass)
        assert len(fake_pe.calls) == 0

        # Wrapper must have executed — ToolMessage present
        messages = (
            result.get("messages", [])
            if isinstance(result, dict)
            else getattr(result, "update", {}).get("messages", [])
        )
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, (
            "P2#1 round-25 regression FAIL: wrapper must execute when mode=RUNNING"
        )
        # Success path — status should NOT be 'error'
        assert tool_messages[0].status != "error", (
            f"P2#1 round-25 regression FAIL: ToolMessage should not be error for "
            f"live-mode approved replay, got status={tool_messages[0].status!r}"
        )


# ---------------------------------------------------------------------------
# Round 34 P1#2: pre-wrapper live-mode recheck on AllowSuccess / Passthrough
# ---------------------------------------------------------------------------


class TestPeDispatchAllowSuccessLiveModeRecheck:
    """P1#2 (round 34): After ``pe.evaluate()`` returns AllowSuccess/Passthrough,
    _pe_dispatch must re-read the session mode before invoking the tool wrapper.

    ``pe.evaluate()`` awaits DB / SSM / policy lookups internally, so the
    ``mode`` captured at the top of the dispatch loop is stale by the time
    we reach the wrapper branch. If the session flipped to TAKEOVER (or a
    terminal state) during ``evaluate()`` we must NOT execute the wrapper —
    parity with the replay path (round 23 P1#1) and the legacy
    ``approved_tool_call_ids`` path (round 25 P2#1).
    """

    async def test_pe_dispatch_allow_success_rejects_when_session_in_takeover(self):
        """SSM flips RUNNING→TAKEOVER between ctx-build and post-evaluate recheck.

        Setup:
          1st SSM read (line ~1389, build EvaluationContext) → RUNNING
          pe.evaluate() returns AllowSuccess
          2nd SSM read (line ~1681, pre-wrapper recheck) → TAKEOVER

        Expected: wrapper NOT invoked, Denied ToolMessage emitted with
        reason code ``session_mode_changed_before_invoke``.
        """
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage
        from unittest.mock import patch

        tool_node_fn = _build_tool_node_fn()

        # SSM returns RUNNING on the first call (ctx build) and TAKEOVER on
        # the second call (the new live-mode recheck before _invoke_wrapper).
        flipping_ssm = AsyncMock()
        flipping_ssm.get_mode_with_revision = AsyncMock(
            side_effect=[
                (SessionStatus.RUNNING, 1),   # ctx build at line ~1389
                (SessionStatus.TAKEOVER, 2),  # pre-wrapper recheck at line ~1681
            ]
        )

        # PE returns AllowSuccess — without the round-34 fix this would
        # directly invoke the wrapper.
        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "allow"

        call_id = "tc_live_mode_drift"
        state = _make_state(
            "file_write",
            {"path": "/should_not_execute.txt", "content": "blocked"},
            call_id=call_id,
        )
        # Crucially NOT in approved_tool_call_ids — we want pe.evaluate to run.
        config = _make_config(fake_pe, flipping_ssm)

        # Spy on _invoke_wrapper to assert it was never called.
        invoke_wrapper_called: list[bool] = []
        original_module = __import__(
            "app.domain.services.graphs.react_graph",
            fromlist=["_invoke_wrapper"],
        )

        async def _spy_invoke_wrapper(*args, **kwargs):
            invoke_wrapper_called.append(True)
            return await original_module._invoke_wrapper(*args, **kwargs)

        with patch(
            "app.domain.services.graphs.react_graph._invoke_wrapper",
            side_effect=_spy_invoke_wrapper,
        ):
            result = await tool_node_fn(state, config)

        # PE.evaluate must have been called (this is the AllowSuccess path).
        assert len(fake_pe.calls) == 1, (
            "P1#2 round-34 FAIL: pe.evaluate should run once (path under test "
            f"is the post-evaluate branch); calls={fake_pe.calls!r}"
        )

        # Wrapper must NOT have been invoked because mode flipped to TAKEOVER.
        assert not invoke_wrapper_called, (
            "P1#2 round-34 FAIL: _invoke_wrapper was called even though SSM "
            "reported TAKEOVER on the post-evaluate recheck — the wrapper must "
            "be skipped when the session left live mode during evaluate()."
        )

        # Both SSM reads must have been awaited (proves the new recheck ran).
        assert flipping_ssm.get_mode_with_revision.await_count == 2, (
            "P1#2 round-34 FAIL: expected exactly 2 SSM reads "
            "(ctx build + pre-wrapper recheck); got "
            f"{flipping_ssm.get_mode_with_revision.await_count}"
        )

        # A Denied ToolMessage must have been emitted.
        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, (
            "P1#2 round-34 FAIL: expected a Denied ToolMessage when session "
            "flips to TAKEOVER after pe.evaluate."
        )
        # Denied → status="error"
        assert tool_messages[0].status == "error", (
            "P1#2 round-34 FAIL: ToolMessage status must be 'error' for the "
            f"Denied outcome; got {tool_messages[0].status!r}"
        )
        content = tool_messages[0].content or ""
        assert (
            "session_mode_changed_before_invoke" in content
            or "MODE_DENIED" in content
            or "TAKEOVER" in content.upper()
        ), (
            "P1#2 round-34 FAIL: ToolMessage content must mention the mode "
            f"change reason; got {content!r}"
        )

    async def test_pe_dispatch_allow_success_executes_when_mode_stays_running(self):
        """Regression guard: when SSM keeps reporting RUNNING on both reads,
        AllowSuccess must still invoke the wrapper exactly once."""
        from langchain_core.messages import ToolMessage

        tool_node_fn = _build_tool_node_fn()

        stable_ssm = AsyncMock()
        stable_ssm.get_mode_with_revision = AsyncMock(
            side_effect=[
                (SessionStatus.RUNNING, 1),  # ctx build
                (SessionStatus.RUNNING, 1),  # pre-wrapper recheck — still live
            ]
        )

        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "allow"

        call_id = "tc_live_mode_stable"
        state = _make_state(
            "file_write",
            {"path": "/ok.txt", "content": "execute"},
            call_id=call_id,
        )
        config = _make_config(fake_pe, stable_ssm)

        result = await tool_node_fn(state, config)

        assert len(fake_pe.calls) == 1
        # Both SSM reads expected (build ctx + pre-wrapper recheck).
        assert stable_ssm.get_mode_with_revision.await_count == 2

        messages = (
            result.update.get("messages", [])
            if hasattr(result, "update")
            else result.get("messages", [])
        )
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        # Wrapper executed → status must NOT be 'error'.
        assert tool_messages[0].status != "error", (
            "P1#2 round-34 regression FAIL: wrapper should execute and emit a "
            "success ToolMessage when mode stays RUNNING across both SSM reads"
        )

    async def test_pe_dispatch_allow_success_fail_closed_when_recheck_ssm_raises(self):
        """If SSM raises on the post-evaluate recheck, fail closed — do NOT
        invoke the wrapper. Surface an [SSM_UNAVAILABLE] error ToolMessage."""
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage
        from unittest.mock import patch

        tool_node_fn = _build_tool_node_fn()

        # 1st read OK, 2nd raises.
        partial_ssm = AsyncMock()
        partial_ssm.get_mode_with_revision = AsyncMock(
            side_effect=[
                (SessionStatus.RUNNING, 1),
                RuntimeError("DB connection lost on recheck"),
            ]
        )

        fake_pe = FakeRecordingPE()
        fake_pe.next_outcome = "allow"

        state = _make_state("file_write", {"path": "/x"})
        config = _make_config(fake_pe, partial_ssm)

        invoke_wrapper_called: list[bool] = []
        original_module = __import__(
            "app.domain.services.graphs.react_graph",
            fromlist=["_invoke_wrapper"],
        )

        async def _spy_invoke_wrapper(*args, **kwargs):
            invoke_wrapper_called.append(True)
            return await original_module._invoke_wrapper(*args, **kwargs)

        with patch(
            "app.domain.services.graphs.react_graph._invoke_wrapper",
            side_effect=_spy_invoke_wrapper,
        ):
            result = await tool_node_fn(state, config)

        # Wrapper must NOT have been called (fail-closed).
        assert not invoke_wrapper_called, (
            "P1#2 round-34 FAIL: _invoke_wrapper called despite SSM recheck "
            "raising — must fail-closed."
        )
        # pe.evaluate ran once (the failure is on the post-evaluate recheck).
        assert len(fake_pe.calls) == 1

        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        assert tool_messages[0].status == "error"
        content = tool_messages[0].content or ""
        assert (
            "SSM_UNAVAILABLE" in content
            or "unavailable" in content.lower()
        ), (
            "P1#2 round-34 FAIL: ToolMessage must indicate SSM unavailability; "
            f"got {content!r}"
        )
