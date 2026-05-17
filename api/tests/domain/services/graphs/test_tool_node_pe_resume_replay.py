"""Phase 10, Task 10.3: tool_node reads pe_resume_outcomes on replay (INV-5 path B).

When pe_resume_outcomes[tool_call_id] is present in state, tool_node must:
- Skip pe.evaluate entirely (replay path B)
- For AllowSuccess → invoke the tool wrapper
- For Denied → skip wrapper, emit deny ToolMessage

Uses the same approach as test_react_graph_pe_dispatch.py: build a minimal
react_graph and extract the tool_node closure.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.domain.models.tool_result import (
    AllowSuccess,
    Denied,
    DecisionReason,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TOOL_WAS_CALLED = False


def _build_tool_node_fn():
    """Build a minimal react_graph and return the tool_node closure.

    Same pattern as test_react_graph_pe_dispatch.py.
    """
    global _TOOL_WAS_CALLED
    from langchain_core.tools import tool as lc_tool
    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        global _TOOL_WAS_CALLED
        _TOOL_WAS_CALLED = True
        return f"wrote {path}"

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"success":true,"result":"done","attachments":[]}')
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    graph = build_react_graph(stub_llm, [file_write])
    return graph.nodes["tool_node"].bound.afunc


def _make_state(
    tool_name: str,
    tool_args: dict,
    call_id: str = "tc1",
    pe_resume_outcomes: dict | None = None,
) -> dict:
    """Build a minimal ReactGraphState dict."""
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
        "pe_resume_outcomes": pe_resume_outcomes or {},
        "pending_ask_outcome": None,
        "pending_ask_tool_call_id": None,
        "pending_ask_artifact": None,
        "pending_ask_tool_args": None,
    }


def _make_config(fake_pe, fake_ssm, *, user_id="u", session_id="s"):
    from app.domain.models.session import SessionStatus
    return {
        "configurable": {
            "permission_engine": fake_pe,
            "session_state_machine": fake_ssm,
            "permission_engine_native_enabled": True,
            "user_id": user_id,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }


def _make_fake_ssm(revision: int = 1):
    from app.domain.models.session import SessionStatus
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(SessionStatus.RUNNING, revision))
    return ssm


# ---------------------------------------------------------------------------
# Task 10.3: replay path — pe_resume_outcomes short-circuits pe.evaluate
# ---------------------------------------------------------------------------

class TestToolNodePeResumeReplay:
    async def test_replay_allow_skips_evaluate_and_invokes_wrapper(self):
        """AllowSuccess in pe_resume_outcomes → tool wrapper is invoked, pe.evaluate skipped."""
        global _TOOL_WAS_CALLED
        _TOOL_WAS_CALLED = False

        tool_node_fn = _build_tool_node_fn()
        fake_pe = AsyncMock()
        fake_ssm = _make_fake_ssm()

        # Simulate a pre-baked AllowSuccess from interrupt_helper
        allow_outcome = AllowSuccess(content="ok", data={"via": "user_click"})
        pe_resume_map = {"tc1": allow_outcome.model_dump(mode="json")}

        state = _make_state(
            "file_write",
            {"path": "/x", "content": "hello"},
            call_id="tc1",
            pe_resume_outcomes=pe_resume_map,
        )
        config = _make_config(fake_pe, fake_ssm)

        cmd = await tool_node_fn(state, config)

        # pe.evaluate must NOT have been called — replay path B short-circuits
        fake_pe.evaluate.assert_not_called()

        # The tool was actually invoked (wrapper ran)
        assert _TOOL_WAS_CALLED is True

        # pe_resume_outcomes entry for tc1 should be cleared in the update
        updated_outcomes = cmd.update.get("pe_resume_outcomes", {})
        assert "tc1" not in updated_outcomes

    async def test_replay_denied_skips_evaluate_and_skips_wrapper(self):
        """Denied in pe_resume_outcomes → pe.evaluate skipped, wrapper NOT called."""
        global _TOOL_WAS_CALLED
        _TOOL_WAS_CALLED = False

        tool_node_fn = _build_tool_node_fn()
        fake_pe = AsyncMock()
        fake_ssm = _make_fake_ssm()

        deny_outcome = Denied(
            content="denied by PE",
            reason=DecisionReason(
                type="approval_policy",
                code="deny:session",
                message="user_denied",
            ),
        )
        pe_resume_map = {"tc1": deny_outcome.model_dump(mode="json")}

        state = _make_state(
            "file_write",
            {"path": "/x"},
            call_id="tc1",
            pe_resume_outcomes=pe_resume_map,
        )
        config = _make_config(fake_pe, fake_ssm)

        cmd = await tool_node_fn(state, config)

        # pe.evaluate must NOT have been called
        fake_pe.evaluate.assert_not_called()

        # Wrapper must NOT have been invoked
        assert _TOOL_WAS_CALLED is False

        # A ToolMessage should be emitted (deny surfaced to agent loop)
        messages = cmd.update.get("messages", [])
        from langchain_core.messages import ToolMessage
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, "Denied outcome must produce a ToolMessage"

        # pe_resume_outcomes entry for tc1 should be cleared
        updated_outcomes = cmd.update.get("pe_resume_outcomes", {})
        assert "tc1" not in updated_outcomes

    async def test_no_replay_calls_evaluate_normally(self):
        """When pe_resume_outcomes is empty, tool_node calls pe.evaluate as usual."""
        global _TOOL_WAS_CALLED
        _TOOL_WAS_CALLED = False

        tool_node_fn = _build_tool_node_fn()
        fake_pe = AsyncMock()
        fake_pe.evaluate = AsyncMock(
            return_value=AllowSuccess(content="auto", data={})
        )
        fake_ssm = _make_fake_ssm()

        # Empty pe_resume_outcomes — normal path
        state = _make_state(
            "file_write",
            {"path": "/x"},
            call_id="tc1",
            pe_resume_outcomes={},
        )
        config = _make_config(fake_pe, fake_ssm)

        await tool_node_fn(state, config)

        # pe.evaluate MUST have been called
        fake_pe.evaluate.assert_awaited_once()


# ---------------------------------------------------------------------------
# P2#3: batch with replay + Asked — early-return Command must clear consumed
# pe_resume_outcomes entries so stale entries do not persist into state.
# ---------------------------------------------------------------------------

class TestBatchReplayPlusAskedClearsConsumed:
    """When a batch has:
      - tool_call 'tc1': pe_resume_outcomes has an AllowSuccess (replay path)
      - tool_call 'tc2': pe.evaluate returns Asked (early return to interrupt_helper)

    The early-return Command must include pe_resume_outcomes WITHOUT the 'tc1'
    entry that was already consumed in the replay step.

    Without the P2#3 fix, 'tc1' stays in state.pe_resume_outcomes. On the next
    _pe_dispatch run (after interrupt resume), the same tool_call_id could match
    the stale entry and skip pe.evaluate altogether — allowing a tool call to
    execute without re-checking permissions.
    """

    async def test_batch_replay_then_asked_clears_consumed_in_early_return(self):
        """P2#3: asked early-return includes pe_resume_outcomes cleanup of replayed tc1."""
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph
        from app.domain.models.tool_result import (
            AllowSuccess,
            Asked,
            DecisionReason,
        )

        @lc_tool
        async def file_write(path: str, content: str = "") -> str:
            """Write to a file."""
            return f"wrote {path}"

        @lc_tool
        async def shell_execute(command: str) -> str:
            """Run a shell command."""
            return "ok"

        stub_llm = AsyncMock()
        from langchain_core.messages import AIMessage
        stub_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content='{"success":true,"result":"done","attachments":[]}')
        )
        stub_llm.bind_tools = MagicMock(return_value=stub_llm)
        graph = build_react_graph(stub_llm, [file_write, shell_execute])
        tool_node_fn = graph.nodes["tool_node"].bound.afunc

        # tc1: AllowSuccess already in pe_resume_outcomes (replay path)
        # tc2: pe.evaluate returns Asked (triggers early return to interrupt_helper)
        tc1_outcome = AllowSuccess(content="replay-ok", data={"via": "user_click"})
        tc2_asked = Asked(
            content="waiting for user",
            reason=DecisionReason(type="risk_enforce", code="medium", message="high risk"),
        )

        fake_pe = AsyncMock()
        # tc2 is the only call that reaches pe.evaluate
        fake_pe.evaluate = AsyncMock(return_value=tc2_asked)
        fake_ssm = _make_fake_ssm()

        # State: two tool calls in the AIMessage, tc1 in pe_resume_outcomes
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "tc1", "name": "file_write", "args": {"path": "/a", "content": "x"}, "type": "tool_call"},
                        {"id": "tc2", "name": "file_write", "args": {"path": "/b", "content": "y"}, "type": "tool_call"},
                    ],
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
            "pe_resume_outcomes": {"tc1": tc1_outcome.model_dump(mode="json")},
            "pending_ask_outcome": None,
            "pending_ask_tool_call_id": None,
            "pending_ask_artifact": None,
            "pending_ask_tool_args": None,
        }
        from app.domain.models.session import SessionStatus
        config = {
            "configurable": {
                "permission_engine": fake_pe,
                "session_state_machine": fake_ssm,
                "permission_engine_native_enabled": True,
                "user_id": "u",
                "session_id": "s",
                "thread_id": "s",
            }
        }

        cmd = await tool_node_fn(state, config)

        # Must be routed to interrupt_helper (Asked outcome for tc2)
        assert cmd.goto == "interrupt_helper", (
            f"Expected goto='interrupt_helper' but got {cmd.goto!r} — "
            "tc2 Asked should trigger interrupt path (P2#3)"
        )

        # The early-return Command update must include pe_resume_outcomes WITHOUT tc1
        # (the entry was consumed during the replay step before tc2 was evaluated).
        update_outcomes = cmd.update.get("pe_resume_outcomes")
        assert update_outcomes is not None, (
            "pe_resume_outcomes key missing from Asked early-return update (P2#3 fix not applied)"
        )
        assert "tc1" not in update_outcomes, (
            f"Stale pe_resume_outcomes[tc1] still present after P2#3 fix — "
            f"consumed entry must be cleared in early-return Command: {update_outcomes}"
        )


# ---------------------------------------------------------------------------
# P1 (round-23): replay must re-check session mode before invoking wrapper
# ---------------------------------------------------------------------------

class TestReplayRejectsWhenSessionInTakeover:
    """When pe_resume_outcomes has a cached AllowSuccess but SSM returns TAKEOVER,
    the replay path must NOT invoke the tool wrapper — it should convert the
    cached outcome to Denied and emit a deny ToolMessage instead.
    """

    async def test_pe_dispatch_replay_rejects_outcome_when_session_in_takeover(self):
        """AllowSuccess cached + SSM returns TAKEOVER → wrapper NOT called, Denied surfaced."""
        global _TOOL_WAS_CALLED
        _TOOL_WAS_CALLED = False

        tool_node_fn = _build_tool_node_fn()
        fake_pe = AsyncMock()

        # SSM returns TAKEOVER — session left live mode after the user approved
        from app.domain.models.session import SessionStatus
        fake_ssm = AsyncMock()
        fake_ssm.get_mode_with_revision = AsyncMock(
            return_value=(SessionStatus.TAKEOVER, 2)
        )

        # Cached AllowSuccess (what the user approved in RUNNING mode)
        allow_outcome = AllowSuccess(content="ok", data={"via": "user_click"})
        pe_resume_map = {"tc1": allow_outcome.model_dump(mode="json")}

        state = _make_state(
            "file_write",
            {"path": "/x", "content": "hello"},
            call_id="tc1",
            pe_resume_outcomes=pe_resume_map,
        )
        config = _make_config(fake_pe, fake_ssm)

        cmd = await tool_node_fn(state, config)

        # pe.evaluate must NOT have been called (replay path short-circuits it)
        fake_pe.evaluate.assert_not_called()

        # The tool wrapper must NOT have been invoked — TAKEOVER should block execution
        assert _TOOL_WAS_CALLED is False, (
            "Tool wrapper was invoked despite session being in TAKEOVER mode at replay time"
        )

        # A ToolMessage must be emitted (deny surfaced to agent loop)
        from langchain_core.messages import ToolMessage
        messages = cmd.update.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, (
            "Expected a deny ToolMessage when session is TAKEOVER at replay time"
        )

        # The consumed entry must still be cleared from pe_resume_outcomes
        updated_outcomes = cmd.update.get("pe_resume_outcomes", {})
        assert "tc1" not in updated_outcomes, (
            "pe_resume_outcomes entry for tc1 should be cleared even when denied at replay"
        )

    async def test_pe_dispatch_replay_allows_when_session_running(self):
        """AllowSuccess cached + SSM returns RUNNING → wrapper IS called normally."""
        global _TOOL_WAS_CALLED
        _TOOL_WAS_CALLED = False

        tool_node_fn = _build_tool_node_fn()
        fake_pe = AsyncMock()
        fake_ssm = _make_fake_ssm()  # returns RUNNING

        allow_outcome = AllowSuccess(content="ok", data={"via": "user_click"})
        pe_resume_map = {"tc1": allow_outcome.model_dump(mode="json")}

        state = _make_state(
            "file_write",
            {"path": "/x", "content": "hello"},
            call_id="tc1",
            pe_resume_outcomes=pe_resume_map,
        )
        config = _make_config(fake_pe, fake_ssm)

        await tool_node_fn(state, config)

        # Tool wrapper should run in RUNNING mode
        assert _TOOL_WAS_CALLED is True, (
            "Tool wrapper should be invoked when session is RUNNING at replay time"
        )
