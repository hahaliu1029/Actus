"""R2 PR-B Commit 1 Day-4 hard gate tests: tool_node split exactly-once.

These 4 tests drive the real LangGraph graph via ``InMemorySaver``
checkpointer, not mocks. They prove the prefix-closed invariant holds
under LangGraph's node-replay semantics.

- **Test A**: ``test_prefix_closed_side_effect_exactly_once`` — I-4.1
  write_tool_a must execute exactly once across interrupt + resume.
- **Test B**: ``test_approve_resume_path`` — I-4.2 approve. Interrupt
  surfaces risk_skill_b; resume with approve; risk_skill_b executes via
  ``approved_tool_call_ids`` bypass; write_tool_c executes; spy_a stays
  at 1 (replay doesn't re-run).
- **Test C**: ``test_deny_resume_path`` — I-4.2 deny. Interrupt surfaces
  risk_skill_b; resume with deny; ``interrupt_helper`` synthesizes a
  ``Denied`` ToolMessage; write_tool_c executes; risk_skill_b body
  NEVER runs.
- **Test D**: ``test_cascading_multi_asked`` — I-4.1.1. Two cascading
  risk tools: B and C. Resume approve on B → tool_node replays, B runs,
  C hits interrupt. Resume approve on C → C runs. All spies land at
  exactly 1.

Project convention — no pytest-asyncio. Async driver wraps coroutines
in ``_run()`` which delegates to ``asyncio.run()`` (mirrors
``tests/domain/services/test_approval_cache.py``).
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.domain.services.risk_assessor import (
    RiskAssessment,
    RiskAssessor,
    RiskLevel,
)
from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)


def _run(coro):
    return asyncio.run(coro)


class _Spy:
    """Records each call to a tool. Used to verify exactly-once semantics."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return f"{self.name} result"


def _stubbed_assess(
    self: RiskAssessor, tool_name: str, tool_args: dict[str, Any]
) -> RiskAssessment:
    """Deterministic stub for RiskAssessor.assess.

    Returns HIGH risk for any tool_name starting with ``risk_`` so the
    existing risk-level gate in ``tool_node`` routes to
    ``interrupt_helper``. Everything else is NONE so write_tool_a /
    write_tool_c execute directly without interrupt.
    """
    if tool_name.startswith("risk_"):
        return RiskAssessment(
            tool_name=tool_name,
            tool_args=tool_args,
            static_level=RiskLevel.HIGH,
            dynamic_level=RiskLevel.HIGH,
            final_level=RiskLevel.HIGH,
            risk_reason="test stub: high risk",
            matched_patterns=[],
            suggested_alternative=None,
            primary_arg="",
            dir_arg=None,
            arg_digest="stub_digest",
        )
    return RiskAssessment(
        tool_name=tool_name,
        tool_args=tool_args,
        static_level=RiskLevel.NONE,
        dynamic_level=RiskLevel.NONE,
        final_level=RiskLevel.NONE,
        risk_reason="test stub: safe",
        matched_patterns=[],
        suggested_alternative=None,
        primary_arg="",
        dir_arg=None,
        arg_digest="",
    )


def _make_mock_llm(tool_calls_batch: list[dict], done_content: str = "done"):
    """Build a mock LLM that returns ``tool_calls_batch`` on first call
    and a plain ``AIMessage(content=done_content)`` on subsequent calls."""
    from unittest.mock import AsyncMock

    call_count = {"n": 0}

    async def mock_ainvoke(messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return AIMessage(content="", tool_calls=tool_calls_batch)
        return AIMessage(content=done_content)

    adapter = AsyncMock()
    adapter.ainvoke = mock_ainvoke
    adapter.bind_tools = MagicMock(return_value=adapter)
    return adapter


def _make_write_tool(name: str, spy: _Spy):
    @langchain_tool
    async def _write(message: str) -> str:
        """A non-risky write tool that records side effects."""
        return spy(message=message)

    _write.name = name
    annotate_and_register_tool_source(_write, source="native", category="message")
    return _write


def _make_risk_tool(name: str, spy: _Spy):
    @langchain_tool
    async def _risk(action: str) -> str:
        """A risky tool that should trigger interrupt_helper."""
        return spy(action=action)

    _risk.name = name
    _risk.metadata = {"risk_level": "high"}
    annotate_and_register_tool_source(_risk, source="native", category="shell")
    return _risk


_INITIAL_STATE_DEFAULTS = {
    "step_description": "test step",
    "original_request": "test",
    "language": "en",
    "attachments": [],
    "image_content_blocks": [],
    "events": [],
    "should_interrupt": False,
    "soft_hint_sent": False,
    "attempt_count": 0,
    "failure_count": 0,
}


def _make_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "tool_confirmation_enabled": True,
            "smart_approve_enabled": False,
            # No approval_cache, no confirmation_manager, no event_queue —
            # the dispatcher handles their absence by routing directly to
            # interrupt_helper.
        }
    }


def _drive_until_interrupt(graph, initial_state: dict, config: dict):
    """Run ``graph.ainvoke`` until it hits a GraphInterrupt.

    Returns the state snapshot after the interrupt (including pending
    ask fields).
    """

    async def _run_once():
        return await graph.ainvoke(initial_state, config=config)

    result = _run(_run_once())
    return result


def _resume(graph, action: str, config: dict):
    async def _run_once():
        return await graph.ainvoke(
            Command(resume={"action": action}), config=config
        )

    return _run(_run_once())


def _state_values(graph, config: dict) -> dict:
    async def _get():
        snap = await graph.aget_state(config)
        return snap.values

    return _run(_get())


@pytest.fixture(autouse=True)
def _patch_risk_assessor():
    """All Day-4 tests use the deterministic risk stub."""
    with patch.object(RiskAssessor, "assess", _stubbed_assess):
        yield


class TestReactGraphNodeSplit:
    """Day-4 hard gate — run tool_node + interrupt_helper against a real graph."""

    def _build_graph(self, tools: list):
        from app.domain.services.graphs.react_graph import build_react_graph

        llm = _make_mock_llm(
            tool_calls_batch=[
                {"id": "call_A", "name": "write_tool_a", "args": {"message": "hello"}},
                {"id": "call_B", "name": "risk_skill_b", "args": {"action": "danger"}},
                {"id": "call_C", "name": "write_tool_c", "args": {"message": "world"}},
            ],
            done_content="done",
        )
        return build_react_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
        )

    def test_A_prefix_closed_side_effect_exactly_once(self):
        """I-4.1: write_tool_a runs exactly once across interrupt + replay."""
        spy_a = _Spy("write_tool_a")
        spy_b = _Spy("risk_skill_b")
        spy_c = _Spy("write_tool_c")
        tools = [
            _make_write_tool("write_tool_a", spy_a),
            _make_risk_tool("risk_skill_b", spy_b),
            _make_write_tool("write_tool_c", spy_c),
        ]
        graph = self._build_graph(tools)
        config = _make_config("test_a_thread")
        initial_state = {
            **_INITIAL_STATE_DEFAULTS,
            "messages": [SystemMessage(content="sys")],
        }

        # First drive — graph runs pre_llm → llm → tool_node → interrupt_helper
        # → GraphInterrupt (paused on risk_skill_b)
        _drive_until_interrupt(graph, initial_state, config)

        # Inspect state post-interrupt
        values = _state_values(graph, config)
        completed = values.get("completed_tool_call_prefix") or []
        pending_id = values.get("pending_ask_tool_call_id")

        # I-4.1: call_A is in the completed prefix, call_B is pending,
        # call_C has not been reached.
        assert "call_A" in completed, (
            f"completed_tool_call_prefix should contain call_A, got {completed}"
        )
        assert "call_B" not in completed, (
            "call_B (pending ask) must not be in completed prefix"
        )
        assert "call_C" not in completed, (
            "call_C (not yet reached) must not be in completed prefix"
        )
        assert pending_id == "call_B", (
            f"pending_ask_tool_call_id should be call_B, got {pending_id}"
        )

        # Spy exactly-once: write_tool_a fired once, risk_skill_b body
        # never ran (Asked comes from policy, not wrapper invocation),
        # write_tool_c never ran.
        assert len(spy_a.calls) == 1, (
            f"write_tool_a must fire exactly once pre-interrupt, got {len(spy_a.calls)}"
        )
        assert len(spy_b.calls) == 0
        assert len(spy_c.calls) == 0

    def test_B_approve_resume_path(self):
        """I-4.2 approve: resume with approve → risk_skill_b runs, spy_a stays at 1."""
        spy_a = _Spy("write_tool_a")
        spy_b = _Spy("risk_skill_b")
        spy_c = _Spy("write_tool_c")
        tools = [
            _make_write_tool("write_tool_a", spy_a),
            _make_risk_tool("risk_skill_b", spy_b),
            _make_write_tool("write_tool_c", spy_c),
        ]
        graph = self._build_graph(tools)
        config = _make_config("test_b_thread")
        initial_state = {
            **_INITIAL_STATE_DEFAULTS,
            "messages": [SystemMessage(content="sys")],
        }

        _drive_until_interrupt(graph, initial_state, config)
        # Confirm pre-interrupt spy state
        assert len(spy_a.calls) == 1
        assert len(spy_b.calls) == 0

        # Resume with approve
        _resume(graph, "approve", config)

        values = _state_values(graph, config)

        # All three tool_calls have completed — prefix is reset to []
        # on the happy-path return, so we can't inspect it directly;
        # we verify via spy counts and the final ToolMessage stream.
        assert len(spy_a.calls) == 1, (
            f"write_tool_a must NOT re-run on replay, got {len(spy_a.calls)} calls"
        )
        assert len(spy_b.calls) == 1, (
            f"risk_skill_b should run exactly once post-approve, got {len(spy_b.calls)}"
        )
        assert len(spy_c.calls) == 1, (
            f"write_tool_c should run exactly once after B resumes, got {len(spy_c.calls)}"
        )

        # No pending ask state remains
        assert values.get("pending_ask_tool_call_id") is None
        assert values.get("pending_ask_outcome") is None

        # At least one ToolMessage per tool_call should be present
        tool_msgs = [
            m for m in values.get("messages", []) if isinstance(m, ToolMessage)
        ]
        tool_msg_ids = {m.tool_call_id for m in tool_msgs}
        assert {"call_A", "call_B", "call_C"}.issubset(tool_msg_ids), (
            f"Expected ToolMessages for A/B/C, got {tool_msg_ids}"
        )

    def test_C_deny_resume_path(self):
        """I-4.2 deny: resume with deny → risk_skill_b body never runs,
        interrupt_helper synthesizes a Denied ToolMessage, write_tool_c runs."""
        spy_a = _Spy("write_tool_a")
        spy_b = _Spy("risk_skill_b")
        spy_c = _Spy("write_tool_c")
        tools = [
            _make_write_tool("write_tool_a", spy_a),
            _make_risk_tool("risk_skill_b", spy_b),
            _make_write_tool("write_tool_c", spy_c),
        ]
        graph = self._build_graph(tools)
        config = _make_config("test_c_thread")
        initial_state = {
            **_INITIAL_STATE_DEFAULTS,
            "messages": [SystemMessage(content="sys")],
        }

        _drive_until_interrupt(graph, initial_state, config)
        assert len(spy_a.calls) == 1
        assert len(spy_b.calls) == 0

        _resume(graph, "deny", config)

        values = _state_values(graph, config)

        assert len(spy_a.calls) == 1, "write_tool_a must not re-run on replay"
        assert len(spy_b.calls) == 0, (
            "risk_skill_b body must NEVER execute on deny path"
        )
        assert len(spy_c.calls) == 1, "write_tool_c must run after deny"

        # Verify a Denied ToolMessage for call_B was produced
        tool_msgs = [
            m for m in values.get("messages", []) if isinstance(m, ToolMessage)
        ]
        b_msgs = [m for m in tool_msgs if m.tool_call_id == "call_B"]
        assert b_msgs, "Expected a ToolMessage for call_B (Denied)"
        assert any(m.status == "error" for m in b_msgs), (
            "call_B ToolMessage should have status='error' (Denied)"
        )

    def test_E_timeout_fallback_resume_path(self):
        """I-4.2 timeout_fallback: resume with action='timeout_fallback'.

        Same shape as Test C (deny) but ``interrupt_helper`` must emit
        the "操作因超时被跳过" content instead of "用户拒绝了此操作".
        Covers the no-response deadline branch that Test C cannot catch
        because it forces action="deny" directly.
        """
        spy_a = _Spy("write_tool_a")
        spy_b = _Spy("risk_skill_b")
        spy_c = _Spy("write_tool_c")
        tools = [
            _make_write_tool("write_tool_a", spy_a),
            _make_risk_tool("risk_skill_b", spy_b),
            _make_write_tool("write_tool_c", spy_c),
        ]
        graph = self._build_graph(tools)
        config = _make_config("test_e_thread")
        initial_state = {
            **_INITIAL_STATE_DEFAULTS,
            "messages": [SystemMessage(content="sys")],
        }

        _drive_until_interrupt(graph, initial_state, config)
        assert len(spy_b.calls) == 0

        _resume(graph, "timeout_fallback", config)

        values = _state_values(graph, config)

        assert len(spy_a.calls) == 1, "A must not re-run on replay"
        assert len(spy_b.calls) == 0, (
            "risk_skill_b body must NEVER execute on timeout_fallback path"
        )
        assert len(spy_c.calls) == 1, "write_tool_c must run after timeout_fallback"

        tool_msgs = [
            m for m in values.get("messages", []) if isinstance(m, ToolMessage)
        ]
        b_msgs = [m for m in tool_msgs if m.tool_call_id == "call_B"]
        assert b_msgs, "Expected a ToolMessage for call_B (timeout_fallback)"
        assert any(m.status == "error" for m in b_msgs), (
            "call_B ToolMessage should have status='error' (Denied variant)"
        )
        # Content differentiates timeout_fallback from deny
        assert any("操作因超时被跳过" in m.content for m in b_msgs), (
            f"Expected timeout_fallback content, got: "
            f"{[m.content for m in b_msgs]}"
        )

    def test_F_deny_preserves_original_tool_args_in_event(self):
        """The ToolEvent emitted by interrupt_helper's deny path must
        carry the original ``function_args`` — not an empty dict — so
        downstream audit logs can report what the user actually denied.

        Regression guard for the pending_ask_tool_args state-field fix.
        """
        from app.domain.models.event import ToolEvent

        spy_a = _Spy("write_tool_a")
        spy_b = _Spy("risk_skill_b")
        spy_c = _Spy("write_tool_c")
        tools = [
            _make_write_tool("write_tool_a", spy_a),
            _make_risk_tool("risk_skill_b", spy_b),
            _make_write_tool("write_tool_c", spy_c),
        ]
        graph = self._build_graph(tools)
        config = _make_config("test_f_thread")
        initial_state = {
            **_INITIAL_STATE_DEFAULTS,
            "messages": [SystemMessage(content="sys")],
        }

        _drive_until_interrupt(graph, initial_state, config)

        # Verify pending_ask_tool_args is written
        values_mid = _state_values(graph, config)
        pending_args = values_mid.get("pending_ask_tool_args")
        assert pending_args == {"action": "danger"}, (
            f"Expected pending_ask_tool_args={{'action': 'danger'}}, got {pending_args}"
        )

        _resume(graph, "deny", config)

        values = _state_values(graph, config)

        # Find the ToolEvent emitted for the denied call_B
        tool_events = [
            e
            for e in values.get("events", [])
            if isinstance(e, ToolEvent) and e.tool_call_id == "call_B"
        ]
        assert tool_events, "Expected at least one ToolEvent for call_B"
        # The last ToolEvent for call_B is the one from the deny path
        # (CALLED status with the synthesized Denied outcome)
        denied_event = tool_events[-1]
        assert denied_event.function_args == {"action": "danger"}, (
            f"Expected function_args preserved, got {denied_event.function_args}"
        )
        # pending_ask_tool_args should be cleaned up after deny
        assert values.get("pending_ask_tool_args") is None

    def test_D_cascading_multi_asked(self):
        """I-4.1.1: two cascading risk tools (B and C).

        Sequence:
          1. Dispatch [A, risk_B, risk_C] → A runs, B hits interrupt, C not reached.
          2. Resume approve on B → replay: A skipped (prefix), B bypass+runs,
             C hits interrupt, B marked done.
          3. Resume approve on C → replay: A/B skipped, C bypass+runs.
        """
        spy_a = _Spy("write_tool_a")
        spy_b = _Spy("risk_skill_b")
        spy_c = _Spy("risk_skill_c")
        tools = [
            _make_write_tool("write_tool_a", spy_a),
            _make_risk_tool("risk_skill_b", spy_b),
            _make_risk_tool("risk_skill_c", spy_c),
        ]

        # Override the LLM mock with a cascading batch (risk_skill_c
        # replaces write_tool_c)
        from app.domain.services.graphs.react_graph import build_react_graph

        llm = _make_mock_llm(
            tool_calls_batch=[
                {"id": "call_A", "name": "write_tool_a", "args": {"message": "hi"}},
                {"id": "call_B", "name": "risk_skill_b", "args": {"action": "x"}},
                {"id": "call_C", "name": "risk_skill_c", "args": {"action": "y"}},
            ],
        )
        graph = build_react_graph(
            llm=llm, tools=tools, checkpointer=InMemorySaver()
        )

        config = _make_config("test_d_thread")
        initial_state = {
            **_INITIAL_STATE_DEFAULTS,
            "messages": [SystemMessage(content="sys")],
        }

        # First interrupt on B
        _drive_until_interrupt(graph, initial_state, config)
        values = _state_values(graph, config)
        assert values.get("pending_ask_tool_call_id") == "call_B"
        assert len(spy_a.calls) == 1
        assert len(spy_b.calls) == 0
        assert len(spy_c.calls) == 0

        # Approve B — tool_node replays, B runs via approved_tool_call_ids,
        # C hits interrupt next
        _resume(graph, "approve", config)
        values = _state_values(graph, config)
        assert values.get("pending_ask_tool_call_id") == "call_C"
        assert len(spy_a.calls) == 1, "A must not re-run on replay"
        assert len(spy_b.calls) == 1, "B runs exactly once post-approve"
        assert len(spy_c.calls) == 0

        # Approve C — replays, C runs via approved_tool_call_ids
        _resume(graph, "approve", config)
        values = _state_values(graph, config)
        assert values.get("pending_ask_tool_call_id") is None
        assert len(spy_a.calls) == 1, "A still only one call total"
        assert len(spy_b.calls) == 1, "B still only one call total"
        assert len(spy_c.calls) == 1, "C runs exactly once post-approve"
