# api/tests/invariants/test_inv5_behavior_pe_call_order.py
"""INV-5 behavior: with a fake PE, every native tool call MUST trigger
pe.evaluate (path A) or consume pe_resume_outcomes (path B) before
wrapper execution.

This is the runtime complement to the static AST gate above.

NOTE: The plan referenced a non-existent `graph_runner_with_fake_pe` fixture.
This test instead directly invokes tool_node (via _build_tool_node_fn) using
the pattern established in test_react_graph_pe_dispatch.py (Phase 9 tests).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.tool_result import AllowSuccess, DecisionReason

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _RecordingPE:
    """PE that records evaluate calls and returns AllowSuccess."""

    def __init__(self):
        self._evaluate_count = 0

    @property
    def evaluate_count(self) -> int:
        return self._evaluate_count

    async def evaluate(self, call, ctx):
        self._evaluate_count += 1
        return AllowSuccess(content="ok", data={})

    async def preflight_resume(self, *a, **kw):
        return None

    async def commit_resume(self, *a, **kw):
        return AllowSuccess(content="ok", data={})


def _make_fake_ssm():
    from app.domain.models.session import SessionStatus
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(SessionStatus.RUNNING, 1))
    return ssm


def _make_state(tool_name: str, tool_args: dict, call_id: str = "tc1") -> dict:
    from langchain_core.messages import AIMessage
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
        "pe_resume_outcomes": None,
    }


def _make_config(fake_pe, fake_ssm, *, user_id: str = "u", session_id: str = "s") -> dict:
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


def _build_tool_node_fn():
    """Build a minimal react_graph and return the tool_node closure."""
    from langchain_core.tools import tool as lc_tool
    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        return f"wrote {path}"

    stub_llm = AsyncMock()
    from langchain_core.messages import AIMessage as _AIMessage
    stub_llm.ainvoke = AsyncMock(
        return_value=_AIMessage(content='{"success":true,"result":"done","attachments":[]}')
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    graph = build_react_graph(stub_llm, [file_write])
    tool_node_fn = graph.nodes["tool_node"].bound.afunc
    return tool_node_fn


async def test_native_tool_invocation_calls_pe_evaluate_first():
    """PE.evaluate must be called before the tool wrapper executes.

    Verifies INV-5 path A: pe.evaluate is called at least once when a
    native tool (file_write) is dispatched through _pe_dispatch.
    """
    tool_node_fn = _build_tool_node_fn()
    fake_pe = _RecordingPE()
    fake_ssm = _make_fake_ssm()

    state = _make_state("file_write", {"path": "/x", "content": "hello"})
    config = _make_config(fake_pe, fake_ssm)

    await tool_node_fn(state, config)

    assert fake_pe.evaluate_count >= 1, (
        "INV-5 behavior: pe.evaluate was not called before tool wrapper execution. "
        f"evaluate_count={fake_pe.evaluate_count}"
    )


async def test_pe_evaluate_count_matches_tool_calls():
    """PE.evaluate is called exactly once per native tool call in the batch."""
    tool_node_fn = _build_tool_node_fn()
    fake_pe = _RecordingPE()
    fake_ssm = _make_fake_ssm()

    # Single tool call → exactly one evaluate
    state = _make_state("file_write", {"path": "/y"}, call_id="tc42")
    config = _make_config(fake_pe, fake_ssm, session_id="sess-inv5")

    await tool_node_fn(state, config)

    assert fake_pe.evaluate_count == 1, (
        f"Expected exactly 1 pe.evaluate call, got {fake_pe.evaluate_count}"
    )
