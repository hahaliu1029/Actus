"""B1-0 characterization anchors C1-C9 (spec §3).

Locks CURRENT behavior before the B1-1/B1-2 refactor. Every test here must
PASS against the unmodified codebase and stay green with all 4 B1 flags OFF
for the rest of the epic (INV-B1-0).

Style: option (b) — call node closures directly (same as
test_react_graph_pe_dispatch.py). Fixtures are private copies per suite
convention (no cross-test-file helper imports).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from app.domain.models.event import (
    MessageEvent,
    ToolConfirmationEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import (
    AllowSuccess,
    Asked,
    DecisionReason,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Private fixture copies (P-10 convention)
# ---------------------------------------------------------------------------

class SequencedPE:
    """Fake PE returning a scripted outcome per evaluate() call, in order."""

    def __init__(self, outcomes: list[str]):
        self.outcomes = list(outcomes)
        self.calls: list[tuple] = []

    async def evaluate(self, call, ctx):
        self.calls.append((call, ctx))
        kind = self.outcomes.pop(0)
        if kind == "allow":
            return AllowSuccess(content="auto", data={})
        if kind == "ask":
            return Asked(
                content="waiting for user",
                reason=DecisionReason(
                    type="risk_enforce", code="medium", message="test"
                ),
            )
        raise AssertionError(f"unexpected scripted outcome {kind!r}")

    async def preflight_resume(self, *a, **kw):
        pass

    async def commit_resume(self, *a, **kw):
        pass


def _make_fake_ssm(mode=SessionStatus.RUNNING, revision=1):
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(mode, revision))
    return ssm


def _tc(name: str, args: dict, call_id: str) -> dict:
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def _make_state(tool_calls: list[dict], **overrides) -> dict:
    state = {
        "messages": [AIMessage(content="", tool_calls=tool_calls)]
        if tool_calls
        else [HumanMessage(content="go")],
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
    state.update(overrides)
    return state


def _make_config(fake_pe, fake_ssm, *, extra: dict | None = None):
    tc_cfg = SimpleNamespace(enabled=True)
    configurable: dict = {
        "permission_engine": fake_pe,
        "session_state_machine": fake_ssm,
        "tool_confirmation_config": tc_cfg,
        "user_id": "u",
        "session_id": "s",
        "thread_id": "s",
    }
    if extra:
        configurable.update(extra)
    return {"configurable": configurable}


def _build_graph_fns(record: list | None = None, shell_result: str = "ok"):
    """Build a minimal react graph; return node closures + stubs.

    ``record`` collects ("start"/"end", tool_name) tuples so serial-vs-
    concurrent execution interleaving is observable (C4).
    """
    from langchain_core.tools import tool as lc_tool

    from app.domain.services.graphs.react_graph import build_react_graph

    rec = record if record is not None else []

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        rec.append(("start", "file_write"))
        await asyncio.sleep(0)
        rec.append(("end", "file_write"))
        return f"wrote {path}"

    @lc_tool
    async def file_read(path: str) -> str:
        """Read a file."""
        rec.append(("start", "file_read"))
        await asyncio.sleep(0)
        rec.append(("end", "file_read"))
        return "content"

    @lc_tool
    async def shell_execute(command: str, exec_dir: str = "") -> str:
        """Run shell command."""
        rec.append(("start", "shell_execute"))
        await asyncio.sleep(0)
        rec.append(("end", "shell_execute"))
        return shell_result

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
    # C1 tripwire: any astream call on the default path is a regression.
    stub_llm.astream = MagicMock(
        side_effect=AssertionError("llm_node must use ainvoke on default path (C1)")
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    graph = build_react_graph(stub_llm, [file_write, file_read, shell_execute])
    return SimpleNamespace(
        graph=graph,
        llm_node=graph.nodes["llm_node"].bound.afunc,
        tool_node=graph.nodes["tool_node"].bound.afunc,
        stub_llm=stub_llm,
        record=rec,
    )


# ---------------------------------------------------------------------------
# C1 — llm_node is atomic ainvoke (never astream) on the default path
# ---------------------------------------------------------------------------

class TestC1LlmNodeAtomicAinvoke:
    async def test_llm_node_uses_ainvoke_never_astream(self):
        fns = _build_graph_fns()
        state = _make_state([])
        result = await fns.llm_node(state, {"configurable": {}})
        fns.stub_llm.ainvoke.assert_awaited_once()
        assert not fns.stub_llm.astream.called
        assert "messages" in result and "events" in result


# ---------------------------------------------------------------------------
# C2 — CALLING events arrive as one batch AFTER the full LLM response,
#      via state-path only; tool_calls non-empty suppresses MessageEvent
# ---------------------------------------------------------------------------

class TestC2CallingBatchStatePath:
    async def test_calling_batch_after_full_response_state_path_only(self):
        fns = _build_graph_fns()
        queue: asyncio.Queue = asyncio.Queue()
        tc1 = _tc("file_write", {"path": "/a", "content": "x"}, "c2t1")
        tc2 = _tc("file_read", {"path": "/b"}, "c2t2")
        fns.stub_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="", tool_calls=[tc1, tc2])
        )
        state = _make_state([])
        result = await fns.llm_node(
            state, {"configurable": {"event_queue": queue}}
        )
        calling = [e for e in result["events"] if isinstance(e, ToolEvent)]
        assert [e.tool_call_id for e in calling] == ["c2t1", "c2t2"]
        assert all(e.status == ToolEventStatus.CALLING for e in calling)
        assert queue.empty(), "CALLING must not use event_queue today"

    async def test_tool_calls_nonempty_suppresses_message_event(self):
        """:1127 baseline — content + tool_calls together emits NO MessageEvent."""
        fns = _build_graph_fns()
        tc1 = _tc("file_write", {"path": "/a"}, "c2t3")
        fns.stub_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="thinking aloud", tool_calls=[tc1])
        )
        result = await fns.llm_node(_make_state([]), {"configurable": {}})
        assert not [e for e in result["events"] if isinstance(e, MessageEvent)]


# ---------------------------------------------------------------------------
# C3 — single-channel exclusivity baseline (allow case)
# ---------------------------------------------------------------------------

class TestC3SingleChannelExclusivity:
    async def test_called_goes_state_path_queue_stays_empty(self):
        fns = _build_graph_fns()
        queue: asyncio.Queue = asyncio.Queue()
        fake_pe = SequencedPE(["allow"])
        state = _make_state([_tc("file_write", {"path": "/a", "content": "x"}, "t1")])
        result = await fns.tool_node(
            state, _make_config(fake_pe, _make_fake_ssm(), extra={"event_queue": queue})
        )
        assert queue.empty(), "ToolEvent must not be enqueued today"
        called = [
            e for e in result.update["events"]
            if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLED
        ]
        assert [e.tool_call_id for e in called] == ["t1"]


# ---------------------------------------------------------------------------
# C4 — strictly serial execution in tc order (both dispatch paths)
# ---------------------------------------------------------------------------

class TestC4SerialExecution:
    async def test_pe_path_serial_in_tc_order(self):
        record: list = []
        fns = _build_graph_fns(record)
        fake_pe = SequencedPE(["allow", "allow"])
        state = _make_state([
            _tc("file_write", {"path": "/a", "content": "x"}, "t1"),
            _tc("file_read", {"path": "/b"}, "t2"),
        ])
        await fns.tool_node(state, _make_config(fake_pe, _make_fake_ssm()))
        assert record == [
            ("start", "file_write"), ("end", "file_write"),
            ("start", "file_read"), ("end", "file_read"),
        ]

    async def test_legacy_path_serial_in_tc_order(self):
        record: list = []
        fns = _build_graph_fns(record)
        state = _make_state([
            _tc("file_write", {"path": "/a", "content": "x"}, "t1"),
            _tc("file_read", {"path": "/b"}, "t2"),
        ])
        # No permission_engine key → legacy direct-execute path.
        await fns.tool_node(
            state, {"configurable": {"user_id": "u", "session_id": "s"}}
        )
        assert record == [
            ("start", "file_write"), ("end", "file_write"),
            ("start", "file_read"), ("end", "file_read"),
        ]


# ---------------------------------------------------------------------------
# C5 — message order: ToolMessage per tc order, deferred HumanMessage last;
#      event order: CALLED per tc in execution order
# ---------------------------------------------------------------------------

class TestC5MessageAndEventOrder:
    async def test_toolmessages_tc_order_deferred_human_last(self):
        # shell output with a valid base64 data-URL → AllowSuccess is converted
        # to Passthrough (_maybe_convert_shell_outcome_with_images) → deferred
        # HumanMessage (Layer 3, _translate_outcome Step 4).
        b64 = "iVBORw0KGgoAAAANSUhEUgAA" * 8  # base64 charset, len % 4 == 0
        fns = _build_graph_fns(
            shell_result=f"console output data:image/png;base64,{b64} tail"
        )
        fake_pe = SequencedPE(["allow", "allow"])
        state = _make_state([
            _tc("shell_execute", {"command": "echo hi"}, "t1"),
            _tc("file_write", {"path": "/a", "content": "x"}, "t2"),
        ])
        result = await fns.tool_node(state, _make_config(fake_pe, _make_fake_ssm()))
        msgs = result.update["messages"]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
        assert [m.tool_call_id for m in tool_msgs] == ["t1", "t2"]
        assert len(human_msgs) == 1, "Passthrough must defer exactly one HumanMessage"
        assert msgs.index(human_msgs[0]) > msgs.index(tool_msgs[-1])
        called = [e for e in result.update["events"] if isinstance(e, ToolEvent)]
        assert [e.tool_call_id for e in called] == ["t1", "t2"]
        assert all(e.status == ToolEventStatus.CALLED for e in called)


# ---------------------------------------------------------------------------
# C6 — exactly-once accounting: Asked interrupt writes executed prefix;
#      clean batch resets prefix/pending fields
# ---------------------------------------------------------------------------

class TestC6PrefixAccounting:
    async def test_asked_interrupt_prefix_holds_executed_ids(self):
        fns = _build_graph_fns()
        fake_pe = SequencedPE(["allow", "ask"])
        queue: asyncio.Queue = asyncio.Queue()
        state = _make_state([
            _tc("file_write", {"path": "/a", "content": "x"}, "t1"),
            _tc("file_read", {"path": "/b"}, "t2"),
        ])
        result = await fns.tool_node(
            state, _make_config(fake_pe, _make_fake_ssm(), extra={"event_queue": queue})
        )
        assert result.goto == "interrupt_helper"
        assert result.update["completed_tool_call_prefix"] == ["t1"]
        assert result.update["pending_ask_tool_call_id"] == "t2"
        assert result.update["pending_ask_outcome"] is not None
        assert result.update["pending_ask_artifact"] is not None
        assert result.update["pending_ask_tool_args"] == {"path": "/b"}
        # Asked early-return does NOT write soft_hint_sent (spec §4.1 asymmetry)
        assert "soft_hint_sent" not in result.update

    async def test_clean_batch_resets_prefix_and_pending(self):
        fns = _build_graph_fns()
        fake_pe = SequencedPE(["allow", "allow"])
        state = _make_state([
            _tc("file_write", {"path": "/a", "content": "x"}, "t1"),
            _tc("file_read", {"path": "/b"}, "t2"),
        ])
        result = await fns.tool_node(state, _make_config(fake_pe, _make_fake_ssm()))
        assert result.goto == "pre_llm_node"
        assert result.update["completed_tool_call_prefix"] == []
        assert result.update["approved_tool_call_ids"] == []
        for key in (
            "pending_ask_outcome", "pending_ask_tool_call_id",
            "pending_ask_artifact", "pending_ask_tool_args",
        ):
            assert result.update[key] is None
        assert result.update["attempt_count"] == 1


# ---------------------------------------------------------------------------
# C7 — MAX_ITERATIONS(30) cap: attempt_count=29 batch completion routes END
# ---------------------------------------------------------------------------

class TestC7AttemptCapRouting:
    async def test_pe_path_routes_end_at_cap(self):
        fns = _build_graph_fns()
        fake_pe = SequencedPE(["allow"])
        state = _make_state(
            [_tc("file_write", {"path": "/a", "content": "x"}, "t1")],
            attempt_count=29,
        )
        result = await fns.tool_node(state, _make_config(fake_pe, _make_fake_ssm()))
        assert result.update["attempt_count"] == 30
        assert result.goto == END

    async def test_legacy_path_routes_end_at_cap(self):
        fns = _build_graph_fns()
        state = _make_state(
            [_tc("file_write", {"path": "/a", "content": "x"}, "t1")],
            attempt_count=29,
        )
        result = await fns.tool_node(
            state, {"configurable": {"user_id": "u", "session_id": "s"}}
        )
        assert result.update["attempt_count"] == 30
        assert result.goto == END


# ---------------------------------------------------------------------------
# C8 — mixed batch → _pe_dispatch None sentinel → legacy whole-batch takeover
# ---------------------------------------------------------------------------

class TestC8MixedBatchLegacyFallback:
    async def test_mixed_batch_zero_pe_evaluate_and_fail_closed_guard(self):
        fns = _build_graph_fns()
        fake_pe = SequencedPE([])  # any evaluate() call would pop-crash → proof of zero calls
        state = _make_state([
            _tc("totally_unknown_tool_xyz", {}, "t1"),  # unresolvable source → batch not PE-eligible
            _tc("file_write", {"path": "/a", "content": "x"}, "t2"),
        ])
        result = await fns.tool_node(state, _make_config(fake_pe, _make_fake_ssm()))
        assert fake_pe.calls == [], "mixed batch must bypass pe.evaluate entirely"
        tool_msgs = {
            m.tool_call_id: m
            for m in result.update["messages"]
            if isinstance(m, ToolMessage)
        }
        assert "Unknown tool" in str(tool_msgs["t1"].content)
        # native_mixed_batch_fail_closed: real native tool denied in legacy fallback
        assert tool_msgs["t2"].status == "error"
        assert "无法与非权限引擎工具" in str(tool_msgs["t2"].content)
        assert result.update["completed_tool_call_prefix"] == []
        assert result.update["attempt_count"] == 1


# ---------------------------------------------------------------------------
# C9 — Asked batch channel timing: ToolConfirmationEvent is ALREADY in the
#      queue at node-return time; prior tool's CALLED only surfaces via the
#      returned update (state-path); no dual-channel emission
#
# NOTE (R17, node-return vs wire): this is a NODE-RETURN-level assertion, not a
# wire-delivery guarantee. In production the queued ToolConfirmationEvent is
# emitted mid-node (before node return), and AgentTaskRunner STOPS consuming on
# ToolConfirmationEvent (agent_task_runner.py:3089/:4527) — so on a flag-OFF
# Asked-interrupt batch the prior tools' state-path CALLED events (present in
# result.update["events"]) are NOT wire-delivered post-confirmation. This is
# today's behavior, preserved byte-for-byte (INV-B1-0). Under B1-1c flag-ON the
# concurrent path emits those CALLED via the QUEUE before the drain enqueues the
# confirmation, so they ARE wire-visible in that mode (see T9
# test_asked_drain_queued_called_precede_confirmation).
# ---------------------------------------------------------------------------

class TestC9AskedChannelTiming:
    async def test_confirmation_in_queue_at_return_called_in_update(self):
        fns = _build_graph_fns()
        fake_pe = SequencedPE(["allow", "ask"])
        queue: asyncio.Queue = asyncio.Queue()
        state = _make_state([
            _tc("file_write", {"path": "/a", "content": "x"}, "t1"),
            _tc("file_read", {"path": "/b"}, "t2"),
        ])
        result = await fns.tool_node(
            state, _make_config(fake_pe, _make_fake_ssm(), extra={"event_queue": queue})
        )
        # By construction the put() happened mid-node, before return — so the
        # confirmation is already waiting in the queue when the state-path
        # events first become visible (node return).
        assert queue.qsize() == 1
        assert isinstance(queue.get_nowait(), ToolConfirmationEvent)
        called = [
            e for e in result.update["events"]
            if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLED
        ]
        assert [e.tool_call_id for e in called] == ["t1"]
        assert not any(
            isinstance(e, ToolConfirmationEvent) for e in result.update["events"]
        ), "confirmation must never dual-emit via state-path"
