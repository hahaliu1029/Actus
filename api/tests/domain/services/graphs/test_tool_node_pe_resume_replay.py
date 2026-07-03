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
# R5-P3 hardening: execution COUNT for the replay-B provenance test — a boolean
# cannot detect a double-execution regression (tool run in phase 1 AND phase 3).
_TOOL_CALL_COUNT = 0


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
    """PE-1 §2.5: include ``tool_confirmation_config`` so the per-call gate
    inside ``_pe_dispatch`` resolves ``is_pe_enabled_for_source`` to True.
    """
    from types import SimpleNamespace

    from app.domain.models.session import SessionStatus  # noqa: F401

    tc_cfg = SimpleNamespace(enabled=True)
    return {
        "configurable": {
            "permission_engine": fake_pe,
            "session_state_machine": fake_ssm,
            "tool_confirmation_config": tc_cfg,
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
        from types import SimpleNamespace
        from app.domain.models.session import SessionStatus  # noqa: F401
        tc_cfg = SimpleNamespace(enabled=True)
        config = {
            "configurable": {
                "permission_engine": fake_pe,
                "session_state_machine": fake_ssm,
                "tool_confirmation_config": tc_cfg,
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


# ---------------------------------------------------------------------------
# B1-1a Step 5b (spec §7 R15#3 + INV-B1-2): replay-B provenance — the cached
# typed outcome is consumed WITHOUT re-running the N1 shell AST validator.
# ---------------------------------------------------------------------------

class TestReplayBProvenanceNoN1Rerun:
    """The legitimate provenance of a cached shell_execute AllowSuccess is the
    ORIGINAL evaluation (N1 validate → PE Asked → interrupt_helper commit_resume
    → pe_resume_outcomes). On the post-resume tool_node replay, replay-path B
    consumes the cached outcome BEFORE the N1 gate — so the validator must run
    exactly ONCE across the whole cycle (the original gate pass), never a second
    time on replay (spec §4.1 INV-B1-2).

    Driver: law A/B hybrid — the pe_resume_outcomes value is produced by the
    REAL interrupt_helper writer path (no hand-seeding), matching
    test_interrupt_helper_pe_commit.py's commit contract.
    """

    @staticmethod
    def _build_graph_fns():
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool as lc_tool

        from app.domain.services.graphs.react_graph import build_react_graph

        global _TOOL_WAS_CALLED, _TOOL_CALL_COUNT
        _TOOL_WAS_CALLED = False
        _TOOL_CALL_COUNT = 0

        @lc_tool
        async def shell_execute(command: str, exec_dir: str = "") -> str:
            """Run a shell command."""
            global _TOOL_WAS_CALLED, _TOOL_CALL_COUNT
            _TOOL_WAS_CALLED = True
            _TOOL_CALL_COUNT += 1
            return "hi"

        stub_llm = AsyncMock()
        stub_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content='{"success":true,"result":"done","attachments":[]}')
        )
        stub_llm.bind_tools = MagicMock(return_value=stub_llm)
        graph = build_react_graph(stub_llm, [shell_execute])
        return (
            graph.nodes["tool_node"].bound.afunc,
            graph.nodes["interrupt_helper"].bound.afunc,
        )

    @staticmethod
    def _shell_state(**overrides) -> dict:
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "id": "tc-shell", "name": "shell_execute",
                    "args": {"command": "echo hi"}, "type": "tool_call",
                }])
            ],
            "llm_input_messages": [],
            "step_description": "test", "original_request": "test", "language": "en",
            "attachments": [], "image_content_blocks": [], "events": [],
            "should_interrupt": False, "soft_hint_sent": False,
            "attempt_count": 0, "failure_count": 0,
            "completed_tool_call_prefix": [], "approved_tool_call_ids": [],
            "pe_resume_outcomes": {},
            "pending_ask_outcome": None, "pending_ask_tool_call_id": None,
            "pending_ask_artifact": None, "pending_ask_tool_args": None,
        }
        state.update(overrides)
        return state

    async def test_replay_b_provenance_no_n1_rerun(self, monkeypatch):
        """INV-B1-2 provenance：replay-B 消费 typed outcome 不重跑 N1。

        断言（本测试的规范部分）：
        1. shell_ast_validator.validate 全程恰好调用 1 次（原始 gate 通道）——
           replay-B 在 N1 块之前消费缓存 outcome，绝不重跑 validator；
        2. pe.evaluate 全程恰好调用 1 次（replay 不重评估）；
        3. 工具本体恰好执行 1 次，最终 ToolMessage 为成功结果。
        """
        import app.domain.services.safety.shell_ast_validator as ast_mod
        from langchain_core.messages import ToolMessage

        from app.domain.models.tool_result import (
            AllowSuccess,
            Asked,
            DecisionReason,
        )

        # `validate` is a function-level local import inside the N1 gate; monkeypatch
        # of the module attribute is picked up because a local `from mod import x`
        # re-reads the module attribute on every call (codex R3 verified).
        real_validate = ast_mod.validate
        calls = {"n": 0}

        def counting_validate(*args, **kwargs):
            calls["n"] += 1
            return real_validate(*args, **kwargs)

        monkeypatch.setattr(ast_mod, "validate", counting_validate)

        tool_node_fn, interrupt_helper_fn = self._build_graph_fns()

        # --- Phase 1: original evaluation → N1 validate (#1) → PE Asked → interrupt.
        asked_pe = AsyncMock()
        asked_pe.evaluate = AsyncMock(return_value=Asked(
            content="waiting for user",
            reason=DecisionReason(type="risk_enforce", code="medium", message="confirm shell"),
        ))
        fake_ssm = _make_fake_ssm()
        original = await tool_node_fn(
            self._shell_state(), _make_config(asked_pe, fake_ssm)
        )
        assert original.goto == "interrupt_helper", "shell_execute Asked must route to interrupt"
        assert asked_pe.evaluate.await_count == 1
        assert calls["n"] == 1, "original gate must run N1 validate exactly once"
        assert _TOOL_WAS_CALLED is False, "tool must not run before approval"

        # --- Phase 2: interrupt_helper real commit_resume → writes pe_resume_outcomes.
        commit_pe = AsyncMock()
        commit_pe.commit_resume = AsyncMock(
            return_value=AllowSuccess(content="ok", data={"via": "user_click"})
        )
        resume_state = self._shell_state(
            pending_ask_outcome=original.update["pending_ask_outcome"],
            pending_ask_tool_call_id=original.update["pending_ask_tool_call_id"],
            pending_ask_artifact=original.update["pending_ask_artifact"],
            pending_ask_tool_args=original.update["pending_ask_tool_args"],
        )
        resume_payload = {
            "tool_call_id": "tc-shell",
            "action": "approve",
            "scope": "session",
            "claim_nonce": "n" * 32,
        }
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            commit_cmd = await interrupt_helper_fn(
                resume_state, _make_config(commit_pe, _make_fake_ssm())
            )
        pe_resume_outcomes = commit_cmd.update["pe_resume_outcomes"]
        assert "tc-shell" in pe_resume_outcomes, "commit_resume must write the typed outcome"

        # --- Phase 3: post-resume tool_node replay → replay-B consumes cached
        #     outcome BEFORE N1; validate must NOT run again, pe.evaluate not re-run.
        replay_pe = AsyncMock()  # its .evaluate must never be called
        replay_state = self._shell_state(
            pe_resume_outcomes=pe_resume_outcomes,
            completed_tool_call_prefix=[],
        )
        replay_cmd = await tool_node_fn(
            replay_state, _make_config(replay_pe, _make_fake_ssm())
        )

        # (1) N1 validator ran exactly once across the WHOLE cycle.
        assert calls["n"] == 1, "replay-B must NOT re-run the N1 validator"
        # (2) pe.evaluate ran exactly once (only in phase 1; replay never re-evaluates).
        replay_pe.evaluate.assert_not_called()
        # (3) the tool executed EXACTLY once across the whole cycle (R5-P3: count,
        #     not boolean — a phase-1+phase-3 double execution must fail here).
        assert _TOOL_WAS_CALLED is True, "replay-B must execute the approved tool"
        assert _TOOL_CALL_COUNT == 1, (
            f"replay-B must execute the approved tool exactly once, got {_TOOL_CALL_COUNT}"
        )
        replay_msgs = [
            m for m in replay_cmd.update["messages"] if isinstance(m, ToolMessage)
        ]
        assert len(replay_msgs) == 1 and replay_msgs[0].status != "error", (
            "replay-B success outcome must yield exactly one non-error ToolMessage"
        )
