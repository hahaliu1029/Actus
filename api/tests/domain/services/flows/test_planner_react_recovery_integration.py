"""PR-1 tests for PlannerReActFlow Recovery wiring (T19c/e/f/o)."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.models.app_config import AgentConfig
from app.domain.services.flows import planner_react
from app.domain.services.flows.planner_react import PlannerReActFlow


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _fake_profile(provider_id: str = "test_provider"):
    from app.domain.services.provider_profiles._base import ProviderProfile
    return ProviderProfile(
        provider_id=provider_id,
        human_name="Test",
        default_api_mode="chat_completions",
        api_mode_fallback_enabled=True,
    )


def _make_flow(profile=None) -> PlannerReActFlow:
    """Construct a minimal PlannerReActFlow for Recovery integration assertions.
    Modeled on tests/domain/services/flows/test_planner_react_di_gate.py:_make_flow.
    The llm mock uses spec=BaseChatModel so Pydantic validation in
    ``ActusRecoveryChatModel.inner`` accepts it as a BaseChatModel instance.
    """
    return PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(spec=BaseChatModel),
        agent_config=AgentConfig(max_iterations=10, max_retries=3, max_search_results=5),
        session_id="test-session",
        browser_accessor=EagerBrowserAccessor(MagicMock()),
        sandbox_accessor=EagerSandboxAccessor(MagicMock()),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        profile=profile,
        _allow_default_prompt_assembler=True,
    )


def test_T19c_flow_accepts_profile_param():
    sig = inspect.signature(PlannerReActFlow.__init__)
    assert "profile" in sig.parameters
    # Must have a default (None) so existing callers without profile still work.
    assert sig.parameters["profile"].default is None


def test_T19c_flow_stores_profile_and_recovery_wrapped_flag():
    profile = _fake_profile()
    flow = _make_flow(profile=profile)
    assert flow._profile is profile
    assert flow._recovery_wrapped is False  # not yet wrapped


async def test_T19e_ensure_graphs_wraps_llm_when_profile_provided(monkeypatch):
    """After _ensure_graphs() runs, self._llm is recovery-wrapped."""
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
    )
    from app.infrastructure.external.llm.actus_fallback_chat_model import (
        ActusFallbackChatModel,
    )

    flow = _make_flow(profile=_fake_profile())
    # _ensure_graphs depends on a checkpointer + tool collection that need
    # async stubs. Stub the heavy machinery to isolate the wrap-call assertion.
    monkeypatch.setattr(flow, "_get_checkpointer", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(flow, "_collect_all_tools", AsyncMock(return_value=[]))
    # build_react_graph and build_main_graph are imported INSIDE the method
    # via module top-level — patch at module attribute path.
    import app.domain.services.flows.planner_react as planner_react_mod
    monkeypatch.setattr(planner_react_mod, "build_react_graph", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(planner_react_mod, "build_main_graph", MagicMock(return_value=MagicMock()))

    await flow._ensure_graphs()

    assert flow._recovery_wrapped is True, "_recovery_wrapped flag must flip True after wrap"
    llm = flow._llm
    if isinstance(llm, ActusFallbackChatModel):
        assert isinstance(llm.primary, ActusRecoveryChatModel)
        assert isinstance(llm.fallback, ActusRecoveryChatModel)
    else:
        assert isinstance(llm, ActusRecoveryChatModel), (
            f"Expected ActusRecoveryChatModel, got {type(llm).__name__}"
        )


async def test_T19f_ensure_graphs_no_wrap_when_profile_is_none(monkeypatch):
    """Backward compat: profile=None → no wrap, _llm passes through unchanged."""
    flow = _make_flow(profile=None)
    monkeypatch.setattr(flow, "_get_checkpointer", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(flow, "_collect_all_tools", AsyncMock(return_value=[]))
    import app.domain.services.flows.planner_react as planner_react_mod
    monkeypatch.setattr(planner_react_mod, "build_react_graph", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(planner_react_mod, "build_main_graph", MagicMock(return_value=MagicMock()))

    original_llm = flow._llm
    await flow._ensure_graphs()
    assert flow._recovery_wrapped is False
    assert flow._llm is original_llm  # untouched


async def test_T19o_ensure_graphs_idempotent_no_nested_recovery(monkeypatch):
    """Calling _ensure_graphs twice does not nest Recovery wrappers."""
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
    )
    from app.infrastructure.external.llm.actus_fallback_chat_model import (
        ActusFallbackChatModel,
    )

    flow = _make_flow(profile=_fake_profile())
    monkeypatch.setattr(flow, "_get_checkpointer", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(flow, "_collect_all_tools", AsyncMock(return_value=[]))
    import app.domain.services.flows.planner_react as planner_react_mod
    monkeypatch.setattr(planner_react_mod, "build_react_graph", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(planner_react_mod, "build_main_graph", MagicMock(return_value=MagicMock()))

    await flow._ensure_graphs()
    first_llm = flow._llm
    await flow._ensure_graphs()
    second_llm = flow._llm

    # Same wrapper instance, NOT a new wrap of the previous wrap (no nesting).
    assert second_llm is first_llm

    # Drilling into .inner: must be the original MagicMock, not another Recovery.
    inner = second_llm.primary.inner if isinstance(second_llm, ActusFallbackChatModel) else second_llm.inner
    assert not isinstance(inner, ActusRecoveryChatModel), (
        f"Recovery nesting detected: inner is {type(inner).__name__}"
    )


# ── PR-2 Task 2.6: _build_on_context_overflow_callback contract tests ─────────


@dataclass
class _FakeCompactionResult:
    messages: list
    level_applied: int
    tokens_before: int
    tokens_after: int


class _FakeCompactor:
    """Records every try_compact call (including config kwarg) so tests can
    assert cost-ledger handler propagation. See Audit Round 9 P1 #1.
    """

    def __init__(self, result: _FakeCompactionResult):
        self._result = result
        self.calls: list[dict] = []

    async def try_compact(self, *, messages, context_window, summary_llm, config=None):
        self.calls.append({
            "messages": messages,
            "context_window": context_window,
            "summary_llm": summary_llm,
            "config": config,
        })
        return self._result


def _make_flow_with_overflow(*, guard_enabled: bool = True, profile=None):
    """Flow fixture with a real-shaped overflow_config and stub compactor seam.
    Uses MagicMock-via-spec for ContextOverflowConfig because we only need
    .model_name and .context_overflow_guard_enabled fields.
    """
    from types import SimpleNamespace
    flow = _make_flow(profile=profile or _fake_profile())
    flow._overflow_config = SimpleNamespace(
        model_name="test-model",
        context_overflow_guard_enabled=guard_enabled,
    )
    flow._summary_llm = None
    return flow


async def test_T19f_callback_uses_total_context_window_not_effective(monkeypatch):
    """Spec §4.5: callback passes resolve_context_window's TOTAL window to
    compactor.try_compact (not effective_window — the compactor's soft/hard
    triggers calibrate against the full model context).
    """
    monkeypatch.setattr(planner_react, "resolve_context_window", lambda n, c: 123456)
    msgs = [HumanMessage(content="x")]
    compactor = _FakeCompactor(_FakeCompactionResult(
        messages=msgs, level_applied=2, tokens_before=900, tokens_after=400,
    ))
    flow = _make_flow_with_overflow()
    flow._compactor = compactor

    cb = flow._build_on_context_overflow_callback()
    out = await cb(msgs, {})

    assert out == msgs
    assert compactor.calls and compactor.calls[-1]["context_window"] == 123456


async def test_T28b_callback_progress_by_tokens_not_message_count(monkeypatch):
    """Same N messages in and out, but tokens_after < tokens_before AND
    level_applied > 0 → callback returns the compressed list. Progress signal
    is tokens, not len(messages)."""
    monkeypatch.setattr(planner_react, "resolve_context_window", lambda n, c: 100000)
    msgs = [HumanMessage(content="x")] * 10
    compactor = _FakeCompactor(_FakeCompactionResult(
        messages=msgs, level_applied=2, tokens_before=9000, tokens_after=4000,
    ))
    flow = _make_flow_with_overflow()
    flow._compactor = compactor

    cb = flow._build_on_context_overflow_callback()
    out = await cb([HumanMessage(content="x")] * 10, {})
    assert out is not None
    assert len(out) == 10  # message count unchanged; progress came from tokens


async def test_T28b_callback_level_applied_zero_returns_none(monkeypatch):
    """compactor says 'didn't need to compact' (level_applied == 0) → None."""
    monkeypatch.setattr(planner_react, "resolve_context_window", lambda n, c: 100000)
    msgs = [HumanMessage(content="x")]
    compactor = _FakeCompactor(_FakeCompactionResult(
        messages=msgs, level_applied=0, tokens_before=900, tokens_after=900,
    ))
    flow = _make_flow_with_overflow()
    flow._compactor = compactor

    cb = flow._build_on_context_overflow_callback()
    out = await cb(msgs, {})
    assert out is None


async def test_T28b_callback_tokens_not_decreasing_returns_none(monkeypatch):
    """tokens_after >= tokens_before → None (no progress) even if level_applied > 0."""
    monkeypatch.setattr(planner_react, "resolve_context_window", lambda n, c: 100000)
    msgs = [HumanMessage(content="x")]
    compactor = _FakeCompactor(_FakeCompactionResult(
        messages=msgs, level_applied=2, tokens_before=5000, tokens_after=5000,
    ))
    flow = _make_flow_with_overflow()
    flow._compactor = compactor

    cb = flow._build_on_context_overflow_callback()
    out = await cb(msgs, {})
    assert out is None


async def test_T28b_callback_forwards_cost_handler_to_compactor(monkeypatch):
    """Audit Round 9 P1 #1: B2 emergency compact MUST register the same
    cost-ledger callback as `_check_overflow` so Level-2 summary LLM calls
    are billed under node_name='context_compaction'.
    """
    monkeypatch.setattr(planner_react, "resolve_context_window", lambda n, c: 100000)
    msgs = [HumanMessage(content="x")] * 4
    compactor = _FakeCompactor(_FakeCompactionResult(
        messages=msgs, level_applied=2, tokens_before=9000, tokens_after=4000,
    ))
    cost_sentinel = object()
    flow = _make_flow_with_overflow()
    flow._compactor = compactor
    object.__setattr__(flow, "_cost_callback_handler", cost_sentinel)

    cb = flow._build_on_context_overflow_callback()
    await cb(msgs, {})
    assert compactor.calls, "compactor not invoked"
    cfg = compactor.calls[-1].get("config")
    assert cfg is not None, "compact_config missing — cost ledger broken"
    assert cost_sentinel in (cfg.get("callbacks") or [])
    assert cfg.get("metadata", {}).get("langgraph_node") == "context_compaction"


async def test_T28b_callback_skips_when_overflow_guard_disabled(monkeypatch):
    """Audit Round 9 P2 #2: `context_overflow_guard_enabled=False` means the
    proactive `_check_overflow` path is off; the B2 emergency path must obey
    the same switch.
    """
    call_log = []
    monkeypatch.setattr(
        planner_react, "resolve_context_window",
        lambda n, c: call_log.append("resolve") or 100000,
    )
    compactor = _FakeCompactor(_FakeCompactionResult(
        messages=[HumanMessage(content="x")], level_applied=2,
        tokens_before=100, tokens_after=50,
    ))
    flow = _make_flow_with_overflow(guard_enabled=False)
    flow._compactor = compactor

    cb = flow._build_on_context_overflow_callback()
    out = await cb([HumanMessage(content="x")], {})
    assert out is None, "guard-off must skip emergency compact"
    assert compactor.calls == [], "compactor was called despite guard-off"
    assert call_log == [], "resolve_context_window invoked despite guard-off"


def test_T35_on_context_overflow_signature_is_two_params():
    """Protocol signature regression gate — spec §3.4 locks it at 2 params."""
    from app.domain.services.recovery._base import OnContextOverflow

    args = OnContextOverflow.__args__  # type: ignore[attr-defined]
    assert len(args) == 3, f"OnContextOverflow must be 2-arg callable; got args={args}"
    # __args__ = (arg1_type, arg2_type, return_type)
    # Argument positions [0] and [1] are `list[BaseMessage]` and `dict`; no third param.


def test_T34b_trigger_recompact_does_not_reference_target_tokens_or_context_window():
    """AST scan: TriggerRecompact.apply must not touch ctx.profile.default_context_window,
    target_tokens, or any context-window concept — budget decisions belong to the Runner
    callback, not Recovery's action."""
    import ast
    import inspect
    from app.domain.services.recovery._actions import TriggerRecompact

    src = inspect.getsource(TriggerRecompact)
    tree = ast.parse(src)
    forbidden = {
        "target_tokens", "default_context_window", "context_window",
        "resolve_context_window", "compute_effective_window",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden, (
                f"TriggerRecompact.apply referenced forbidden attr {node.attr!r} — "
                f"compact-budget decisions belong to the Runner callback"
            )
        if isinstance(node, ast.Name):
            assert node.id not in forbidden, (
                f"TriggerRecompact.apply referenced forbidden name {node.id!r}"
            )
