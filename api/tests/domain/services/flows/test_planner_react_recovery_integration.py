"""PR-1 tests for PlannerReActFlow Recovery wiring (T19c/e/f/o)."""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models import BaseChatModel

from app.domain.models.app_config import AgentConfig
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
        browser=MagicMock(),
        sandbox=MagicMock(),
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
