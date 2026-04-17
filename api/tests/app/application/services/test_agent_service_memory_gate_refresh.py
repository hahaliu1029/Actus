"""PR-4+8 regression: ``AgentService._refresh_config`` must rebuild gate
breaker + daily_cap when the snapshot's gate LLM identity changes.

Bug scenario this test prevents:
- Initial config: ``settings.memory_gate_llm="summary_llm"`` but
  ``app_config.summary_model`` is empty → ``snapshot.memory_gate_llm=None``
  → ``_build_agent_service`` pins breaker/daily_cap to None.
- Deployer hot-edits ``config.yaml`` to set ``summary_model`` → next
  request triggers ``_load_app_config`` reload → new snapshot with
  non-None gate LLM → ``_refresh_config`` fires.
- Without the rebuild hook, ``self._memory_gate_breaker`` stays at None
  → ``_create_task`` passes None to flow → gate path runs unprotected
  (no circuit breaker on LLM errors, no daily cap on auto-promotion).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.application.services.agent_service import AgentService, _ConfigSnapshot
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    MCPConfig,
    SkillRiskPolicy,
    ToolRuntimeConfig,
)
from app.domain.models.context_overflow_config import ContextOverflowConfig

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_snapshot(gate_llm) -> _ConfigSnapshot:
    """Minimal snapshot — only the fields ``_refresh_config`` touches."""
    return _ConfigSnapshot(
        llm=MagicMock(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        skill_risk_policy=SkillRiskPolicy(),
        overflow_config=ContextOverflowConfig(),
        summary_llm=None,
        vision_fallback_model=None,
        skill_creator_service=MagicMock(),
        supports_vision=False,
        supports_pdf_input=False,
        file_understanding_config=None,
        tool_runtime=ToolRuntimeConfig(),
        memory_gate_llm=gate_llm,
    )


def _make_agent_service(
    *,
    initial_breaker,
    initial_daily_cap,
    initial_gate_llm,
    rebuild_fn,
) -> AgentService:
    """Bypass full __init__ (requires many real deps) and set only the
    attrs ``_refresh_config`` reads."""
    svc = AgentService.__new__(AgentService)
    svc._config_snapshot = _make_snapshot(initial_gate_llm)
    svc._memory_gate_breaker = initial_breaker
    svc._memory_gate_daily_cap = initial_daily_cap
    svc._memory_gate_rebuild_fn = rebuild_fn
    return svc


class TestRefreshGateRebuild:

    def test_off_to_on_builds_breaker_and_cap(self) -> None:
        """Deployer enables gate mid-flight: snapshot goes None → LLM.
        Breaker + cap must materialize or else flow bypasses them."""
        new_llm = MagicMock(name="new_gate_llm")

        calls: list = []

        def rebuild(snap):
            calls.append(snap.memory_gate_llm)
            return (object(), object())  # distinctive non-None sentinels

        svc = _make_agent_service(
            initial_breaker=None,
            initial_daily_cap=None,
            initial_gate_llm=None,
            rebuild_fn=rebuild,
        )
        svc._refresh_config(_make_snapshot(new_llm))

        assert calls == [new_llm], "rebuild_fn must fire with new snapshot"
        assert svc._memory_gate_breaker is not None
        assert svc._memory_gate_daily_cap is not None

    def test_on_to_off_clears_breaker_and_cap(self) -> None:
        """Deployer disables gate (removes summary_model). Breaker +
        cap should clear to None or the old instances linger referencing
        a dead LLM that will never be called again."""
        old_llm = MagicMock(name="old_gate_llm")
        old_breaker = MagicMock(name="stale_breaker")
        old_cap = MagicMock(name="stale_cap")

        def rebuild(snap):
            return (None, None)

        svc = _make_agent_service(
            initial_breaker=old_breaker,
            initial_daily_cap=old_cap,
            initial_gate_llm=old_llm,
            rebuild_fn=rebuild,
        )
        svc._refresh_config(_make_snapshot(None))

        assert svc._memory_gate_breaker is None
        assert svc._memory_gate_daily_cap is None

    def test_llm_swap_rebuilds_both(self) -> None:
        """Switching from one gate LLM to another resets breaker state
        (error counter doesn't transfer across LLMs) and daily cap
        (cap instance should bind to whatever the new LLM needs)."""
        old_llm = MagicMock(name="old_llm")
        new_llm = MagicMock(name="new_llm")
        old_breaker = MagicMock()
        old_cap = MagicMock()
        new_breaker = MagicMock()
        new_cap = MagicMock()

        def rebuild(snap):
            assert snap.memory_gate_llm is new_llm
            return (new_breaker, new_cap)

        svc = _make_agent_service(
            initial_breaker=old_breaker,
            initial_daily_cap=old_cap,
            initial_gate_llm=old_llm,
            rebuild_fn=rebuild,
        )
        svc._refresh_config(_make_snapshot(new_llm))

        assert svc._memory_gate_breaker is new_breaker
        assert svc._memory_gate_daily_cap is new_cap

    def test_same_llm_preserves_breaker_state(self) -> None:
        """Config refresh that touches other fields but keeps gate LLM
        identity intact must NOT reset the breaker — an ongoing LLM
        failure streak shouldn't be wiped because operator bumped an
        unrelated flag."""
        gate_llm = MagicMock(name="gate_llm")
        existing_breaker = MagicMock(name="existing_breaker")
        existing_cap = MagicMock(name="existing_cap")

        rebuild_called = False

        def rebuild(snap):
            nonlocal rebuild_called
            rebuild_called = True
            return (MagicMock(), MagicMock())

        svc = _make_agent_service(
            initial_breaker=existing_breaker,
            initial_daily_cap=existing_cap,
            initial_gate_llm=gate_llm,
            rebuild_fn=rebuild,
        )
        svc._refresh_config(_make_snapshot(gate_llm))  # same identity

        assert not rebuild_called, (
            "rebuild should be skipped when gate_llm identity unchanged"
        )
        assert svc._memory_gate_breaker is existing_breaker
        assert svc._memory_gate_daily_cap is existing_cap

    def test_no_rebuild_fn_still_swaps_snapshot(self) -> None:
        """Legacy construction (no factory injected) must not crash
        during refresh — it just can't auto-rebuild. Snapshot still
        swaps; breaker/cap stay at their init values."""
        svc = _make_agent_service(
            initial_breaker=None,
            initial_daily_cap=None,
            initial_gate_llm=None,
            rebuild_fn=None,
        )
        new_llm = MagicMock()
        svc._refresh_config(_make_snapshot(new_llm))

        assert svc._config_snapshot.memory_gate_llm is new_llm
        assert svc._memory_gate_breaker is None  # unchanged (no factory)
        assert svc._memory_gate_daily_cap is None
