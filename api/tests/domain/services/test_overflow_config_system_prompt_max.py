"""B5 C9 / M2-PR0: ContextOverflowConfig.system_prompt_max_tokens projection + usage.

Verifies:
1. ``ContextOverflowConfig.system_prompt_max_tokens`` has the right default
   (10000 post M2-PR0; was 3500 pre-M2-PR0)
2. ``ContextOverflowConfig.from_llm_config`` projects the field from ``LLMConfig``
3. ``AgentTaskRunner._build_prompt_assembler`` reads the config value and
   falls back to the canonical default when config is absent.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.domain.models.app_config import AgentConfig, LLMConfig
from app.domain.models.context_overflow_config import ContextOverflowConfig


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


# ---- ContextOverflowConfig field ---------------------------------------- #


def test_system_prompt_max_tokens_default_is_10000() -> None:
    """M2-PR0: default bumped from 3500 → 10000 to leave room for the
    three memory sections (user/rule/fact_index). Existing deployments
    that want the pre-M2 value can still set 3500 explicitly."""
    config = ContextOverflowConfig()
    assert config.system_prompt_max_tokens == 10000


def test_system_prompt_max_tokens_accepts_custom_value() -> None:
    config = ContextOverflowConfig(system_prompt_max_tokens=5000)
    assert config.system_prompt_max_tokens == 5000


def test_system_prompt_max_tokens_validates_non_negative() -> None:
    with pytest.raises(ValueError):
        ContextOverflowConfig(system_prompt_max_tokens=-1)


# ---- LLMConfig → ContextOverflowConfig projection ---------------------- #


def test_from_llm_config_projects_system_prompt_max_tokens() -> None:
    """The field must flow from LLMConfig through from_llm_config into the
    overflow config used by AgentTaskRunner."""
    llm_config = LLMConfig(system_prompt_max_tokens=4096)
    overflow = ContextOverflowConfig.from_llm_config(llm_config)
    assert overflow.system_prompt_max_tokens == 4096


def test_from_llm_config_uses_default_when_field_absent() -> None:
    """LLMConfig default (10000 post M2-PR0) propagates through."""
    llm_config = LLMConfig()  # no system_prompt_max_tokens override
    overflow = ContextOverflowConfig.from_llm_config(llm_config)
    assert overflow.system_prompt_max_tokens == 10000


# ---- AgentTaskRunner._build_prompt_assembler reads config -------------- #


class _FakeOverflowConfig:
    """Minimal duck-typed ContextOverflowConfig for the builder test."""

    def __init__(
        self,
        *,
        system_prompt_max_tokens: int,
        token_estimator: str = "hybrid",
        model_name: str = "",
    ) -> None:
        self.system_prompt_max_tokens = system_prompt_max_tokens
        self.token_estimator = token_estimator
        self.model_name = model_name


def test_build_prompt_assembler_reads_config_value() -> None:
    """The helper must pick up ``system_prompt_max_tokens`` from the
    injected overflow_config, not the pre-C9 hardcoded 3500.

    Post-C7.5: no feature flag gate — the helper always returns a real
    assembler.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    # Construct a partially-initialized runner (don't go through __init__;
    # we only need _build_prompt_assembler to see _overflow_config)
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._overflow_config = _FakeOverflowConfig(
        system_prompt_max_tokens=4096,
        token_estimator="hybrid",
        model_name="",
    )

    assembler = runner._build_prompt_assembler()
    assert assembler is not None
    assert assembler._budget.max_tokens == 4096


def test_build_prompt_assembler_falls_back_to_canonical_default_when_config_missing() -> None:
    """Defensive: ``_overflow_config=None`` path must still return a usable
    assembler. Post M2-PR0 the fallback matches the canonical default
    (10000) so production and fallback paths agree."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._overflow_config = None

    assembler = runner._build_prompt_assembler()
    assert assembler is not None
    assert assembler._budget.max_tokens == 10000


# ---- M2-PR0: main_graph fallback parity -------------------------------- #


def test_build_main_graph_fallback_budget_matches_canonical_default() -> None:
    """M2-PR0 regression: ``build_main_graph(_allow_default_prompt_assembler=True)``
    constructs a minimal assembler with ``SystemPromptBudget(max_tokens=10000)``
    so the test-only fallback matches the canonical default. Guards against
    future drift between production config (ContextOverflowConfig) and the
    hardcoded test fallback in ``main_graph.py:182``.
    """
    from unittest.mock import patch

    from app.domain.services.graphs.main_graph import build_main_graph

    captured: dict[str, int] = {}

    # Patch SystemPromptBudget where main_graph imports it (inside the
    # function body) so we capture the max_tokens argument used for the
    # fallback assembler, without having to invoke the full graph.
    with patch(
        "app.domain.services.prompts.budget.SystemPromptBudget",
        autospec=True,
    ) as mock_budget:
        mock_budget.return_value = MagicMock(max_tokens=10000)
        try:
            build_main_graph(
                _allow_default_prompt_assembler=True,
                planner_llm=MagicMock(),
                react_graph=MagicMock(),
                summary_llm=MagicMock(),
                uow_factory=MagicMock(),
                session_id="test-m2-pr0",
            )
        except Exception:
            # We don't care whether the graph compiles end-to-end; we only
            # need to verify the budget constructor was called with 10000.
            pass

        # Verify SystemPromptBudget was invoked with max_tokens=10000.
        assert mock_budget.called, "SystemPromptBudget should be called in fallback path"
        # Any of the calls should have max_tokens=10000. (build_main_graph may
        # call it once directly; we assert at least one such call.)
        matching_calls = [
            call for call in mock_budget.call_args_list
            if call.kwargs.get("max_tokens") == 10000
        ]
        assert matching_calls, (
            f"Expected SystemPromptBudget(max_tokens=10000) in fallback; "
            f"got calls: {mock_budget.call_args_list}"
        )
        captured["observed"] = 10000

    assert captured["observed"] == 10000
