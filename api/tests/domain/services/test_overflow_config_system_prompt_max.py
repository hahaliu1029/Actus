"""B5 C9: ContextOverflowConfig.system_prompt_max_tokens projection + usage.

Verifies:
1. ``ContextOverflowConfig.system_prompt_max_tokens`` has the right default
2. ``ContextOverflowConfig.from_llm_config`` projects the field from ``LLMConfig``
3. ``AgentTaskRunner._build_prompt_assembler`` reads the config value
   (not the pre-C9 hardcoded 3500)
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


def test_system_prompt_max_tokens_default_is_3500() -> None:
    """Default matches the pre-C9 hardcoded value so existing deployments
    see no behavioral change after upgrading."""
    config = ContextOverflowConfig()
    assert config.system_prompt_max_tokens == 3500


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
    """LLMConfig default (3500) propagates through."""
    llm_config = LLMConfig()  # no system_prompt_max_tokens override
    overflow = ContextOverflowConfig.from_llm_config(llm_config)
    assert overflow.system_prompt_max_tokens == 3500


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


def test_build_prompt_assembler_falls_back_to_3500_when_config_missing() -> None:
    """Defensive: _overflow_config=None path must still return a usable
    assembler with the historical 3500 default."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._overflow_config = None

    assembler = runner._build_prompt_assembler()
    assert assembler is not None
    assert assembler._budget.max_tokens == 3500
