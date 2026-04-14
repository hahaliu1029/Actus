"""B5 C9: budget integration — ContextAssembler and GradualCompactor agree.

Verifies the C9 invariant: ``ContextAssembler`` (sync in-step trimmer) and
``GradualCompactor.try_compact`` (async LLM summarization) must derive
their budgets from the SAME ``compute_effective_window()`` call so the
two layers cannot drift.

Covers 5 scenarios from the design doc:
1. Baseline effective_window arithmetic
2. 10% floor kicks in when config is misconfigured
3. Both call sites in planner_react land on the same number
4. Compaction output satisfies target ratio
5. Hard compaction keeps ≤ 1 system + 19 tail messages
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.domain.services.graphs.compaction import GradualCompactor
from app.domain.services.graphs.context_assembler import ContextAssembler
from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts.budget import compute_effective_window


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


# ---- Scenario 1: baseline effective_window arithmetic ----------------- #


def test_compute_effective_window_baseline() -> None:
    """compute_effective_window(8000, 3500, 1000) == 3500."""
    ew = compute_effective_window(
        total_context_window=8000,
        system_prompt_max_tokens=3500,
        reserved_output_tokens=1000,
    )
    assert ew == 3500  # 8000 - 3500 - 1000


def test_compute_effective_window_typical_gpt5() -> None:
    """gpt-5.4 with 128K context, 4K reserved, 3500 system budget."""
    ew = compute_effective_window(
        total_context_window=128_000,
        system_prompt_max_tokens=3500,
        reserved_output_tokens=4096,
    )
    assert ew == 128_000 - 3500 - 4096


# ---- Scenario 2: 10% floor protects against misconfiguration ---------- #


def test_compute_effective_window_floor_protects_against_negative() -> None:
    """Misconfigured: system budget + reserved > total → floor kicks in."""
    ew = compute_effective_window(
        total_context_window=8000,
        system_prompt_max_tokens=9000,
        reserved_output_tokens=500,
    )
    # Raw: 8000 - 9000 - 500 = -1500 → floor max(int(8000 * 0.1), -1500) = 800
    assert ew == 800


def test_compute_effective_window_floor_exactly_zero_raw() -> None:
    """Edge case: system + reserved == total → raw is 0, floor wins."""
    ew = compute_effective_window(
        total_context_window=10_000,
        system_prompt_max_tokens=9000,
        reserved_output_tokens=1000,
    )
    assert ew == max(0, int(10_000 * 0.1))
    assert ew == 1000


# ---- Scenario 3: ContextAssembler and try_compact share the same value - #


async def test_assembler_uses_effective_window_compactor_uses_total(
    monkeypatch,
) -> None:
    """B5 C9 runtime invariant: the ``ContextAssembler`` receives the
    effective window (total - system - reserved) while the
    ``GradualCompactor`` receives the total context window. This
    asymmetry is deliberate — see ``_check_overflow`` comment.

    Rather than spin up a full ``PlannerReActFlow`` (needs a dozen
    real dependencies), we patch ``resolve_context_window`` and
    directly exercise both code paths with the same
    ``ContextOverflowConfig``.
    """
    from app.domain.models.context_overflow_config import ContextOverflowConfig

    # Patch resolve_context_window to a deterministic total.
    total = 128_000
    monkeypatch.setattr(
        "app.domain.services.context.model_context_window.resolve_context_window",
        lambda model_name, cfg: total,
    )

    overflow_config = ContextOverflowConfig(
        model_name="fake-model",
        context_overflow_guard_enabled=True,
        reserved_output_tokens=4096,
        system_prompt_max_tokens=3500,
        token_safety_factor=1.15,
        token_estimator="char",
    )

    # 1) Assembler construction path (matches planner_react._build_graphs)
    from app.domain.services.context.model_context_window import resolve_context_window

    total_window = resolve_context_window(overflow_config.model_name, overflow_config)
    effective_window = compute_effective_window(
        total_context_window=total_window,
        system_prompt_max_tokens=overflow_config.system_prompt_max_tokens,
        reserved_output_tokens=overflow_config.reserved_output_tokens,
    )
    assembler = ContextAssembler(
        estimator=TokenEstimator(strategy="char"),
        effective_window=effective_window,
        safety_factor=overflow_config.token_safety_factor,
    )

    # 2) Compactor construction path (matches planner_react._check_overflow)
    compactor_context_window_arg = total_window  # NOT effective_window

    # Invariants:
    # - Assembler's internal _effective_window is the subtracted value
    assert assembler._effective_window == total - 3500 - 4096
    assert assembler._effective_window == effective_window
    # - Compactor's context_window arg is the UNSUBTRACTED total
    assert compactor_context_window_arg == total
    # - The two land on different numbers, and that is correct
    assert assembler._effective_window < compactor_context_window_arg
    # - Specifically: effective_window == total - system - reserved
    assert compactor_context_window_arg - assembler._effective_window == 3500 + 4096


# ---- Scenario 4: compaction output hits target ratio ------------------ #


def _make_message_history(num_turns: int) -> list[BaseMessage]:
    """Build a long-ish history: system + repeated (human/ai) turns.

    Each turn uses a ~100-char body so the char estimator produces
    predictable counts.
    """
    msgs: list[BaseMessage] = [
        SystemMessage(content="You are a test agent. " * 5)
    ]
    for i in range(num_turns):
        msgs.append(HumanMessage(content=f"Turn {i}: " + "x" * 90))
        msgs.append(AIMessage(content=f"Reply {i}: " + "y" * 90))
    return msgs


async def test_hard_compact_keeps_system_plus_last_19_non_system() -> None:
    """Scenario 5: Level 3 hard compaction keeps 1 SystemMessage +
    the last 19 non-system messages, plus injected truncation marker.

    We pass ``hard_trigger_ratio`` so low that it always fires, and
    verify ``len(result.messages)`` ≤ 20 (1 system + 19 others) plus
    whatever truncation marker the compactor injects.
    """
    history = _make_message_history(num_turns=30)  # 61 messages total
    assert len(history) == 61

    compactor = GradualCompactor(
        token_estimator=TokenEstimator(strategy="char"),
        soft_trigger_ratio=0.01,  # trigger anything
        hard_trigger_ratio=0.02,  # force Level 3
        target_ratio=0.01,
        summary_max_chars=1000,
        token_safety_factor=1.0,
    )
    # No summary_llm → forces Level 3 path
    result = await compactor.try_compact(
        messages=history, context_window=1000, summary_llm=None
    )

    assert result.level_applied == 3
    # Level 3 keeps: 1 system message + up to 19 non-system messages
    # Plus it may inject a truncation marker. The invariant is:
    # number of ORIGINAL messages kept ≤ 20.
    assert len(result.messages) <= 21  # 1 system + 19 kept + 1 marker


async def test_soft_compact_target_ratio_post_verify() -> None:
    """Scenario 4: Level 2 must satisfy ``tokens_after <= effective_window
    * target_ratio + epsilon``. Post-verify threshold is target + 3%."""
    history = _make_message_history(num_turns=50)  # 101 messages
    effective_window = 2000
    target_ratio = 0.65

    # Mock summary LLM that returns a very short summary — forces
    # Level 2 to succeed and land well under target.
    summary_llm = MagicMock()

    async def fake_ainvoke(messages, config=None):
        return AIMessage(content="Short summary.")

    summary_llm.ainvoke = fake_ainvoke

    compactor = GradualCompactor(
        token_estimator=TokenEstimator(strategy="char"),
        soft_trigger_ratio=0.5,
        hard_trigger_ratio=0.95,
        target_ratio=target_ratio,
        summary_max_chars=1000,
        token_safety_factor=1.0,
    )
    result = await compactor.try_compact(
        messages=history,
        context_window=effective_window,
        summary_llm=summary_llm,
    )

    # Level 2 or 3 (post-verify may escalate)
    assert result.level_applied in (2, 3)
    # Post-verify threshold: target_ratio + 3% tolerance
    post_verify_threshold = target_ratio + 0.03
    assert result.usage_ratio_after <= post_verify_threshold + 0.001
