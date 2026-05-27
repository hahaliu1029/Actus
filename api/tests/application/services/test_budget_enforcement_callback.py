"""C2 PR-6 Task 6.2 — BudgetEnforcementCallback unit tests.

Pins the §14.3 #6 + §14.3.1 contract:

- Cumulative USD across ``on_llm_end`` invocations; trip on ``>= cap``.
- Trip → ``runner.request_stop(StopReason.TOKEN_BUDGET)`` exactly once
  (idempotent post-trip; subsequent calls no-op).
- Missing ``usage_metadata`` → silent skip (no cost, no trip).
- Pricing exceptions are swallowed with a WARNING (don't crash the LLM path).
- Exact-cap equality trips (predicate is ``>=``).

The callback is modeled on the live B4 ``CostCallbackHandler.on_llm_end`` hook
(api/app/domain/services/cost_callback_handler.py:375) — same LangChain entry
point, but with the cumulative-budget side-effect instead of a CostRecord
INSERT.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Mapping, Optional
from unittest.mock import MagicMock

import pytest

from app.application.services.budget_enforcement_callback import (
    BudgetEnforcementCallback,
)
from app.application.services.coordinator_child_runner import StopReason


pytestmark = pytest.mark.anyio


# ── Helpers ──────────────────────────────────────────────────────────────────


def _response_with_usage(usage: Optional[Mapping[str, Any]]) -> Any:
    """LangChain LLMResult-shaped object exposing ``.usage_metadata`` at the
    top level — the "test-fallback" shape supported by the callback's
    two-path ``_extract_usage_metadata`` helper.

    The canonical production path is
    ``response.generations[0][0].message.usage_metadata`` — pinned by
    ``test_extracts_usage_from_deep_langchain_path`` below. This simpler
    shape stays useful because the bulk of the cumulative-budget /
    idempotency / pricing-error fan-out is shape-agnostic, and the
    SimpleNamespace mocks are cheaper than building a full LLMResult per
    test.
    """
    return SimpleNamespace(usage_metadata=usage)


def _response_with_deep_usage(
    usage: Optional[Mapping[str, Any]],
) -> Any:
    """Real LangChain ``LLMResult`` shape:
    ``response.generations[0][0].message.usage_metadata``.

    Pins the production-path read so a future refactor that drops the
    deep-path branch from the helper would surface here. The top-level
    ``usage_metadata`` attribute is deliberately set to ``None`` so the
    helper MUST traverse generations[0][0].message to find the data.
    """
    gen = MagicMock()
    gen.message = MagicMock()
    gen.message.usage_metadata = usage
    response = MagicMock()
    response.usage_metadata = None  # force deep-path traversal
    response.generations = [[gen]]
    return response


def _mk_runner() -> MagicMock:
    runner = MagicMock()
    runner.request_stop = MagicMock()
    return runner


# ── Tests ────────────────────────────────────────────────────────────────────


async def test_accumulates_cost_under_cap() -> None:
    """Single on_llm_end call below cap → request_stop NOT called."""
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=Decimal("0.25"))

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))

    pricing.compute_cost.assert_called_once()
    runner.request_stop.assert_not_called()


async def test_trips_when_cumulative_exceeds_cap() -> None:
    """Two on_llm_end calls; the second pushes total past cap → request_stop fires."""
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(
        side_effect=[Decimal("0.60"), Decimal("0.55")],
    )

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))
    runner.request_stop.assert_not_called()  # 0.60 < 1.0

    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))
    # 0.60 + 0.55 = 1.15 ≥ 1.0 → trip
    runner.request_stop.assert_called_once_with(StopReason.TOKEN_BUDGET)


async def test_idempotent_after_trip() -> None:
    """Post-trip, additional on_llm_end calls do not re-invoke request_stop."""
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=Decimal("2.0"))

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    # First call: 2.0 ≥ 1.0 → trip
    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))
    assert runner.request_stop.call_count == 1

    # Second call post-trip: should be a no-op (no pricing call, no stop call)
    pricing.compute_cost.reset_mock()
    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))
    pricing.compute_cost.assert_not_called()
    assert runner.request_stop.call_count == 1

    # Third call: still idempotent
    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))
    pricing.compute_cost.assert_not_called()
    assert runner.request_stop.call_count == 1


async def test_missing_usage_metadata_is_skip() -> None:
    """``usage_metadata=None`` → no cost added, no trip."""
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=Decimal("99.0"))

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=0.01, pricing=pricing,
    )
    await cb.on_llm_end(_response_with_usage(None))

    pricing.compute_cost.assert_not_called()
    runner.request_stop.assert_not_called()


async def test_response_without_usage_metadata_attr_is_skip() -> None:
    """A response missing the ``usage_metadata`` attribute is treated like None.

    Defensive: the LangChain LLMResult / AIMessage shapes vary across
    providers; ``getattr(..., default=None)`` guards against the missing-attr
    path the way the B4 ``_extract_usage_metadata`` helper does.
    """
    runner = _mk_runner()
    pricing = MagicMock()

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    await cb.on_llm_end(object())  # no usage_metadata attribute

    pricing.compute_cost.assert_not_called()
    runner.request_stop.assert_not_called()


async def test_pricing_exception_is_swallowed_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """pricing.compute_cost raising → cumulative not incremented; warning logged."""
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(side_effect=ValueError("bad pricing"))

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    with caplog.at_level(logging.WARNING):
        await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))

    runner.request_stop.assert_not_called()
    # At least one warning record from this callback present.
    assert any(
        "BudgetEnforcementCallback" in r.getMessage()
        or "pricing" in r.getMessage().lower()
        for r in caplog.records
    ), f"Expected pricing-exception warning; got: {[r.getMessage() for r in caplog.records]}"


async def test_exact_cap_trips() -> None:
    """Cumulative reaching cap exactly (e.g. 1.0 == 1.0) trips — predicate is ``>=``."""
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=Decimal("1.0"))

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))

    runner.request_stop.assert_called_once_with(StopReason.TOKEN_BUDGET)


async def test_pricing_returns_none_is_skip() -> None:
    """pricing.compute_cost returning None (unpriced model) → no cost added, no trip.

    Mirrors the static_pricing.compute_cost contract: ``None`` return signals
    "no usage data" or "no price entry" — must not be treated as zero-cost
    nor crash arithmetic.
    """
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=None)

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=0.01, pricing=pricing,
    )
    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))

    pricing.compute_cost.assert_called_once()
    runner.request_stop.assert_not_called()


async def test_float_pricing_return_supported() -> None:
    """pricing.compute_cost may return float (callers in tests) — should still trip."""
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=1.5)

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    await cb.on_llm_end(_response_with_usage({"input_tokens": 100}))

    runner.request_stop.assert_called_once_with(StopReason.TOKEN_BUDGET)


# ── Canonical LLMResult shape ────────────────────────────────────────────────
#
# These tests pin the production path the callback MUST use to read
# ``usage_metadata``. A real LangChain ``LLMResult`` does NOT carry a
# top-level ``.usage_metadata`` attribute — the canonical reader (matching
# ``CostCallbackHandler._extract_usage_metadata`` at
# ``app/domain/services/cost_callback_handler.py:536``) traverses
# ``response.generations[0][0].message.usage_metadata``. Without the
# two-path helper, every ``on_llm_end`` in production would extract
# ``None`` → return early → cap NEVER trips.


async def test_extracts_usage_from_deep_langchain_path() -> None:
    """Real ``LLMResult`` shape:
    ``response.generations[0][0].message.usage_metadata``.

    The top-level ``response.usage_metadata`` is explicitly ``None`` so the
    helper MUST descend through ``generations`` to find the data — pins the
    production-path read.
    """
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=2.0)

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    response = _response_with_deep_usage(
        {"input_tokens": 100, "output_tokens": 50},
    )

    await cb.on_llm_end(response)

    # 2.0 ≥ 1.0 → trip via deep-path extraction.
    pricing.compute_cost.assert_called_once_with(
        {"input_tokens": 100, "output_tokens": 50},
    )
    runner.request_stop.assert_called_once_with(StopReason.TOKEN_BUDGET)


async def test_deep_path_with_none_message_falls_back_to_top_level() -> None:
    """If ``generations[0][0].message`` is None but the top-level
    ``response.usage_metadata`` is populated, the helper should still find
    the usage data via the fallback branch.

    This guards against the corner case where a half-built ``LLMResult``
    (e.g. an adapter that constructs ``generations`` without populating
    ``.message``) carries usage at the top level instead.
    """
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=Decimal("1.0"))

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    gen = MagicMock()
    gen.message = None  # explicit None on the deep path
    response = MagicMock()
    response.generations = [[gen]]
    response.usage_metadata = {"input_tokens": 50}  # fallback shape

    await cb.on_llm_end(response)

    pricing.compute_cost.assert_called_once_with({"input_tokens": 50})
    runner.request_stop.assert_called_once_with(StopReason.TOKEN_BUDGET)


async def test_deep_path_with_empty_generations_falls_back_to_top_level() -> None:
    """``response.generations = []`` raises IndexError on ``[0][0]`` → the
    helper catches it and falls back to the top-level shape.
    """
    runner = _mk_runner()
    pricing = MagicMock()
    pricing.compute_cost = MagicMock(return_value=Decimal("1.0"))

    cb = BudgetEnforcementCallback(
        runner=runner, max_token_cost_usd=1.0, pricing=pricing,
    )
    response = MagicMock()
    response.generations = []  # IndexError on [0][0]
    response.usage_metadata = {"input_tokens": 33}

    await cb.on_llm_end(response)

    pricing.compute_cost.assert_called_once_with({"input_tokens": 33})
    runner.request_stop.assert_called_once_with(StopReason.TOKEN_BUDGET)
