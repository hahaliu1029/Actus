"""[C2 PR-6 §14.3 #6 + §14.3.1] BudgetEnforcementCallback for child token cost.

Hook into LangChain ``on_llm_end`` (live B4 ``CostCallbackHandler`` uses the
same hook — see ``app/domain/services/cost_callback_handler.py:375``). Each
call extracts the response's ``usage_metadata`` via the same two-path reader
the B4 handler uses (deep ``response.generations[0][0].message.usage_metadata``
plus a top-level fallback for simplified unit-test mocks), computes its USD
cost via the injected pricing object, accumulates it, and once the running
total crosses ``max_token_cost_usd`` calls
``runner.request_stop(StopReason.TOKEN_BUDGET)``.

The runner's ``request_stop`` is first-wins idempotent, and the spec §14.3.1
budget finalizer routes any TOKEN_BUDGET / WALLCLOCK_BUDGET stop into
RESULT_READY(NEEDS_AUTHORIZATION, reason=budget_exhausted) so the parent can
choose to raise the cap and resume.

v1 overshoot tolerance (spec §14.3 #6 r2): enforcement runs per LLM call, not
per streaming chunk — once a call is in flight we cannot abort it (the live
LLM client does not expose a mid-stream cancel). The accepted v1 cost is "one
LLM call worth of tokens above cap" before the next call is blocked.

Pricing contract: ``pricing.compute_cost(usage_metadata) -> Optional[float|Decimal]``.
None return means "no usage data" or "model not priced"; both are treated as
"no cost added" rather than as zero-cost (silent zero-cost would let an
unpriced provider drain forever). Production wiring closes the
``(model, provider)`` over the static_pricing helpers — see
``app/domain/services/pricing/static_pricing.py::compute_cost``.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

if TYPE_CHECKING:
    from app.application.services.coordinator_child_runner import (
        CoordinatorChildRunner,
    )


logger = logging.getLogger(__name__)


def _extract_usage_metadata(response: Any) -> Optional[Mapping[str, Any]]:
    """Read ``usage_metadata`` from a LangChain ``LLMResult``.

    Two-path extraction matching the canonical reader at
    ``app/domain/services/cost_callback_handler.py::CostCallbackHandler._extract_usage_metadata``:

    1. **Production path** — ``response.generations[0][0].message.usage_metadata``.
       This is the shape a real LangChain ``LLMResult`` exposes when the
       OpenAI / Anthropic adapters surface usage telemetry; the top-level
       ``response`` object does NOT carry ``.usage_metadata``.
    2. **Test-fallback path** — ``response.usage_metadata``. The existing
       unit tests for this callback build a ``SimpleNamespace(usage_metadata=…)``
       directly; we honour that simplified shape so old tests keep passing.

    Returns ``None`` if neither path yields a value — callers treat that as a
    silent skip (no cost added, no trip).

    The deep path is tried FIRST so production wiring (where the test-fallback
    attribute also exists on some mocks but carries the wrong data) reads from
    the authoritative location.
    """
    try:
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg is not None:
            um = getattr(msg, "usage_metadata", None)
            if um is not None:
                return um
    except (IndexError, AttributeError, TypeError):
        # ``response`` has no ``generations`` / it's not subscriptable / the
        # nested shape doesn't match — fall through to the simple attr path.
        pass
    return getattr(response, "usage_metadata", None)


class PricingFn(Protocol):
    """Pricing surface the budget callback depends on.

    A thin wrapper around ``app.domain.services.pricing.static_pricing.compute_cost``
    that pre-binds the per-call ``price`` dict (since static_pricing requires
    both ``usage_metadata`` AND ``price``). The callback does NOT do model
    lookup itself — that's the caller's job at construction time, so swapping
    pricing tables / cost regimes is a one-line wiring change rather than a
    callback rewrite.
    """

    def compute_cost(
        self, usage_metadata: Any,
    ) -> Optional[Decimal | float]:  # noqa: D401 — Protocol signature
        ...


class BudgetEnforcementCallback(AsyncCallbackHandler):
    """Accumulates USD across LLM calls; trips ``request_stop`` at cap.

    [C2b budget D8] MUST subclass ``AsyncCallbackHandler``: the LangChain
    dispatch path is bare-class-hostile at THREE points — ``ahandle_event``
    reads ``handler.run_inline`` outside any try (AttributeError raises),
    the per-handler dispatch probes ``handler.ignore_llm``, and the call
    passes ``run_id=``/``parent_run_id=``/``tags=`` kwargs (the latter two
    swallowed by the manager's ``except Exception``). Whichever point fires,
    a bare class never bills on the real dispatch path. ``raise_error``
    stays at the inherited ``False`` default — handler exceptions must not
    kill the LLM run (INV-B3; double-insured by the pricing try/except
    below).

    Idempotent: after the first trip, subsequent ``on_llm_end`` calls are
    no-ops. The runner's own ``request_stop`` is also idempotent (first-wins
    on ``_stop_reason``), so even if the trip-check raced with a parent cancel
    the recorded stop reason remains the first one set.
    """

    def __init__(
        self,
        *,
        runner: "CoordinatorChildRunner",
        max_token_cost_usd: float,
        pricing: PricingFn,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._cap: float = float(max_token_cost_usd)
        self._pricing = pricing
        self._cumulative_usd: float = 0.0
        self._tripped: bool = False

    @property
    def cumulative_usd(self) -> float:
        """[C2b budget D5] Read-only running USD total — consumed by
        ``CoordinatorChildRunner._build_budget_evidence`` so the parent's
        NEEDS_AUTHORIZATION envelope can carry the observed spend."""
        return self._cumulative_usd

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """LangChain ``on_llm_end`` hook — accumulate cost; trip at cap.

        [C2b budget D8] ``run_id`` is DELIBERATELY ``UUID | None = None`` —
        an LSP-legal precondition weakening vs the base class's required
        ``run_id``: this callback keys no state by run_id (no ``_pending``
        map like CostCallbackHandler's), the production manager always
        passes one, and the existing 12 unit tests call
        ``await cb.on_llm_end(response)`` single-arg.

        Behavior:
        - Already tripped → return immediately.
        - ``usage_metadata`` absent / None on BOTH the deep
          ``response.generations[0][0].message.usage_metadata`` (canonical
          LangChain ``LLMResult`` path) AND the top-level
          ``response.usage_metadata`` (test-fallback) → silent skip
          (no cost data; cannot bill). Extraction is delegated to the
          module-level ``_extract_usage_metadata`` helper, which mirrors
          ``CostCallbackHandler._extract_usage_metadata`` in
          ``app/domain/services/cost_callback_handler.py``.
        - ``pricing.compute_cost`` raises → swallow with WARNING and return
          (a broken pricing path must not crash the LLM run; the watchdog
          backstop still trips on wallclock).
        - ``pricing.compute_cost`` returns None → treat as "no priced cost
          available"; cumulative unchanged, no trip.
        - Cumulative ≥ cap → log INFO, call
          ``runner.request_stop(StopReason.TOKEN_BUDGET)``, set ``_tripped``.

        The cap predicate is ``>=`` (exact equality also trips) — matches the
        spec wording and avoids a corner case where Decimal-equality lands
        exactly on the cap and a strict ``>`` would let one more call through.
        """
        if self._tripped:
            return

        usage = _extract_usage_metadata(response)
        if usage is None:
            return

        try:
            cost = self._pricing.compute_cost(usage)
        except Exception as exc:  # noqa: BLE001 — pricing errors must not kill the run
            logger.warning(
                "BudgetEnforcementCallback: pricing.compute_cost raised; "
                "skipping this call's cost (cumulative=%.4f, cap=%.4f): %s",
                self._cumulative_usd, self._cap, exc,
            )
            return

        if cost is None:
            # Unpriced model / missing usage detail — honest skip rather than
            # silent zero-cost (which would let an unpriced provider drain
            # forever past the nominal cap).
            return

        # Coerce to float for the accumulator. The v1 overshoot tolerance
        # (one LLM call) makes float rounding non-load-bearing; Decimal would
        # add complexity here for no measured precision benefit.
        self._cumulative_usd += float(cost)

        if self._cumulative_usd >= self._cap:
            # Lazy import — avoids the top-level circular with the runner
            # module (which already imports from the listener module, and
            # would import from us once wiring lands in PR-6 Task 6.x).
            from app.application.services.coordinator_child_runner import (
                StopReason,
            )

            logger.info(
                "BudgetEnforcementCallback: budget cap reached "
                "(cumulative=%.4f, cap=%.4f) → request_stop(TOKEN_BUDGET)",
                self._cumulative_usd, self._cap,
            )
            self._runner.request_stop(StopReason.TOKEN_BUDGET)
            self._tripped = True
