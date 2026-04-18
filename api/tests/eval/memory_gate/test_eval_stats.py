"""M2-PR1: tests for eval harness statistical helpers.

Intentionally unit tests (no LLM, no network) so they run in the default
CI suite. The ``test_memory_gate_eval`` sibling is ``pytest.mark.slow``
and opts in only when ``EVAL_MEMORY_GATE_LLM`` is set.
"""
from __future__ import annotations

import math

import pytest

from tests.eval.memory_gate.stats import (
    cohens_kappa,
    wilson_ci,
    wilson_ci_lower,
)


# ---- Wilson score interval --------------------------------------------- #


class TestWilsonCI:
    def test_zero_n_returns_full_interval(self) -> None:
        """N=0 means no data → no confidence → the full [0, 1] interval."""
        assert wilson_ci(0, 0) == (0.0, 1.0)

    def test_perfect_success_lower_bound_strictly_below_one(self) -> None:
        """k=n=10 → point estimate 1.0, but Wilson lower bound must be
        strictly < 1.0 (we're not 100% sure the true rate is 1.0 after
        only 10 trials)."""
        lower, upper = wilson_ci(10, 10)
        assert lower < 1.0
        assert upper == 1.0

    def test_zero_successes_lower_bound_is_zero(self) -> None:
        """k=0 → lower bound clamps to 0.0."""
        lower, upper = wilson_ci(0, 10)
        assert lower == 0.0
        assert upper > 0.0
        assert upper < 1.0

    def test_50_50_centered_around_half(self) -> None:
        """Classic coin flip: 50/100 successes → interval ~symmetric around 0.5."""
        lower, upper = wilson_ci(50, 100)
        assert 0.40 < lower < 0.45
        assert 0.55 < upper < 0.60

    def test_m2_hard_gate_target_precision(self) -> None:
        """M2 acceptance gate: precision Wilson lower bound >= 0.70.

        Sanity-check that 80/100 keeps (point = 0.80) clears the 0.70
        lower-bound gate, while 70/100 (point = 0.70) does NOT (because
        the lower bound will be strictly below 0.70).
        """
        clears_lower = wilson_ci_lower(80, 100)
        fails_lower = wilson_ci_lower(70, 100)
        assert clears_lower > 0.70, (
            f"80/100 should clear the 0.70 hard gate; got {clears_lower}"
        )
        assert fails_lower < 0.70, (
            f"70/100 should NOT clear 0.70 (point = lower bound only when "
            f"n→∞); got {fails_lower}"
        )

    def test_reference_value_against_statsmodels(self) -> None:
        """Against a reference Wilson 95% CI table:

        For k=75, n=100, z=1.9599640 the interval is approximately
        (0.65697, 0.82455) — matches
        ``statsmodels.stats.proportion.proportion_confint(75, 100, method='wilson')``.
        Tolerance 1e-3 allows for z-constant rounding across libraries.
        """
        lower, upper = wilson_ci(75, 100)
        assert math.isclose(lower, 0.65697, abs_tol=1e-3), (
            f"lower bound drifted: {lower}"
        )
        assert math.isclose(upper, 0.82455, abs_tol=1e-3), (
            f"upper bound drifted: {upper}"
        )

    def test_raises_on_invalid_successes(self) -> None:
        with pytest.raises(ValueError):
            wilson_ci(11, 10)  # k > n
        with pytest.raises(ValueError):
            wilson_ci(-1, 10)
        with pytest.raises(ValueError):
            wilson_ci(0, -1)


# ---- Cohen's kappa ----------------------------------------------------- #


class TestCohensKappa:
    def test_perfect_agreement_is_one(self) -> None:
        """a == b on every item → kappa = 1.0."""
        labels = ["keep", "drop", "keep", "drop", "keep"]
        assert cohens_kappa(labels, labels) == 1.0

    def test_chance_level_agreement_exact_zero(self) -> None:
        """Constructed example with po exactly equal to pe → kappa = 0.

        20 items. A keeps positions 0-9. B keeps positions 0-4 and 15-19.
        So keep-overlap = {0..4} = 5, drop-overlap = {10..14} = 5.
        Total agreement = 10/20 = 0.5.
        Marginal: P(a=keep) = P(b=keep) = 0.5 → pe = 0.5.
        Therefore kappa = (po - pe) / (1 - pe) = 0/0.5 = 0 exactly.
        """
        labels_a = ["keep"] * 10 + ["drop"] * 10
        labels_b = ["keep"] * 5 + ["drop"] * 10 + ["keep"] * 5
        kappa = cohens_kappa(labels_a, labels_b)
        assert kappa == 0.0, (
            f"chance-level construction should yield kappa=0 exactly; got {kappa}"
        )

    def test_m2_control_set_gate(self) -> None:
        """M2 design: kappa >= 0.7 → rubric stable. Simulate a 30-sample
        control set with one disagreement → kappa should clear 0.7."""
        labels_a = ["keep"] * 15 + ["drop"] * 15
        labels_b = ["keep"] * 15 + ["drop"] * 14 + ["keep"]
        # 1 disagreement out of 30
        kappa = cohens_kappa(labels_a, labels_b)
        assert kappa > 0.70, f"single-disagreement kappa should clear 0.70; got {kappa}"

    def test_perfect_disagreement_is_negative(self) -> None:
        """Binary labels inverted → kappa < 0 (worse than chance)."""
        labels_a = ["keep", "keep", "drop", "drop", "keep"]
        labels_b = ["drop", "drop", "keep", "keep", "drop"]
        assert cohens_kappa(labels_a, labels_b) < 0

    def test_all_same_label_edge_case(self) -> None:
        """Both annotators pick the same single label everywhere →
        pe = 1.0; convention matches sklearn (kappa = 1.0 when po = 1.0)."""
        labels = ["keep"] * 10
        assert cohens_kappa(labels, labels) == 1.0

    def test_raises_on_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            cohens_kappa(["keep"], ["keep", "drop"])

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError):
            cohens_kappa([], [])
