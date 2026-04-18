"""M2-PR1: statistical helpers for the memory gate eval harness.

Two building blocks the eval suite needs but nothing else in the repo
uses yet:

- ``wilson_ci_lower(k, n)`` / ``wilson_ci(k, n)`` — Wilson score interval
  for a binomial proportion (precision). M2 uses the **lower bound** as a
  hard gate: point precision can fluctuate on N=100 samples, but a lower
  bound captures "what precision can we claim with 95% confidence".
- ``cohens_kappa(a, b)`` — annotator agreement between two labelings of
  the same samples. Used for the weekly control-set drift check
  (kappa >= 0.7 → rubric is stable; below → rubric needs tightening).

Both implemented from scratch to avoid pulling in ``statsmodels`` / ``scipy``
just for the eval harness. Formulas are standard; see module tests for
reference values against published tables.

**Important scope note**: these helpers live under ``tests/eval/`` because
they're eval-harness infrastructure, not production code. They must not
be imported from ``app/``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

# ---- Wilson score interval --------------------------------------------- #

# 95% two-sided confidence → z-score = 1.9599639845...
# Hard-coded so we don't pull scipy.stats just for this one constant.
_Z_95 = 1.959963984540054


def wilson_ci(
    successes: int,
    n: int,
    *,
    z: float = _Z_95,
) -> tuple[float, float]:
    """Return the (lower, upper) Wilson score interval for a binomial
    proportion.

    Args:
        successes: count of positive outcomes (0 <= successes <= n)
        n: total sample size (n >= 0)
        z: z-score for the desired confidence; default 1.9600 (95% two-sided)

    Returns:
        ``(lower, upper)`` both in ``[0.0, 1.0]``. When ``n == 0`` returns
        ``(0.0, 1.0)`` (no information). This matches the convention used
        by ``statsmodels.stats.proportion.proportion_confint(method='wilson')``.

    Reference formula:

        p_hat = k / n
        denom = 1 + z^2/n
        center = (p_hat + z^2/(2n)) / denom
        halfwidth = z * sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2)) / denom
        (lower, upper) = (center - halfwidth, center + halfwidth)
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(
            f"successes must be in [0, n]; got successes={successes}, n={n}"
        )
    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n
    z_sq = z * z
    denom = 1.0 + z_sq / n
    center = (p_hat + z_sq / (2.0 * n)) / denom
    halfwidth = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z_sq / (4.0 * n * n))
    ) / denom
    lower = max(0.0, center - halfwidth)
    upper = min(1.0, center + halfwidth)

    # Boundary-exact convention (matches ``statsmodels.stats.proportion``):
    # when all trials succeed, the upper bound is exactly 1.0; when none
    # succeed, the lower is exactly 0.0. Without this, floating-point
    # drift in ``center + halfwidth`` can land at 0.9999... for k=n=10,
    # which is technically correct (epsilon-below-1.0) but surprising
    # for downstream code that checks ``upper == 1.0``.
    if successes == n:
        upper = 1.0
    if successes == 0:
        lower = 0.0

    return (lower, upper)


def wilson_ci_lower(successes: int, n: int, *, z: float = _Z_95) -> float:
    """Convenience wrapper: return just the lower bound.

    M2 hard gate: ``wilson_ci_lower(tp, tp+fp) >= 0.70``.
    """
    return wilson_ci(successes, n, z=z)[0]


# ---- Cohen's kappa ----------------------------------------------------- #


def cohens_kappa(
    labels_a: Sequence[str | bool | int],
    labels_b: Sequence[str | bool | int],
) -> float:
    """Return Cohen's kappa for two sequences of categorical labels.

    Args:
        labels_a: first annotator's labels (same length as ``labels_b``)
        labels_b: second annotator's labels

    Returns:
        kappa in ``[-1.0, 1.0]``. Interpretation:
        - 1.0   = perfect agreement
        - 0.0   = agreement at chance level
        - < 0   = worse than chance (annotators systematically disagree)

    Raises:
        ValueError: if the sequences are empty or different lengths.

    Formula (categorical, binary-or-multiclass uniform weights):

        po = fraction of items where a[i] == b[i]  (observed agreement)
        pe = sum over categories c of  P(a=c) * P(b=c)
              where P(a=c) = count of c in labels_a / n
        kappa = (po - pe) / (1 - pe)

        If pe == 1.0 (both annotators used exactly one label for everything),
        define kappa = 1.0 if po == 1.0 else 0.0. This matches scikit-learn's
        ``cohen_kappa_score`` edge-case handling.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"labels_a and labels_b must have same length; "
            f"got {len(labels_a)} vs {len(labels_b)}"
        )
    n = len(labels_a)
    if n == 0:
        raise ValueError("cannot compute kappa on empty label sequences")

    # Observed agreement
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = observed / n

    # Expected agreement under independence
    categories = set(labels_a) | set(labels_b)
    pe = 0.0
    for c in categories:
        p_a = sum(1 for x in labels_a if x == c) / n
        p_b = sum(1 for x in labels_b if x == c) / n
        pe += p_a * p_b

    if pe >= 1.0:
        # Edge case: both annotators used the same single label everywhere.
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)
