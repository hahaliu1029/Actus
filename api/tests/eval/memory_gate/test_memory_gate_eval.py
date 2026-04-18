"""Memory gate eval harness — M1 visibility + M2 hard gate.

**Off by default** — every test is gated via ``pytest.mark.slow``. Run manually:

    cd api && uv run pytest tests/eval/memory_gate/ -v -m slow

Two-tier assertion model:

1. ``test_gate_precision_baseline`` (M1, carry-over) — point estimate
   precision / recall / F1 printed to stdout. **Never fails**; visibility
   only. Useful when iterating on the prompt: a one-run numeric trace
   without a CI gate blocking the author.
2. ``test_gate_wilson_hard_gate`` (M2 PR-5, new) — asserts the Wilson
   95% CI lower bound on precision clears ``0.70`` at
   ``threshold=0.7`` on the union of core + private datasets. This is
   the M2 acceptance criterion from the labelling rubric §130-140.
   Skipped automatically when the dataset is too small to be meaningful
   (``n < 40`` — see skip-condition comment inline).
3. ``test_gate_adversarial_resistance`` (M2 PR-5, new) — scores the
   adversarial suite separately with a **dual specificity + sensitivity**
   gate. Plain accuracy on a drop-heavy distribution (17/20 drop) would
   pass an always-drop gate at 0.85, so we split: specificity bar
   locks "don't get fooled by deceptive drops"; sensitivity bar locks
   "don't over-reject borderline keeps". Both must clear.

All tests skip with an explanatory message when ``EVAL_MEMORY_GATE_LLM``
is unset — the harness has no useful behavior without a real LLM.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.domain.services.memory_gate import (
    MemoryGateClassifier,
    MemoryGateInput,
    filter_kept_decisions,
)
from tests.eval.memory_gate.paths import (
    private_available,
    private_dataset_path,
    synthetic_adversarial_available,
    synthetic_adversarial_path,
    synthetic_dataset_path,
)
from tests.eval.memory_gate.stats import wilson_ci_lower

pytestmark = [pytest.mark.anyio, pytest.mark.slow]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# M1: 25 synthetic + paraphrased. M2 PR-5: 45 core + 20 adversarial in a
# sibling file + optional ≥ 200 private set under ACTUS_EVAL_DATA_DIR.


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_core_samples() -> list[dict]:
    """Load core synthetic + optional private real dataset.

    Adversarial is intentionally excluded here — it has its own test
    (``test_gate_adversarial_resistance``) so a precision dip on
    deliberately deceptive samples doesn't swamp the Wilson CI on the
    representative distribution.
    """
    rows = _load_jsonl(synthetic_dataset_path())
    if private_available():
        rows.extend(_load_jsonl(private_dataset_path()))
    return rows


def _resolve_llm():
    """Pick an LLM for eval based on env var; skip if none configured.

    ``EVAL_MEMORY_GATE_LLM`` = ``chat_llm`` | ``summary_llm`` (the
    same string keys the deployer uses for ``settings.memory_gate_llm``).
    Building the adapter requires a valid ``config.yaml`` — we import
    lazily so the rest of the module stays importable on a bare check.
    """
    key = os.environ.get("EVAL_MEMORY_GATE_LLM")
    if not key:
        pytest.skip(
            "Set EVAL_MEMORY_GATE_LLM=chat_llm|summary_llm to run the gate "
            "eval harness against a real LLM."
        )

    from app.interfaces.service_dependencies import (
        _build_config_snapshot,
        _load_app_config,
    )

    snapshot = _build_config_snapshot(_load_app_config())
    if key == "chat_llm":
        return snapshot.llm
    if key == "summary_llm":
        llm = snapshot.summary_llm or snapshot.llm
        return llm
    pytest.skip(f"EVAL_MEMORY_GATE_LLM={key!r} not recognized")


async def _run_classifier(samples: list[dict], classifier: MemoryGateClassifier):
    inputs = [
        MemoryGateInput(chunk_index=i, text=s["text"])
        for i, s in enumerate(samples)
    ]
    decisions = await classifier.classify(inputs)
    return decisions


def _score(
    samples: list[dict],
    decisions,
    *,
    threshold: float,
) -> tuple[int, int, int, int]:
    """Compute (tp, fp, fn, tn) aligned by chunk_index.

    Production-aligned semantic: a sample is "kept" iff
    ``verdict == 'keep' AND confidence >= threshold``. Pure-confidence
    scoring mis-counts high-confidence drops as false positives; see
    ``tests/eval/memory_gate/tune_threshold.py::_score_at_threshold`` for
    the spike-regression commentary.
    """
    kept = filter_kept_decisions(decisions, threshold=threshold)
    kept_indices = {d.chunk_index for d in kept}
    tp = fp = fn = tn = 0
    for i, s in enumerate(samples):
        predicted_keep = i in kept_indices
        gold_keep = s["expected_verdict"] == "keep"
        if predicted_keep and gold_keep:
            tp += 1
        elif predicted_keep and not gold_keep:
            fp += 1
        elif not predicted_keep and gold_keep:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


async def test_gate_precision_baseline() -> None:
    """Point-estimate precision on the core dataset — M1 visibility only.

    Reports:
    - overall precision / recall / F1 at threshold=0.7 (M1 default)
    - counts of keep/drop predictions vs ground truth
    - per-category confusion (only for predicted-keep rows, since
      category is undefined/ignored for drops)

    Prints to stdout; does not hard-fail. That is the M1 contract — the
    hard gate lives in ``test_gate_wilson_hard_gate``. Keep this test in
    place because the point estimate is the signal the solo author
    watches while iterating on the prompt.
    """
    samples = _load_core_samples()
    assert len(samples) >= 20, f"dataset shrunk unexpectedly: {len(samples)}"

    llm = _resolve_llm()
    classifier = MemoryGateClassifier(llm)
    decisions = await _run_classifier(samples, classifier)

    tp, fp, fn, tn = _score(samples, decisions, threshold=0.7)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Per-category accuracy among predicted-keep
    kept = filter_kept_decisions(decisions, threshold=0.7)
    by_cat: dict[str, tuple[int, int]] = {}
    for d in kept:
        gold_cat = samples[d.chunk_index]["expected_category"]
        t, n = by_cat.get(d.category, (0, 0))
        n += 1
        if d.category == gold_cat:
            t += 1
        by_cat[d.category] = (t, n)

    print("\n" + "=" * 60)
    print(f"memory_gate eval @ threshold=0.7, N={len(samples)}")
    print(f"  precision = {precision:.3f}  (tp={tp}, fp={fp})")
    print(f"  recall    = {recall:.3f}  (tp={tp}, fn={fn})")
    print(f"  F1        = {f1:.3f}")
    print("  category accuracy (among predicted-keep):")
    for cat, (t, n) in sorted(by_cat.items()):
        print(f"    {cat:6s}: {t}/{n}  ({t / n:.2f} accuracy)")
    print("=" * 60)


async def test_gate_wilson_hard_gate() -> None:
    """M2 acceptance criterion — Wilson 95% CI lower on precision ≥ 0.70.

    **This is the gate the M2 ship depends on.** Rubric §130-140 defines
    the union of the public synthetic set AND the private real set at
    ``threshold=0.7`` as the representative distribution. If the lower
    bound drops below 0.70 the gate's promotion pipeline is spraying
    enough junk into prompt memory that we'd rather fail-closed and
    force a prompt-engineering pass.

    Skip conditions:

    1. No LLM configured (handled by ``_resolve_llm``).
    2. **Private dataset missing.** Rubric acceptance is defined on the
       public+private union, not public-only. Without the private set
       we could still compute a Wilson CI on the 45 public rows alone,
       but that CI would represent a synthetic-only distribution, not
       the deployment reality. Claiming "M2 hard gate passed" on a
       synthetic-only result would be a false-green — so we skip. The
       public-only signal is still reported via
       ``test_gate_precision_baseline`` (visibility, not gated).
    3. ``n < 40``: fallback defense even when private exists but is
       too small to meaningfully anchor a 0.70 lower bound.

    Failure mode rationale: we assert on lower bound, NOT point estimate,
    because a point estimate of 0.72 on n=50 with the unlucky split
    (36/50 correct) has Wilson lower ≈ 0.58 — not defensible as "we know
    precision is ≥ 0.70 with 95% confidence". The lower-bound gate forces
    either more data or a tighter gate before shipping.
    """
    if not private_available():
        pytest.skip(
            "M2 hard gate acceptance requires the public + private union "
            f"(rubric §130-140); private set at {private_dataset_path()} "
            "is not populated. Set ACTUS_EVAL_DATA_DIR and run "
            "`cp your-labeled-set.jsonl "
            "$ACTUS_EVAL_DATA_DIR/memory_gate/real/dataset.jsonl` to "
            "enable. Public-only precision is still visible via "
            "test_gate_precision_baseline."
        )

    # File-exists check isn't enough — an empty ``dataset.jsonl`` would
    # pass ``private_available()`` but contribute zero rows, letting the
    # synthetic-only 45 rows sneak past the n<40 fallback and give a
    # false "M2 hard gate cleared" on a public-only distribution. Load
    # the file and require at least one row of private data before the
    # hard gate runs.
    private_rows = _load_jsonl(private_dataset_path())
    if not private_rows:
        pytest.skip(
            f"Private dataset at {private_dataset_path()} exists but "
            f"contains zero rows. Hard gate acceptance needs the public + "
            f"private union (rubric §130-140) — an empty private file "
            f"would degrade the gate to synthetic-only and mislabel the "
            f"result. Populate the file or remove it to fall back to "
            f"test_gate_precision_baseline."
        )

    samples = _load_core_samples()
    if len(samples) < 40:
        pytest.skip(
            f"hard gate needs n >= 40 for Wilson CI to be meaningful; "
            f"got {len(samples)}. Populate ACTUS_EVAL_DATA_DIR private "
            f"set with more rows or wait for PR-5 follow-up dataset growth."
        )

    llm = _resolve_llm()
    classifier = MemoryGateClassifier(llm)
    decisions = await _run_classifier(samples, classifier)
    tp, fp, fn, tn = _score(samples, decisions, threshold=0.7)

    denom = tp + fp
    if denom == 0:
        # Gate dropped every sample — precision undefined. Fail loudly:
        # this means the gate is so conservative that the pipeline is
        # producing zero promotion value, which is also not shippable.
        pytest.fail(
            f"gate dropped every sample ({tp+fp+fn+tn} total, 0 kept); "
            f"recall = 0. Check prompt / threshold / LLM degradation."
        )

    point = tp / denom
    lower = wilson_ci_lower(tp, denom)

    print("\n" + "=" * 60)
    print(f"memory_gate WILSON HARD GATE @ threshold=0.7, N={len(samples)}")
    print(f"  precision point = {point:.3f}  (tp={tp}, fp={fp})")
    print(f"  Wilson 95% lower= {lower:.3f}")
    print(f"  hard gate target= 0.700")
    print("=" * 60)

    assert lower >= 0.70, (
        f"M2 hard gate failed: Wilson 95% lower bound on precision = "
        f"{lower:.3f} < 0.70 target. "
        f"tp={tp} fp={fp} fn={fn} tn={tn} at threshold=0.7 on N={len(samples)}. "
        f"Run `uv run python -m tests.eval.memory_gate.tune_threshold "
        f"--llm chat_llm --sweep 0.5:0.9:0.05` to see whether a higher "
        f"threshold clears the gate, or iterate on the prompt in "
        f"app/domain/services/memory_gate.py."
    )


async def test_gate_adversarial_resistance() -> None:
    """Adversarial suite — dual specificity + sensitivity gates (M2 PR-5).

    Loads ``synthetic/adversarial.jsonl`` — 20 hand-crafted samples
    designed to fool the gate (task-local masquerading as rules,
    hypotheticals, agent-output mimicry, retractions, prompt-injection
    attempts, and borderline keeps).

    Why two assertions instead of accuracy: the suite is drop-heavy
    (17/20 drop, 3/20 keep). Plain accuracy would let an "always-drop"
    gate score 17/20 = 0.85 and pass, even though that gate has zero
    recall on borderline keeps. Split the bars:

    - **specificity** = TN / (TN + FP) — fraction of deceptive drops
      the gate correctly rejected. Locks "don't get fooled."
    - **sensitivity** = TP / (TP + FN) — fraction of borderline keeps
      the gate correctly kept. Locks "don't over-reject."

    Both must clear to pass. An over-conservative gate fails sensitivity;
    a fooled gate fails specificity.

    Threshold is 0.7 same as the main gate — we don't tune per suite.

    Skip conditions: no LLM configured, or adversarial file missing
    (someone deleted it locally during bisection).
    """
    if not synthetic_adversarial_available():
        pytest.skip(
            f"adversarial suite missing at {synthetic_adversarial_path()} — "
            f"PR-5 shipped it; did someone delete it locally?"
        )

    samples = _load_jsonl(synthetic_adversarial_path())
    assert len(samples) >= 15, (
        f"adversarial suite shrunk unexpectedly: {len(samples)}"
    )

    llm = _resolve_llm()
    classifier = MemoryGateClassifier(llm)
    decisions = await _run_classifier(samples, classifier)
    tp, fp, fn, tn = _score(samples, decisions, threshold=0.7)
    total = tp + fp + fn + tn

    total_drops = tn + fp
    total_keeps = tp + fn
    specificity = tn / total_drops if total_drops else 1.0
    sensitivity = tp / total_keeps if total_keeps else 1.0

    # Bars anchored to the PR-5 17/20 drop — 3/20 keep composition:
    # specificity >= 14/17 ≈ 0.824 → gate was fooled on ≤ 3 deceptive drops.
    # sensitivity >= 2/3 ≈ 0.667 → gate over-rejected ≤ 1 borderline keep.
    # Stored as pre-computed decimals so dataset growth doesn't silently
    # weaken the test — bump explicitly when confidence grows.
    MIN_SPECIFICITY = 14 / 17
    MIN_SENSITIVITY = 2 / 3

    print("\n" + "=" * 60)
    print(f"memory_gate ADVERSARIAL @ threshold=0.7, N={total}")
    print(
        f"  specificity = {specificity:.3f}  (tn={tn}/{total_drops})"
        f"  ← gate resists deceptive drops (min {MIN_SPECIFICITY:.3f})"
    )
    print(
        f"  sensitivity = {sensitivity:.3f}  (tp={tp}/{total_keeps})"
        f"  ← gate catches borderline keeps (min {MIN_SENSITIVITY:.3f})"
    )
    print(f"  false positives  = {fp}  (gate KEPT a deceptive drop)")
    print(f"  false negatives  = {fn}  (gate DROPPED a borderline keep)")
    print("=" * 60)

    assert specificity >= MIN_SPECIFICITY, (
        f"adversarial specificity = {specificity:.3f} < "
        f"{MIN_SPECIFICITY:.3f} target. Gate was fooled into keeping "
        f"{fp} deceptive drop(s) out of {total_drops}. Inspect which "
        f"a* samples tripped the gate (task-local, hypothetical, "
        f"injection, retraction, agent-mimicry)."
    )
    assert sensitivity >= MIN_SENSITIVITY, (
        f"adversarial sensitivity = {sensitivity:.3f} < "
        f"{MIN_SENSITIVITY:.3f} target. Gate over-rejected {fn} "
        f"borderline keep(s) out of {total_keeps}. Inspect a17-a19 "
        f"(team-stack fact, contrastive toolchain, nuanced language "
        f"preference). An always-drop gate fails here even if it "
        f"scores high on plain accuracy."
    )
