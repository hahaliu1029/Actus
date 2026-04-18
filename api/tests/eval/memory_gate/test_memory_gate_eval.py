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
3. ``test_gate_adversarial_resistance`` (M2 PR-5 + gap-step-3,
   **parametrized per suite**) — loads each of the five suite files
   from ``synthetic/adversarial/`` (design doc §621: ``ambiguous``,
   ``sarcasm``, ``temporary``, ``contradictions``, ``testing``) and
   applies a "≤ 1 mis-classification" accuracy bar scaled to the
   suite's current sample count. The pre-step-3 single-gate version
   used specificity+sensitivity on the 20-row union; the per-suite
   split surfaces WHICH pattern class failed when the gate regresses,
   and provides the structural hook for the design §634 Wilson CI
   ≥ 0.80 per-suite target (activates once gap #2 grows each suite
   to 20-30 samples).

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
    ADVERSARIAL_SUITE_NAMES,
    private_available,
    private_dataset_path,
    synthetic_adversarial_available,
    synthetic_adversarial_suite_path,
    synthetic_adversarial_suite_paths,
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


# Per-suite minimum bars — step 3 split. Kept lenient because each
# suite currently has only 3-5 samples, so a single mis-classification
# would already swing the metric by 20-33pp. The design target (§634)
# is Wilson lower CI ≥ 0.80 per suite, which requires 20-30 samples
# per suite — scheduled for gap #2 dataset growth, not this PR.
#
# Current floor is "at most ONE mis-classification per suite", expressed
# as (n-1)/n so future sample additions auto-tighten the bar:
# - suite with 3 samples: 2/3 ≈ 0.667
# - suite with 5 samples: 4/5 = 0.80
# - suite with 10 samples: 9/10 = 0.90
#
# Bars measure verdict accuracy (specificity on drop-only suites,
# combined on mixed suites). See per-suite rationale in the test body.
def _min_accuracy_bar(n: int) -> float:
    """Allow ≤ 1 mis-classification out of n samples."""
    if n <= 0:
        return 1.0
    if n == 1:
        return 1.0  # a 1-sample suite cannot tolerate a miss
    return (n - 1) / n


@pytest.mark.parametrize("suite_name", ADVERSARIAL_SUITE_NAMES)
async def test_gate_adversarial_resistance(suite_name: str) -> None:
    """Adversarial per-suite resistance — design doc §621 layout.

    Parametrized across the five suites (``ambiguous``, ``sarcasm``,
    ``temporary``, ``contradictions``, ``testing``). Each suite file
    lives at ``synthetic/adversarial/{suite}.jsonl``. A suite is skipped
    if its file is missing (dataset mid-migration).

    Scoring approach (revised from the pre-split single-gate version):

    - **Drop-only suites** (``temporary``, ``contradictions``,
      ``sarcasm``, ``testing``): all samples have ``expected_verdict=drop``,
      so specificity alone captures gate robustness. Accuracy ≡
      specificity here.
    - **Mixed suite** (``ambiguous``): contains both drops and
      borderline keeps, so accuracy = (tp + tn) / total.

    Per-suite bar: ``≤ 1 mis-classification`` (see ``_min_accuracy_bar``
    rationale above). Tightens automatically as suites grow.

    Skip conditions:
    - No LLM configured (handled by ``_resolve_llm``).
    - Suite file missing (e.g. you deleted ``sarcasm.jsonl`` while
      relabelling).

    Design doc §634 target of Wilson lower CI ≥ 0.80 per suite requires
    20-30 samples per suite; gap #2 grew each suite to 22 (2026-04-19).
    All 5 suites pass single-batch after the gap #2 prompt fix
    (explicit hypothetical/temporary markers in ``_SYSTEM_PROMPT``).
    """
    suite_paths = synthetic_adversarial_suite_paths()
    suite_path = suite_paths.get(suite_name)
    if suite_path is None:
        pytest.skip(
            f"adversarial suite '{suite_name}' not available at "
            f"{synthetic_adversarial_suite_path(suite_name)} — "
            f"dataset mid-migration?"
        )

    samples = _load_jsonl(suite_path)
    assert len(samples) >= 3, (
        f"suite '{suite_name}' has {len(samples)} samples; minimum 3 "
        f"to keep the accuracy bar meaningful (see _min_accuracy_bar)"
    )

    llm = _resolve_llm()
    classifier = MemoryGateClassifier(llm)
    decisions = await _run_classifier(samples, classifier)
    tp, fp, fn, tn = _score(samples, decisions, threshold=0.7)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0

    min_bar = _min_accuracy_bar(total)
    allowed_misses = total - int(round(total * min_bar))

    print("\n" + "=" * 60)
    print(f"memory_gate ADVERSARIAL [{suite_name}] @ threshold=0.7, N={total}")
    print(f"  accuracy = {accuracy:.3f}  (tp={tp}, tn={tn})")
    print(f"  misses   = {fp + fn} (fp={fp}, fn={fn})  max allowed={allowed_misses}")
    print(f"  min bar  = {min_bar:.3f}")
    print("=" * 60)

    assert accuracy >= min_bar, (
        f"adversarial suite '{suite_name}' accuracy = {accuracy:.3f} < "
        f"{min_bar:.3f} target. Misses: fp={fp} (gate KEPT a deceptive "
        f"drop), fn={fn} (gate DROPPED a borderline keep). Inspect "
        f"{suite_path.name} to identify which row(s) tripped the gate. "
        f"Per-suite Wilson CI ≥ 0.80 target (design §634) will activate "
        f"after gap #2 grows each suite to 20-30 samples."
    )
