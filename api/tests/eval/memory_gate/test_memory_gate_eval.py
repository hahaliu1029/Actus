"""M1 minimal eval harness for the memory LLM quality gate.

**Off by default** — gated via ``pytest.mark.slow``. Run manually:

    cd api && uv run pytest tests/eval/memory_gate/ -v -m slow

Resolution of "real LLM" here is intentionally narrow. M1's goal is to
give the solo author an early signal on precision before M2 scales the
dataset to 200+ samples with Wilson CI. What this harness *does*:

- Loads the 25-ish synthetic samples from ``synthetic/dataset.jsonl``.
- Runs them through the real ``MemoryGateClassifier`` with an LLM
  selected by environment (``EVAL_MEMORY_GATE_LLM=chat_llm|summary_llm``
  or falls back to skipping the test with an explanatory message).
- Reports aggregate precision (predicted-keep ∩ expected-keep) /
  predicted-keep and recall (... / expected-keep), plus the confusion
  matrix per category, printed to stdout.
- **Does not** hard-fail on precision below a target. M2 owns the hard
  gate via Wilson CI lower bound; at M1 we just want visibility.

What this harness *does not* do (deferred):

- Wilson 95% CI — needs at least 100 samples to be meaningful
- Private adversarial suites (scraped from real dev sessions, gitignored)
- Control-set kappa drift monitoring
- Per-model threshold sweep
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

pytestmark = [pytest.mark.anyio, pytest.mark.slow]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# 25 条 synthetic + paraphrased；M2 扩到 100 公开集 + 私有集。
_DATASET = Path(__file__).parent / "synthetic" / "dataset.jsonl"


def _load_samples() -> list[dict]:
    rows: list[dict] = []
    with _DATASET.open() as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


async def test_gate_precision_baseline() -> None:
    """Point-estimate precision on the synthetic dataset.

    Reports:
    - overall precision / recall / F1 at threshold=0.7 (M1 default)
    - counts of keep/drop predictions vs ground truth
    - per-category confusion (only for predicted-keep rows, since
      category is undefined/ignored for drops)

    Prints to stdout; does not hard-fail. Precision < 0.5 warrants a
    second look but doesn't block the test suite because dataset N=25
    means each mis-prediction swings the metric by 4pp.
    """
    samples = _load_samples()
    assert len(samples) >= 20, f"dataset shrunk unexpectedly: {len(samples)}"

    llm = _resolve_llm()
    classifier = MemoryGateClassifier(llm)
    inputs = [
        MemoryGateInput(chunk_index=i, text=s["text"])
        for i, s in enumerate(samples)
    ]
    decisions = await classifier.classify(inputs)

    # At M1 default threshold
    kept = filter_kept_decisions(decisions, threshold=0.7)
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

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Per-category accuracy among predicted-keep
    by_cat: dict[str, tuple[int, int]] = {}
    decision_by_idx = {d.chunk_index: d for d in kept}
    for i in kept_indices:
        d = decision_by_idx[i]
        gold_cat = samples[i]["expected_category"]
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
