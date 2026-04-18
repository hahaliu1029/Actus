"""M2-PR1: threshold sweep CLI for the memory gate.

Runs the gate over a dataset at multiple thresholds and prints a table
of precision / recall / F1 with the Wilson 95% CI lower bound for
precision. Used by the solo author to pick a threshold that keeps the
M2 hard gate (``Wilson CI lower >= 0.70``) while maximizing recall.

**Not a pytest test** — this is a standalone script. Invoke via:

    cd api && uv run python -m tests.eval.memory_gate.tune_threshold \\
        --llm chat_llm \\
        --sweep 0.5:0.9:0.1

The LLM must be selectable from the app config (``chat_llm``,
``summary_llm``, etc.). Use ``--public-only`` to skip the private dataset
(useful for contributors who don't have the local ``~/.actus/eval/``
directory populated).

Cost / failure-mode note: the threshold sweep is **post-hoc** — the LLM
runs exactly **once** per invocation, with all N samples in a single
``MemoryGateClassifier.classify(inputs)`` batch call. The sweep then
re-partitions that single run's confidences at each threshold. What to
worry about is therefore **not** the number of calls (it's always 1);
it's the single-prompt / structured-output size cap. On N=300 samples
the batch prompt can push past context limits and the structured
``list[MemoryGateDecision]`` output can exceed max-output tokens. Use
``--limit`` to probe with a smaller slice if the single-call batch is
blowing up; chunk-level batching across multiple LLM calls is a
follow-up for a richer sweep harness.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.eval.memory_gate.paths import (
    private_available,
    private_dataset_path,
    synthetic_dataset_path,
)
from tests.eval.memory_gate.stats import wilson_ci_lower


# ---- Dataset loading --------------------------------------------------- #


@dataclass(frozen=True)
class Sample:
    id: str
    text: str
    expected_keep: bool


def _load_jsonl(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                Sample(
                    id=str(obj["id"]),
                    text=str(obj["text"]),
                    expected_keep=(obj.get("expected_verdict") == "keep"),
                )
            )
    return rows


def _load_combined(*, public_only: bool) -> list[Sample]:
    """Load public synthetic + optional private real dataset.

    Returns samples in a stable order (public first, then private) so
    output is reproducible across runs.
    """
    rows = _load_jsonl(synthetic_dataset_path())
    if not public_only and private_available():
        rows.extend(_load_jsonl(private_dataset_path()))
    return rows


# ---- Sweep math -------------------------------------------------------- #


def _parse_sweep(spec: str) -> list[float]:
    """Parse ``low:high:step`` into an inclusive list of thresholds.

    Example: ``0.5:0.9:0.1`` → ``[0.5, 0.6, 0.7, 0.8, 0.9]``. Rounded
    to 2 decimals to avoid floating-point drift in the output table.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--sweep expects low:high:step; got {spec!r}"
        )
    try:
        lo, hi, step = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--sweep components must be floats: {spec!r}"
        ) from exc
    if step <= 0:
        raise argparse.ArgumentTypeError(
            f"--sweep step must be positive; got {step}"
        )
    if hi < lo:
        raise argparse.ArgumentTypeError(
            f"--sweep high ({hi}) must be >= low ({lo})"
        )

    out: list[float] = []
    current = lo
    # Inclusive upper bound with epsilon to survive float drift
    while current <= hi + 1e-9:
        out.append(round(current, 2))
        current += step
    return out


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def precision_lower(self) -> float:
        """Wilson 95% CI lower bound on precision."""
        return wilson_ci_lower(self.tp, self.tp + self.fp)


def _score_at_threshold(
    predictions: list[tuple[bool, float]],
    expected: list[bool],
    threshold: float,
) -> ThresholdResult:
    """Given per-sample ``(gate_decision, gate_confidence)`` and expected
    keep labels, compute confusion counts at a threshold.

    The gate outputs a confidence; samples are "kept" when
    ``confidence >= threshold``.
    """
    tp = fp = fn = tn = 0
    for (_, conf), exp in zip(predictions, expected):
        predicted_keep = conf >= threshold
        if predicted_keep and exp:
            tp += 1
        elif predicted_keep and not exp:
            fp += 1
        elif not predicted_keep and exp:
            fn += 1
        else:
            tn += 1
    return ThresholdResult(
        threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn
    )


# ---- LLM wiring -------------------------------------------------------- #


async def _classify_all(samples: list[Sample], llm) -> list[tuple[bool, float]]:
    """Run the gate classifier once and return (keep_decision, confidence) per
    sample. Threshold sweep happens post-hoc on the confidences.

    Importing lazily so the module stays importable even when the full
    app isn't available (e.g., if someone runs ``--help`` without config).
    """
    from app.domain.services.memory_gate import (
        MemoryGateClassifier,
        MemoryGateInput,
    )

    classifier = MemoryGateClassifier(llm)
    inputs = [
        MemoryGateInput(chunk_index=i, text=s.text)
        for i, s in enumerate(samples)
    ]
    decisions = await classifier.classify(inputs)
    # Align decisions back to input order by chunk_index. The decision's
    # own internal ``verdict`` (keep/drop) isn't used by the sweep —
    # we only need ``confidence`` so each threshold in the sweep can
    # re-decide keep/drop post-hoc. The bool half of the tuple preserves
    # the gate's own verdict for future introspection / logging.
    by_idx = {d.chunk_index: d for d in decisions}
    out: list[tuple[bool, float]] = []
    for i in range(len(samples)):
        d = by_idx.get(i)
        if d is None:
            # Gate may have dropped a sample from its batch output (e.g.,
            # LLM returned a shorter list). Treat as "low-confidence drop"
            # so the sweep sees it as a negative prediction.
            out.append((False, 0.0))
        else:
            out.append((d.verdict == "keep", d.confidence))
    return out


def _resolve_llm(key: str):
    """Resolve an LLM adapter by config key (mirrors ``test_memory_gate_eval``).

    Kept in the script (not in ``paths.py``) because it imports heavy
    app-layer modules; tests that only exercise path helpers shouldn't
    pay for this import cost.
    """
    from app.interfaces.service_dependencies import (
        _build_config_snapshot,
        _load_app_config,
    )

    snapshot = _build_config_snapshot(_load_app_config())
    if key == "chat_llm":
        return snapshot.llm
    if key == "summary_llm":
        return snapshot.summary_llm or snapshot.llm
    raise SystemExit(
        f"Unknown --llm {key!r}; expected chat_llm or summary_llm"
    )


# ---- Main -------------------------------------------------------------- #


def _format_table(results: list[ThresholdResult]) -> str:
    """Return a plain-text ASCII table of the sweep results.

    Columns: threshold | precision (point / lower CI) | recall | F1 | TP/FP/FN/TN
    """
    lines = [
        "threshold | precision   (≥CI lo) | recall  | F1      | TP  FP  FN  TN",
        "----------+----------------------+---------+---------+----------------",
    ]
    for r in results:
        lines.append(
            f"  {r.threshold:.2f}    | "
            f"{r.precision:.3f}       ({r.precision_lower:.3f}) | "
            f"{r.recall:.3f}   | {r.f1:.3f}   | "
            f"{r.tp:>3} {r.fp:>3} {r.fn:>3} {r.tn:>3}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests.eval.memory_gate.tune_threshold",
        description=(
            "Sweep gate thresholds and report precision / recall / F1 + "
            "Wilson 95% CI lower bound on precision."
        ),
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="LLM key from app_config (chat_llm or summary_llm)",
    )
    parser.add_argument(
        "--sweep",
        default="0.5:0.9:0.1",
        help="Threshold range low:high:step (default 0.5:0.9:0.1)",
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help=(
            "Skip the private dataset even if it exists locally. Default "
            "merges public + private when ACTUS_EVAL_DATA_DIR has data."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total samples (default: all). Useful for quick probes.",
    )
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    samples = _load_combined(public_only=args.public_only)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        print("no samples loaded; check synthetic/dataset.jsonl", file=sys.stderr)
        return 1

    thresholds = _parse_sweep(args.sweep)

    print(f"loaded {len(samples)} samples (public + private combined="
          f"{'yes' if not args.public_only and private_available() else 'no'})")
    print(f"sweeping thresholds: {thresholds}")

    llm = _resolve_llm(args.llm)
    predictions = await _classify_all(samples, llm)
    expected = [s.expected_keep for s in samples]

    results = [
        _score_at_threshold(predictions, expected, t) for t in thresholds
    ]
    print()
    print(_format_table(results))

    # Flag the M2 hard gate crossings
    print()
    passing = [r for r in results if r.precision_lower >= 0.70]
    if passing:
        best = max(passing, key=lambda r: r.recall)
        print(
            f"M2 hard gate (precision lower CI ≥ 0.70) clears at threshold "
            f"{best.threshold:.2f} with recall {best.recall:.3f}."
        )
    else:
        print(
            "WARN: no threshold clears the M2 hard gate "
            "(precision Wilson lower CI ≥ 0.70). Consider prompt engineering."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
