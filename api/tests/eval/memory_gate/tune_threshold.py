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
    synthetic_adversarial_available,
    synthetic_adversarial_suite_paths,
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


def _load_combined(
    *,
    public_only: bool,
    with_adversarial: bool = False,
) -> list[Sample]:
    """Load public synthetic + optional private real dataset + optional
    adversarial suites (union across all 5).

    Returns samples in a stable order (core public → private → adversarial)
    so output is reproducible across runs. Adversarial rows are loaded
    LAST so operators scrolling the per-row output can visually see the
    cutover by id prefix (``a*``). Within the adversarial block, suite
    order follows ``ADVERSARIAL_SUITE_NAMES``
    (ambiguous, sarcasm, temporary, contradictions, testing).

    ``with_adversarial=False`` by default because the primary hard-gate
    calculation (Wilson CI on union) should be computed on the core
    distribution; adversarial is a secondary, separately-reported signal.
    """
    rows = _load_jsonl(synthetic_dataset_path())
    if not public_only and private_available():
        rows.extend(_load_jsonl(private_dataset_path()))
    if with_adversarial:
        for _name, path in synthetic_adversarial_suite_paths().items():
            rows.extend(_load_jsonl(path))
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
    """Given per-sample ``(verdict_is_keep, gate_confidence)`` and expected
    keep labels, compute confusion counts at a threshold.

    **Production-aligned semantic** (matches ``filter_kept_decisions`` in
    ``domain/services/memory_gate.py``): a sample is "kept" iff
    ``verdict == "keep" AND confidence >= threshold``. A verdict of "drop"
    is treated as predicted_keep=False regardless of confidence — gate
    routinely assigns high confidence (0.9+) to both its keep AND drop
    decisions, and a pure-confidence sweep would mis-count every
    high-confidence drop as a positive.

    The pre-PR-2 revision of this function used a confidence-only rule
    (ignoring verdict). It reported apparent precision = base rate on the
    M2 spike run because the gate's drop verdicts (correctly classified)
    were being counted as false positives. Verdict-aware scoring restored
    agreement between the CLI table and ``filter_kept_decisions``.
    """
    tp = fp = fn = tn = 0
    for (verdict_is_keep, conf), exp in zip(predictions, expected):
        predicted_keep = verdict_is_keep and conf >= threshold
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
    # Align decisions back to input order by chunk_index. The bool half
    # of the tuple (``d.verdict == "keep"``) AND the confidence together
    # drive scoring in ``_score_at_threshold`` — this matches the
    # production gate rule (``verdict == "keep" AND conf >= threshold``).
    # Confidence alone is not sufficient because the gate routinely
    # assigns high confidence to both keep AND drop decisions.
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
        "--with-adversarial",
        action="store_true",
        help=(
            "Include the union of synthetic/adversarial/*.jsonl suite "
            "files. Rows are appended after core + private in suite order "
            "(ambiguous, sarcasm, temporary, contradictions, testing) so "
            "operators can visually spot the ``a*`` id block at the tail. "
            "Per-suite scoring (design §634 Wilson ≥ 0.80) runs under "
            "``pytest -m slow test_gate_adversarial_resistance``."
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
    # Load public and private EXPLICITLY so we can distinguish
    # "private file exists" from "private rows reached the scoring
    # batch". The latter is what the rubric §130-140 hard-gate verdict
    # depends on: a private file that's empty OR one whose rows were
    # truncated out by --limit both yield "synthetic-only precision"
    # and must NOT be labelled as the union. Previously the function
    # used ``_load_combined`` + a ``private_row_count`` check against
    # the raw file, which false-greened the --limit-truncates-private
    # case flagged by round-4 codex review.
    public_rows = _load_jsonl(synthetic_dataset_path())
    private_file_exists = not args.public_only and private_available()
    private_rows: list[Sample] = []
    if private_file_exists:
        private_rows = _load_jsonl(private_dataset_path())
    private_rows_total = len(private_rows)

    # Union preserving (public first, private second) order — matches
    # _load_combined's concatenation so the slice-based split below is
    # positional and doesn't need per-Sample provenance tracking.
    core_samples = public_rows + private_rows
    if not core_samples:
        print("no samples loaded; check synthetic/dataset.jsonl", file=sys.stderr)
        return 1

    adv_samples: list[Sample] = []
    if args.with_adversarial and synthetic_adversarial_available():
        # Union across all 5 suite files. Individual-suite scoring
        # happens in pytest (``test_gate_adversarial_resistance``) —
        # the CLI stays a single combined table for exploratory use.
        for _name, path in synthetic_adversarial_suite_paths().items():
            adv_samples.extend(_load_jsonl(path))

    # ``--limit`` caps the TOTAL batched prompt size (help text promises
    # "Cap total samples"). Apply it to ``core + adversarial`` so a quick
    # probe like ``--limit 10 --with-adversarial`` doesn't silently
    # balloon to 30. Core comes first (it's the representative
    # distribution — most probes care about core); adversarial fills the
    # remainder only if there's headroom.
    if args.limit is not None:
        total = max(0, args.limit)
        core_samples = core_samples[:total]
        remaining = total - len(core_samples)
        adv_samples = adv_samples[: max(0, remaining)]

    # POST-LIMIT composition: this is what the LLM actually scores, and
    # what the hard-gate verdict must reflect. len(core_samples) -
    # len(public_rows) is the count of private rows still in the batch
    # because the union kept public-first order and --limit truncates
    # from the tail.
    private_in_batch = max(0, len(core_samples) - len(public_rows))
    has_real_private = private_in_batch > 0

    # Union was intended (user didn't pass --public-only) but didn't
    # materialize in the actual batch. THREE sub-cases handled by the
    # branch below:
    # (a) private file missing entirely (round-5 finding)
    # (b) private file exists but is empty (round-3 finding)
    # (c) private rows truncated out by --limit (round-4 finding)
    union_implied_but_public_only_in_batch = (
        not args.public_only and not has_real_private
    )

    thresholds = _parse_sweep(args.sweep)

    adv_note = f" (+{len(adv_samples)} adversarial, separate report)" if adv_samples else ""
    print(
        f"loaded {len(core_samples)} core samples "
        f"(public + private combined="
        f"{'yes' if has_real_private else 'no'}"
        f"){adv_note}"
    )
    print(f"sweeping thresholds: {thresholds}")

    # Single LLM batch call: batch core + adversarial together, then
    # split predictions before scoring. Keeps cost at 1 LLM call
    # regardless of --with-adversarial.
    llm = _resolve_llm(args.llm)
    all_samples = core_samples + adv_samples
    predictions = await _classify_all(all_samples, llm)

    core_predictions = predictions[: len(core_samples)]
    core_expected = [s.expected_keep for s in core_samples]
    core_results = [
        _score_at_threshold(core_predictions, core_expected, t)
        for t in thresholds
    ]

    # Section header reflects the actual scope of the scoring batch,
    # not an assumed union. Three cases:
    # - has_real_private: union materialized → header announces the
    #   M2 hard gate as applicable
    # - args.public_only: user explicitly opted out of union; header
    #   notes public-only scope but still allows the verdict below
    #   (round-1 agreement: --public-only user owns the scope)
    # - otherwise: union was implied but didn't materialize → header
    #   says visibility only so operators can't misread the table as
    #   a union-scoped result
    if has_real_private:
        scope_label = "public + private"
        gate_label = "gated by M2 hard gate"
    elif args.public_only:
        scope_label = "public only"
        gate_label = "gated by M2 hard gate (public-only scope)"
    else:
        scope_label = "public only"
        gate_label = "visibility only, union not available"

    print()
    print(
        f"===== CORE ({scope_label}) N={len(core_samples)} — "
        f"{gate_label} ====="
    )
    print(_format_table(core_results))

    # Hard gate verdict — ONLY on the core distribution. This is the
    # contract: rubric §130-140 says acceptance is on public+private
    # union, and the adversarial suite is explicitly out of scope.
    print()
    if union_implied_but_public_only_in_batch:
        # Sweep above is synthetic-only precision. Emitting "M2 hard
        # gate clears at X" here would mislead the operator into
        # thinking the rubric acceptance criterion passed, when it only
        # holds on the public subset. Suppress the verdict and surface
        # the fix paths. Distinguish the three sub-cases so the message
        # is actionable:
        if not private_available():
            # Round-5: private file doesn't exist at all
            print(
                f"WARN: private dataset not found at "
                f"{private_dataset_path()}. Sweep above reflects "
                f"synthetic-only precision, NOT the public + private "
                f"union required by rubric §130-140. NOT emitting M2 "
                f"hard-gate verdict. Create the file (populate "
                f"$ACTUS_EVAL_DATA_DIR/memory_gate/real/dataset.jsonl) "
                f"or pass --public-only to acknowledge public-only scope."
            )
        elif private_rows_total == 0:
            # Round-3: file exists but empty
            print(
                f"WARN: private dataset at {private_dataset_path()} "
                f"exists but contains 0 rows. Sweep above reflects "
                f"synthetic-only precision, NOT the public + private "
                f"union required by rubric §130-140. NOT emitting M2 "
                f"hard-gate verdict. Populate the file, remove it, or "
                f"pass --public-only to get a public-only verdict."
            )
        else:
            # Round-4: --limit truncated the private rows out of the batch
            print(
                f"WARN: --limit {args.limit} truncated the batch to "
                f"{len(core_samples)} sample(s), all from the public "
                f"set; private dataset has {private_rows_total} row(s) "
                f"but none reached the sweep. Sweep above reflects "
                f"synthetic-only precision, NOT the public + private "
                f"union required by rubric §130-140. NOT emitting M2 "
                f"hard-gate verdict. Raise --limit above "
                f"{len(public_rows)} or pass --public-only."
            )
    else:
        passing = [r for r in core_results if r.precision_lower >= 0.70]
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

    # Adversarial separate report — no hard-gate verdict. If someone
    # wants a pass/fail on this suite they should run
    # ``pytest -m slow tests/eval/memory_gate/test_memory_gate_eval.py
    # ::test_gate_adversarial_resistance`` which applies specificity +
    # sensitivity bars tuned to the deliberately-tricky distribution.
    if adv_samples:
        adv_predictions = predictions[len(core_samples):]
        adv_expected = [s.expected_keep for s in adv_samples]
        adv_results = [
            _score_at_threshold(adv_predictions, adv_expected, t)
            for t in thresholds
        ]
        print()
        print(
            f"===== ADVERSARIAL N={len(adv_samples)} — "
            f"visibility only, NOT gated by M2 hard gate ====="
        )
        print(_format_table(adv_results))
        print(
            "NOTE: adversarial precision/recall are not directly comparable "
            "to core. Drop-heavy composition (17/20) means a pure-drop gate "
            "scores high on accuracy-like metrics. Use "
            "test_gate_adversarial_resistance for specificity+sensitivity "
            "assertions."
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
