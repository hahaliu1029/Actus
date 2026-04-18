"""M2-PR1: unit tests for the pure logic in tune_threshold.py.

Covers the parts of the CLI that don't require an LLM or the app
config — dataset loading, sweep parsing, confusion-matrix scoring,
and table formatting. The ``_amain`` end-to-end entry point is exercised
only by the manual CLI run; ``_classify_all`` is exercised here with a
mock classifier so that refactors of ``MemoryGateDecision`` that break
attribute access get caught at test time rather than first real run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest

from tests.eval.memory_gate.tune_threshold import (
    Sample,
    ThresholdResult,
    _classify_all,
    _format_table,
    _load_combined,
    _load_jsonl,
    _parse_sweep,
    _score_at_threshold,
)


# ---- Sweep parsing ----------------------------------------------------- #


class TestParseSweep:
    def test_default_range(self) -> None:
        assert _parse_sweep("0.5:0.9:0.1") == [0.5, 0.6, 0.7, 0.8, 0.9]

    def test_inclusive_upper_bound(self) -> None:
        """Upper bound included when step divides evenly."""
        assert _parse_sweep("0.7:0.8:0.05") == [0.7, 0.75, 0.8]

    def test_float_drift_does_not_drop_endpoint(self) -> None:
        """Classic floating-point gotcha: 0.1+0.1+0.1 can overshoot 0.3
        by epsilon; the epsilon guard in the parser must keep 0.3 in."""
        result = _parse_sweep("0.1:0.3:0.1")
        assert 0.3 in result
        assert result == [0.1, 0.2, 0.3]

    def test_single_point_sweep(self) -> None:
        """low == high works — useful for single-threshold regression."""
        assert _parse_sweep("0.7:0.7:0.1") == [0.7]

    def test_raises_on_bad_shape(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_sweep("0.5,0.9,0.1")  # comma separated
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_sweep("0.5:0.9")  # missing step

    def test_raises_on_non_float(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_sweep("a:b:c")

    def test_raises_on_zero_step(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_sweep("0.5:0.9:0.0")

    def test_raises_on_reversed_range(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_sweep("0.9:0.5:0.1")


# ---- Confusion matrix --------------------------------------------------- #


class TestScoreAtThreshold:
    def test_perfect_prediction(self) -> None:
        predictions = [(True, 0.9), (True, 0.85), (False, 0.1)]
        expected = [True, True, False]
        r = _score_at_threshold(predictions, expected, threshold=0.5)
        assert r.tp == 2
        assert r.fp == 0
        assert r.fn == 0
        assert r.tn == 1
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_threshold_affects_classification(self) -> None:
        """Same predictions, different thresholds → different confusion."""
        predictions = [(True, 0.6), (True, 0.8)]
        expected = [True, False]

        # At threshold 0.5: both predicted keep → tp=1, fp=1
        r_low = _score_at_threshold(predictions, expected, threshold=0.5)
        assert r_low.tp == 1
        assert r_low.fp == 1
        # Precision = 0.5, recall = 1.0
        assert r_low.precision == 0.5
        assert r_low.recall == 1.0

        # At threshold 0.7: only confidence=0.8 kept → tp=0, fp=1
        r_high = _score_at_threshold(predictions, expected, threshold=0.7)
        assert r_high.tp == 0
        assert r_high.fp == 1
        assert r_high.fn == 1
        assert r_high.precision == 0.0

    def test_all_negative_prediction_metrics_safe(self) -> None:
        """tp+fp = 0 → precision defined as 0.0 (not nan)."""
        predictions = [(False, 0.1), (False, 0.2)]
        expected = [True, False]
        r = _score_at_threshold(predictions, expected, threshold=0.5)
        assert r.tp == 0
        assert r.fp == 0
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0

    def test_precision_lower_bound_links_to_wilson(self) -> None:
        """``ThresholdResult.precision_lower`` is the Wilson CI lower on
        tp/(tp+fp) — spot-check against the M2 hard-gate scenario."""
        predictions = [(True, 0.9)] * 80 + [(True, 0.9)] * 20  # 80 kept
        expected = [True] * 80 + [False] * 20  # 80 correct, 20 wrong
        r = _score_at_threshold(predictions, expected, threshold=0.5)
        assert r.tp == 80
        assert r.fp == 20
        assert r.precision == 0.80
        # Known: Wilson CI 95% lower for 80/100 is ~0.7111
        assert 0.70 < r.precision_lower < 0.72


# ---- Dataset loading --------------------------------------------------- #


class TestLoadJsonl:
    def test_parses_keep_and_drop_verdicts(self, tmp_path: Path) -> None:
        p = tmp_path / "d.jsonl"
        p.write_text(
            '\n'.join([
                '{"id": "a1", "text": "用户：我用 Go", "expected_verdict": "keep", "expected_category": "user"}',
                '{"id": "a2", "text": "把按钮改成蓝色", "expected_verdict": "drop", "expected_category": "user"}',
                '',  # blank line tolerated
            ])
        )
        rows = _load_jsonl(p)
        assert len(rows) == 2
        assert rows[0].id == "a1"
        assert rows[0].expected_keep is True
        assert rows[1].expected_keep is False

    def test_missing_verdict_treated_as_drop(self, tmp_path: Path) -> None:
        """Defensive: dataset row without ``expected_verdict`` → drop
        (keep would silently inflate precision)."""
        p = tmp_path / "d.jsonl"
        p.write_text('{"id": "x1", "text": "no verdict"}\n')
        rows = _load_jsonl(p)
        assert rows[0].expected_keep is False


class TestLoadCombined:
    def test_public_only_skips_private(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """``--public-only`` → ignore the private dir even if it exists."""
        # Point private to a populated tmp_path with a private dataset
        monkeypatch.setenv("ACTUS_EVAL_DATA_DIR", str(tmp_path))
        priv = tmp_path / "memory_gate" / "real"
        priv.mkdir(parents=True)
        (priv / "dataset.jsonl").write_text(
            '{"id": "priv1", "text": "p", "expected_verdict": "keep"}\n'
        )
        # public_only should skip the private file
        rows = _load_combined(public_only=True)
        # Public dataset has 25 samples; none from private
        ids = [r.id for r in rows]
        assert "priv1" not in ids

    def test_union_includes_private_when_available(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ACTUS_EVAL_DATA_DIR", str(tmp_path))
        priv = tmp_path / "memory_gate" / "real"
        priv.mkdir(parents=True)
        (priv / "dataset.jsonl").write_text(
            '{"id": "priv1", "text": "p", "expected_verdict": "drop"}\n'
        )
        rows = _load_combined(public_only=False)
        ids = [r.id for r in rows]
        assert "priv1" in ids
        # Public samples still present
        assert len(rows) > 1

    def test_missing_private_is_ok(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """When private dir doesn't exist, load_combined quietly returns
        public-only (CI / contributor path). Asserts by count: union of
        public + missing private should equal public size, and no id from
        our explicit private marker shows up.
        """
        monkeypatch.setenv(
            "ACTUS_EVAL_DATA_DIR", str(tmp_path / "does-not-exist")
        )
        rows_missing = _load_combined(public_only=False)
        rows_public_only = _load_combined(public_only=True)
        assert len(rows_missing) == len(rows_public_only), (
            f"missing-private should match public-only size; "
            f"missing={len(rows_missing)} public-only={len(rows_public_only)}"
        )
        # Private samples would carry a distinctive id prefix ('priv'/'real')
        # by author convention — guard against any that accidentally land in.
        assert not any(
            r.id.startswith(("priv_", "real_")) for r in rows_missing
        ), "private-prefixed ids must not leak when private dir is absent"


# ---- Classifier wiring (attribute-access contract) --------------------- #


class TestClassifyAll:
    """Exercise ``_classify_all`` with a mock classifier so breakage of
    the ``MemoryGateDecision`` attribute surface gets caught here rather
    than first real LLM run. This class pins the contract:

    - ``_classify_all`` reads ``d.chunk_index``, ``d.verdict``, ``d.confidence``
      (NOT ``d.keep`` — that field does not exist on the real domain
      dataclass).
    - Missing chunk_index in the response → synthesized ``(False, 0.0)``.
    - Order follows input index, not classifier output order.
    """

    def test_maps_verdict_and_confidence_in_order(self) -> None:
        from app.domain.services.memory_gate import MemoryGateDecision

        class _FakeClassifier:
            async def classify(self, inputs):  # noqa: D401
                # Return decisions out of order to verify index-based alignment
                return [
                    MemoryGateDecision(
                        chunk_index=1,
                        verdict="keep",
                        category="user",
                        confidence=0.92,
                    ),
                    MemoryGateDecision(
                        chunk_index=0,
                        verdict="drop",
                        category="user",
                        confidence=0.15,
                    ),
                ]

        samples = [
            Sample(id="a", text="hi", expected_keep=False),
            Sample(id="b", text="hey", expected_keep=True),
        ]

        # Patch MemoryGateClassifier to our fake so `_classify_all`'s
        # constructor call returns the fake.
        import tests.eval.memory_gate.tune_threshold as mod

        # Monkey-patch the lazy import by stubbing the referenced symbols
        # inside _classify_all; the cleanest hook is to override the
        # domain import at call time via a classifier factory. Since
        # _classify_all does `from app.domain.services.memory_gate import
        # MemoryGateClassifier`, we substitute that attribute on the
        # module where it's looked up.
        from app.domain.services import memory_gate as gate_mod

        original = gate_mod.MemoryGateClassifier
        gate_mod.MemoryGateClassifier = lambda _llm: _FakeClassifier()
        try:
            # _resolve_llm is never called here since we pass llm=None
            out = asyncio.run(_classify_all(samples, llm=None))
        finally:
            gate_mod.MemoryGateClassifier = original

        # Output order = input order (index 0 first, then 1)
        assert out == [(False, 0.15), (True, 0.92)]

    def test_missing_decision_synthesizes_drop(self) -> None:
        from app.domain.services.memory_gate import MemoryGateDecision

        class _PartialClassifier:
            async def classify(self, inputs):  # noqa: D401
                # Only decision 0 — decision 1 missing from batch output
                return [
                    MemoryGateDecision(
                        chunk_index=0,
                        verdict="keep",
                        category="user",
                        confidence=0.80,
                    ),
                ]

        samples = [
            Sample(id="a", text="hi", expected_keep=True),
            Sample(id="b", text="missing", expected_keep=True),
        ]
        from app.domain.services import memory_gate as gate_mod

        original = gate_mod.MemoryGateClassifier
        gate_mod.MemoryGateClassifier = lambda _llm: _PartialClassifier()
        try:
            out = asyncio.run(_classify_all(samples, llm=None))
        finally:
            gate_mod.MemoryGateClassifier = original

        # Index 1 synthesized as (False, 0.0)
        assert out[0] == (True, 0.80)
        assert out[1] == (False, 0.0)


# ---- Table formatting --------------------------------------------------- #


class TestFormatTable:
    def test_header_and_row_alignment(self) -> None:
        r = ThresholdResult(threshold=0.7, tp=80, fp=20, fn=5, tn=95)
        table = _format_table([r])
        # Header present
        assert "threshold" in table
        assert "precision" in table
        assert "Wilson" not in table  # (just the header abbrev "≥CI lo")
        # Row has all counts
        assert "  0.70" in table
        assert "0.800" in table  # precision
        assert " 80 " in table  # tp
        assert " 20 " in table  # fp

    def test_multi_row_output(self) -> None:
        rows = [
            ThresholdResult(threshold=0.5, tp=90, fp=40, fn=2, tn=68),
            ThresholdResult(threshold=0.7, tp=80, fp=20, fn=5, tn=95),
            ThresholdResult(threshold=0.9, tp=50, fp=5, fn=10, tn=135),
        ]
        table = _format_table(rows)
        lines = table.splitlines()
        # Header + separator + 3 rows
        assert len(lines) == 5
