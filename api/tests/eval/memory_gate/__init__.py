"""Memory gate eval harness (M1 → M2).

Contents:
- ``synthetic/dataset.jsonl`` — 52 public core samples (45 from PR-5 +
  marker-as-hint regression rows added in gap #2 closure; the regression
  set is selected dynamically by ``notes`` prefix, not by ID range, so
  the count drifts as samples are added — see
  ``test_gate_marker_hint_regression`` in ``test_memory_gate_eval.py``
  for the live count). Target 100 at M2 ship.
- ``synthetic/adversarial/{ambiguous,sarcasm,temporary,contradictions,
  testing}.jsonl`` — 110 hand-crafted adversarial samples (22 per suite)
  split across the five suites from design doc §621. Gap #2 grew each
  suite from 3-5 to 22 samples; per-suite Wilson precision CI ≥ 0.80
  (design §634) is **not yet** satisfied — see
  ``synthetic/adversarial/README.md`` "Per-suite Wilson precision gate
  status" for the math (only ``ambiguous`` is precision-scorable, with
  13 keeps that cap Wilson lower at 0.772 even at 100% precision).
- ``labelling_rubric.md`` — verdict/category definitions, M2 semantics
- ``stats.py`` — Wilson CI + Cohen's kappa helpers (no external deps)
- ``paths.py`` — public + private dataset + adversarial path resolution
- ``tune_threshold.py`` — CLI that sweeps thresholds and prints P/R/F1 + CI
- ``test_memory_gate_eval.py`` — pytest.mark.slow suite: baseline
  (visibility), Wilson hard gate (M2 acceptance), and adversarial
  resistance
- ``test_annotator_consistency.py`` — pytest.mark.slow weekly drift
  check (kappa on control-set re-labels)
- ``test_eval_stats.py`` / ``test_eval_paths.py`` / ``test_tune_threshold.py``
  — unit tests for the helpers, run in default CI
"""
RUBRIC_VERSION = "v1-m2"
"""Tag applied to any dataset scored against this rubric. Bump when the
semantics in ``labelling_rubric.md`` change — downstream CI artifacts
and eval snapshots pin on the version so historic results stay
interpretable.

Previous versions:
- ``v0-m1`` — initial M1 ship, 25-sample synthetic set. Semantics
  carried forward unchanged in ``v1-m2``; the bump only reflects the
  expansion of dataset layout, adversarial suite, and control-set
  drift conventions landed in PR-5."""
