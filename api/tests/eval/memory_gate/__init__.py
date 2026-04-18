"""Memory gate eval harness (M1 → M2).

Contents:
- ``synthetic/dataset.jsonl`` — 45 public core samples (PR-5 expanded
  from 25); target 100 at M2 ship
- ``synthetic/adversarial.jsonl`` — 20 hand-crafted adversarial samples
  (PR-5) covering task-local masquerade, hypotheticals, agent-output
  mimicry, retractions, prompt-injection attempts, and borderline keeps
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
