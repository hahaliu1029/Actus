"""Eval metrics output for the SmartApprove dashboard JSON.

Run: ``uv run python -m tests.eval.permission_smart_approve.metrics report``
to write ``monitoring/dashboards/permission_engine_eval_latest.json``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _load_corpus() -> list[dict]:
    corpus_dir = Path(__file__).parent / "corpus"
    return [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(corpus_dir.glob("*.json"))]


def report() -> dict:
    entries = _load_corpus()
    by_source = Counter(e.get("tool_source", "unknown") for e in entries)
    by_expected = Counter(e.get("expected_decision", "unknown") for e in entries)
    return {
        "version": 1,
        "total": len(entries),
        "by_source": dict(by_source),
        "by_expected_decision": dict(by_expected),
    }


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "report":
        print("usage: python -m tests.eval.permission_smart_approve.metrics report",
              file=sys.stderr)
        return 2
    # parents[4] = repo root (metrics.py lives at api/tests/eval/permission_smart_approve/).
    out_path = Path(__file__).resolve().parents[4] / "monitoring" / "dashboards" / \
        "permission_engine_eval_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report(), indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
