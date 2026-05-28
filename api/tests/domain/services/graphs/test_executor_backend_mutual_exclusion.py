"""[C2 PR-9 §15.2 Gate #1] Executor dispatches mutually-exclusive backends.

Static AST gate: ``executor_node`` in ``main_graph.py`` must dispatch through
both backend paths — the parallel-coordinator path AND the inline react path.
Any refactor that collapses both branches into a single backend (or renames
either signal) must update this gate so reviewers notice that the dispatch
invariant has changed.

Heuristic adaptation (vs. plan §15.2 listing):
    The plan's listing scanned for two named helper functions:
    ``_run_parallel_backend`` and ``_run_react_backend``. In the current
    code (PR-3 §7.2 split) only the parallel path was extracted into a
    named helper — ``_run_parallel_backend`` (main_graph.py:63). The react
    path is **inlined** inside ``executor_node`` and reaches its subgraph
    via ``step_react.astream(...)`` (main_graph.py:912 at the time of
    writing). So this gate verifies the asymmetric structural invariant:

      1. ``executor_node`` contains a call to ``_run_parallel_backend``
         (parallel-coordinator branch).
      2. ``executor_node`` contains an attribute call ``step_react.astream``
         (the canonical inline-react branch signal).

    If a future PR extracts the react path into ``_run_react_backend`` (or
    similar), both signals can be detected — replace heuristic (2) with a
    name-presence check on the new helper.

This is intentionally a *name-presence* gate — it does NOT verify that the
two paths sit in mutually-exclusive ``if/else`` branches. That level of
control-flow analysis is exercised by
``test_executor_dispatch_branch.py`` (runtime path tests).
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.coordinator_graph


# Resolve repo-relative ``api/app/...`` paths from this test file's location so
# the gate works regardless of pytest's CWD (``api/`` vs repo root).
_API_ROOT = Path(__file__).resolve().parents[4]
_MAIN_GRAPH = _API_ROOT / "app" / "domain" / "services" / "graphs" / "main_graph.py"


def test_executor_node_dispatches_exclusive_backends():
    src = _MAIN_GRAPH.read_text()
    tree = ast.parse(src)
    found_parallel = False
    found_react = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "executor_node":
            for s in ast.walk(node):
                # (1) Parallel coordinator backend — named helper call.
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Name):
                    if s.func.id == "_run_parallel_backend":
                        found_parallel = True
                # (2) Inline react backend signal — ``step_react.astream(...)``
                # (or any ``.astream`` call on the local ``step_react`` binding).
                # If the react path is later extracted into a named helper
                # (e.g. ``_run_react_backend``), update this branch to match
                # the new helper's ``ast.Name`` id.
                if (
                    isinstance(s, ast.Call)
                    and isinstance(s.func, ast.Attribute)
                    and s.func.attr == "astream"
                    and isinstance(s.func.value, ast.Name)
                    and s.func.value.id == "step_react"
                ):
                    found_react = True
    assert found_parallel, (
        "executor_node must call ``_run_parallel_backend`` (parallel coordinator branch)"
    )
    assert found_react, (
        "executor_node must reach the inline react backend via ``step_react.astream(...)`` "
        "(or — if extracted — call a named ``_run_react_backend`` helper; update this gate)"
    )
