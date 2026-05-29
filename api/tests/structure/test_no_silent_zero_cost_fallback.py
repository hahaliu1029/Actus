"""[PR-9b-B Task B4] Structural guard — no silent-zero CostAggregate fallback
in ``parallel_execution_subgraph.reducer_node``.

INV-B2 (no silent-zero pattern): the reducer node MUST NOT coerce a missing /
None ``cost_summary`` into a fresh ``CostAggregate()`` and emit it onto the
wire as if it were authoritative cost data. The B4 contract requires the
reducer to pull cost via ``cost_rollup_service.aggregate(...)`` (INV-B1 — cost
authoritative from the ledger), and to surface aggregate failures via
``diagnostics_summary='cost_unavailable: ...'`` (INV-B3).

This test does a string + AST scan of ``parallel_execution_subgraph.py`` so
the pattern stays forbidden even if the surrounding code is refactored.

Specifically banned (the literal silent-zero pattern PR-9b-B removes):

    cost_total = getattr(output, "cost_summary", None) or CostAggregate()

The variant ``... or CostAggregate()`` adjacent to a ``getattr`` reading
``cost_summary`` is what we ban — that is the documented INV-B2 violation.

WorkerResult.__init__'s ``self.cost_summary = cost_summary or CostAggregate()``
is OK (carrier-init pattern, NOT a wire-emit cost computation).
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBGRAPH_PATH = (
    REPO_ROOT
    / "api"
    / "app"
    / "domain"
    / "services"
    / "graphs"
    / "parallel_execution_subgraph.py"
)


def _read_source() -> str:
    assert SUBGRAPH_PATH.exists(), (
        f"parallel_execution_subgraph.py missing at {SUBGRAPH_PATH}"
    )
    return SUBGRAPH_PATH.read_text(encoding="utf-8")


def test_no_silent_zero_cost_aggregate_string_pattern() -> None:
    """The literal silent-zero pattern must not appear anywhere in the file.

    Catches the exact ``getattr(output, "cost_summary", ...) or CostAggregate()``
    construction the B4 PR removes.
    """
    src = _read_source()
    # Ban the exact composite token sequence the silent-zero pattern uses.
    banned = 'getattr(output, "cost_summary"'
    occurrences = src.count(banned)
    assert occurrences == 0, (
        f"INV-B2 violation: parallel_execution_subgraph.py still contains "
        f"the silent-zero pattern `{banned}` "
        f"({occurrences} occurrence(s)). The reducer MUST pull cost via "
        f"cost_rollup_service.aggregate(...) and surface failure via "
        f"diagnostics_summary='cost_unavailable: ...'."
    )


def test_no_or_cost_aggregate_fallback_in_reducer_node() -> None:
    """AST scan: inside ``reducer_node``, no ``<expr> or CostAggregate()`` form.

    The ``WorkerResult.__init__`` carrier `or CostAggregate()` is OK — that
    sits in the WorkerResult class body, NOT in reducer_node. We walk only
    the reducer_node AST subtree.
    """
    src = _read_source()
    tree = ast.parse(src)

    reducer_node_fn: ast.AsyncFunctionDef | ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.name == "reducer_node":
                reducer_node_fn = node
                break

    assert reducer_node_fn is not None, (
        "reducer_node function not found in parallel_execution_subgraph.py — "
        "this guard relies on the function being present."
    )

    violations: list[str] = []
    for sub in ast.walk(reducer_node_fn):
        # Match ``<lhs> or CostAggregate()`` BoolOp(op=Or, ...).
        if not isinstance(sub, ast.BoolOp):
            continue
        if not isinstance(sub.op, ast.Or):
            continue
        for value in sub.values:
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            if isinstance(func, ast.Name) and func.id == "CostAggregate":
                violations.append(
                    f"line {sub.lineno}: `... or CostAggregate()` inside "
                    f"reducer_node"
                )

    assert not violations, (
        "INV-B2 violation: reducer_node contains silent-zero "
        "`or CostAggregate()` fallback:\n  - "
        + "\n  - ".join(violations)
        + "\n\nReplace with cost_rollup_service.aggregate(...) pull; "
        "on aggregate failure set cost_total=CostAggregate() AND append "
        "'cost_unavailable: ...' to diagnostics_summary (INV-B3)."
    )


def test_reducer_node_calls_aggregate() -> None:
    """Positive structural assertion: reducer_node body must invoke
    ``aggregate(...)`` on a cost-rollup-service reference.

    Caught via AST: an ``await <something>.aggregate(...)`` call inside the
    reducer_node body. Pins the contract that B4 actually wired the pull.
    """
    src = _read_source()
    tree = ast.parse(src)

    reducer_node_fn: ast.AsyncFunctionDef | ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.name == "reducer_node":
                reducer_node_fn = node
                break

    assert reducer_node_fn is not None

    saw_aggregate_call = False
    for sub in ast.walk(reducer_node_fn):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Attribute) and func.attr == "aggregate":
            saw_aggregate_call = True
            break

    assert saw_aggregate_call, (
        "INV-B1 violation: reducer_node does not call `.aggregate(...)` "
        "anywhere. The B4 contract requires the cost authority to come from "
        "cost_rollup_service.aggregate(coordinator_run_id=..., "
        "child_session_ids=...), NOT from the envelope cost_summary."
    )
