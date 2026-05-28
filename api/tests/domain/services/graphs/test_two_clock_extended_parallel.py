"""[C2 PR-9 §15.2 Gate #2] Two-Clock 扩展 — parallel backend / reducer / worker
must NOT write ``state["skill_context"]``.

Background: PR-8 sealed ``executor_node`` (see
``test_executor_no_skill_context_writeback.py``) so that ``updater_node`` is
the only writer of ``state.skill_context``. This gate extends the same
invariant to the parallel execution path — neither the outer
``_run_parallel_backend`` helper nor the inner subgraph's ``reducer_node`` /
``worker_node`` may write back ``state["skill_context"]``.

This gate is COMPLEMENTARY (not parallel) to the PR-8
``test_executor_no_skill_context_writeback`` suite, which guards
``executor_node`` via a different heuristic: it scans ``Command(update={...})``
dict-literal keys (see ``_extract_dict_keys_in_command_update``). The two
heuristics catch DIFFERENT attack surfaces:

- PR-8 gate: ``return Command(update={"skill_context": ...})`` — dict literal
- This gate (PR-9): ``state["skill_context"] = ...`` — subscript ASSIGN
  (``ast.Store`` context only — reads are NOT flagged)

Known gaps (intentionally accepted; runtime tests cover the rest):
- Variable-based construction (``var = {...}; Command(update=var)``)
- Variable-key subscript: ``KEY = "skill_context"; state[KEY] = ...`` --
   ``ast.Subscript(slice=ast.Name(id="KEY"))`` does not match the literal
   ``ast.Constant`` slice check. Promoting magic strings to module-level
   constants would silently bypass this gate.
- Dict unpacking (``Command(update={**base, "skill_context": ...})``)
- ``state.update({"skill_context": ...})`` -- batched-write via dict method.
   Produces ``ast.Call`` (not ``ast.Subscript``), so the subscript scanner
   does not fire. A future refactor that consolidates many ``state[X]=Y``
   into a single ``state.update({...})`` block would silently bypass this
   gate.
- Helper functions called from the target that themselves write skill_context
  (no control-flow analysis)
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.coordinator_graph


# Resolve repo-relative ``api/app/...`` paths from this test file's location so
# the gate works regardless of pytest's CWD (``api/`` vs repo root).
_API_ROOT = Path(__file__).resolve().parents[4]
_MAIN_GRAPH = _API_ROOT / "app" / "domain" / "services" / "graphs" / "main_graph.py"
_SUBGRAPH = (
    _API_ROOT / "app" / "domain" / "services" / "graphs" / "parallel_execution_subgraph.py"
)


def _forbidden_writes(file_path: Path, function_names: list[str]) -> list[str]:
    src = file_path.read_text()
    tree = ast.parse(src)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_names:
            for s in ast.walk(node):
                # detect: state["skill_context"] = ...
                # (READ access via state["skill_context"] is allowed; only writes are forbidden)
                if (
                    isinstance(s, ast.Subscript)
                    and isinstance(s.slice, ast.Constant)
                    and s.slice.value == "skill_context"
                    and isinstance(s.ctx, ast.Store)
                ):
                    violations.append(f"{node.name}:{s.lineno}")
    return violations


def test_run_parallel_backend_no_skill_context_write():
    assert not _forbidden_writes(_MAIN_GRAPH, ["_run_parallel_backend"])


def test_reducer_node_no_skill_context_write():
    assert not _forbidden_writes(_SUBGRAPH, ["reducer_node"])


def test_worker_node_no_skill_context_write():
    assert not _forbidden_writes(_SUBGRAPH, ["worker_node"])
