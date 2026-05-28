"""[C2 PR-9 §15.2 Gate #5] ``DefaultPermissionEngine.evaluate`` must invoke
``ChildScopeGate.check_in_scope`` BEFORE source dispatch.

Spec invariant: the child scope gate is the *prologue* of the permission
evaluation — it must run **before** any source-specific risk assessment or
writer touch. The PR-2 implementation at
``app/domain/services/permission/default_engine.py`` enforces this by
gating on ``ctx.child_permission_context`` at the very top of ``evaluate``
(currently around line 314-348), and only dispatching to
``self._sources.get(call.tool_source)`` later (line 421 at the time of
writing). This AST gate locks that ordering in.

Heuristic adaptation (vs. plan §15.2 listing):
    The plan's listing scanned for a ``for ... in <attr-with-name-containing-"source">``
    loop and asserted that ``check_in_scope`` was called before the loop's
    ``lineno``. The current ``evaluate`` implementation does **not** iterate
    sources — it does a single dict lookup ``self._sources.get(call.tool_source)``
    (frozen at PE-1 §2.4 step 5.5; see
    ``default_engine.py:417-434``). So the heuristic here scans for the
    canonical source-dispatch line: an attribute access whose ``.attr`` is
    ``_sources`` (matches both ``self._sources.get(...)`` and
    ``self._sources[...]``). If the implementation ever reverts to a loop,
    the alternate ``for`` heuristic still fires as a fallback.

KNOWN FALSE-NEGATIVE RISK (intentional, documented per PR-9 task spec):
    1. If ``_sources`` is renamed (e.g. to ``_source_registry``), the
       dispatch-line heuristic will not match → assertion ``assert
       source_dispatch_lines`` will fail. That is desired — a reviewer will
       notice and update the gate.
    2. If ``check_in_scope`` is renamed at the gate API, the prologue
       heuristic will not match → assertion ``assert check_in_scope_lines``
       will fail. Same: desired.
    3. The gate measures **line ordering**, not control-flow ordering. A
       ``check_in_scope`` call inside a *later* code branch (e.g. inside a
       sub-method called from evaluate) would not be detected, because we
       only walk the direct AST of ``evaluate`` itself. The runtime
       complement is ``test_default_engine_child_prologue.py:test_prologue_runs_before_source_loop``
       which mocks every collaborator and asserts no downstream writer is
       touched when the gate denies.
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.coordinator_pure


# Resolve repo-relative ``api/app/...`` paths from this test file's location so
# the gate works regardless of pytest's CWD (``api/`` vs repo root).
_API_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ENGINE = (
    _API_ROOT / "app" / "domain" / "services" / "permission" / "default_engine.py"
)


def test_child_gate_called_before_source_loop():
    """[r3 P1-5] AST scan: find ``check_in_scope`` call BEFORE source dispatch.

    Source dispatch = either a ``for`` loop iterating an attribute whose
    name contains "source", OR an attribute access on ``_sources`` (current
    ``self._sources.get(call.tool_source)`` pattern).
    """
    src = _DEFAULT_ENGINE.read_text()
    tree = ast.parse(src)
    found_evaluate = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "evaluate":
            found_evaluate = True
            check_in_scope_lines = []
            source_dispatch_lines = []
            for s in ast.walk(node):
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute):
                    if s.func.attr == "check_in_scope":
                        check_in_scope_lines.append(s.lineno)
                # Heuristic A: ``for x in <attr-with-"source">`` (legacy / fallback).
                if isinstance(s, (ast.For, ast.AsyncFor)) and isinstance(s.iter, ast.Attribute):
                    if "source" in s.iter.attr.lower():
                        source_dispatch_lines.append(s.lineno)
                # Heuristic B: any attribute access whose ``.attr`` contains
                # "_sources" — matches both ``self._sources.get(...)`` and
                # ``self._sources[...]`` (current PE-1 §2.4 dispatch shape).
                if isinstance(s, ast.Attribute) and "_sources" in s.attr:
                    source_dispatch_lines.append(s.lineno)
            assert check_in_scope_lines, "ChildScopeGate.check_in_scope not called in evaluate"
            assert source_dispatch_lines, "source dispatch (loop OR _sources attribute) not found in evaluate"
            assert min(check_in_scope_lines) < min(source_dispatch_lines), (
                f"ChildScopeGate must run BEFORE source dispatch "
                f"(gate@{min(check_in_scope_lines)} vs dispatch@{min(source_dispatch_lines)})"
            )
    assert found_evaluate, "evaluate method not found in DefaultPermissionEngine"
