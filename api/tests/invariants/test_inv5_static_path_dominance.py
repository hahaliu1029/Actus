# api/tests/invariants/test_inv5_static_path_dominance.py
"""INV-5 static: every _invoke_wrapper(...) callsite in react_graph.py
must have a 'PE-dominated' ancestor — either:
  (a) await pe.evaluate(...)  (path A), OR
  (b) state.get('pe_resume_outcomes')[...] read + variant branch (path B)

WHY: ensures no native tool is executed without permission engine
having decided. 'Function contains evaluate' is NOT sufficient — it
must dominate the wrapper call on the control flow path.

WHAT: parse the AST, find every Call to _invoke_wrapper, walk its
enclosing FunctionDef body for any pe.evaluate / pe_resume_outcomes
expression appearing strictly earlier (lexically).

SKIP LIST: functions in INV5_SKIP_FUNCTION_NAMES are exempted because
they are legacy fail-open paths that delegate to _pe_dispatch (which
enforces dominance) when PE is enabled. The legacy body handles the
PE-disabled path and is expected to call _invoke_wrapper without
pe.evaluate. PE-3 will remove these bodies.
"""

import ast
from pathlib import Path

from tests.invariants._whitelists import (
    INV5_CALLSITE_SAFETY_NET_WHITELIST,
    INV5_REACT_GRAPH_PATH,
    INV5_SKIP_FUNCTION_NAMES,
    REPO_ROOT,
)

_PE_EVALUATE_PATTERN = "evaluate"
_PE_RESUME_KEY = "pe_resume_outcomes"


def _is_pe_evaluate_await(node: ast.AST) -> bool:
    # Match `await pe.evaluate(...)` or `await self._pe.evaluate(...)`
    if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == _PE_EVALUATE_PATTERN:
            return True
    return False


def _references_pe_resume_outcomes(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and child.value == _PE_RESUME_KEY:
            return True
    return False


def _find_callsites(tree: ast.AST, target_name: str):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == target_name:
            sites.append(node)
        # Also catch attribute form: _invoke_wrapper called as helper
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == target_name:
            sites.append(node)
    return sites


def _find_enclosing_function(tree: ast.AST, target: ast.AST):
    """Return the *innermost* FunctionDef/AsyncFunctionDef enclosing target.

    PE-0 round 32 P2 fix: the previous version returned the first match from
    ``ast.walk``, which is the *outermost* function (because module-level
    functions come before their nested helpers in walk order). For nested
    helpers like ``_pe_dispatch`` defined inside ``build_react_graph``, that
    means the dominance scan ran against ``build_react_graph``'s scope —
    which sees ``pe_resume_outcomes`` references from sibling nested helpers
    (e.g. ``_pe_dispatch`` itself) that lexically precede the callsite but
    are not reachable in the actual call's lexical scope. Picking the
    *innermost* enclosing function fixes this.
    """
    target_line = getattr(target, "lineno", None)
    if target_line is None:
        return None
    best = None
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = getattr(func, "lineno", None)
        end = getattr(func, "end_lineno", None)
        if start is None or end is None:
            continue
        if start <= target_line <= end:
            # `target` is lexically inside this function. Prefer the one
            # with the *latest* starting line — i.e. the deepest nested
            # function whose body contains target.
            if best is None or start > getattr(best, "lineno", -1):
                best = func
    return best


def test_invoke_wrapper_dominated_by_pe():
    path = REPO_ROOT / INV5_REACT_GRAPH_PATH
    tree = ast.parse(path.read_text())
    callsites = _find_callsites(tree, "_invoke_wrapper")

    # PE-0 round 32: build a (function_name, lineno) -> justification lookup
    # for documented unreachable safety-net callsites.
    safety_net_lookup: dict[tuple[str, int], str] = {
        (fn, lineno): justification
        for (fn, lineno, _sunset, justification)
        in INV5_CALLSITE_SAFETY_NET_WHITELIST
    }
    safety_net_hits: set[tuple[str, int]] = set()

    violations = []
    for call in callsites:
        func = _find_enclosing_function(tree, call)
        if func is None:
            violations.append(f"line {call.lineno}: not inside any function")
            continue

        # Skip functions in the legacy exemption list
        func_name = getattr(func, "name", "")
        if func_name in INV5_SKIP_FUNCTION_NAMES:
            continue
        # Also skip any function whose name starts with _legacy_
        if func_name.startswith("_legacy_"):
            continue

        # PE-0 round 32 P2 fix: dominance requires the prior expression's
        # *entire subtree* to lie strictly before the callsite. Using only
        # `lineno < call.lineno` is unsound: an ancestor node like
        # ``for tc in batch:`` has `lineno` before the callsite but its
        # subtree spans past the callsite, so ``ast.walk(parent)`` inside
        # ``_references_pe_resume_outcomes`` would also see post-callsite
        # nodes (e.g. ``if state.get('pe_resume_outcomes'): ...`` placed
        # *after* the callsite in the same for-body) and produce a false
        # positive dominance pass.
        #
        # Fix: require `end_lineno < call.lineno` so the node and its full
        # AST subtree are guaranteed to be in source order before the
        # callsite. Python 3.8+ AST exposes `end_lineno` on statements and
        # expressions; nodes without it (e.g. operators, contexts) are
        # filtered out by the `hasattr` guard.
        prior_stmts = []
        for stmt in ast.walk(func):
            if not hasattr(stmt, "lineno"):
                continue
            end_lineno = getattr(stmt, "end_lineno", stmt.lineno)
            if end_lineno >= call.lineno:
                # Subtree may extend past callsite — recursing into it could
                # match a node that lexically follows the callsite. Skip.
                continue
            prior_stmts.append(stmt)
        dominated = any(
            _is_pe_evaluate_await(s) or _references_pe_resume_outcomes(s)
            for s in prior_stmts
        )
        if dominated:
            continue

        # Documented unreachable safety-net? Whitelisted with justification.
        key = (func_name, call.lineno)
        if key in safety_net_lookup:
            safety_net_hits.add(key)
            continue

        violations.append(
            f"{INV5_REACT_GRAPH_PATH}:{call.lineno} (in {func_name}) — "
            "_invoke_wrapper without prior pe.evaluate or pe_resume_outcomes"
        )

    assert not violations, "INV-5 static violations:\n" + "\n".join(violations)

    # Guard against stale whitelist entries — every documented safety-net
    # must still exist at the recorded line; otherwise the entry has drifted
    # and should be removed or updated.
    stale = set(safety_net_lookup.keys()) - safety_net_hits
    assert not stale, (
        "INV-5 stale safety-net whitelist entries (no matching callsite "
        f"found): {sorted(stale)}"
    )


# ----------------------------------------------------------------------
# PE-0 round 32 P2 regression tests: prove the dominance scan rejects
# the specific false-positive shape that the previous `lineno <` filter
# silently accepted.
# ----------------------------------------------------------------------


def _scan_synthetic_source(source: str) -> list[str]:
    """Run the same dominance check used by the real test on inline source
    so we can construct adversarial AST shapes without touching react_graph.
    Returns the list of unsatisfied callsites as ``func_name:lineno``.
    """
    tree = ast.parse(source)
    callsites = _find_callsites(tree, "_invoke_wrapper")
    bad = []
    for call in callsites:
        func = _find_enclosing_function(tree, call)
        if func is None:
            bad.append(f"<no-func>:{call.lineno}")
            continue
        prior_stmts = []
        for stmt in ast.walk(func):
            if not hasattr(stmt, "lineno"):
                continue
            end_lineno = getattr(stmt, "end_lineno", stmt.lineno)
            if end_lineno >= call.lineno:
                continue
            prior_stmts.append(stmt)
        dominated = any(
            _is_pe_evaluate_await(s) or _references_pe_resume_outcomes(s)
            for s in prior_stmts
        )
        if not dominated:
            bad.append(f"{getattr(func, 'name', '?')}:{call.lineno}")
    return bad


def test_dominance_rejects_post_callsite_resume_in_same_for_body():
    """False-positive shape: parent ``for`` lineno precedes the callsite but
    its subtree contains a ``pe_resume_outcomes`` reference *after* the
    callsite. The old ``lineno <`` filter would walk into the for-subtree
    via ``_references_pe_resume_outcomes`` and incorrectly mark the call
    dominated. The new ``end_lineno <`` filter must reject it.
    """
    source = (
        "async def fn(state):\n"
        "    for tc in batch:\n"
        "        await _invoke_wrapper(tc)\n"
        "        if state.get('pe_resume_outcomes'):\n"
        "            pass\n"
    )
    bad = _scan_synthetic_source(source)
    # The callsite at line 3 must be flagged as undominated.
    assert any(entry.endswith(":3") for entry in bad), (
        "Dominance scan accepted a callsite whose only pe_resume_outcomes "
        "reference is later in the same for-body. Expected to flag it. "
        f"Got: {bad}"
    )


def test_dominance_accepts_real_prior_pe_resume_outcomes():
    """Sanity: dominance still passes when ``pe_resume_outcomes`` truly
    appears before the callsite at the statement level.
    """
    source = (
        "async def fn(state):\n"
        "    outcomes = state.get('pe_resume_outcomes')\n"
        "    for tc in batch:\n"
        "        await _invoke_wrapper(tc)\n"
    )
    bad = _scan_synthetic_source(source)
    assert not bad, (
        f"Dominance scan rejected a callsite that has a real prior "
        f"pe_resume_outcomes statement: {bad}"
    )


def test_dominance_accepts_real_prior_pe_evaluate_await():
    """Sanity: dominance still passes when ``await pe.evaluate(...)``
    appears as a complete statement before the callsite.
    """
    source = (
        "async def fn(pe, tc):\n"
        "    decision = await pe.evaluate(tc)\n"
        "    await _invoke_wrapper(tc)\n"
    )
    bad = _scan_synthetic_source(source)
    assert not bad, (
        f"Dominance scan rejected a callsite with a real prior "
        f"await pe.evaluate(...) statement: {bad}"
    )


def test_enclosing_function_picks_innermost():
    """``_find_enclosing_function`` must return the innermost nested
    function — not the module-level wrapper — so the dominance scan only
    sees lexical scope reachable from the callsite.
    """
    source = (
        "def outer():\n"
        "    state_get_pe_resume_outcomes = 'pe_resume_outcomes'\n"
        "    def inner():\n"
        "        _invoke_wrapper()\n"
        "    return inner\n"
    )
    tree = ast.parse(source)
    callsites = _find_callsites(tree, "_invoke_wrapper")
    assert len(callsites) == 1
    func = _find_enclosing_function(tree, callsites[0])
    assert func is not None
    assert func.name == "inner", (
        f"Expected innermost enclosing function 'inner', got {func.name!r}"
    )
