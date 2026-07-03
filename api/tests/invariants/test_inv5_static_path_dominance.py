# api/tests/invariants/test_inv5_static_path_dominance.py
"""INV-5 v2 static (B1-1a): named-sink model + A/B/C token dominance.

Rule 1 (sink): every _invoke_wrapper(...) callsite in each INV5_SCAN_PATHS
file must have, on its AST ancestor-function chain, a function named EXACTLY
_make_execute_thunk or _legacy_make_execute_thunk (INV5_SINK_FACTORY_NAMES).
No prefix exemptions — `_legacy_backdoor()` style bypasses are rejected.
Final (Task 4): INV5_SKIP_FUNCTION_NAMES is EMPTY — tool_node's legacy body
now routes execution through _legacy_make_execute_thunk, so no function-level
exemption remains.

Rule 3 (executor zero-escape): EVERY .py file in the executor package —
RECURSIVELY (executor_package_files() uses rglob, so subdirectory modules like
executor/subpkg/backdoor.py are scanned too; codex R1-P2 widened to siblings,
R2-P2 to subdirectories) — is scanned to prove it contains ZERO execution
surface: no _invoke_wrapper, no factory reference, no *.evaluate call, and only
allowlisted imports. Intra-package imports are allowed ONLY as absolute
allowlisted prefixes OR level==1 relative imports (same-package, fully scanned);
level>=2 relative imports (`from ..graphs import ...`) climb OUT of the scanned
package and are REJECTED (codex R2-P2).

Rule 2 (dominance, re-aimed): every `_make_execute_thunk(...)` CALLSITE must
have, lexically earlier in its innermost enclosing function, one of the three
authorization tokens:
  A: `await <x>.evaluate(...)`         — fresh PE evaluation
  B: constant "pe_resume_outcomes"     — typed resume replay
  C: constant "approved_tool_call_ids" — legacy interrupt pre-approval
`_legacy_make_execute_thunk` callsites are EXEMPT from dominance (fail-open
legacy semantics, precisely scoped to that one factory's callsites).

Known limit (spec §4.1 v2 rule 4): this is lexical dominance, not control-flow
dominance — same axis of strength as the pre-v2 scanner. The semantic layer is
the behavior-test trio (test_inv5_behavior_pe_call_order.py paths A/B/C).
"""

import ast
from pathlib import Path

from tests.invariants._whitelists import (
    INV5_CALLSITE_SAFETY_NET_WHITELIST,
    INV5_EXECUTOR_MODULE_PATH,
    INV5_REACT_GRAPH_PATH,
    INV5_SCAN_PATHS,
    INV5_SINK_FACTORY_NAMES,
    INV5_SKIP_FUNCTION_NAMES,
    REPO_ROOT,
    executor_package_files,
)

_PE_EVALUATE_PATTERN = "evaluate"
_PE_RESUME_KEY = "pe_resume_outcomes"
_PRE_APPROVED_KEY = "approved_tool_call_ids"


def _is_pe_evaluate_await(node: ast.AST) -> bool:
    if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == _PE_EVALUATE_PATTERN:
            return True
    return False


def _references_authorization_token(node: ast.AST) -> bool:
    """Path B or C: a constant reference to either state key."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and child.value in (
            _PE_RESUME_KEY, _PRE_APPROVED_KEY,
        ):
            return True
    return False


def _find_callsites(tree: ast.AST, target_name: str):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == target_name:
            sites.append(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == target_name:
            sites.append(node)
    return sites


def _enclosing_function_chain(tree: ast.AST, target: ast.AST):
    """All FunctionDef/AsyncFunctionDef whose span contains target,
    ordered outermost → innermost."""
    target_line = getattr(target, "lineno", None)
    if target_line is None:
        return []
    chain = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = getattr(func, "lineno", None)
        end = getattr(func, "end_lineno", None)
        if start is None or end is None:
            continue
        if start <= target_line <= end:
            chain.append(func)
    chain.sort(key=lambda f: f.lineno)
    return chain


def _find_enclosing_function(tree: ast.AST, target: ast.AST):
    chain = _enclosing_function_chain(tree, target)
    return chain[-1] if chain else None


def _dominated(func: ast.AST, call: ast.Call) -> bool:
    """Lexically-prior full-subtree (end_lineno < call.lineno) A/B/C token."""
    prior_stmts = []
    for stmt in ast.walk(func):
        if not hasattr(stmt, "lineno"):
            continue
        end_lineno = getattr(stmt, "end_lineno", stmt.lineno)
        if end_lineno >= call.lineno:
            continue
        prior_stmts.append(stmt)
    return any(
        _is_pe_evaluate_await(s) or _references_authorization_token(s)
        for s in prior_stmts
    )


def _load_tree() -> ast.AST:
    path = REPO_ROOT / INV5_REACT_GRAPH_PATH
    return ast.parse(path.read_text())


# ----------------------------------------------------------------------
# Rule 1: named-sink — _invoke_wrapper only reachable via the two factories
# ----------------------------------------------------------------------

def test_invoke_wrapper_only_reachable_via_named_factories():
    safety_net_lookup = {
        (fn, lineno): justification
        for (fn, lineno, _sunset, justification)
        in INV5_CALLSITE_SAFETY_NET_WHITELIST
    }
    safety_net_hits: set[tuple[str, int]] = set()

    violations = []
    for rel in sorted(INV5_SCAN_PATHS):
        tree = ast.parse((REPO_ROOT / rel).read_text())
        callsites = _find_callsites(tree, "_invoke_wrapper")
        # anti-vacuity: only react_graph must carry wrapper callsites; the
        # executor file having ZERO callsites is the EXPECTED zero-escape state.
        if rel == INV5_REACT_GRAPH_PATH:
            assert callsites, (
                f"no _invoke_wrapper callsites in {rel} — scanner target drifted"
            )
        for call in callsites:
            chain = _enclosing_function_chain(tree, call)
            if not chain:
                violations.append(f"{rel}:{call.lineno}: not inside any function")
                continue
            innermost = chain[-1]
            if innermost.name in INV5_SKIP_FUNCTION_NAMES:
                continue
            chain_names = {f.name for f in chain}
            if chain_names & INV5_SINK_FACTORY_NAMES:
                continue
            key = (innermost.name, call.lineno)
            if key in safety_net_lookup:
                safety_net_hits.add(key)
                continue
            violations.append(
                f"{rel}:{call.lineno} (in {innermost.name}) — "
                "_invoke_wrapper outside the named thunk factories "
                f"(ancestor chain: {sorted(chain_names)})"
            )

    assert not violations, "INV-5 v2 sink violations:\n" + "\n".join(violations)
    stale = set(safety_net_lookup.keys()) - safety_net_hits
    assert not stale, f"INV-5 stale safety-net whitelist entries: {sorted(stale)}"


# ----------------------------------------------------------------------
# Rule 2: dominance re-aimed at _make_execute_thunk callsites (A/B/C tokens)
# ----------------------------------------------------------------------

def test_pe_factory_callsites_dominated_by_authorization_tokens():
    tree = _load_tree()
    callsites = _find_callsites(tree, "_make_execute_thunk")
    assert callsites, (
        "no _make_execute_thunk callsites found — either the factory was "
        "renamed (update INV5_SINK_FACTORY_NAMES: SECURITY review) or the "
        "PE path stopped using the two-phase seam"
    )
    violations = []
    for call in callsites:
        func = _find_enclosing_function(tree, call)
        if func is None:
            violations.append(f"line {call.lineno}: not inside any function")
            continue
        if not _dominated(func, call):
            violations.append(
                f"{INV5_REACT_GRAPH_PATH}:{call.lineno} (in {func.name}) — "
                "_make_execute_thunk without prior pe.evaluate / "
                "pe_resume_outcomes / approved_tool_call_ids"
            )
    assert not violations, "INV-5 v2 dominance violations:\n" + "\n".join(violations)


# ----------------------------------------------------------------------
# Synthetic-source regression battery (adapted from PE-0 round 32 + v2 shapes)
# ----------------------------------------------------------------------

def _scan_synthetic_sink(source: str) -> list[str]:
    tree = ast.parse(source)
    bad = []
    for call in _find_callsites(tree, "_invoke_wrapper"):
        chain = _enclosing_function_chain(tree, call)
        names = {f.name for f in chain}
        if not (names & INV5_SINK_FACTORY_NAMES):
            inner = chain[-1].name if chain else "<no-func>"
            bad.append(f"{inner}:{call.lineno}")
    return bad


def _scan_synthetic_dominance(source: str) -> list[str]:
    tree = ast.parse(source)
    bad = []
    for call in _find_callsites(tree, "_make_execute_thunk"):
        func = _find_enclosing_function(tree, call)
        if func is None or not _dominated(func, call):
            bad.append(f"{getattr(func, 'name', '?')}:{call.lineno}")
    return bad


def test_sink_rejects_legacy_backdoor_prefix_bypass():
    """v2 kills the old `_legacy_*` prefix exemption."""
    source = (
        "async def _legacy_backdoor(tc):\n"
        "    await _invoke_wrapper(tc)\n"
    )
    assert _scan_synthetic_sink(source) == ["_legacy_backdoor:2"]


def test_sink_rejects_bare_callsite_outside_factories():
    source = (
        "async def _pe_gate(tc):\n"
        "    decision = await pe.evaluate(tc)\n"
        "    await _invoke_wrapper(tc)\n"
    )
    assert _scan_synthetic_sink(source) == ["_pe_gate:3"]


def test_sink_accepts_thunk_nested_in_named_factory():
    source = (
        "def _make_execute_thunk(tc):\n"
        "    async def _execute_thunk():\n"
        "        return await _invoke_wrapper(tc)\n"
        "    return _execute_thunk\n"
    )
    assert _scan_synthetic_sink(source) == []


def test_dominance_rejects_factory_call_without_prior_token():
    source = (
        "async def _pe_gate(tc):\n"
        "    return Execute(thunk=_make_execute_thunk(tc))\n"
    )
    assert _scan_synthetic_dominance(source) == ["_pe_gate:2"]


def test_dominance_accepts_path_a_evaluate():
    source = (
        "async def _pe_gate(tc):\n"
        "    outcome = await pe.evaluate(tc)\n"
        "    return Execute(thunk=_make_execute_thunk(tc))\n"
    )
    assert _scan_synthetic_dominance(source) == []


def test_dominance_accepts_path_b_resume_outcomes():
    source = (
        "async def _pe_gate(tc, state):\n"
        "    replay = state.get('pe_resume_outcomes') or {}\n"
        "    return Execute(thunk=_make_execute_thunk(tc))\n"
    )
    assert _scan_synthetic_dominance(source) == []


def test_dominance_accepts_path_c_pre_approved():
    source = (
        "async def _pe_gate(tc, state):\n"
        "    approved = set(state.get('approved_tool_call_ids') or [])\n"
        "    return Execute(thunk=_make_execute_thunk(tc))\n"
    )
    assert _scan_synthetic_dominance(source) == []


def test_dominance_rejects_post_callsite_token_in_same_for_body():
    """PE-0 round 32 end_lineno mechanics preserved in v2."""
    source = (
        "async def fn(state):\n"
        "    for tc in batch:\n"
        "        _make_execute_thunk(tc)\n"
        "        if state.get('pe_resume_outcomes'):\n"
        "            pass\n"
    )
    bad = _scan_synthetic_dominance(source)
    assert any(entry.endswith(":3") for entry in bad)


def test_enclosing_function_picks_innermost():
    source = (
        "def outer():\n"
        "    x = 'pe_resume_outcomes'\n"
        "    def inner():\n"
        "        _make_execute_thunk()\n"
        "    return inner\n"
    )
    tree = ast.parse(source)
    callsites = _find_callsites(tree, "_make_execute_thunk")
    func = _find_enclosing_function(tree, callsites[0])
    assert func is not None and func.name == "inner"


# ----------------------------------------------------------------------
# Rule 3: executor module has ZERO escape surface (B1-1a Task 4)
# ----------------------------------------------------------------------

def test_executor_module_has_zero_escape_surface():
    """INV-5 v2 rule 3: NO file in the executor package may construct or reach
    execution. Iterates over EVERY package file (codex R1-P2 — sibling modules
    like tool_call_stream_collector.py / __init__.py were previously
    unscanned)."""
    package_files = executor_package_files()
    # anti-vacuity: the primary module must be in the globbed set (guards
    # against a broken glob silently scanning nothing).
    assert INV5_EXECUTOR_MODULE_PATH in package_files, (
        "executor package glob did not include batch_tool_executor.py — "
        f"scan surface drifted (found: {package_files})"
    )
    for rel in package_files:
        src = (REPO_ROOT / rel).read_text()
        tree = ast.parse(src)
        assert "_invoke_wrapper" not in src, f"{rel} references _invoke_wrapper"
        for name in INV5_SINK_FACTORY_NAMES:
            assert name not in src, f"{rel} must not reference factory {name}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "evaluate", (
                    f"{rel} must not call *.evaluate (line {node.lineno})"
                )


def test_legacy_factory_callsites_exist_and_are_dominance_exempt():
    """_legacy_make_execute_thunk callsites exist (anti-vacuity) and are the
    ONLY dominance-exempt surface (fail-open semantics, precisely scoped)."""
    tree = _load_tree()
    legacy_sites = _find_callsites(tree, "_legacy_make_execute_thunk")
    assert legacy_sites, (
        "no _legacy_make_execute_thunk callsites — legacy body drifted off "
        "the two-phase seam (SECURITY review required)"
    )
    # 支配测试（test_pe_factory_callsites_dominated_by_authorization_tokens）
    # 按名只扫 _make_execute_thunk —— 本测试锁定豁免面的存在性与唯一性。


# ----------------------------------------------------------------------
# Rule 3+ (R6#1): factory nesting + wrapper-inside-inner-thunk static guard
# ----------------------------------------------------------------------

_FACTORY_EXPECTED_PARENT = {
    "_make_execute_thunk": "_pe_dispatch",
    "_legacy_make_execute_thunk": "tool_node",
}


def test_factories_nested_in_dispatch_and_wrapper_inside_inner_thunk():
    """R6#1（spec §4.1 R3#4 的静态化）：两个工厂必须嵌套定义在
    _pe_dispatch / tool_node 内部；且工厂 span 内的每个 _invoke_wrapper
    callsite 的最内层包裹函数必须是工厂**内部嵌套**的 thunk 函数——
    工厂体直调 wrapper（per-call 直接执行、无 thunk 交付 executor）被拒。"""
    tree = _load_tree()
    factories = [
        f for f in ast.walk(tree)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
        and f.name in INV5_SINK_FACTORY_NAMES
    ]
    assert {f.name for f in factories} == set(INV5_SINK_FACTORY_NAMES), (
        "missing factory definition(s) — two-phase seam drifted"
    )
    for factory in factories:
        parent_names = {
            fn.name for fn in _enclosing_function_chain(tree, factory)
        } - {factory.name}
        expected = _FACTORY_EXPECTED_PARENT[factory.name]
        assert expected in parent_names, (
            f"{factory.name} (line {factory.lineno}) must be nested inside "
            f"{expected} — module-level/foreign-scope factory is a bypass "
            "(SECURITY review required)"
        )
        wrapper_calls = [
            c for c in _find_callsites(tree, "_invoke_wrapper")
            if factory.lineno <= c.lineno <= (factory.end_lineno or factory.lineno)
        ]
        assert wrapper_calls, f"{factory.name} contains no wrapper callsite"
        for call in wrapper_calls:
            innermost = _find_enclosing_function(tree, call)
            assert innermost is not None and innermost is not factory and \
                innermost.lineno > factory.lineno, (
                    f"{factory.name}: wrapper callsite at line {call.lineno} "
                    "must live in a NESTED thunk function, not the factory "
                    "body itself (direct per-call execution bypass)"
                )


def _scan_synthetic_factory_nesting(source: str) -> list[str]:
    """Replica of test_factories_nested_... on synthetic source; returns the
    list of rejection reasons (empty = all shapes accepted)."""
    tree = ast.parse(source)
    bad: list[str] = []
    factories = [
        f for f in ast.walk(tree)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
        and f.name in INV5_SINK_FACTORY_NAMES
    ]
    for factory in factories:
        parent_names = {
            fn.name for fn in _enclosing_function_chain(tree, factory)
        } - {factory.name}
        expected = _FACTORY_EXPECTED_PARENT[factory.name]
        if expected not in parent_names:
            bad.append(f"{factory.name}:not-nested-in-{expected}")
            continue
        wrapper_calls = [
            c for c in _find_callsites(tree, "_invoke_wrapper")
            if factory.lineno <= c.lineno <= (factory.end_lineno or factory.lineno)
        ]
        for call in wrapper_calls:
            innermost = _find_enclosing_function(tree, call)
            if not (innermost is not None and innermost is not factory
                    and innermost.lineno > factory.lineno):
                bad.append(f"{factory.name}:wrapper-in-factory-body:{call.lineno}")
    return bad


def test_factory_nesting_rejects_module_level_factory():
    """Adversarial: a module-level same-named factory (no _pe_dispatch/
    tool_node parent) is rejected by the nesting check."""
    source = (
        "def _make_execute_thunk(tc):\n"
        "    async def _execute_thunk():\n"
        "        return await _invoke_wrapper(tc)\n"
        "    return _execute_thunk\n"
    )
    bad = _scan_synthetic_factory_nesting(source)
    assert bad == ["_make_execute_thunk:not-nested-in-_pe_dispatch"]


def test_factory_nesting_rejects_wrapper_in_factory_body():
    """Adversarial: factory body directly calls _invoke_wrapper (no nested
    thunk delivering to the executor) is rejected."""
    source = (
        "async def _pe_dispatch(state, config):\n"
        "    async def _make_execute_thunk(tc):\n"
        "        return await _invoke_wrapper(tc)\n"
        "    return _make_execute_thunk\n"
    )
    bad = _scan_synthetic_factory_nesting(source)
    assert bad == ["_make_execute_thunk:wrapper-in-factory-body:3"]


# ----------------------------------------------------------------------
# Rule 3+ (R6#2): executor module import allowlist (P-6 purity AST-ised)
# ----------------------------------------------------------------------

_EXECUTOR_IMPORT_ALLOWLIST = (
    "__future__", "asyncio", "dataclasses", "typing", "logging", "json",
    "langchain_core", "app.domain.models",
    # intra-package re-exports (__init__.py pulls its own siblings up); these
    # stay WITHIN the scanned package so they cannot be an escape surface.
    "app.domain.services.executor",
)


def _import_allowed(node: ast.AST, mod: str) -> bool:
    """Allow allowlisted absolute imports OR a SAME-PACKAGE relative import
    (ast.ImportFrom with level==1) — a `from . import x` / `from .sibling import
    y` / `from .subpkg.mod import z` reaches only package members, all of which
    are themselves rglob-scanned, so it is NOT an escape surface.

    codex R2-P2: level>=2 is REJECTED. `from ..graphs import react_graph as rg`
    (level==2) climbs OUT of the executor package into a sibling package that is
    NOT scanned, laundering _invoke_wrapper via attribute access with no
    forbidden literal token. Only level==1 (rooted at the executor package
    itself, fully under the rglob scan) is safe. The ban on
    importlib/__import__/langgraph/react_graph is enforced separately (token
    scan + allowlist prefix)."""
    if isinstance(node, ast.ImportFrom) and (node.level or 0) >= 1:
        return (node.level or 0) == 1
    # codex R3-P2: EXACT-or-dotted-prefix match, NOT tuple startswith. A raw
    # `mod.startswith(("app.domain.services.executor", ...))` also matches
    # same-prefix SIBLING modules OUTSIDE the allowlisted namespaces — e.g.
    # `app.domain.services.executor_backdoor` (a sibling module, NOT under the
    # rglob-scanned `executor` package) or `langchain_core_evil`. `mod == entry`
    # covers importing the namespace itself; `mod.startswith(entry + ".")`
    # covers genuine sub-members; neither admits `entry` + a non-`.` char.
    return any(
        mod == entry or mod.startswith(entry + ".")
        for entry in _EXECUTOR_IMPORT_ALLOWLIST
    )


def test_executor_module_import_allowlist():
    """R6#2：executor 包内每个模块只允许白名单 import——禁 react_graph/langgraph/
    importlib 及 `__import__` 动态逃逸（P-6 纯净契约的 AST 化）。iterates over
    EVERY package file (codex R1-P2). 包内 relative/absolute 自引用被放行
    （仍在被扫描的包内，非逃逸面）。"""
    package_files = executor_package_files()
    assert INV5_EXECUTOR_MODULE_PATH in package_files, (
        "executor package glob did not include batch_tool_executor.py — "
        f"scan surface drifted (found: {package_files})"
    )
    for rel in package_files:
        src = (REPO_ROOT / rel).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for mod in modules:
                assert _import_allowed(node, mod), (
                    f"{rel} imports {mod!r} (line {node.lineno}) — outside "
                    "the allowlist (SECURITY review required)"
                )
        assert "__import__" not in src and "importlib" not in src, (
            f"dynamic import escape in {rel}"
        )


# ----------------------------------------------------------------------
# Non-vacuity (codex R1-P2): prove the WIDENED package scan REJECTS a forbidden
# escape planted in a NON-batch-executor sibling module (e.g. the collector).
# These mirror the real scan's per-file logic on synthetic source — a green
# result would mean the widening is load-bearing (it rejects real escapes),
# not vacuous.
# ----------------------------------------------------------------------

def _scan_synthetic_zero_escape(source: str) -> list[str]:
    """Replica of the Rule-3 per-file body on synthetic source; returns the
    list of escape-surface reasons (empty = clean)."""
    bad: list[str] = []
    tree = ast.parse(source)
    if "_invoke_wrapper" in source:
        bad.append("references _invoke_wrapper")
    for name in INV5_SINK_FACTORY_NAMES:
        if name in source:
            bad.append(f"references factory {name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "evaluate":
                bad.append(f"*.evaluate call (line {node.lineno})")
    return bad


def _scan_synthetic_import_allowlist(source: str) -> list[str]:
    """Replica of the Rule-3+ import-allowlist per-file body on synthetic
    source; returns rejection reasons (empty = all imports allowed)."""
    bad: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for mod in modules:
            if not _import_allowed(node, mod):
                bad.append(f"import {mod!r}:{node.lineno}")
    if "__import__" in source or "importlib" in source:
        bad.append("dynamic import escape")
    return bad


def test_package_scan_rejects_invoke_wrapper_in_sibling_collector():
    """A sibling (non-batch-executor) module gaining an _invoke_wrapper
    reference must be flagged — proves the widened scan is load-bearing."""
    source = (
        "class ToolCallStreamCollector:\n"
        "    async def escape(self, tc):\n"
        "        return await _invoke_wrapper(tc)\n"
    )
    assert _scan_synthetic_zero_escape(source) == ["references _invoke_wrapper"]


def test_package_scan_rejects_pe_evaluate_in_sibling_collector():
    """A sibling module gaining a *.evaluate call must be flagged."""
    source = (
        "class ToolCallStreamCollector:\n"
        "    async def escape(self, pe, tc):\n"
        "        return await pe.evaluate(tc)\n"
    )
    assert _scan_synthetic_zero_escape(source) == ["*.evaluate call (line 3)"]


def test_package_scan_rejects_importlib_in_sibling_collector():
    """A sibling module gaining `import importlib` must be flagged by the
    widened import-allowlist scan (dynamic-import escape)."""
    source = "import importlib\n"
    bad = _scan_synthetic_import_allowlist(source)
    assert any("importlib" in b for b in bad)
    assert "dynamic import escape" in bad


def test_package_scan_rejects_react_graph_import_in_sibling_collector():
    """A sibling module reaching back into react_graph must be flagged."""
    source = "from app.domain.services.graphs.react_graph import _invoke_wrapper\n"
    bad = _scan_synthetic_import_allowlist(source)
    assert bad == ["import 'app.domain.services.graphs.react_graph':1"]


def test_package_scan_accepts_intra_package_reexport_init():
    """The real __init__.py shape (absolute intra-package re-exports) must be
    ACCEPTED — the ban must not weaken to catch legitimate self-imports."""
    source = (
        "from app.domain.services.executor.batch_tool_executor import Execute\n"
        "from app.domain.services.executor.tool_call_stream_collector import (\n"
        "    ToolCallStreamCollector,\n"
        ")\n"
    )
    assert _scan_synthetic_import_allowlist(source) == []


def test_package_scan_accepts_relative_intra_package_import():
    """A relative intra-package import (`from .sibling import x`) is accepted —
    it reaches only scanned siblings, so it is not an escape surface."""
    source = "from .batch_tool_executor import Execute\n"
    assert _scan_synthetic_import_allowlist(source) == []


def test_package_scan_rejects_relative_out_of_package_reach():
    """codex R2-P2: a relative import that climbs OUT of the package
    (`from ..graphs import react_graph as rg`, level==2) reaches a sibling
    package that is NOT rglob-scanned, laundering _invoke_wrapper via attribute
    access with no forbidden literal token. The tightened policy REJECTS
    level>=2 — only level==1 (rooted at the executor package, fully scanned) is
    the safe carve-out. This locks the rejection the test name has always
    claimed (previously it only asserted the `import langgraph` ban and
    acknowledged the level>=2 gap in a comment)."""
    source = "from ..graphs import react_graph as rg\n"
    bad = _scan_synthetic_import_allowlist(source)
    assert bad == ["import 'graphs':1"]


def test_package_scan_rejects_deeper_relative_out_of_package_reach():
    """codex R2-P2: even level==3 (`from ...prompts import x`) — climbing two
    packages up — is rejected. Any level>=2 leaves the executor package."""
    source = "from ...prompts import assembler\n"
    bad = _scan_synthetic_import_allowlist(source)
    assert bad == ["import 'prompts':1"]


def test_package_scan_accepts_relative_subpackage_import():
    """codex R2-P2: a level==1 import reaching into a package SUBDIRECTORY
    (`from .subpkg.mod import x`) is ACCEPTED — subpkg is now rglob-scanned, so
    it stays within the scanned package."""
    source = "from .subpkg.helper import Execute\n"
    assert _scan_synthetic_import_allowlist(source) == []


def test_package_scan_rejects_same_prefix_sibling_module_import():
    """codex R3-P2: a same-prefix SIBLING module outside the scanned package —
    `app.domain.services.executor_backdoor` — must be REJECTED. It shares the
    `app.domain.services.executor` prefix but is NOT under the rglob-scanned
    `executor` package (which is `app.domain.services.executor.*`), so a naive
    tuple `startswith` would wrongly admit it as an escape surface. The
    exact-or-dotted-prefix matcher closes the hole."""
    absolute = "import app.domain.services.executor_backdoor\n"
    assert _scan_synthetic_import_allowlist(absolute) == [
        "import 'app.domain.services.executor_backdoor':1"
    ]
    from_form = "from app.domain.services.executor_backdoor import x\n"
    assert _scan_synthetic_import_allowlist(from_form) == [
        "import 'app.domain.services.executor_backdoor':1"
    ]


def test_package_scan_rejects_same_prefix_sibling_langchain_core():
    """codex R3-P2: the same prefix hole also existed for the `langchain_core`
    entry — `langchain_core_evil` shares the prefix but is a different top-level
    package. Exact-or-dotted-prefix rejects it while still accepting genuine
    `langchain_core.*` members (positive control below)."""
    source = "import langchain_core_evil\n"
    assert _scan_synthetic_import_allowlist(source) == [
        "import 'langchain_core_evil':1"
    ]


def test_package_scan_accepts_executor_submember_after_tightening():
    """Positive control for codex R3-P2: a genuine executor sub-member import
    (`from app.domain.services.executor.batch_tool_executor import
    BatchToolExecutor`) and a bare `langchain_core.*` member stay ACCEPTED — the
    tightened matcher must not over-reject legitimate allowlist members."""
    submember = (
        "from app.domain.services.executor.batch_tool_executor import "
        "BatchToolExecutor\n"
    )
    assert _scan_synthetic_import_allowlist(submember) == []
    langchain_member = "from langchain_core.messages import AIMessage\n"
    assert _scan_synthetic_import_allowlist(langchain_member) == []
    # importing the allowlisted namespace itself (exact match) is accepted
    namespace = "import langchain_core\n"
    assert _scan_synthetic_import_allowlist(namespace) == []


def test_executor_package_glob_covers_all_current_siblings():
    """Anti-vacuity for the dynamic glob: the current package really contains
    the collector + __init__ + batch executor, and all are in the scan set."""
    files = set(executor_package_files())
    expected = {
        "api/app/domain/services/executor/__init__.py",
        "api/app/domain/services/executor/batch_tool_executor.py",
        "api/app/domain/services/executor/tool_call_stream_collector.py",
    }
    assert expected <= files, (
        f"glob missing expected package files: {expected - files}"
    )
    # every globbed file is also in the Rule-1 sink scan surface
    assert files <= INV5_SCAN_PATHS, (
        f"executor files not in INV5_SCAN_PATHS: {files - INV5_SCAN_PATHS}"
    )


def _rglob_package_files(pkg: Path) -> set[str]:
    """Replica of executor_package_files()'s recursive collection on an
    arbitrary package root; returns relpaths (POSIX) under pkg."""
    return {
        str(p.relative_to(pkg)).replace("\\", "/")
        for p in pkg.rglob("*.py")
        if "__pycache__" not in p.parts
    }


def test_recursive_glob_catches_subdirectory_module(tmp_path):
    """codex R2-P2: prove the recursive glob is load-bearing — a module placed
    in a package SUBDIRECTORY (`subpkg/backdoor.py`) IS collected. Under the old
    top-level `glob('*.py')` this file would NEVER be scanned, leaving a
    subdirectory escape surface. Also proves __pycache__ is excluded."""
    pkg = tmp_path / "executor"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "batch_tool_executor.py").write_text("import asyncio\n")
    subpkg = pkg / "subpkg"
    subpkg.mkdir()
    (subpkg / "__init__.py").write_text("")
    (subpkg / "backdoor.py").write_text("import importlib\n")
    # a compiled artifact under __pycache__ must be excluded
    pycache = pkg / "__pycache__"
    pycache.mkdir()
    (pycache / "batch_tool_executor.py").write_text("# stray\n")

    collected = _rglob_package_files(pkg)
    assert "subpkg/backdoor.py" in collected, (
        "recursive glob failed to reach a subdirectory module — subdir escape "
        "surface would be unscanned"
    )
    assert "__init__.py" in collected
    assert "batch_tool_executor.py" in collected
    assert not any("__pycache__" in f for f in collected), (
        f"__pycache__ leaked into scan set: {collected}"
    )
    # and the planted subdir escape (import importlib) is caught by the
    # per-file scan the recursive glob now feeds
    escape_src = (subpkg / "backdoor.py").read_text()
    assert "dynamic import escape" in _scan_synthetic_import_allowlist(escape_src)
