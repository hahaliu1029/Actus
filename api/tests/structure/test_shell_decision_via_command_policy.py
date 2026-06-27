"""C5b §8.7 (INV-1) — structural lock: every shell allow/deny decision flows through
the command-policy evaluator, fed by the shared build_command_policy, and the old
`if not <validation_result>.allowed:` predicate is gone.

Inherent limitation (same as the N1 structural test): static AST scanning asserts
structure/ordering, not runtime control flow. Runtime coverage: the routing-identity
tests in test_react_graph_policy_observe.py and test_skill_tool_ast_validator.py.
"""
from __future__ import annotations

import ast
import pathlib


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _find_function_recursive(tree: ast.AST, name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _find_method(tree: ast.AST, *, class_name: str, method_name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return item
    return None


def _calls_to(scope: ast.AST, name: str) -> list[ast.Call]:
    return [
        n for n in ast.walk(scope)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


def _constructs_command_policy(scope: ast.AST) -> bool:
    """True iff `scope` directly constructs CommandPolicy(...) — bare OR attribute-qualified
    (`sp.CommandPolicy(...)`) — so a module-qualified decoy can't evade the lock. [codex planR8 P3]"""
    for n in ast.walk(scope):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if (isinstance(f, ast.Name) and f.id == "CommandPolicy") or (
            isinstance(f, ast.Attribute) and f.attr == "CommandPolicy"
        ):
            return True
    return False


def _name_bound_to_call(fn: ast.AST, call_node: ast.Call) -> str | None:
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Assign)
            and n.value is call_node
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            return n.targets[0].id
    return None


def _calls_in_direct_body(if_node: ast.If, name: str) -> list[ast.Call]:
    """Calls to `name` in GUARANTEED-EXECUTED position of the denial branch's DIRECT body:
    the immediate value of a top-level `x = name(...)`, `return name(...)`, `name(...)`, or
    `await name(...)` statement. A call buried where it does NOT actually run — inside a nested
    block (dead `if False:` / loop / try), or in a non-executed sub-expression
    (`False and name(...)`, a ternary `name(...) if c else None`, a lambda) — is NOT credited,
    so a decoy denial branch that never actually denies cannot satisfy the lock. This is a
    statement-SHAPE check, not a broad ast.walk, precisely to defeat that decoy class.
    [codex implR1 P3 + implR2 P3]"""
    calls: list[ast.Call] = []
    for stmt in if_node.body:
        if isinstance(stmt, (ast.Assign, ast.Return, ast.Expr)):
            value = stmt.value
        else:
            continue  # block stmts / bare `continue` / etc. — not a call-bearing simple stmt
        if isinstance(value, ast.Await):
            value = value.value  # unwrap `await finalizer(...)`
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == name:
            calls.append(value)
    return calls


def _assert_gate_decides_via_policy(
    fn: ast.AST, *, result_var: str, denial_helper: str, finalizer: str | None
) -> None:
    # (a) evaluate_command is called; its policy= is build_command_policy(...) (or a
    #     local bound to one); and the fn constructs NO CommandPolicy(...) directly.
    eval_calls = _calls_to(fn, "evaluate_command")
    assert eval_calls, "gate must call evaluate_command()"
    assert not _constructs_command_policy(fn), "gate must NOT construct CommandPolicy() directly"
    assert _calls_to(fn, "build_command_policy"), "gate must build policy via build_command_policy()"

    ev = eval_calls[0]

    # (a3) evaluate_command runs at/after validate() (spec §8.7 ordering). [codex planR2 P3]
    validate_calls = _calls_to(fn, "validate")
    assert validate_calls, "gate must call validate() before deciding"
    assert ev.lineno >= min(c.lineno for c in validate_calls), (
        "evaluate_command must run at/after validate()"
    )

    policy_kw = next((k for k in ev.keywords if k.arg == "policy"), None)
    assert policy_kw is not None, "evaluate_command must be called with policy=..."
    pol = policy_kw.value
    if isinstance(pol, ast.Call):
        assert isinstance(pol.func, ast.Name) and pol.func.id == "build_command_policy", (
            "evaluate_command(policy=...) must be a build_command_policy(...) call"
        )
    elif isinstance(pol, ast.Name):
        bound = any(
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == pol.id
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "build_command_policy"
            for n in ast.walk(fn)
        )
        assert bound, "evaluate_command(policy=<name>) must be bound to build_command_policy(...)"
    else:
        raise AssertionError("evaluate_command(policy=...) must be a build_command_policy call/name")

    # (a2) capture the denial `if not <decision>.allowed:` branch, where <decision> is the
    #      name bound to evaluate_command(...) (or an inline evaluate_command(...).allowed).
    decision_var = _name_bound_to_call(fn, ev)
    denial_if = None
    for node in ast.walk(fn):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)):
            continue
        operand = node.test.operand
        if not (isinstance(operand, ast.Attribute) and operand.attr == "allowed"):
            continue
        val = operand.value
        if (decision_var is not None and isinstance(val, ast.Name) and val.id == decision_var) or (
            isinstance(val, ast.Call) and isinstance(val.func, ast.Name)
            and val.func.id == "evaluate_command"
        ):
            denial_if = node
            break
    assert denial_if is not None, "the denial branch must read `not <evaluate_command result>.allowed`"

    # (a2') the `not <decision>.allowed` branch must UNCONDITIONALLY produce the denial AND
    #       terminate — checked on the DIRECT branch body (not ast.walk), so NEITHER a decoy
    #       `if not <decision>.allowed: pass` + a SEPARATE re-derived denial [codex planR5 P2]
    #       NOR a dead-nested `if False: <denial>; continue` (whose terminal never fires, so
    #       control falls through to execute the command) can satisfy the lock. [codex implR1 P3]
    assert _calls_in_direct_body(denial_if, denial_helper), (
        f"the `not {decision_var}.allowed` branch must DIRECTLY call {denial_helper}(...) (the real denial)"
    )
    if finalizer is not None:  # PE / legacy gates: finalize + continue the loop
        assert _calls_in_direct_body(denial_if, finalizer), (
            f"the denial branch must DIRECTLY call {finalizer}(...)"
        )
        assert any(isinstance(n, ast.Continue) for n in denial_if.body), (
            "the denial branch must `continue` after finalizing (direct child, not nested)"
        )
    else:  # native (_invoke_native): return the legacy denial
        assert any(isinstance(n, ast.Return) for n in denial_if.body), (
            "the native denial branch must `return` the denial (direct child, not nested)"
        )

    # (b) NO If.test reads `.allowed` on the validation-result var (old predicate is gone).
    #     `.allowed` is permitted only as a ValidationResultView(allowed=...) Call kwarg,
    #     which is never an If.test.
    # (b2) NO If.test re-derives the decision from <result_var>.code or DENY_VALIDATION_CODES
    #      (kills the bypass `if ast_result.code in DENY_VALIDATION_CODES:`). [codex planR5 P2]
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        for sub in ast.walk(node.test):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == "allowed"
                and isinstance(sub.value, ast.Name)
                and sub.value.id == result_var
            ):
                raise AssertionError(
                    f"decision must not branch on {result_var}.allowed — decide via "
                    f"evaluate_command(...) instead (the old predicate must be gone)"
                )
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == "code"
                and isinstance(sub.value, ast.Name)
                and sub.value.id == result_var
            ):
                raise AssertionError(
                    f"decision must not re-derive from {result_var}.code in an If.test — "
                    f"the evaluate_command policy decision must drive the branch"
                )
            if isinstance(sub, ast.Name) and sub.id == "DENY_VALIDATION_CODES":
                raise AssertionError(
                    "decision must not re-derive from DENY_VALIDATION_CODES in the gate"
                )


def _react_graph_tree() -> ast.AST:
    src = (_repo_root() / "api/app/domain/services/graphs/react_graph.py").read_text()
    return ast.parse(src)


def test_pe_dispatch_decides_via_command_policy() -> None:
    fn = _find_function_recursive(_react_graph_tree(), "_pe_dispatch")
    assert fn is not None, "_pe_dispatch closure missing from react_graph.py"
    _assert_gate_decides_via_policy(
        fn, result_var="_ast_result", denial_helper="to_typed_denied", finalizer="_finalize_pe_outcome"
    )


def test_tool_node_decides_via_command_policy() -> None:
    fn = _find_function_recursive(_react_graph_tree(), "tool_node")
    assert fn is not None, "tool_node closure missing from react_graph.py"
    _assert_gate_decides_via_policy(
        fn, result_var="ast_result", denial_helper="to_typed_denied", finalizer="_finalize_outcome"
    )


def test_invoke_native_decides_via_command_policy() -> None:
    src = (_repo_root() / "api/app/domain/services/tools/skill.py").read_text()
    fn = _find_method(ast.parse(src), class_name="SkillTool", method_name="_invoke_native")
    assert fn is not None, "SkillTool._invoke_native missing from skill.py"
    _assert_gate_decides_via_policy(
        fn, result_var="ast_result", denial_helper="to_legacy_tool_result", finalizer=None
    )


def test_compiler_builds_command_policy_via_shared_builder() -> None:
    # (c) compile_tool_call uses build_command_policy and the compiler module
    #     constructs NO CommandPolicy(...) directly (the only construction flows
    #     through the shared builder; compile_container_create never builds one).
    src = (_repo_root() / "api/app/domain/services/safety/sandbox_policy_compiler.py").read_text()
    tree = ast.parse(src)
    fn = _find_method(tree, class_name="SandboxPolicyCompiler", method_name="compile_tool_call")
    assert fn is not None, "SandboxPolicyCompiler.compile_tool_call missing"
    assert _calls_to(fn, "build_command_policy"), "compile_tool_call must use build_command_policy()"
    # (c2) verdict is derived from the evaluator, not v.allowed (codex planR1 P2):
    assert _calls_to(fn, "evaluate_command"), "compile_tool_call must derive verdict via evaluate_command()"
    # (c3) verdict must NOT be read from inp.validation (`v`) — forbid any `v.allowed` access so a
    #      sloppy impl can't add an UNUSED evaluate_command() yet keep `verdict="ok" if v.allowed`.
    #      The evaluator result is bound to a DIFFERENT name (e.g. `_decision`), so `_decision.allowed`
    #      stays permitted. [codex planR3 P2]
    for _n in ast.walk(fn):
        if (isinstance(_n, ast.Attribute) and _n.attr == "allowed"
                and isinstance(_n.value, ast.Name) and _n.value.id == "v"):
            raise AssertionError(
                "compile_tool_call must not read v.allowed; derive verdict via evaluate_command"
            )
    assert not _constructs_command_policy(tree), (
        "sandbox_policy_compiler.py must not construct CommandPolicy() directly; "
        "use build_command_policy()"
    )
