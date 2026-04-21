"""N1 AST CI gate — prevents future refactors from removing the validator call
or reordering it after the legacy risk gate.

Inherent limitation (spec §7.7): static AST scanning cannot assert runtime
control flow, only textual ordering and structural nesting. Runtime coverage
is provided by the integration tests in
`api/tests/domain/services/graphs/test_react_graph_ast_stage_s.py`.
"""
from __future__ import annotations

import ast
import pathlib


def _repo_root() -> pathlib.Path:
    """Resolve repo root regardless of pytest cwd (CI + local)."""
    # This file is at api/tests/structure/test_*.py; repo root is 3 parents up.
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


def _first_call_line(fn_ast: ast.AST, *, call_names: set[str]) -> int | None:
    candidates: list[int] = []
    for n in ast.walk(fn_ast):
        if not isinstance(n, ast.Call):
            continue
        matched = (
            (isinstance(n.func, ast.Name) and n.func.id in call_names)
            or (isinstance(n.func, ast.Attribute) and n.func.attr in call_names)
        )
        if matched:
            candidates.append(n.lineno)
    return min(candidates) if candidates else None


def test_tool_node_calls_ast_validator_before_risk_gate() -> None:
    src = (_repo_root() / "api/app/domain/services/graphs/react_graph.py").read_text()
    tree = ast.parse(src)
    tool_node_fn = _find_function_recursive(tree, "tool_node")
    assert tool_node_fn is not None, "tool_node closure missing from react_graph.py"

    first_validate = _first_call_line(tool_node_fn, call_names={"validate"})
    first_risk = _first_call_line(tool_node_fn, call_names={"assess"})

    assert first_validate is not None, (
        "tool_node shell branch must call shell_ast_validator.validate(). "
        "Do NOT remove N1 AST validator integration."
    )
    if first_risk is not None:
        assert first_validate < first_risk, (
            f"AST validator at line {first_validate} must precede "
            f"legacy _risk_assessor.assess() at line {first_risk}."
        )

    # Structural check: validate() call must be inside an `if` block whose
    # test compares `tool_source.category == "shell"`. Accepts both the bare
    # Compare form and any BoolOp(And, …) form that includes it — so the
    # audit-round-3 narrowing (`… and tool_name == "shell_execute"`) still
    # satisfies this invariant.
    def _is_category_shell_compare(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "category"
            and any(
                isinstance(cmp, ast.Constant) and cmp.value == "shell"
                for cmp in node.comparators
            )
        )

    def _test_expr_has_category_shell(test_node: ast.AST) -> bool:
        if _is_category_shell_compare(test_node):
            return True
        # Walk BoolOp / nested compares to find the category==shell compare.
        for sub in ast.walk(test_node):
            if _is_category_shell_compare(sub):
                return True
        return False

    validate_inside_shell_branch = False
    for node in ast.walk(tool_node_fn):
        if not isinstance(node, ast.If):
            continue
        if not _test_expr_has_category_shell(node.test):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and (
                (isinstance(sub.func, ast.Name) and sub.func.id == "validate")
                or (isinstance(sub.func, ast.Attribute) and sub.func.attr == "validate")
            ):
                validate_inside_shell_branch = True
                break
        if validate_inside_shell_branch:
            break
    assert validate_inside_shell_branch, (
        "validate() must be inside an `if` whose test contains "
        "`tool_source.category == 'shell'` (bare Compare or inside a BoolOp)."
    )


def test_invoke_native_calls_ast_validator_before_contains_blocked() -> None:
    src = (_repo_root() / "api/app/domain/services/tools/skill.py").read_text()
    tree = ast.parse(src)
    fn = _find_method(tree, class_name="SkillTool", method_name="_invoke_native")
    assert fn is not None, "SkillTool._invoke_native not found"

    first_validate = _first_call_line(fn, call_names={"validate"})
    first_contains_blocked = _first_call_line(fn, call_names={"_contains_blocked_command"})

    assert first_validate is not None, "N1 validator missing in _invoke_native"
    if first_contains_blocked is not None:
        assert first_validate < first_contains_blocked, (
            "validate() must precede legacy _contains_blocked_command."
        )


def test_validator_module_imports_only_allowed_deps() -> None:
    """N1 domain purity: shell_ast_validator.py may only import
    bashlex + posixpath + stdlib + domain.models.tool_result.
    """
    src = (_repo_root() / "api/app/domain/services/safety/shell_ast_validator.py").read_text()
    tree = ast.parse(src)
    forbidden_prefixes = ("app.application", "app.infrastructure", "core.config")
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            for forbid in forbidden_prefixes:
                assert not mod.startswith(forbid), (
                    f"validator must not import from {mod} (domain purity)"
                )
        elif isinstance(n, ast.Import):
            for alias in n.names:
                for forbid in forbidden_prefixes:
                    assert not alias.name.startswith(forbid), (
                        f"validator must not import {alias.name} (domain purity)"
                    )
