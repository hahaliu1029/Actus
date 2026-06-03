"""N1 — production tool_node shell-AST gate structural assertions.

The async ``_stage_s_ast_validate`` stub + its 3 unit tests were removed in
PE-4a (dead scaffolding deleted with ``_run_policy_chain``). The two
remaining tests AST-scan the LIVE ``tool_node`` to lock the production N1
shell AST validator gate.
"""
from __future__ import annotations


def test_production_tool_node_has_ast_validator_call_in_shell_branch():
    """Structural assertion: tool_node source contains validate() call.

    Full end-to-end behavior tested in Task 30's AST CI gate +
    existing integration tests with compiled react_graph.
    """
    import ast
    from pathlib import Path

    # CWD-independent path: walk up from this test file to the api/ root,
    # mirroring the B5 CI gate convention
    # (see test_executor_no_skill_context_writeback.py).
    src_path = (
        Path(__file__).resolve().parents[4]
        / "app"
        / "domain"
        / "services"
        / "graphs"
        / "react_graph.py"
    )
    src = src_path.read_text()
    tree = ast.parse(src)

    tool_node_fn = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "tool_node":
            tool_node_fn = node
            break
    assert tool_node_fn is not None

    has_validate_call = any(
        (isinstance(n, ast.Call)
         and ((isinstance(n.func, ast.Name) and n.func.id == "validate")
              or (isinstance(n.func, ast.Attribute) and n.func.attr == "validate")))
        for n in ast.walk(tool_node_fn)
    )
    assert has_validate_call, (
        "tool_node must contain a validate() call after N1 integration"
    )


def test_production_tool_node_gate_narrow_to_shell_execute():
    """Audit (round 3) P2: the N1 gate predicate must compare against the
    exact tool_name ``"shell_execute"``, not just ``tool_source.category ==
    "shell"``. Otherwise shell_read_output / shell_wait_process /
    shell_write_input / shell_kill_process all pass through validate("")
    and inflate ``ast_validations_total`` with empty-command successes —
    making ``parser_failure_rate`` depend on polling traffic instead of
    real AST validations.
    """
    import ast
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parents[4]
        / "app"
        / "domain"
        / "services"
        / "graphs"
        / "react_graph.py"
    )
    src = src_path.read_text()
    tree = ast.parse(src)

    tool_node_fn = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "tool_node"
        ):
            tool_node_fn = node
            break
    assert tool_node_fn is not None

    def _has_shell_execute_compare(test_node: ast.AST) -> bool:
        for sub in ast.walk(test_node):
            if not isinstance(sub, ast.Compare):
                continue
            left = sub.left
            is_tool_name = isinstance(left, ast.Name) and left.id == "tool_name"
            has_shell_execute = any(
                isinstance(c, ast.Constant) and c.value == "shell_execute"
                for c in sub.comparators
            )
            if is_tool_name and has_shell_execute:
                return True
        return False

    gate_has_narrow_compare = any(
        isinstance(n, ast.If) and _has_shell_execute_compare(n.test)
        for n in ast.walk(tool_node_fn)
    )
    assert gate_has_narrow_compare, (
        "N1 gate must include `tool_name == 'shell_execute'` to avoid "
        "counting shell_read_output / shell_wait_process / shell_write_input "
        "/ shell_kill_process as AST validations (audit P2)."
    )
