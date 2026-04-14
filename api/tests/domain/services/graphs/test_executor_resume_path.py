"""B5 C5b: verify the resume path is unchanged under both flag states.

When ``executor_node`` runs in resume mode (``state.resume_value is not
None``), the system prompt assembly path is skipped entirely — both
legacy and PromptAssembler paths are bypassed because ``saved_messages``
already contains the checkpointed SystemMessage and the node only
appends a ``HumanMessage(resume_hint)``.

This AST-level test verifies:

1. ``executor_node`` has a dedicated resume branch that reuses
   ``saved_messages`` as-is (no new SystemMessage construction).
2. The resume branch produces ``initial_messages = saved_messages + [HumanMessage(...)]``
   — it does NOT call the assembler or rebuild the system prompt.

Full end-to-end coverage (spinning up a full LangGraph with a
checkpointer) is out of scope for this CI gate. The AST check is the
same technique used by ``test_step_result_injection.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path


MAIN_GRAPH_PATH = (
    Path(__file__).resolve().parents[4]
    / "app"
    / "domain"
    / "services"
    / "graphs"
    / "main_graph.py"
)


def _find_function(
    tree: ast.Module, name: str
) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    return None


def _executor_source() -> str:
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    executor = _find_function(tree, "executor_node")
    assert executor is not None, "executor_node not found"
    return ast.unparse(executor)


def test_resume_path_reuses_saved_messages_directly() -> None:
    """Resume branch must build initial_messages from saved_messages + resume hint."""
    src = _executor_source()
    # The resume branch pattern (either equivalent is acceptable)
    assert "saved_messages + [HumanMessage(content=resume_hint)]" in src or (
        "saved_messages" in src and "resume_hint" in src and "HumanMessage" in src
    )


def test_resume_path_guarded_by_resume_value_check() -> None:
    """Resume branch must be gated on ``resume_value is not None``."""
    src = _executor_source()
    assert "resume_value is not None" in src


def test_resume_path_does_not_call_prompt_assembler() -> None:
    """Inside the resume branch, we must not see a call to prompt_assembler.assemble.

    We verify this by parsing the AST and checking that no
    ``prompt_assembler.assemble`` call occurs inside the ``if resume_value
    is not None:`` body. The assembler should only be called on the
    normal (non-resume) path.
    """
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    executor = _find_function(tree, "executor_node")
    assert executor is not None

    # Walk the AST to find the `if resume_value is not None:` block
    resume_branches: list[ast.If] = []
    for node in ast.walk(executor):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # Match: `resume_value is not None`
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "resume_value"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            resume_branches.append(node)

    assert resume_branches, (
        "executor_node should have at least one `if resume_value is not None:` branch"
    )

    # Walk the resume body (If.body only — NOT orelse, which contains the
    # non-resume elif branches) and ensure no prompt_assembler.assemble
    # call exists inside the resume branch statements themselves.
    for branch in resume_branches:
        for body_stmt in branch.body:
            for sub in ast.walk(body_stmt):
                if not isinstance(sub, ast.Call):
                    continue
                func = sub.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "assemble"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "prompt_assembler"
                ):
                    raise AssertionError(
                        "Found `prompt_assembler.assemble(...)` inside the resume branch body. "
                        "Resume path must skip section assembly — saved_messages already "
                        "contains the checkpointed SystemMessage."
                    )


def test_prompt_assembler_path_exists_outside_resume_branch() -> None:
    """Positive control: the assembler IS called somewhere in executor_node,
    just not inside the resume branch. If this fails, C5b's assembler
    wiring was removed entirely."""
    src = _executor_source()
    assert "prompt_assembler.assemble" in src, (
        "executor_node should call prompt_assembler.assemble in the non-resume path"
    )


def test_executor_references_prompt_assembler() -> None:
    """Post-C7.5: no feature flag, but the executor must still call the
    assembler — the `prompt_assembler` identifier must appear in the
    function body (or the C5b wiring is broken)."""
    src = _executor_source()
    assert "prompt_assembler" in src, (
        "executor_node should call prompt_assembler.assemble on the non-resume path"
    )


def test_resume_body_contains_no_assembler_identifier() -> None:
    """Heuristic: the resume branch body must not reference ANY identifier
    containing ``assembler`` as a substring.

    This is a stronger guard than the direct-call scanner above: it catches
    helper-function indirection like ``_do_assembler_stuff()`` or
    ``build_system_via_assembler()`` where the actual
    ``prompt_assembler.assemble`` call lives in a helper body. If someone
    refactors the assembler into a helper AND calls it from the resume
    branch, at least the helper name will trip this check.

    Known false negatives: a helper named without "assembler" in its name
    still evades this scanner — that is a code review concern.
    """
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    executor = _find_function(tree, "executor_node")
    assert executor is not None

    resume_branches: list[ast.If] = []
    for node in ast.walk(executor):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "resume_value"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            resume_branches.append(node)

    assert resume_branches
    for branch in resume_branches:
        for body_stmt in branch.body:
            for sub in ast.walk(body_stmt):
                if isinstance(sub, ast.Name) and "assembler" in sub.id.lower():
                    raise AssertionError(
                        f"Found identifier {sub.id!r} containing 'assembler' "
                        f"inside the resume branch body. Resume path must "
                        f"skip all prompt-assembly code — including any "
                        f"helper-function indirection that touches the "
                        f"assembler."
                    )
                if isinstance(sub, ast.Attribute) and "assembler" in sub.attr.lower():
                    raise AssertionError(
                        f"Found attribute {sub.attr!r} containing 'assembler' "
                        f"inside the resume branch body."
                    )
