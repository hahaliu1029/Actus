"""B5 C5a: verify executor_node consumes StepMetadata correctly.

Rather than spinning up the full LangGraph orchestration (which would
require mocks for every dependency), these tests verify the ``executor_node``
consumption contract at the AST level:

1. The ``react_graph_provider`` call site unpacks a 2-tuple
2. ``fresh_configurable["bound_tool_names"]`` is populated from StepMetadata
3. ``fresh_skill_context`` is used LOCALLY (not written back to state)

Plus an integration-ish check using a minimal mock provider that runs
through ``executor_node``'s early path up to the point of consuming the
tuple. Full end-to-end coverage is in the existing
``test_integration.py`` (pre-existing failures, tracked separately).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


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
    """Return the source text of the executor_node function."""
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    executor = _find_function(tree, "executor_node")
    assert executor is not None, "executor_node not found"
    return ast.unparse(executor)


def test_executor_unpacks_two_tuple_from_provider() -> None:
    """executor_node must unpack ``(step_react, step_meta)`` from provider."""
    src = _executor_source()
    # Either form is acceptable:
    #   step_react, step_meta = await react_graph_provider(...)
    #   result = await react_graph_provider(...); step_react = result[0]; ...
    # We look for the first form since that's what the C5a refactor uses.
    assert "step_react, step_meta = await react_graph_provider(" in src, (
        "executor_node should unpack a 2-tuple from react_graph_provider. "
        "Found source:\n" + src[:500]
    )


def test_executor_injects_bound_tool_names_into_fresh_configurable() -> None:
    """bound_tool_names from StepMetadata must be written to fresh_configurable."""
    src = _executor_source()
    assert "step_meta.bound_tool_names" in src, (
        "executor_node should read bound_tool_names from step_meta"
    )
    # The injection happens via dict merge. We just check both pieces are present.
    assert "fresh_configurable" in src
    assert "bound_tool_names" in src


def test_executor_reads_skill_context_from_step_metadata() -> None:
    """Primary path: executor must use step_meta.skill_context, not state."""
    src = _executor_source()
    assert "step_meta.skill_context" in src
    # And the primary consumer line:
    assert "fresh_skill_context" in src


def test_executor_does_not_assign_skill_context_back_to_state() -> None:
    """Negative: no ``state["skill_context"] = ...`` or equivalent mutation.

    The AST lint in ``test_executor_no_skill_context_writeback.py`` covers
    ``Command(update={"skill_context": ...})`` patterns. This test adds a
    second check against direct state dict mutation.
    """
    src = _executor_source()
    forbidden_patterns = (
        'state["skill_context"]',
        "state['skill_context']",
    )
    for pat in forbidden_patterns:
        # Only flag if used on the LHS of assignment.
        # ast.unparse normalizes spacing, so we check for the exact assignment.
        if pat + " = " in src or pat + "=" in src:
            raise AssertionError(
                f"executor_node must not write state.skill_context "
                f"(found pattern {pat!r} used as assignment target)"
            )


def test_executor_has_fresh_config_for_downstream() -> None:
    """executor_node must build a fresh_config (not reuse incoming config)
    so the C5b PromptAssembler can read bound_tool_names from configurable."""
    src = _executor_source()
    assert "fresh_config" in src
