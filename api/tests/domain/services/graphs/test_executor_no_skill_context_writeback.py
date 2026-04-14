"""Guard against DIRECT writebacks of skill_context in executor_node.

B5 C5a: CI gate enforcing the two-clock architecture. ONLY ``updater_node``
is allowed to write ``state.skill_context`` (via ``skill_context_refresher``).
``executor_node`` consumes ``StepMetadata.skill_context`` locally and
must not echo it back into state.

**Coverage boundary**: this test ONLY catches the obvious case —
a ``dict`` literal with key ``skill_context`` inside a
``Command(update=...)`` call inside ``executor_node``'s function body.

It does NOT catch:

- Indirect construction (helper function builds the dict)
- Variable-based construction (``var = {...}; Command(update=var)``)
- Dict unpacking tricks (``dict(**base, skill_context=...)``)

Indirect writebacks are enforced by code review — see the "Prompt Assembly
Invariants" section in ``CONTRIBUTING.md``.
"""
from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_KEYS_IN_EXECUTOR = frozenset({"skill_context"})
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
    """Walk the AST and return the first function (sync or async) with this name."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    return None


def _extract_dict_keys_in_command_update(func_node: ast.AST) -> set[str]:
    """Collect all string keys passed as ``Command(update={...})`` dict literals.

    Only catches DIRECT dict literals — variable indirection is out of scope
    for this scanner (documented in the module docstring).
    """
    keys: set[str] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        # Match Command(...) — both `Command(...)` and `langgraph.types.Command(...)`
        called = node.func
        if isinstance(called, ast.Name) and called.id == "Command":
            pass
        elif isinstance(called, ast.Attribute) and called.attr == "Command":
            pass
        else:
            continue
        # Find the `update` kwarg
        for kw in node.keywords:
            if kw.arg != "update":
                continue
            if isinstance(kw.value, ast.Dict):
                for k in kw.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.add(k.value)
    return keys


def test_main_graph_path_exists() -> None:
    """Sanity check: the scanner target file must exist before we try to parse it."""
    assert MAIN_GRAPH_PATH.exists(), (
        f"main_graph.py not found at {MAIN_GRAPH_PATH}. "
        "If the file was moved, update MAIN_GRAPH_PATH in this test."
    )


def test_executor_node_does_not_write_skill_context() -> None:
    """``executor_node`` must not write ``skill_context`` to state.

    Only ``updater_node`` is allowed to do so (via ``skill_context_refresher``).
    See B5 design doc "Two-clock architecture" and CONTRIBUTING.md
    "Prompt Assembly Invariants" section.
    """
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    executor = _find_function(tree, "executor_node")
    assert executor is not None, "executor_node not found in main_graph.py"

    keys = _extract_dict_keys_in_command_update(executor)
    forbidden = keys & FORBIDDEN_KEYS_IN_EXECUTOR
    assert not forbidden, (
        f"executor_node must not write {sorted(forbidden)} to state. "
        f"Only updater_node (via skill_context_refresher) may update skill_context. "
        f"See CONTRIBUTING.md 'Prompt Assembly Invariants' section."
    )


def test_section_assembly_meta_keys_are_safe() -> None:
    """Guard against ``section_assembly_meta`` ever containing ``skill_context``.

    ``executor_node`` spreads ``**section_assembly_meta`` into its
    ``Command(update=...)`` dicts. The AST scanner only inspects literal
    dict keys, so a ``**`` spread with a ``skill_context`` key would evade
    detection. This test inspects every dict-literal assignment to a
    variable named ``section_assembly_meta`` in ``executor_node`` and
    asserts the keys are in a known safe set.
    """
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    executor = _find_function(tree, "executor_node")
    assert executor is not None

    safe_keys = {"system_prompt_version_hash", "system_prompt_tokens"}
    found_any = False
    for node in ast.walk(executor):
        # Match: section_assembly_meta = {...}
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "section_assembly_meta":
            continue
        # Only validate dict-literal assignments. Type annotations
        # (ast.AnnAssign) for the initial `{}` are handled separately below.
        if not isinstance(node.value, ast.Dict):
            continue
        found_any = True
        for k in node.value.keys:
            assert isinstance(k, ast.Constant) and isinstance(k.value, str), (
                "section_assembly_meta should only use string literal keys"
            )
            assert k.value in safe_keys, (
                f"section_assembly_meta contains disallowed key {k.value!r}. "
                f"Only {sorted(safe_keys)} are permitted (no skill_context)."
            )
    assert found_any, (
        "Expected to find at least one ``section_assembly_meta = {...}`` "
        "literal assignment in executor_node"
    )


def test_updater_node_is_allowed_to_write_skill_context() -> None:
    """Positive control: ``updater_node`` SHOULD write ``skill_context``.

    If this test fails, someone removed the legitimate write path — the
    two-clock fallback is broken and ``executor_node``'s legacy path
    (when ``react_graph_provider`` is None) will no longer see updated
    skill contexts across loop iterations.
    """
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    updater = _find_function(tree, "updater_node")
    assert updater is not None, "updater_node not found"
    keys = _extract_dict_keys_in_command_update(updater)
    assert "skill_context" in keys, (
        "updater_node should write skill_context (it's the authoritative "
        "fallback source). If this test fails, someone removed the "
        "legitimate write path."
    )
