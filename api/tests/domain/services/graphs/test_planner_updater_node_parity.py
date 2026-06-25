"""B5 C6: parity tests for planner_node + updater_node with the assembler flag.

Same strategy as C5b's ``test_executor_prompt_assembler_parity.py``:
invoke the prompt assembly logic directly (not through the full
LangGraph node) and verify both flag states produce system content
with the key structural markers.

Plus AST-level tests that verify the planner_node and updater_node
have the flag branch + legacy fallback pattern, mirroring C5b's
``test_step_result_injection.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts import get_prompt_section_bundle
from app.domain.services.prompts.assembler import PromptAssembler
from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.render_context import build_render_context
from app.domain.services.prompts.section import PromptMode


MAIN_GRAPH_PATH = (
    Path(__file__).resolve().parents[4]
    / "app"
    / "domain"
    / "services"
    / "graphs"
    / "main_graph.py"
)


def _make_prompt_assembler() -> PromptAssembler:
    return PromptAssembler(
        budget=SystemPromptBudget(max_tokens=10_000),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )


def _representative_state() -> dict:
    """State with tool summary embedded in skill_context (legacy pattern)."""
    return {
        "language": "zh",
        "message": "帮我写一个 Python 脚本",
        "conversation_summaries": ["第一轮：用户问了 X"],
        "skill_context": (
            "## Active Skills\n"
            "- skill_foo: do foo things\n\n"
            "## Available Tool Summary\n"
            "- shell: shell_execute\n- file: file_read"
        ),
    }


# ---- Direct-call structural tests (assembler-only, post-C7.5) -------- #


def _assembler_planner_system_content(state: dict) -> str:
    section_bundle = get_prompt_section_bundle(state.get("language", "zh"))
    agent_config = MagicMock(supports_vision=True, supports_pdf_input=True)
    ctx = build_render_context(state, {"configurable": {}}, agent_config)
    result = _make_prompt_assembler().assemble(
        section_bundle.planner, ctx, PromptMode.FULL
    )
    return result.text


def test_planner_contains_identity() -> None:
    state = _representative_state()
    assembler_text = _assembler_planner_system_content(state)
    assert "任务规划智能体" in assembler_text


def test_planner_contains_tool_summary() -> None:
    state = _representative_state()
    assembler_text = _assembler_planner_system_content(state)
    assert "## Available Tool Summary" in assembler_text
    assert "shell_execute" in assembler_text


def test_planner_contains_conversation_summaries() -> None:
    state = _representative_state()
    assembler_text = _assembler_planner_system_content(state)
    assert "## 历史对话摘要" in assembler_text
    assert "第一轮：用户问了 X" in assembler_text


def test_planner_assembler_excludes_active_skills_prefix() -> None:
    """The tool_summary_legacy section extracts ONLY from the marker onward.
    ``## Active Skills`` (in the input state but before the marker) must
    NOT appear in the assembled planner system prompt."""
    state = _representative_state()
    assembler_text = _assembler_planner_system_content(state)
    assert "## Active Skills" not in assembler_text
    assert "skill_foo: do foo things" not in assembler_text


def test_planner_no_skill_context_still_renders_identity() -> None:
    """With empty skill_context and no summaries, only planner_identity renders."""
    state = {"language": "zh", "message": "test"}
    assembler_text = _assembler_planner_system_content(state)
    assert "任务规划智能体" in assembler_text
    assert "## Available Tool Summary" not in assembler_text
    assert "## 历史对话摘要" not in assembler_text


def test_planner_en_bundle_dispatches_english() -> None:
    state = _representative_state()
    state["language"] = "en"
    state["conversation_summaries"] = ["Round 1: user asked X"]
    assembler_text = _assembler_planner_system_content(state)
    assert "task planner agent" in assembler_text.lower()
    assert "任务规划智能体" not in assembler_text
    # Conversation summaries header dispatches too
    assert "## Conversation History Summary" in assembler_text


def test_updater_bundle_produces_same_structure_as_planner() -> None:
    """Updater uses the same 3 sections → same structural output."""
    state = _representative_state()
    section_bundle = get_prompt_section_bundle("zh")
    agent_config = MagicMock(supports_vision=True, supports_pdf_input=True)
    ctx = build_render_context(state, {"configurable": {}}, agent_config)

    planner_text = (
        _make_prompt_assembler()
        .assemble(section_bundle.planner, ctx, PromptMode.FULL)
        .text
    )
    updater_text = (
        _make_prompt_assembler()
        .assemble(section_bundle.updater, ctx, PromptMode.FULL)
        .text
    )
    # Same section set → identical text (both contain identity + tool_summary + summaries)
    assert planner_text == updater_text


def test_updater_includes_conversation_summaries() -> None:
    """Post-C7.5: the updater registry includes ``conversation_summaries_section``,
    so the assembler-path updater system prompt contains conversation
    summaries when state has them. The legacy updater used to omit them
    (intentional asymmetry documented in C6), but that path no longer
    exists — the new behavior is now the only behavior.
    """
    state = _representative_state()
    assert state["conversation_summaries"]

    section_bundle = get_prompt_section_bundle("zh")
    agent_config = MagicMock(supports_vision=True, supports_pdf_input=True)
    ctx = build_render_context(state, {"configurable": {}}, agent_config)
    assembler_text = (
        _make_prompt_assembler()
        .assemble(section_bundle.updater, ctx, PromptMode.FULL)
        .text
    )
    assert "## 历史对话摘要" in assembler_text
    assert "第一轮：用户问了 X" in assembler_text


# ---- AST-level node structure tests ---------------------------------- #


def _find_function(
    tree: ast.Module, name: str
) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    return None


def _node_source(name: str) -> str:
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = _find_function(tree, name)
    assert fn is not None, f"{name} not found"
    return ast.unparse(fn)


def test_planner_node_references_prompt_assembler() -> None:
    """Post-C7.5: no feature flag, but planner_node must still call the
    assembler (no legacy path remains)."""
    src = _node_source("planner_node")
    assert "prompt_assembler" in src


def test_updater_node_references_prompt_assembler() -> None:
    src = _node_source("updater_node")
    assert "prompt_assembler" in src


def test_planner_node_has_no_legacy_helper() -> None:
    """Post-C7.5: the legacy ``_build_legacy_planner_system_content``
    helper is deleted — no fallback path remains."""
    src = _node_source("planner_node")
    assert "_build_legacy_planner_system_content" not in src
    # Negative: the deleted legacy constant reference must be gone
    assert "PLANNER_SYSTEM_PROMPT" not in src


def test_updater_node_has_no_legacy_helper() -> None:
    src = _node_source("updater_node")
    assert "_build_legacy_updater_system_content" not in src
    assert "PLANNER_SYSTEM_PROMPT" not in src


def test_planner_node_calls_section_bundle_planner() -> None:
    """planner_node must use ``section_bundle.planner`` (not executor/updater)."""
    src = _node_source("planner_node")
    assert "section_bundle.planner" in src


def test_updater_node_calls_section_bundle_updater() -> None:
    """updater_node must use ``section_bundle.updater`` (not planner/executor)."""
    src = _node_source("updater_node")
    assert "section_bundle.updater" in src


def test_updater_node_passes_team_members_into_render_context() -> None:
    """EPIC-FIX-1: updater_node is the re-plan loop that RE-EMITS
    role-bearing ``parallel_work_units``. The agent-team teaching section
    is registered in BOTH the planner AND updater registries
    (bundles/{en,zh}.py), so updater_node must mirror planner_node's
    best-effort STRUCTURAL-ONLY team-teaching load and pass
    ``team_members=`` into its ``build_render_context`` call — otherwise
    the section renders inert (text=None) on every replan round and
    role-tagging degrades exactly where the planner had taught it.

    This is an AST-level assertion: walk updater_node, find the
    ``build_render_context(...)`` call, and require a ``team_members``
    keyword argument.
    """
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = _find_function(tree, "updater_node")
    assert fn is not None

    found_render_ctx_call = False
    found_team_members_kw = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_brc = (
            isinstance(func, ast.Name) and func.id == "build_render_context"
        ) or (
            isinstance(func, ast.Attribute)
            and func.attr == "build_render_context"
        )
        if not is_brc:
            continue
        found_render_ctx_call = True
        if any(kw.arg == "team_members" for kw in node.keywords):
            found_team_members_kw = True

    assert found_render_ctx_call, (
        "updater_node must call build_render_context"
    )
    assert found_team_members_kw, (
        "updater_node must pass team_members= into build_render_context "
        "(mirror planner_node's STRUCTURAL-ONLY team-teaching load) so the "
        "agent-team teaching section reaches the assembled updater prompt"
    )


def test_planner_node_does_not_write_skill_context() -> None:
    """Two-clock invariant: planner_node must not write skill_context to
    state. Only updater_node is allowed (via skill_context_refresher) —
    see CONTRIBUTING.md "Prompt Assembly 不变式" and
    test_executor_no_skill_context_writeback.py for the executor side.
    """
    source = MAIN_GRAPH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    fn = _find_function(tree, "planner_node")
    assert fn is not None

    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Command":
            pass
        elif isinstance(func, ast.Attribute) and func.attr == "Command":
            pass
        else:
            continue
        for kw in node.keywords:
            if kw.arg != "update":
                continue
            if isinstance(kw.value, ast.Dict):
                for k in kw.value.keys:
                    if (
                        isinstance(k, ast.Constant)
                        and isinstance(k.value, str)
                        and k.value == "skill_context"
                    ):
                        raise AssertionError(
                            "planner_node writes skill_context to state — "
                            "only updater_node may do so (two-clock architecture)"
                        )

    # Planner also uses a dict literal return (not Command) — check that too
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for k in node.value.keys:
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and k.value == "skill_context"
                ):
                    raise AssertionError(
                        "planner_node returns skill_context in its update dict"
                    )
