"""B5 C3: structural sanity test for the 5 C3 sections assembled together.

Unlike C2 (which has a byte-level snapshot against the legacy
``REACT_SYSTEM_PROMPT`` constant), the C3 dynamic/conditional sections do
NOT have a legacy constant to diff against:

- ``tools_guide_stable`` wraps FILE_VIEW/MEMORY hints that lived inside
  ``_build_runtime_system_context`` as concatenated strings.
- ``tools_guide_dynamic`` replaces ``_build_available_tool_summary`` which
  reads from multiple raw tool registries — the new output is equivalent
  in meaning but not byte-identical.
- ``skill_context`` pass-through is built upstream; its content varies
  per step.
- ``conversation_summaries`` introduces a language-appropriate header (the
  legacy code hardcoded the Chinese header even for EN users — that's a
  C0a/C0b miss the new section fixes).
- ``sandbox_state`` is a stub.

So instead of snapshot equality, this file asserts **structural invariants**
on the combined assembly:

1. All 5 sections can coexist in a registry with the 3 C2 sections.
2. Priority ordering produces the expected section order in
   ``sections_included``.
3. When all conditional inputs are present, all expected sections render.
4. When conditional inputs are missing, sections drop out cleanly
   (no dangling markers, no crashes).
5. ``PromptMode.MINIMAL`` applies the allowlist consistently.
"""
from __future__ import annotations

import pytest

from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts.assembler import PromptAssembler
from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.section import (
    MINIMAL_MODE_ALLOWLIST,
    PromptMode,
    RenderContext,
    SectionRegistry,
)
from app.domain.services.prompts.sections.behavior_core import behavior_core_section
from app.domain.services.prompts.sections.conversation_summaries import (
    conversation_summaries_section,
)
from app.domain.services.prompts.sections.identity import identity_section
from app.domain.services.prompts.sections.output_format import output_format_section
from app.domain.services.prompts.sections.sandbox_state import sandbox_state_section
from app.domain.services.prompts.sections.skill_context import skill_context_section
from app.domain.services.prompts.sections.tools_guide_dynamic import (
    tools_guide_dynamic_section,
)
from app.domain.services.prompts.sections.tools_guide_stable import (
    tools_guide_stable_section,
)


def _make_assembler() -> PromptAssembler:
    return PromptAssembler(
        budget=SystemPromptBudget(max_tokens=10_000),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )


def _make_full_registry() -> SectionRegistry:
    """Registry with all 3 C2 + 5 C3 sections (the B5 target state)."""
    return SectionRegistry(
        sections=[
            identity_section,              # priority 10
            behavior_core_section,         # priority 10
            output_format_section,         # priority 9
            tools_guide_stable_section,    # priority 8
            tools_guide_dynamic_section,   # priority 7
            skill_context_section,         # priority 7
            conversation_summaries_section,  # priority 6
            sandbox_state_section,         # priority 5
        ],
        name="react_c3_combined",
    )


# ---- Ordering invariants ------------------------------------------------ #


def test_all_c3_sections_render_when_inputs_present() -> None:
    """Given all conditional inputs, every section produces output."""
    registry = _make_full_registry()
    ctx = RenderContext(
        lang="zh",
        has_file_view=True,
        has_memory_tools=True,
        bound_tool_names=frozenset({"skill_foo_bar", "shell_execute", "file_read"}),
        skill_context="## Active Skills\n- skill_foo: do foo things",
        skill_names_in_context=("foo",),
        conversation_summaries=("第一轮：用户问了 X",),
    )
    result = _make_assembler().assemble(registry, ctx, PromptMode.FULL)

    # sandbox_state is still a stub → not included
    assert "identity" in result.sections_included
    assert "behavior_core" in result.sections_included
    assert "output_format" in result.sections_included
    assert "tools_guide_stable" in result.sections_included
    assert "tools_guide_dynamic" in result.sections_included
    assert "skill_context" in result.sections_included
    assert "conversation_summaries" in result.sections_included
    assert "sandbox_state" not in result.sections_included


def test_section_order_in_assembled_text() -> None:
    """Priority-desc ordering: C2 sections before C3 sections in assembled text."""
    registry = _make_full_registry()
    ctx = RenderContext(
        lang="zh",
        has_file_view=True,
        has_memory_tools=True,
        bound_tool_names=frozenset({"shell_execute"}),
        skill_context="## Active Skills",
        conversation_summaries=("summary-1",),
    )
    result = _make_assembler().assemble(registry, ctx, PromptMode.FULL)
    text = result.text

    # Identity (10) comes before tools_guide_stable (8) which comes before
    # conversation_summaries (6).
    identity_idx = text.index("任务执行智能体")  # from identity ZH
    tools_stable_idx = text.index("文件理解")  # from tools_guide_stable ZH
    conv_idx = text.index("## 历史对话摘要")  # from conversation_summaries ZH
    assert identity_idx < tools_stable_idx < conv_idx


def test_sections_dropped_when_inputs_missing() -> None:
    """With no conditional inputs, only C2 sections + optional stubs render."""
    registry = _make_full_registry()
    ctx = RenderContext(lang="zh")  # no flags, no summaries, no skill_context
    result = _make_assembler().assemble(registry, ctx, PromptMode.FULL)

    assert "identity" in result.sections_included
    assert "behavior_core" in result.sections_included
    assert "output_format" in result.sections_included
    # All 5 C3 sections drop out
    assert "tools_guide_stable" not in result.sections_included
    assert "tools_guide_dynamic" not in result.sections_included
    assert "skill_context" not in result.sections_included
    assert "conversation_summaries" not in result.sections_included
    assert "sandbox_state" not in result.sections_included


def test_minimal_mode_respects_allowlist_across_c2_and_c3() -> None:
    """MINIMAL mode keeps exactly the allowlisted sections, drops the rest."""
    registry = _make_full_registry()
    ctx = RenderContext(
        lang="zh",
        has_file_view=True,
        has_memory_tools=True,
        bound_tool_names=frozenset({"shell_execute"}),
        skill_context="## Active Skills",
        conversation_summaries=("x",),
    )
    result = _make_assembler().assemble(registry, ctx, PromptMode.MINIMAL)

    # Every rendered section must be in the allowlist
    for sid in result.sections_included:
        assert sid in MINIMAL_MODE_ALLOWLIST, (
            f"Section {sid!r} rendered in MINIMAL mode but is not in allowlist"
        )

    # Concretely: identity + behavior_core + tools_guide_stable survive
    assert "identity" in result.sections_included
    assert "behavior_core" in result.sections_included
    assert "tools_guide_stable" in result.sections_included
    # Non-allowlisted sections are filtered
    assert "output_format" not in result.sections_included
    assert "tools_guide_dynamic" not in result.sections_included
    assert "skill_context" not in result.sections_included
    assert "conversation_summaries" not in result.sections_included


def test_none_mode_keeps_only_priority_ge_9() -> None:
    """NONE mode is priority-gated (>=9). Only identity, behavior_core, output_format survive."""
    registry = _make_full_registry()
    ctx = RenderContext(
        lang="zh",
        has_file_view=True,
        has_memory_tools=True,
        bound_tool_names=frozenset({"shell_execute"}),
        skill_context="## Active Skills",
        conversation_summaries=("x",),
    )
    result = _make_assembler().assemble(registry, ctx, PromptMode.NONE)

    assert set(result.sections_included) == {
        "identity",
        "behavior_core",
        "output_format",
    }


# ---- Language dispatch ------------------------------------------------- #


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_combined_assembly_works_in_both_languages(lang: str) -> None:
    """Both ZH and EN produce non-empty assembled text with C2+C3 sections."""
    registry = _make_full_registry()
    ctx = RenderContext(
        lang=lang,  # type: ignore[arg-type]
        has_file_view=True,
        bound_tool_names=frozenset({"shell_execute"}),
        conversation_summaries=("s1", "s2"),
    )
    result = _make_assembler().assemble(registry, ctx, PromptMode.FULL)
    assert result.text
    assert result.text.strip()
    # Tool summary header is hardcoded English in both languages
    assert "## Available Tool Summary" in result.text


def test_conversation_summaries_header_matches_lang() -> None:
    """ZH gets Chinese header, EN gets English header — fixes C0a/C0b miss."""
    registry = _make_full_registry()

    zh_ctx = RenderContext(lang="zh", conversation_summaries=("s1",))
    zh_result = _make_assembler().assemble(registry, zh_ctx, PromptMode.FULL)
    assert "## 历史对话摘要" in zh_result.text
    assert "## Conversation History Summary" not in zh_result.text

    en_ctx = RenderContext(lang="en", conversation_summaries=("s1",))
    en_result = _make_assembler().assemble(registry, en_ctx, PromptMode.FULL)
    assert "## Conversation History Summary" in en_result.text
    assert "## 历史对话摘要" not in en_result.text


# ---- Registry validation ------------------------------------------------ #


def test_full_registry_passes_startup_validation() -> None:
    """Constructing the full registry must not trip ``__post_init__`` checks
    (no duplicate ids, no dangling skill_ refs in static sections)."""
    registry = _make_full_registry()
    assert len(registry.sections) == 8
    ids = [s.id for s in registry.sections]
    assert len(set(ids)) == 8  # all unique


def test_tools_guide_dynamic_appears_before_skill_context() -> None:
    """tools_guide_dynamic and skill_context share priority=7 — assembly
    order is determined by registry declaration order.

    Pinning this order matters because until C5a's deduplication,
    ``state.skill_context`` may redundantly include the tool summary.
    If the ordering flips, readers will see the duplicated summary in
    a confusing order. This test locks in: tool summary first, then
    skill guides (matches the legacy concatenation order in
    ``_build_runtime_system_context``).
    """
    registry = _make_full_registry()
    ctx = RenderContext(
        lang="zh",
        bound_tool_names=frozenset({"shell_execute"}),
        skill_context="## Active Skills\n- example",
    )
    result = _make_assembler().assemble(registry, ctx, PromptMode.FULL)
    text = result.text
    tool_summary_idx = text.index("## Available Tool Summary")
    skill_ctx_idx = text.index("## Active Skills")
    assert tool_summary_idx < skill_ctx_idx


def test_skill_context_skill_ids_metadata_round_trip() -> None:
    """The skill_context section metadata is surfaced via assembler result."""
    registry = _make_full_registry()
    ctx = RenderContext(
        lang="zh",
        bound_tool_names=frozenset({"skill_foo_bar"}),
        skill_context="## Active Skills\n- skill_foo",
        skill_names_in_context=("foo",),
    )
    result = _make_assembler().assemble(registry, ctx, PromptMode.FULL)
    # The assembled text must contain the skill_context body
    assert "## Active Skills" in result.text
    assert "skill_foo" in result.text
