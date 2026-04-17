"""B5 C3: per-section unit tests for the 5 C3 dynamic/conditional sections.

Each section is tested in isolation against synthetic RenderContext fixtures.
The combined assembly snapshot test lives in test_sections_c3_combined.py.
"""
from __future__ import annotations

import pytest

from app.domain.services.prompts.section import RenderContext, SectionOutput
from app.domain.services.prompts.sections.conversation_summaries import (
    conversation_summaries_section,
)
from app.domain.services.prompts.sections.sandbox_state import sandbox_state_section
from app.domain.services.prompts.sections.skill_context import skill_context_section
from app.domain.services.prompts.sections.tools_guide_dynamic import (
    tools_guide_dynamic_section,
)
from app.domain.services.prompts.sections.tools_guide_stable import (
    tools_guide_stable_section,
)


# ---- tools_guide_stable ------------------------------------------------ #


class TestToolsGuideStable:
    def test_returns_none_when_neither_flag_set(self) -> None:
        ctx = RenderContext(lang="zh", has_file_view=False, has_memory_tools=False)
        output = tools_guide_stable_section.render(ctx)
        assert output.text is None

    def test_only_file_view(self) -> None:
        ctx = RenderContext(lang="zh", has_file_view=True, has_memory_tools=False)
        output = tools_guide_stable_section.render(ctx)
        assert output.text is not None
        assert "文件理解" in output.text
        assert "file_view" in output.text
        assert "记忆工具" not in output.text

    def test_only_memory_tools(self) -> None:
        ctx = RenderContext(lang="zh", has_file_view=False, has_memory_tools=True)
        output = tools_guide_stable_section.render(ctx)
        assert output.text is not None
        assert "记忆工具" in output.text
        assert "memory_search" in output.text
        assert "文件理解" not in output.text

    def test_both_flags_set(self) -> None:
        ctx = RenderContext(lang="zh", has_file_view=True, has_memory_tools=True)
        output = tools_guide_stable_section.render(ctx)
        assert output.text is not None
        assert "文件理解" in output.text
        assert "记忆工具" in output.text
        # File view should appear before memory tools (order matches legacy)
        assert output.text.index("文件理解") < output.text.index("记忆工具")

    def test_en_renders_english(self) -> None:
        ctx = RenderContext(lang="en", has_file_view=True, has_memory_tools=True)
        output = tools_guide_stable_section.render(ctx)
        assert output.text is not None
        assert "File understanding" in output.text
        assert "Memory Tools" in output.text
        assert "文件理解" not in output.text
        assert "记忆工具" not in output.text

    def test_section_metadata(self) -> None:
        s = tools_guide_stable_section
        assert s.id == "tools_guide_stable"
        assert s.priority == 8
        assert s.cacheable is True
        assert s.dynamic is False

    def test_in_minimal_allowlist(self) -> None:
        from app.domain.services.prompts.section import MINIMAL_MODE_ALLOWLIST

        assert tools_guide_stable_section.id in MINIMAL_MODE_ALLOWLIST

    def test_memory_save_hint_gated_on_memory_save_bound(self) -> None:
        """Only teach memory_save when it's actually in bound_tool_names.

        Regression for the bug where ``has_memory_tools=True`` alone (which
        covers memory_search/memory_get) would still emit the memory_save
        guidance, sending the LLM into the unknown-tool path whenever save
        wiring is partial (Redis down, session_id missing, etc.).
        """
        # has_memory_tools True but memory_save NOT bound → no save guidance
        ctx = RenderContext(
            lang="zh",
            has_file_view=False,
            has_memory_tools=True,
            bound_tool_names=frozenset({"memory_search", "memory_get"}),
        )
        output = tools_guide_stable_section.render(ctx)
        assert output.text is not None
        assert "memory_search" in output.text
        assert "memory_save" not in output.text
        assert output.metadata["has_memory_save"] is False

    def test_memory_save_hint_shown_when_save_bound(self) -> None:
        ctx = RenderContext(
            lang="zh",
            has_file_view=False,
            has_memory_tools=True,
            bound_tool_names=frozenset(
                {"memory_search", "memory_get", "memory_save"}
            ),
        )
        output = tools_guide_stable_section.render(ctx)
        assert output.text is not None
        assert "memory_save" in output.text
        assert output.metadata["has_memory_save"] is True

    def test_memory_save_hint_gated_in_english(self) -> None:
        ctx = RenderContext(
            lang="en",
            has_file_view=False,
            has_memory_tools=True,
            bound_tool_names=frozenset({"memory_search", "memory_get"}),
        )
        output = tools_guide_stable_section.render(ctx)
        assert output.text is not None
        assert "memory_search" in output.text
        assert "memory_save" not in output.text


# ---- tools_guide_dynamic ----------------------------------------------- #


class TestToolsGuideDynamic:
    def test_empty_bound_tool_names_returns_none(self) -> None:
        ctx = RenderContext(lang="zh", bound_tool_names=frozenset())
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is None

    def test_single_native_category(self) -> None:
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset({"shell_execute", "shell_read_output"}),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is not None
        assert "## Available Tool Summary" in output.text
        assert "- shell: shell_execute, shell_read_output" in output.text

    def test_multiple_categories_in_display_order(self) -> None:
        """Categories appear in the canonical _DISPLAY_ORDER, not insertion order."""
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset(
                {
                    "browser_navigate",  # 3rd
                    "shell_execute",  # 1st
                    "file_read",  # 2nd
                }
            ),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is not None
        text = output.text
        # Order: shell, file, browser (from _DISPLAY_ORDER)
        shell_idx = text.index("- shell:")
        file_idx = text.index("- file:")
        browser_idx = text.index("- browser:")
        assert shell_idx < file_idx < browser_idx

    def test_skill_tools_grouped(self) -> None:
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset(
                {"skill_foo_bar", "skill_baz_qux", "shell_execute"}
            ),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is not None
        assert "- skill: skill_baz_qux, skill_foo_bar" in output.text  # alphabetical

    def test_skill_creator_tools_categorized(self) -> None:
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset(
                {"brainstorm_skill", "generate_skill", "install_skill"}
            ),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is not None
        assert "- skill creator: brainstorm_skill, generate_skill, install_skill" in output.text

    def test_a2a_tools_categorized(self) -> None:
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset(
                {"get_remote_agent_cards", "call_remote_agent"}
            ),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is not None
        assert "- a2a: call_remote_agent, get_remote_agent_cards" in output.text

    def test_mcp_discovery_separate_from_mcp(self) -> None:
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset(
                {"list_mcp_tools", "get_mcp_tool", "mcp_amap_maps_weather"}
            ),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is not None
        assert "- mcp: mcp_amap_maps_weather" in output.text
        assert "- mcp discovery: get_mcp_tool, list_mcp_tools" in output.text

    def test_double_underscore_skill_name_handled(self) -> None:
        """Edge case: skill_{slug}_{tool} with double underscore from un-normalized slug."""
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset({"skill_foo__bar_tool"}),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is not None
        assert "skill_foo__bar_tool" in output.text

    def test_metadata_includes_bound_tool_names(self) -> None:
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset({"file_view", "shell_execute"}),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.metadata["bound_tool_names_used"] == ["file_view", "shell_execute"]

    def test_unknown_tool_falls_into_other_category(self) -> None:
        """Tools not matching any prefix go to 'other' bucket."""
        ctx = RenderContext(
            lang="zh",
            bound_tool_names=frozenset({"weird_unknown_tool", "shell_execute"}),
        )
        output = tools_guide_dynamic_section.render(ctx)
        assert output.text is not None
        # The unknown tool MUST land in the 'other' category, not silently
        # be swallowed into one of the known prefix buckets.
        assert "- other: weird_unknown_tool" in output.text

    def test_section_metadata(self) -> None:
        s = tools_guide_dynamic_section
        assert s.id == "tools_guide_dynamic"
        assert s.priority == 7
        assert s.cacheable is False
        assert s.dynamic is True

    def test_NOT_in_minimal_allowlist(self) -> None:
        from app.domain.services.prompts.section import MINIMAL_MODE_ALLOWLIST

        assert tools_guide_dynamic_section.id not in MINIMAL_MODE_ALLOWLIST


# ---- skill_context ----------------------------------------------------- #


class TestSkillContext:
    def test_empty_returns_none(self) -> None:
        ctx = RenderContext(lang="zh", skill_context=None)
        output = skill_context_section.render(ctx)
        assert output.text is None

    def test_blank_string_returns_none(self) -> None:
        ctx = RenderContext(lang="zh", skill_context="   \n\n  ")
        output = skill_context_section.render(ctx)
        assert output.text is None

    def test_pass_through(self) -> None:
        markdown = "## Active Skills\n- skill_foo: do foo things"
        ctx = RenderContext(
            lang="zh",
            skill_context=markdown,
            bound_tool_names=frozenset({"skill_foo_bar"}),  # avoid invariant trip
            skill_names_in_context=("foo",),
        )
        output = skill_context_section.render(ctx)
        assert output.text == markdown.strip()
        assert output.metadata["skill_ids_used"] == ["foo"]

    def test_strips_leading_trailing_whitespace(self) -> None:
        ctx = RenderContext(
            lang="zh",
            skill_context="\n\n  ## Active Skills\n  ",
        )
        output = skill_context_section.render(ctx)
        assert output.text == "## Active Skills"

    def test_section_metadata(self) -> None:
        s = skill_context_section
        assert s.id == "skill_context"
        assert s.priority == 7
        assert s.cacheable is False
        assert s.dynamic is True

    def test_NOT_in_minimal_allowlist(self) -> None:
        from app.domain.services.prompts.section import MINIMAL_MODE_ALLOWLIST

        assert skill_context_section.id not in MINIMAL_MODE_ALLOWLIST

    # -- Codex audit HIGH #2: strip embedded `## Available Tool Summary` -- #

    def test_strips_tool_summary_block(self) -> None:
        """When ``state.skill_context`` contains both skill guides and
        the embedded ``## Available Tool Summary`` block (as produced by
        ``_build_runtime_system_context``), the section must emit ONLY
        the skill-guide portion. The tool summary is the authoritative
        responsibility of ``tools_guide_dynamic_section``."""
        combined = (
            "## Active Skills\n"
            "- skill_foo: do foo things\n"
            "- skill_bar: do bar things\n\n"
            "## Available Tool Summary\n"
            "- shell: shell_execute\n"
            "- file: file_read, file_write\n"
            "- skill: skill_foo_bar"
        )
        ctx = RenderContext(
            lang="zh",
            skill_context=combined,
            bound_tool_names=frozenset({"skill_foo_bar"}),
        )
        output = skill_context_section.render(ctx)
        assert output.text is not None
        assert "## Active Skills" in output.text
        assert "skill_foo" in output.text
        assert "skill_bar" in output.text
        # Tool summary block must be absent
        assert "## Available Tool Summary" not in output.text
        assert "shell_execute" not in output.text
        assert "file_read" not in output.text

    def test_only_tool_summary_returns_none(self) -> None:
        """Edge case: if the blob is JUST a tool summary with no skill
        guides, the section should return None (nothing to emit)."""
        ctx = RenderContext(
            lang="zh",
            skill_context="## Available Tool Summary\n- shell: shell_execute",
        )
        output = skill_context_section.render(ctx)
        assert output.text is None

    def test_combined_assembly_has_one_tool_summary(self) -> None:
        """End-to-end verification: assemble the executor registry with
        a realistic skill_context that duplicates the tool summary in
        both state.skill_context AND the bound_tool_names, and confirm
        the final assembled text has exactly ONE ``## Available Tool Summary``
        marker (produced by tools_guide_dynamic_section). Pre-fix, the
        skill_context pass-through caused this to be 2."""
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from app.domain.services.prompts.assembler import PromptAssembler
        from app.domain.services.prompts.budget import SystemPromptBudget
        from app.domain.services.prompts.bundles.zh import (
            ZH_EXECUTOR_REGISTRY,
        )
        from app.domain.services.prompts.section import PromptMode

        combined_skill_context = (
            "## Active Skills\n"
            "- skill_foo: do foo things\n\n"
            "## Available Tool Summary\n"
            "- shell: shell_execute\n"
            "- skill: skill_foo_bar"
        )
        ctx = RenderContext(
            lang="zh",
            has_file_view=True,
            bound_tool_names=frozenset({"shell_execute", "skill_foo_bar"}),
            skill_context=combined_skill_context,
            skill_names_in_context=("foo",),
        )
        assembler = PromptAssembler(
            budget=SystemPromptBudget(max_tokens=10_000),
            token_estimator=TokenEstimator(strategy="hybrid"),
            telemetry=None,
        )
        result = assembler.assemble(ZH_EXECUTOR_REGISTRY, ctx, PromptMode.FULL)
        marker_count = result.text.count("## Available Tool Summary")
        assert marker_count == 1, (
            f"Expected exactly 1 '## Available Tool Summary' marker in the "
            f"assembled prompt, got {marker_count}. Pre-fix bug: "
            f"skill_context pass-through duplicated the tool summary block."
        )


# ---- conversation_summaries -------------------------------------------- #


class TestConversationSummaries:
    def test_empty_returns_none(self) -> None:
        ctx = RenderContext(lang="zh", conversation_summaries=())
        output = conversation_summaries_section.render(ctx)
        assert output.text is None

    def test_zh_header(self) -> None:
        ctx = RenderContext(
            lang="zh",
            conversation_summaries=("第一轮：用户问了 X", "第二轮：助手回答了 Y"),
        )
        output = conversation_summaries_section.render(ctx)
        assert output.text is not None
        assert output.text.startswith("## 历史对话摘要\n")
        assert "第一轮：用户问了 X" in output.text
        assert "第二轮：助手回答了 Y" in output.text

    def test_en_header(self) -> None:
        ctx = RenderContext(
            lang="en",
            conversation_summaries=("Round 1: user asked X", "Round 2: assistant said Y"),
        )
        output = conversation_summaries_section.render(ctx)
        assert output.text is not None
        assert output.text.startswith("## Conversation History Summary\n")
        assert "Round 1: user asked X" in output.text
        # English header should NOT contain CJK
        first_line = output.text.split("\n", 1)[0]
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in first_line)

    def test_summaries_joined_with_double_newline(self) -> None:
        ctx = RenderContext(
            lang="zh",
            conversation_summaries=("a", "b", "c"),
        )
        output = conversation_summaries_section.render(ctx)
        assert output.text is not None
        # Content after header should be "a\n\nb\n\nc"
        body = output.text.split("\n", 1)[1]
        assert body == "a\n\nb\n\nc"

    def test_metadata_includes_count(self) -> None:
        ctx = RenderContext(
            lang="zh",
            conversation_summaries=("a", "b", "c"),
        )
        output = conversation_summaries_section.render(ctx)
        assert output.metadata["summary_count"] == 3

    def test_section_metadata(self) -> None:
        s = conversation_summaries_section
        assert s.id == "conversation_summaries"
        assert s.priority == 6
        assert s.cacheable is False
        assert s.dynamic is True


# ---- sandbox_state ----------------------------------------------------- #


class TestSandboxState:
    def test_always_returns_none(self) -> None:
        # Try a few contexts — should always return None (stub)
        for ctx in [
            RenderContext(lang="zh"),
            RenderContext(lang="en", has_file_view=True),
            RenderContext(lang="zh", message="test", skill_context="some context"),
        ]:
            output = sandbox_state_section.render(ctx)
            assert output.text is None

    def test_section_metadata(self) -> None:
        s = sandbox_state_section
        assert s.id == "sandbox_state"
        assert s.priority == 5
        assert s.cacheable is False
        assert s.dynamic is True

    def test_NOT_in_minimal_allowlist(self) -> None:
        from app.domain.services.prompts.section import MINIMAL_MODE_ALLOWLIST

        assert sandbox_state_section.id not in MINIMAL_MODE_ALLOWLIST
