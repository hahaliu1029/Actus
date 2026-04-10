"""B5 C6: per-section unit tests for planner_identity + planner_tool_summary_legacy."""
from __future__ import annotations

import pytest

from app.domain.services.prompts.section import RenderContext
from app.domain.services.prompts.sections.planner_identity import (
    planner_identity_section,
)
from app.domain.services.prompts.sections.planner_tool_summary_legacy import (
    planner_tool_summary_legacy_section,
)


# ---- planner_identity -------------------------------------------------- #


class TestPlannerIdentity:
    def test_zh_renders(self) -> None:
        ctx = RenderContext(lang="zh")
        output = planner_identity_section.render(ctx)
        assert output.text is not None
        assert "任务规划智能体" in output.text
        assert "Task Planner Agent" in output.text  # ZH text embeds the English label

    def test_en_renders(self) -> None:
        ctx = RenderContext(lang="en")
        output = planner_identity_section.render(ctx)
        assert output.text is not None
        assert "task planner agent" in output.text.lower()
        assert "任务规划智能体" not in output.text

    def test_always_renders_regardless_of_ctx_fields(self) -> None:
        """Even with minimal ctx, planner_identity must render (priority 10)."""
        ctx = RenderContext(lang="zh")
        output = planner_identity_section.render(ctx)
        assert output.text is not None
        assert output.text.strip()

    def test_section_metadata(self) -> None:
        s = planner_identity_section
        assert s.id == "planner_identity"
        assert s.priority == 10
        assert s.cacheable is True
        assert s.dynamic is False

    def test_in_minimal_allowlist(self) -> None:
        """planner_identity must survive MINIMAL mode (for sub-agent planning)."""
        from app.domain.services.prompts.section import MINIMAL_MODE_ALLOWLIST

        assert planner_identity_section.id in MINIMAL_MODE_ALLOWLIST

    def test_zh_and_en_text_differ(self) -> None:
        zh_ctx = RenderContext(lang="zh")
        en_ctx = RenderContext(lang="en")
        assert (
            planner_identity_section.render(zh_ctx).text
            != planner_identity_section.render(en_ctx).text
        )


# ---- planner_tool_summary_legacy --------------------------------------- #


class TestPlannerToolSummaryLegacy:
    def test_empty_skill_context_returns_none(self) -> None:
        ctx = RenderContext(lang="zh", skill_context=None)
        output = planner_tool_summary_legacy_section.render(ctx)
        assert output.text is None

    def test_blank_string_returns_none(self) -> None:
        ctx = RenderContext(lang="zh", skill_context="")
        output = planner_tool_summary_legacy_section.render(ctx)
        assert output.text is None

    def test_no_marker_returns_none(self) -> None:
        ctx = RenderContext(
            lang="zh",
            skill_context="## Active Skills\n- skill_foo: do foo\n(no tool summary here)",
        )
        output = planner_tool_summary_legacy_section.render(ctx)
        assert output.text is None

    def test_extracts_from_marker_onward(self) -> None:
        skill_ctx = (
            "## Active Skills\n- skill_foo: do foo\n\n"
            "## Available Tool Summary\n- shell: shell_execute\n- file: file_read"
        )
        ctx = RenderContext(lang="zh", skill_context=skill_ctx)
        output = planner_tool_summary_legacy_section.render(ctx)
        assert output.text is not None
        assert output.text.startswith("## Available Tool Summary")
        assert "shell_execute" in output.text
        assert "## Active Skills" not in output.text  # trimmed prefix
        assert "skill_foo" not in output.text

    def test_marker_at_start(self) -> None:
        """Marker at position 0 should still produce valid extraction."""
        ctx = RenderContext(
            lang="zh",
            skill_context="## Available Tool Summary\n- shell: shell_execute",
        )
        output = planner_tool_summary_legacy_section.render(ctx)
        assert output.text is not None
        assert output.text.startswith("## Available Tool Summary")

    def test_metadata_indicates_legacy_source(self) -> None:
        ctx = RenderContext(
            lang="zh",
            skill_context="## Available Tool Summary\n- shell: shell_execute",
        )
        output = planner_tool_summary_legacy_section.render(ctx)
        assert output.metadata["tool_summary_source"] == "state.skill_context"

    def test_section_metadata(self) -> None:
        s = planner_tool_summary_legacy_section
        assert s.id == "planner_tool_summary_legacy"
        assert s.priority == 7
        assert s.cacheable is False
        assert s.dynamic is True

    def test_NOT_in_minimal_allowlist(self) -> None:
        from app.domain.services.prompts.section import MINIMAL_MODE_ALLOWLIST

        assert planner_tool_summary_legacy_section.id not in MINIMAL_MODE_ALLOWLIST

    def test_lang_independent_extraction(self) -> None:
        """The marker extraction is language-independent (ZH and EN use the
        same `## Available Tool Summary` header)."""
        skill_ctx = "## Available Tool Summary\n- shell: shell_execute"
        zh_output = planner_tool_summary_legacy_section.render(
            RenderContext(lang="zh", skill_context=skill_ctx)
        )
        en_output = planner_tool_summary_legacy_section.render(
            RenderContext(lang="en", skill_context=skill_ctx)
        )
        assert zh_output.text == en_output.text
