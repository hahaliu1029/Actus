"""planner_tool_summary_legacy section — extract tool summary from state.skill_context.

B5 C6: the planner_node and updater_node currently build the tool summary
inside the **executor's** ``_build_runtime_system_context`` and store it
into ``state.skill_context``. When the planner runs, it reads
``state.skill_context`` and extracts the ``## Available Tool Summary``
substring to know which tools are available.

This section preserves that legacy behavior as a pass-through: it reads
``ctx.skill_context``, finds the ``## Available Tool Summary`` marker,
and emits everything from that marker onward.

**Why not reuse ``tools_guide_dynamic``**: that section reads
``ctx.bound_tool_names``, which is empty at planner time (planner
doesn't go through ``react_graph_provider``). Using it here would
produce an empty output — a semantic regression. The long-term fix is
to populate ``bound_tool_names`` for the planner independently, but
that's out of B5 scope.

Returns ``SectionOutput(text=None)`` when ``ctx.skill_context`` is empty
or does not contain the ``## Available Tool Summary`` marker.

priority=7, dynamic=True, NOT in MINIMAL_MODE_ALLOWLIST.
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


_TOOL_SUMMARY_MARKER = "## Available Tool Summary"


def _render(ctx: RenderContext) -> SectionOutput:
    """Extract the tool summary substring from ``ctx.skill_context``."""
    skill_context = ctx.skill_context or ""
    if _TOOL_SUMMARY_MARKER not in skill_context:
        return SectionOutput(text=None)
    start = skill_context.index(_TOOL_SUMMARY_MARKER)
    summary_text = skill_context[start:].strip()
    if not summary_text:
        return SectionOutput(text=None)
    return SectionOutput(
        text=summary_text,
        metadata={"tool_summary_source": "state.skill_context"},
    )


planner_tool_summary_legacy_section = Section(
    id="planner_tool_summary_legacy",
    priority=7,
    cacheable=False,
    dynamic=True,
    render=_render,
)
