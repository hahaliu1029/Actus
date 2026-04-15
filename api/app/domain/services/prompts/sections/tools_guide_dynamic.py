"""tools_guide_dynamic section — Available Tool Summary built from bound_tool_names.

B5 C3: this is the section that replaces the legacy
``agent_task_runner._build_available_tool_summary()``. The legacy reads from
multiple raw tool registries; the new version reads from the unified
``ctx.bound_tool_names`` set, which is the authoritative source provided by
``_build_step_react_graph`` (C5a).

The section returns ``SectionOutput(text=None)`` if ``bound_tool_names`` is
empty (no tools bound), so the assembler skips it.

**The section title "## Available Tool Summary" is hardcoded English** —
LLMs use it as a structural marker; translating it would break the
``Available Tool Summary`` reference in behavior_core (which both ZH and
EN behavior_core mention by name).
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)
from app.domain.services.tools.tool_source_resolver import (
    ToolSourceUnknownError,
    resolve_tool_source,
)


# Display order for category groups (categories not in this list are appended at the end)
_DISPLAY_ORDER: tuple[str, ...] = (
    "shell",
    "file",
    "browser",
    "message",
    "search",
    "memory",
    "skill",
    "skill creator",
    "skill guide",
    "mcp",
    "mcp discovery",
    "a2a",
)


def _categorize(tool_name: str) -> str | None:
    """Return the category label for a tool name via the ToolSource resolver."""
    try:
        return resolve_tool_source(tool_name).category
    except ToolSourceUnknownError:
        return None


def _group_tools_by_category(
    bound_tool_names: frozenset[str],
) -> dict[str, list[str]]:
    """Group bound tool names into category buckets, sorted within each group.

    Tools whose name cannot be resolved by the ToolSource resolver (i.e.
    ``_categorize`` returns ``None``) are bucketed under ``"other"`` so the
    summary surfaces them instead of silently swallowing them.
    """
    groups: dict[str, list[str]] = {}
    for name in sorted(bound_tool_names):
        category = _categorize(name) or "other"
        groups.setdefault(category, []).append(name)
    return groups


def _render(ctx: RenderContext) -> SectionOutput:
    """Build the Available Tool Summary section from ``ctx.bound_tool_names``."""
    if not ctx.bound_tool_names:
        return SectionOutput(text=None)

    groups = _group_tools_by_category(ctx.bound_tool_names)
    if not groups:
        return SectionOutput(text=None)

    # Render in canonical display order; unknown categories at the end
    lines = ["## Available Tool Summary"]
    seen: set[str] = set()
    for category in _DISPLAY_ORDER:
        names = groups.get(category)
        if names:
            lines.append(f"- {category}: {', '.join(names)}")
            seen.add(category)
    # Any remaining categories not in _DISPLAY_ORDER (e.g. "other")
    for category, names in groups.items():
        if category in seen:
            continue
        lines.append(f"- {category}: {', '.join(names)}")

    text = "\n".join(lines)
    return SectionOutput(
        text=text,
        metadata={"bound_tool_names_used": sorted(ctx.bound_tool_names)},
    )


tools_guide_dynamic_section = Section(
    id="tools_guide_dynamic",
    priority=7,
    cacheable=False,
    dynamic=True,
    render=_render,
)
