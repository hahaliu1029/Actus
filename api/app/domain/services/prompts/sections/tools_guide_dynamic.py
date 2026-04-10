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


# Map tool name prefix → category label shown in the summary.
# Order matters: keys are checked in declaration order, first match wins.
#
# Naming invariant (enforced by convention, not code):
# - Native sandbox tools use clean prefixes: shell_*, file_*, browser_*,
#   message_*, search_*, memory_*.
# - MCP tools are ALWAYS `mcp_*` (registry layer prefixes them).
# - Skill tools are ALWAYS `skill_*` (dynamic skill tool factory prefixes them).
# - A2A and skill-creator tools have bespoke names and live in the standalone
#   frozensets above.
# If a future MCP tool is registered with a non-`mcp_` prefix (e.g. `search_brave`),
# it will be mis-categorized here. New tool namespaces must either follow the
# prefix convention or be added to a bespoke frozenset.
_PREFIX_TO_CATEGORY = (
    ("shell_", "shell"),
    ("file_", "file"),
    ("browser_", "browser"),
    ("message_", "message"),
    ("search_", "search"),
    ("memory_", "memory"),
    ("mcp_", "mcp"),
    ("skill_", "skill"),
)

# Standalone tools that don't follow a clean prefix convention.
_A2A_TOOLS = frozenset({"get_remote_agent_cards", "call_remote_agent"})
_SKILL_CREATOR_TOOLS = frozenset({"brainstorm_skill", "generate_skill", "install_skill"})
_MCP_DISCOVERY_TOOLS = frozenset({"list_mcp_tools", "get_mcp_tool"})
_SKILL_GUIDE_TOOLS = frozenset({"get_skill_guide"})

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


def _categorize(tool_name: str) -> str:
    """Return the category label for a tool name."""
    if tool_name in _A2A_TOOLS:
        return "a2a"
    if tool_name in _SKILL_CREATOR_TOOLS:
        return "skill creator"
    if tool_name in _SKILL_GUIDE_TOOLS:
        return "skill guide"
    if tool_name in _MCP_DISCOVERY_TOOLS:
        return "mcp discovery"
    for prefix, category in _PREFIX_TO_CATEGORY:
        if tool_name.startswith(prefix):
            return category
    return "other"


def _group_tools_by_category(
    bound_tool_names: frozenset[str],
) -> dict[str, list[str]]:
    """Group bound tool names into category buckets, sorted within each group."""
    groups: dict[str, list[str]] = {}
    for name in sorted(bound_tool_names):
        category = _categorize(name)
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
