"""Tool-filter presets — named allowlists for ``AgentTaskRunner._tool_filter``.

T12 / Phase 1 PR-X: child sessions (e.g. subagent research) carry a persisted
preset name on ``Session.tool_filter_preset`` so that ``_create_task``
reconstruction paths (resume after pod restart, FINISHING re-entry, orphan
sweeper, preflight rebuild) can re-derive the in-memory allowlist instead of
silently falling back to ``tool_filter=None`` (the F8 security gap).

Adding a new preset:

1. Add the entry to ``TOOL_FILTER_PRESETS`` here.
2. Widen the ``Literal[...]`` on ``Session.tool_filter_preset``.
3. Extend the CHECK constraint via an Alembic migration so DB only accepts
   the new value.

``resolve_preset`` fails closed on an unknown name (raises ``ValueError``).
The CHECK constraint already prevents unknown values from reaching the DB,
so an unknown lookup signals a code/data inconsistency that should surface
loudly rather than degrade silently into an unfiltered session.
"""
from __future__ import annotations

from typing import FrozenSet, Mapping, Optional


SUBAGENT_RESEARCH_ALLOWED_TOOLS: FrozenSet[str] = frozenset({
    "search_web",
    "file_read",
    "file_list",
    "file_view",
    "list_mcp_tools",
    "get_mcp_tool",
    "get_skill_guide",
    "memory_search",
    "memory_get",
    "shell_read_output",
})


# [C2 PR-1 Task 1.6] Coordinator step worker allowlist. The coordinator child
# session executes a single planned step under a constrained tool surface:
# read-anything + typed file writes, but NO shell/browser/user-interaction/
# memory_save (writes go through file tools so reducer_node can detect them).
COORDINATOR_STEP_BASE_ALLOWED_TOOLS: FrozenSet[str] = frozenset({
    # READ
    "search_web",
    "file_read",
    "file_list",
    "file_view",
    # typed WRITE only
    "file_write",
    "file_str_replace",
    # META (tool/skill discovery)
    "list_mcp_tools",
    "get_mcp_tool",
    "get_skill_guide",
    # MEMORY READ
    "memory_search",
    "memory_get",
})


TOOL_FILTER_PRESETS: Mapping[str, FrozenSet[str]] = {
    "subagent_research": SUBAGENT_RESEARCH_ALLOWED_TOOLS,
    "coordinator_step": COORDINATOR_STEP_BASE_ALLOWED_TOOLS,
}


def resolve_preset(name: Optional[str]) -> Optional[FrozenSet[str]]:
    """Return the allowlist for ``name`` or ``None`` if ``name`` is ``None``.

    Raises ``ValueError`` for unknown preset names so a mis-persisted /
    rolled-back preset surfaces loudly during task reconstruction rather
    than silently letting the session run unfiltered (F8 protection).
    """
    if name is None:
        return None
    try:
        return TOOL_FILTER_PRESETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tool_filter_preset: {name!r}. "
            f"Known presets: {sorted(TOOL_FILTER_PRESETS)}"
        ) from exc
