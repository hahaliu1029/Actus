"""build_render_context — bridges LangGraph state into a Section RenderContext.

B5 C1: defines the function signature and pure helpers. The actual call
sites in main_graph.py executor_node / planner_node / updater_node are
wired in C5b/C6.

C1 ships the function but no caller. C1 tests cover the pure helper
``_infer_provider`` and the field-mapping logic against synthetic state /
config dicts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from app.domain.services.prompts.section import RenderContext
from app.domain.services.tools.tool_source_resolver import (
    ToolSourceUnknownError,
    resolve_tool_source,
)

if TYPE_CHECKING:
    from app.domain.models.memory_recall import RecalledMemory
    from app.domain.services.prompts.memory_snapshot import MemorySnapshot


def _infer_provider(llm: Any) -> Literal["openai", "anthropic"]:
    """Read the canonical ``provider_name`` attribute off the LLM adapter.

    Falls back to ``"openai"`` for adapters that don't expose the attribute
    (legacy or test mocks). C0a added ``provider_name = "openai"`` to all
    three production adapters.
    """
    value = getattr(llm, "provider_name", "openai")
    if value not in ("openai", "anthropic"):
        return "openai"
    return value  # type: ignore[return-value]


def _categorize_tools(bound_tool_names: frozenset[str]) -> frozenset[str]:
    """Return the set of categories represented in ``bound_tool_names``.

    Uses ``tool_source_resolver`` as the single source of truth. Any name
    that cannot be resolved (truly unknown) is silently skipped — the
    categorize function should not crash render_context on a rogue name.

    Returns ``frozenset`` to keep ``RenderContext`` immutable.
    """
    categories: set[str] = set()
    for name in bound_tool_names:
        try:
            categories.add(resolve_tool_source(name).category)
        except ToolSourceUnknownError:
            continue
    return frozenset(categories)


def _has_category(bound_tool_names: frozenset[str], target: str) -> bool:
    """Return True iff any bound tool resolves to ``target`` category.

    Uses ``category ==`` (not ``source ==``) so discovery-only steps
    (only ``list_mcp_tools`` / ``get_mcp_tool`` bound) are correctly
    classified as ``category == "mcp discovery"`` — NOT ``category == "mcp"``
    — and therefore do NOT flip ``mcp_active`` to True.
    """
    for name in bound_tool_names:
        try:
            if resolve_tool_source(name).category == target:
                return True
        except ToolSourceUnknownError:
            continue
    return False


def _format_attachments_for_context(state: dict) -> str | None:
    """Render attachments as plain text for sections that want to mention them.

    Sections that need rich attachment metadata should read ``state.attachments``
    directly via the ``RenderContext`` extension mechanism — this helper is for
    the simple "list of file paths" use case.
    """
    attachments = state.get("attachments")
    if not attachments:
        return None
    if isinstance(attachments, list):
        return "\n".join(str(a) for a in attachments)
    return str(attachments)


def _extract_skill_names_from_state(state: dict) -> tuple[str, ...]:
    """Read the authoritative skill id list from state, falling back to ()."""
    value = state.get("skill_names_in_context")
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def build_render_context(
    state: dict,
    config: dict,
    agent_config: Any,
    *,
    memory_snapshot: "MemorySnapshot | None" = None,
    team_members: "tuple[tuple[str, str], ...] | None" = None,
    recalled_memory: "RecalledMemory | None" = None,
    parallel_dispatch_allowed: bool = True,
) -> RenderContext:
    """Build a ``RenderContext`` from LangGraph state + config + AgentConfig.

    C1 only defines the function. C5b wires the call site in
    ``main_graph.executor_node`` (and the other nodes in C6).

    Required ``configurable`` fields (set by C5a / C6):
    - ``llm``: BaseChatModel instance with ``provider_name`` attribute
    - ``bound_tool_names``: ``frozenset[str]`` of currently bound tool names

    The function is defensive — missing fields fall back to safe defaults
    (empty sets, ``"zh"``, etc.) so partial state during graph initialization
    doesn't crash assembly.

    M2 PR-4: ``memory_snapshot`` is an optional pre-built bundle of
    category-bucketed memory chunks. Callers that have user_id + repo
    await ``build_memory_snapshot`` upstream (in the async node) and pass
    the result here as a kwarg. Absent (the test / no-memory default)
    leaves the three memory sections inert.

    B8: ``recalled_memory`` is the query-time recall bundle for the
    planner's recalled_memory section. Only the two planner entrances
    pass it; every other call site takes the None default and renders
    byte-identically.
    """
    configurable = (config.get("configurable") if config else None) or {}
    llm = configurable.get("llm")
    provider = _infer_provider(llm)

    bound_tool_names = configurable.get("bound_tool_names")
    if not isinstance(bound_tool_names, frozenset):
        bound_tool_names = frozenset(bound_tool_names or ())

    skill_names = _extract_skill_names_from_state(state)

    current_step = state.get("current_step")
    step_description = (
        getattr(current_step, "description", None) if current_step else None
    )

    summaries_raw = state.get("conversation_summaries") or ()
    summaries = tuple(summaries_raw) if isinstance(summaries_raw, (list, tuple)) else ()

    return RenderContext(
        lang=state.get("language", "zh"),
        provider=provider,
        step_description=step_description,
        message=state.get("message"),
        attachments_text=_format_attachments_for_context(state),
        skill_context=state.get("skill_context"),
        skill_names_in_context=skill_names,
        bound_tool_names=bound_tool_names,
        conversation_summaries=summaries,
        has_file_view="file_view" in bound_tool_names,
        has_memory_tools=_has_category(bound_tool_names, "memory"),
        has_vision=getattr(agent_config, "supports_vision", True),
        has_pdf=getattr(agent_config, "supports_pdf_input", True),
        tool_categories=_categorize_tools(bound_tool_names),
        mcp_active=_has_category(bound_tool_names, "mcp"),
        a2a_active=_has_category(bound_tool_names, "a2a"),
        memory_snapshot=memory_snapshot,
        team_members=team_members,
        recalled_memory=recalled_memory,
        parallel_dispatch_allowed=parallel_dispatch_allowed,
    )
