"""build_render_context — bridges LangGraph state into a Section RenderContext.

B5 C1: defines the function signature and pure helpers. The actual call
sites in main_graph.py executor_node / planner_node / updater_node are
wired in C5b/C6.

C1 ships the function but no caller. C1 tests cover the pure helper
``_infer_provider`` and the field-mapping logic against synthetic state /
config dicts.
"""
from __future__ import annotations

from typing import Any, Literal

from app.domain.services.prompts.section import RenderContext

_MEMORY_TOOL_NAMES = frozenset({"memory_search", "memory_get"})
_A2A_TOOL_NAMES = frozenset({"get_remote_agent_cards", "call_remote_agent"})

_NATIVE_TOOL_PREFIX_TO_CATEGORY = {
    "shell_": "shell",
    "file_": "file",
    "browser_": "browser",
    "message_": "message",
    "memory_": "memory",
    "search_": "search",
    "mcp_": "mcp",
    "skill_": "skill",
}


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
    """Group bound tool names into category buckets for ``ctx.tool_categories``.

    Returns ``frozenset`` to keep ``RenderContext`` immutable.
    """
    categories: set[str] = set()
    for name in bound_tool_names:
        if name in _A2A_TOOL_NAMES:
            categories.add("a2a")
            continue
        for prefix, category in _NATIVE_TOOL_PREFIX_TO_CATEGORY.items():
            if name.startswith(prefix):
                categories.add(category)
                break
    return frozenset(categories)


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
        has_memory_tools=bool(_MEMORY_TOOL_NAMES & bound_tool_names),
        has_vision=getattr(agent_config, "supports_vision", True),
        has_pdf=getattr(agent_config, "supports_pdf_input", True),
        tool_categories=_categorize_tools(bound_tool_names),
        mcp_active=any(n.startswith("mcp_") for n in bound_tool_names),
        a2a_active=bool(_A2A_TOOL_NAMES & bound_tool_names),
    )
