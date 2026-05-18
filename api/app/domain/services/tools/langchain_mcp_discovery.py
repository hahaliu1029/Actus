"""MCP tool discovery: list_mcp_tools + get_mcp_tool for progressive loading.

Follows the same pattern as Skill progressive loading:
- Layer 1 (metadata): list_mcp_tools() shows server + tool names + descriptions
- Layer 2 (schema): get_mcp_tool(name) returns full parameters and activates the tool
- Layer 3 (call): activated tool is bound in the next plan step's react_graph
"""

from __future__ import annotations

import logging
from typing import Any, Callable, FrozenSet, Optional

from langchain_core.tools import StructuredTool

from app.domain.models.tool_result import AllowSuccess, ToolOutcome
from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)

logger = logging.getLogger(__name__)


def create_mcp_discovery_tools(
    mcp_tool_ref: Callable[[], Any],
    activated_tools_ref: Callable[[], set[str]],
    tool_filter: Optional[FrozenSet[str]] = None,
) -> list[StructuredTool]:
    """Create MCP discovery tools for progressive loading.

    Parameters
    ----------
    mcp_tool_ref : callable returning the MCPTool instance
    activated_tools_ref : callable returning the mutable activated tools set
    tool_filter : optional frozenset of allowed tool names. When provided
        (``is not None``; ``frozenset()`` means deny-all):
          - ``list_mcp_tools`` only surfaces MCP tool names whose canonical
            name is in the allowlist — names absent from the allowlist are
            silently omitted from the discovery output.
          - ``get_mcp_tool`` refuses to activate (and refuses to return a
            schema for) any name absent from the allowlist; instead returns
            a denial message that exposes only the requested name (which the
            caller already chose to mention) and no schema/metadata.
        ``None`` (default) preserves legacy "no filter" behaviour and is
        the correct value at the top-level / parent-agent layer.
    """

    async def _list_mcp_tools(server_name: str = "") -> tuple[str, ToolOutcome]:
        """列出可用的 MCP 工具。不传参数返回所有概览；传入服务器名返回该服务器详情。"""
        mcp = mcp_tool_ref()
        all_tools = mcp.get_tools()
        if not all_tools:
            outcome = AllowSuccess(content="No MCP tools available.")
            return outcome.content, outcome

        lines = ["## Available MCP Tools\n"]
        for schema in all_tools:
            fn = schema.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            if not name:
                continue
            if server_name:
                # Normalize: accept both "amap-maps" and "mcp_amap-maps"
                prefix = server_name if server_name.startswith("mcp_") else f"mcp_{server_name}"
                if not name.startswith(f"{prefix}_"):
                    continue
            # Phase 1 minimal subagent — tool_filter enforcement:
            # skip any tool whose canonical name is not in the allowlist
            # so the metadata never reaches the LLM.
            if tool_filter is not None and name not in tool_filter:
                continue
            lines.append(f"- **{name}**: {desc}")

        if len(lines) == 1:
            content = f"No MCP tools found for server '{server_name}'."
            outcome = AllowSuccess(content=content)
            return outcome.content, outcome

        # P1 #5 fix — Make the activation-hint allowlist-aware so the
        # blocked meta-tool name (``get_mcp_tool``) never leaks via the
        # ``list_mcp_tools`` body when the subagent's allowlist excludes
        # it. ``tool_filter is None`` preserves legacy parent-agent
        # behaviour (hint always emitted).
        if tool_filter is None or "get_mcp_tool" in tool_filter:
            lines.append(
                "\nUse `get_mcp_tool(tool_name)` to get full parameter details "
                "and activate a tool."
            )
        content = "\n".join(lines)
        outcome = AllowSuccess(content=content)
        return outcome.content, outcome

    async def _get_mcp_tool(tool_name: str) -> tuple[str, ToolOutcome]:
        """获取 MCP 工具完整参数定义并激活。激活后该工具将在下一个 plan step 可直接调用。"""
        # Phase 1 minimal subagent — tool_filter enforcement:
        # refuse activation BEFORE iterating the MCP catalog so we never
        # leak schema details (parameters / description) for a blocked
        # tool. The error message echoes only the requested name, which
        # the LLM already supplied as the ``tool_name`` argument — so we
        # don't leak any new information.
        if tool_filter is not None and tool_name not in tool_filter:
            # P1 #5 fix — Make the discovery hint allowlist-aware. If
            # ``list_mcp_tools`` is itself blocked, omit the hint entirely
            # so we don't leak the blocked meta-tool name in the denial
            # message. When neither hint is allowed, the message is just
            # the bare unavailability notice.
            base = (
                f"MCP tool '{tool_name}' is not available in this agent's "
                "tool allowlist."
            )
            if "list_mcp_tools" in tool_filter:
                content = (
                    f"{base} Use `list_mcp_tools()` to see the tools that "
                    "are accessible."
                )
            else:
                content = base
            outcome = AllowSuccess(content=content)
            return outcome.content, outcome

        mcp = mcp_tool_ref()
        activated = activated_tools_ref()

        for schema in mcp.get_tools():
            fn = schema.get("function", {})
            if fn.get("name") == tool_name:
                activated.add(tool_name)
                logger.info("[MCPDiscovery] Activated tool '%s'", tool_name)

                params = fn.get("parameters", {})
                desc = fn.get("description", "")
                props = params.get("properties", {})
                required = params.get("required", [])

                lines = [
                    f"## {tool_name}",
                    f"**Description**: {desc}",
                    "",
                    "**Parameters**:",
                ]
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "any")
                    pdesc = pdef.get("description", "")
                    req = " (required)" if pname in required else ""
                    lines.append(f"- `{pname}` ({ptype}){req}: {pdesc}")

                if not props:
                    lines.append("- (no parameters)")

                lines.append("")
                lines.append(
                    f"Tool `{tool_name}` has been activated and will be available "
                    "for direct calling in the **next plan step**."
                )
                content = "\n".join(lines)
                outcome = AllowSuccess(content=content, data=schema if isinstance(schema, dict) else None)
                return outcome.content, outcome

        # P1 #5 fix — same allowlist-aware treatment for the "not found"
        # branch: don't expose ``list_mcp_tools`` as a hint if the
        # subagent allowlist excludes it.
        not_found_base = f"MCP tool '{tool_name}' not found."
        if tool_filter is None or "list_mcp_tools" in tool_filter:
            content = (
                f"{not_found_base} Use `list_mcp_tools()` to see available tools."
            )
        else:
            content = not_found_base
        outcome = AllowSuccess(content=content)
        return outcome.content, outcome

    list_mcp_tools = StructuredTool.from_function(
        coroutine=_list_mcp_tools,
        name="list_mcp_tools",
        description=(
            "列出可用的 MCP 工具。不传参数返回所有服务器工具概览；"
            "传入服务器名前缀返回该服务器的工具详情。"
        ),
        response_format="content_and_artifact",
    )
    get_mcp_tool = StructuredTool.from_function(
        coroutine=_get_mcp_tool,
        name="get_mcp_tool",
        description=(
            "获取指定 MCP 工具的完整参数定义并激活。"
            "激活后该工具将在下一个 plan step 中可直接调用。"
            "传入工具全名（如 'mcp_amap-maps_maps_weather'）。"
        ),
        response_format="content_and_artifact",
    )

    tools = [list_mcp_tools, get_mcp_tool]
    for t in tools:
        annotate_and_register_tool_source(t, source="mcp", category="mcp discovery")
    return tools
