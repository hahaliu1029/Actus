"""MCP tool discovery: list_mcp_tools + get_mcp_tool for progressive loading.

Follows the same pattern as Skill progressive loading:
- Layer 1 (metadata): list_mcp_tools() shows server + tool names + descriptions
- Layer 2 (schema): get_mcp_tool(name) returns full parameters and activates the tool
- Layer 3 (call): activated tool is bound in the next plan step's react_graph
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)

logger = logging.getLogger(__name__)


def create_mcp_discovery_tools(
    mcp_tool_ref: Callable[[], Any],
    activated_tools_ref: Callable[[], set[str]],
) -> list[StructuredTool]:
    """Create MCP discovery tools for progressive loading.

    Parameters
    ----------
    mcp_tool_ref : callable returning the MCPTool instance
    activated_tools_ref : callable returning the mutable activated tools set
    """

    async def _list_mcp_tools(server_name: str = "") -> str:
        """列出可用的 MCP 工具。不传参数返回所有概览；传入服务器名返回该服务器详情。"""
        mcp = mcp_tool_ref()
        all_tools = mcp.get_tools()
        if not all_tools:
            return "No MCP tools available."

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
            lines.append(f"- **{name}**: {desc}")

        if len(lines) == 1:
            return f"No MCP tools found for server '{server_name}'."

        lines.append(
            "\nUse `get_mcp_tool(tool_name)` to get full parameter details "
            "and activate a tool."
        )
        return "\n".join(lines)

    async def _get_mcp_tool(tool_name: str) -> str:
        """获取 MCP 工具完整参数定义并激活。激活后该工具将在下一个 plan step 可直接调用。"""
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
                return "\n".join(lines)

        return (
            f"MCP tool '{tool_name}' not found. "
            "Use `list_mcp_tools()` to see available tools."
        )

    list_mcp_tools = StructuredTool.from_function(
        coroutine=_list_mcp_tools,
        name="list_mcp_tools",
        description=(
            "列出可用的 MCP 工具。不传参数返回所有服务器工具概览；"
            "传入服务器名前缀返回该服务器的工具详情。"
        ),
    )
    get_mcp_tool = StructuredTool.from_function(
        coroutine=_get_mcp_tool,
        name="get_mcp_tool",
        description=(
            "获取指定 MCP 工具的完整参数定义并激活。"
            "激活后该工具将在下一个 plan step 中可直接调用。"
            "传入工具全名（如 'mcp_amap-maps_maps_weather'）。"
        ),
    )

    tools = [list_mcp_tools, get_mcp_tool]
    for t in tools:
        annotate_and_register_tool_source(t, source="mcp", category="mcp discovery")
    return tools
