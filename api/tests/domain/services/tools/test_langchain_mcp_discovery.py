"""Tests for MCP discovery tools (list_mcp_tools, get_mcp_tool)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.domain.services.tools.langchain_mcp_discovery import create_mcp_discovery_tools

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_mcp_tool():
    mock = MagicMock()
    mock.get_tools.return_value = [
        {
            "function": {
                "name": "mcp_amap_weather",
                "description": "[amap] Weather query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        },
        {
            "function": {
                "name": "mcp_amap_geo",
                "description": "[amap] Geocoding",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address": {"type": "string", "description": "Address"},
                    },
                },
            }
        },
    ]
    return mock


class TestListMcpTools:

    async def test_list_all_servers(self):
        mcp = _make_mcp_tool()
        activated = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
        )
        list_tool = next(t for t in tools if t.name == "list_mcp_tools")
        result = await list_tool.ainvoke({"server_name": ""})
        assert "mcp_amap_weather" in result
        assert "mcp_amap_geo" in result

    async def test_list_empty_mcp(self):
        mcp = MagicMock()
        mcp.get_tools.return_value = []
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: set(),
        )
        list_tool = next(t for t in tools if t.name == "list_mcp_tools")
        result = await list_tool.ainvoke({"server_name": ""})
        assert "no mcp tools" in result.lower()


class TestGetMcpTool:

    async def test_activate_tool(self):
        mcp = _make_mcp_tool()
        activated = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
        )
        get_tool = next(t for t in tools if t.name == "get_mcp_tool")
        result = await get_tool.ainvoke({"tool_name": "mcp_amap_weather"})
        assert "mcp_amap_weather" in result
        assert "city" in result
        assert "mcp_amap_weather" in activated

    async def test_activate_nonexistent_tool(self):
        mcp = _make_mcp_tool()
        activated = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
        )
        get_tool = next(t for t in tools if t.name == "get_mcp_tool")
        result = await get_tool.ainvoke({"tool_name": "nonexistent"})
        assert "not found" in result.lower()
        assert "nonexistent" not in activated

    async def test_activate_idempotent(self):
        mcp = _make_mcp_tool()
        activated = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
        )
        get_tool = next(t for t in tools if t.name == "get_mcp_tool")
        await get_tool.ainvoke({"tool_name": "mcp_amap_weather"})
        await get_tool.ainvoke({"tool_name": "mcp_amap_weather"})
        assert len(activated) == 1

    async def test_result_mentions_next_step(self):
        """Result should mention the tool will be available in the next step."""
        mcp = _make_mcp_tool()
        activated = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
        )
        get_tool = next(t for t in tools if t.name == "get_mcp_tool")
        result = await get_tool.ainvoke({"tool_name": "mcp_amap_weather"})
        assert "next" in result.lower()

    async def test_activate_shows_required_params(self):
        """Required parameters should be marked."""
        mcp = _make_mcp_tool()
        activated = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
        )
        get_tool = next(t for t in tools if t.name == "get_mcp_tool")
        result = await get_tool.ainvoke({"tool_name": "mcp_amap_weather"})
        assert "(required)" in result


class TestExcludedServers:
    """D1a G2 (R1#19): governance-blocked servers filtered from discovery."""

    def _make_two_server_mcp(self):
        mock = MagicMock()
        mock.get_tools.return_value = [
            {
                "function": {
                    "name": "mcp_ok_weather",
                    "description": "[ok] weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            },
            {
                "function": {
                    "name": "mcp_bad_lookup",
                    "description": "[bad] SHOULD NOT LEAK",
                    "parameters": {
                        "type": "object",
                        "properties": {"apikey": {"type": "string"}},
                    },
                }
            },
        ]
        mock.tool_server_bindings.return_value = {
            "mcp_ok_weather": "srv-ok",
            "mcp_bad_lookup": "srv-bad",
        }
        return mock

    async def test_list_omits_blocked_server_tools(self):
        mcp = self._make_two_server_mcp()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: set(),
            excluded_servers={"srv-bad"},
        )
        list_tool = next(t for t in tools if t.name == "list_mcp_tools")
        result = await list_tool.ainvoke({"server_name": ""})
        assert "mcp_ok_weather" in result
        assert "mcp_bad_lookup" not in result, (
            f"blocked-server tool leaked into list output:\n{result}"
        )
        assert "SHOULD NOT LEAK" not in result

    async def test_get_refuses_blocked_server_activation(self):
        mcp = self._make_two_server_mcp()
        activated: set[str] = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
            excluded_servers={"srv-bad"},
        )
        get_tool = next(t for t in tools if t.name == "get_mcp_tool")
        result = await get_tool.ainvoke({"tool_name": "mcp_bad_lookup"})
        assert "mcp_bad_lookup" not in activated, (
            f"blocked-server tool was activated: {activated}"
        )
        # schema param name must not leak in the denial
        assert "apikey" not in result

    async def test_none_excluded_keeps_legacy_behavior(self):
        mcp = self._make_two_server_mcp()
        activated: set[str] = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
            excluded_servers=None,
        )
        list_tool = next(t for t in tools if t.name == "list_mcp_tools")
        get_tool = next(t for t in tools if t.name == "get_mcp_tool")
        result = await list_tool.ainvoke({"server_name": ""})
        assert "mcp_ok_weather" in result and "mcp_bad_lookup" in result
        await get_tool.ainvoke({"tool_name": "mcp_bad_lookup"})
        assert "mcp_bad_lookup" in activated

    async def test_allowed_server_tool_still_activates(self):
        mcp = self._make_two_server_mcp()
        activated: set[str] = set()
        tools = create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp,
            activated_tools_ref=lambda: activated,
            excluded_servers={"srv-bad"},
        )
        get_tool = next(t for t in tools if t.name == "get_mcp_tool")
        result = await get_tool.ainvoke({"tool_name": "mcp_ok_weather"})
        assert "mcp_ok_weather" in activated
        assert "mcp_ok_weather" in result
