import asyncio

import pytest
from app.domain.models.app_config import MCPConfig, MCPServerConfig, MCPTransport
from app.domain.services.tools.mcp import MCPClientManager, MCPTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_connect_mcp_servers_continues_when_single_server_times_out(
    monkeypatch,
) -> None:
    manager = MCPClientManager(
        MCPConfig(
            mcpServers={
                "slow-server": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://slow.example.com/mcp",
                ),
                "fast-server": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://fast.example.com/mcp",
                ),
            }
        )
    )
    called: list[str] = []

    async def fake_connect(server_name: str, server_config: MCPServerConfig) -> None:
        called.append(server_name)
        if server_name == "slow-server":
            await asyncio.sleep(0.5)

    monkeypatch.setattr(manager, "_connect_mcp_server", fake_connect)
    monkeypatch.setattr(
        "app.domain.services.tools.mcp.MCP_SERVER_CONNECT_TIMEOUT_SECONDS",
        0.05,
    )

    await asyncio.wait_for(manager._connect_mcp_servers(), timeout=0.2)

    assert called == ["slow-server", "fast-server"]


async def test_connect_mcp_servers_skips_disabled_servers(monkeypatch) -> None:
    manager = MCPClientManager(
        MCPConfig(
            mcpServers={
                "disabled-server": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=False,
                    url="https://disabled.example.com/mcp",
                ),
                "enabled-server": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://enabled.example.com/mcp",
                ),
            }
        )
    )
    called: list[str] = []

    async def fake_connect(server_name: str, server_config: MCPServerConfig) -> None:
        called.append(server_name)

    monkeypatch.setattr(manager, "_connect_mcp_server", fake_connect)

    await manager._connect_mcp_servers()

    assert called == ["enabled-server"]


async def test_mcp_tool_initialize_degrades_when_manager_init_fails(monkeypatch) -> None:
    async def fake_initialize(self) -> None:
        raise RuntimeError("boom")

    async def fake_cleanup(self) -> None:
        return None

    monkeypatch.setattr(MCPClientManager, "initialize", fake_initialize)
    monkeypatch.setattr(MCPClientManager, "cleanup", fake_cleanup)

    tool = MCPTool()
    config = MCPConfig(
        mcpServers={
            "bad-server": MCPServerConfig(
                transport=MCPTransport.STREAMABLE_HTTP,
                enabled=True,
                url="https://bad.example.com/mcp",
            )
        }
    )

    await tool.initialize(config)

    assert tool.get_tools() == []


async def test_mcp_manager_invoke_returns_timeout_when_call_tool_hangs(
    monkeypatch,
) -> None:
    class _SlowSession:
        async def call_tool(self, _tool_name: str, _arguments: dict) -> object:
            await asyncio.sleep(0.5)
            return object()

    manager = MCPClientManager(
        MCPConfig(
            mcpServers={
                "amap-mcp-server": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://amap.example.com/mcp",
                )
            }
        )
    )
    manager._clients["amap-mcp-server"] = _SlowSession()  # type: ignore[assignment]
    monkeypatch.setattr(
        "app.domain.services.tools.mcp.MCP_TOOL_CALL_TIMEOUT_SECONDS",
        0.05,
    )

    result = await asyncio.wait_for(
        manager.invoke(
            "mcp_amap-mcp-server_maps_direction_transit_integrated_by_address",
            {"origin": "A", "destination": "B"},
        ),
        timeout=0.2,
    )

    assert result.success is False
    assert "超时" in (result.message or "")
