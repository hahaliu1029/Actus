"""B9 P-7：公开只读已连接清单（spec §4 R2#4）。"""
from unittest.mock import MagicMock

from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.mcp import MCPClientManager, MCPTool


def test_manager_connected_server_ids_reads_clients():
    mgr = MCPClientManager(mcp_config=None)
    mgr._clients = {"srv-a": MagicMock(), "srv-b": MagicMock()}
    assert sorted(mgr.connected_server_ids()) == ["srv-a", "srv-b"]


def test_mcp_tool_forwards_and_handles_none_manager():
    tool = MCPTool()
    assert tool.connected_server_ids() == []      # manager 未初始化
    tool._manager = MagicMock()
    tool._manager.connected_server_ids.return_value = ["srv-a"]
    assert tool.connected_server_ids() == ["srv-a"]


def test_a2a_tool_reads_agent_cards_keys():
    tool = A2ATool()
    assert tool.connected_server_ids() == []
    mgr = MagicMock()
    mgr.agent_cards = {"a2a-id-1": MagicMock()}
    # A2ATool 的 manager 属性挂法以 a2a.py 实际字段为准（grep "self.manager\|self._manager"）
    tool.manager = mgr
    assert tool.connected_server_ids() == ["a2a-id-1"]
