"""Characterization tests for PlannerReActFlow tool collection methods."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.flows.planner_react import PlannerReActFlow
from app.domain.services.tools._supervisor_tool_wrapper import (
    SupervisorAwareToolWrapper,
)


def _make_flow(**overrides):
    """Create a PlannerReActFlow with minimal stubs. Override any kwarg."""
    defaults = dict(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=MagicMock(),
        session_id="test-session",
        browser_accessor=EagerBrowserAccessor(MagicMock()),
        sandbox_accessor=EagerSandboxAccessor(MagicMock()),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(),
        a2a_tool=MagicMock(),
        skill_tool=MagicMock(),
        checkpointer=MagicMock(),
    )
    defaults.update(overrides)
    return PlannerReActFlow(**defaults)


class _ToolArgs(BaseModel):
    value: str


def _make_tool(name: str):
    async def _run(value: str) -> str:
        return value

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=f"{name} desc",
        args_schema=_ToolArgs,
    )


class TestCollectNativeTools:
    def test_returns_native_tool_names(self):
        flow = _make_flow()
        tools = flow._collect_native_tools()
        names = {t.name for t in tools}
        # Spot-check representative tools from each category
        assert "shell_execute" in names
        assert "file_read" in names
        assert "browser_navigate" in names
        assert "message_notify_user" in names
        assert "search_web" in names
        assert len(tools) > 10  # native tools are ~24 total

    def test_passes_execution_supervisor_to_native_factory(self, monkeypatch):
        flow = _make_flow()
        supervisor = object()
        flow._execution_supervisor = supervisor
        captured = {}

        def _fake_create_native_tools(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(
            "app.domain.services.flows.planner_react.create_native_tools",
            _fake_create_native_tools,
        )

        flow._collect_native_tools()

        assert captured["supervisor"] is supervisor


class TestCollectMcpTools:
    @pytest.mark.anyio
    async def test_small_set_binds_all(self):
        """MCP tools <= 15: all bound directly, no discovery tools."""
        mcp_tool = MagicMock()
        # Return 3 tool schemas (well under threshold of 15)
        mcp_tool.get_tools.return_value = [
            {"function": {"name": f"mcp_test_tool_{i}"}} for i in range(3)
        ]
        flow = _make_flow(mcp_tool=mcp_tool)
        tools = await flow._collect_mcp_tools()
        names = {t.name for t in tools}
        assert "list_mcp_tools" not in names  # no discovery
        assert "get_mcp_tool" not in names

    @pytest.mark.anyio
    async def test_large_set_uses_discovery(self):
        """MCP tools > 15: only always_bind + discovery tools, non-always_bind excluded."""
        mcp_tool = MagicMock()
        mcp_tool.get_tools.return_value = [
            {"function": {"name": f"mcp_test_tool_{i}"}} for i in range(20)
        ]
        flow = _make_flow(mcp_tool=mcp_tool)
        flow._mcp_always_bind_names = {"mcp_test_tool_0"}
        flow._mcp_tool_ref = lambda: mcp_tool
        flow._activated_mcp_tools_ref = lambda: set()
        tools = await flow._collect_mcp_tools()
        names = {t.name for t in tools}
        # Discovery tools present
        assert "list_mcp_tools" in names
        assert "get_mcp_tool" in names
        # always_bind tool is bound
        assert "mcp_test_tool_0" in names
        # Non-always_bind tools are excluded
        assert "mcp_test_tool_1" not in names
        assert "mcp_test_tool_19" not in names

    @pytest.mark.anyio
    async def test_none_mcp_tool_returns_empty(self):
        """mcp_tool=None should return [] without crashing."""
        flow = _make_flow(mcp_tool=None)
        tools = await flow._collect_mcp_tools()
        assert tools == []


class TestCollectMcpToolsExcludedServers:
    """D1a G2 / R2#F5: PlannerReActFlow drops governance-blocked servers from
    BOTH the direct-bind (<=15) and discovery (>15) collection paths."""

    @pytest.mark.anyio
    async def test_direct_bind_excludes_blocked_servers(self):
        """Small set: blocked-server tools never enter the direct-bind list."""
        mcp_tool = MagicMock()
        mcp_tool.get_tools.return_value = [
            {"function": {"name": "mcp_ok_a", "description": "", "parameters": {}}},
            {"function": {"name": "mcp_ok_b", "description": "", "parameters": {}}},
            {"function": {"name": "mcp_bad_c", "description": "", "parameters": {}}},
        ]
        mcp_tool.tool_server_bindings.return_value = {
            "mcp_ok_a": "srv-ok",
            "mcp_ok_b": "srv-ok",
            "mcp_bad_c": "srv-bad",
        }
        flow = _make_flow(mcp_tool=mcp_tool, excluded_mcp_servers={"srv-bad"})
        tools = await flow._collect_mcp_tools()
        names = {t.name for t in tools}
        assert "mcp_ok_a" in names and "mcp_ok_b" in names
        assert "mcp_bad_c" not in names, (
            f"blocked-server tool leaked into planner direct-bind: {names}"
        )

    @pytest.mark.anyio
    async def test_no_excluded_keeps_legacy_direct_bind(self):
        """Empty excluded set = identity (all direct-bound, legacy behavior)."""
        mcp_tool = MagicMock()
        mcp_tool.get_tools.return_value = [
            {"function": {"name": "mcp_ok_a", "description": "", "parameters": {}}},
            {"function": {"name": "mcp_bad_c", "description": "", "parameters": {}}},
        ]
        mcp_tool.tool_server_bindings.return_value = {
            "mcp_ok_a": "srv-ok",
            "mcp_bad_c": "srv-bad",
        }
        flow = _make_flow(mcp_tool=mcp_tool)  # excluded_mcp_servers defaults to empty
        tools = await flow._collect_mcp_tools()
        names = {t.name for t in tools}
        assert names == {"mcp_ok_a", "mcp_bad_c"}

    @pytest.mark.anyio
    async def test_discovery_path_receives_excluded_and_filters_always_bind(
        self, monkeypatch
    ):
        """Large set: discovery factory gets excluded_servers, and blocked
        always-bind tools are stripped from the direct-bind path."""
        mcp_tool = MagicMock()
        mcp_tool.get_tools.return_value = [
            {"function": {"name": f"mcp_test_tool_{i}"}} for i in range(20)
        ]
        mcp_tool.tool_server_bindings.return_value = {
            f"mcp_test_tool_{i}": ("srv-bad" if i == 0 else "srv-ok")
            for i in range(20)
        }
        flow = _make_flow(mcp_tool=mcp_tool, excluded_mcp_servers={"srv-bad"})
        flow._mcp_always_bind_names = {"mcp_test_tool_0", "mcp_test_tool_1"}
        flow._mcp_tool_ref = lambda: mcp_tool
        flow._activated_mcp_tools_ref = lambda: set()

        captured: dict = {}

        def _fake_discovery(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(
            "app.domain.services.tools.langchain_mcp_discovery.create_mcp_discovery_tools",
            _fake_discovery,
        )

        tools = await flow._collect_mcp_tools()
        # Discovery factory received the blocked server set.
        assert captured.get("excluded_servers") == {"srv-bad"}
        names = {t.name for t in tools}
        # Blocked always-bind tool stripped; allowed always-bind tool kept.
        assert "mcp_test_tool_0" not in names
        assert "mcp_test_tool_1" in names


class TestCollectA2aTools:
    def test_returns_a2a_tool_names(self):
        a2a_tool = MagicMock()
        a2a_tool.manager = MagicMock()  # non-None manager so tools are created
        flow = _make_flow(a2a_tool=a2a_tool)
        tools = flow._collect_a2a_tools()
        names = {t.name for t in tools}
        assert "get_remote_agent_cards" in names
        assert "call_remote_agent" in names


class TestCollectSkillCreationTools:
    def test_with_guide(self):
        """skill_pool_getter set -> includes get_skill_guide."""
        flow = _make_flow(
            brainstorm_skill_tool=MagicMock(),
            create_skill_tool=MagicMock(),
        )
        flow._skill_pool_getter = lambda: []
        flow._file_listings_getter = lambda: {}
        tools = flow._collect_skill_creation_tools()
        names = {t.name for t in tools}
        assert "get_skill_guide" in names

    def test_without_guide(self):
        """skill_pool_getter is None -> no get_skill_guide."""
        flow = _make_flow()
        flow._skill_pool_getter = None
        tools = flow._collect_skill_creation_tools()
        names = {t.name for t in tools}
        assert "get_skill_guide" not in names


class TestCollectAllTools:
    @pytest.mark.anyio
    async def test_order_native_mcp_a2a_skill(self):
        """Aggregator returns tools in strict order: native -> MCP -> A2A -> skill creation."""
        mcp_tool = MagicMock()
        mcp_tool.get_tools.return_value = [
            {"function": {"name": "mcp_test_tool_0"}}
        ]
        a2a_tool = MagicMock()
        a2a_tool.manager = MagicMock()
        flow = _make_flow(
            mcp_tool=mcp_tool,
            a2a_tool=a2a_tool,
            brainstorm_skill_tool=MagicMock(),
            create_skill_tool=MagicMock(),
        )

        tools = await flow._collect_all_tools()
        names = [t.name for t in tools]

        # Boundary indices for each category
        native_last = max(i for i, n in enumerate(names) if n in {
            "shell_execute", "file_read", "browser_navigate",
            "message_notify_user", "search_web",
        })
        mcp_indices = [i for i, n in enumerate(names) if n.startswith("mcp_")]
        assert mcp_indices, "Expected at least one MCP tool"
        mcp_first = min(mcp_indices)
        mcp_last = max(mcp_indices)
        a2a_first = min(i for i, n in enumerate(names) if n in {
            "get_remote_agent_cards", "call_remote_agent",
        })
        a2a_last = max(i for i, n in enumerate(names) if n in {
            "get_remote_agent_cards", "call_remote_agent",
        })
        skill_first = min(i for i, n in enumerate(names) if n in {
            "brainstorm_skill", "generate_skill", "install_skill",
        })

        # Verify strict ordering: native < MCP < A2A < skill creation
        assert native_last < mcp_first, (
            f"Native should come before MCP: native_last={native_last}, mcp_first={mcp_first}"
        )
        assert mcp_last < a2a_first, (
            f"MCP should come before A2A: mcp_last={mcp_last}, a2a_first={a2a_first}"
        )
        assert a2a_last < skill_first, (
            f"A2A should come before skill creation: a2a_last={a2a_last}, skill_first={skill_first}"
        )

    @pytest.mark.anyio
    async def test_wraps_aggregate_tools_without_double_wrapping(self, monkeypatch):
        """Aggregate wrapping covers non-native tools and keeps native wrappers idempotent."""
        flow = _make_flow()
        supervisor = object()
        flow._execution_supervisor = supervisor
        already_wrapped = SupervisorAwareToolWrapper(
            inner=_make_tool("native_wrapped"),
            supervisor=supervisor,
        )
        raw_mcp = _make_tool("mcp_raw")
        raw_memory = _make_tool("memory_search")

        monkeypatch.setattr(flow, "_collect_native_tools", lambda: [already_wrapped])
        monkeypatch.setattr(
            flow,
            "_collect_mcp_tools",
            AsyncMock(return_value=[raw_mcp]),
        )
        monkeypatch.setattr(flow, "_collect_a2a_tools", lambda: [])
        monkeypatch.setattr(flow, "_collect_skill_creation_tools", lambda: [])
        monkeypatch.setattr(flow, "_collect_memory_tools", lambda: [raw_memory])

        tools = await flow._collect_all_tools()

        assert tools[0] is already_wrapped
        assert all(isinstance(tool, SupervisorAwareToolWrapper) for tool in tools)
        assert [tool.name for tool in tools] == [
            "native_wrapped",
            "mcp_raw",
            "memory_search",
        ]
        assert flow._has_memory_tools is True


class TestCollectMemoryTools:
    """C6: _collect_memory_tools creates memory_search + memory_get."""

    def test_with_memory_deps_returns_two_tools(self) -> None:
        flow = _make_flow(
            memory_embedding_provider=AsyncMock(),
            memory_session_factory=MagicMock(),
            memory_repo_factory=MagicMock(),
        )
        tools = flow._collect_memory_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "memory_search" in names
        assert "memory_get" in names

    def test_without_deps_returns_empty(self) -> None:
        flow = _make_flow()
        tools = flow._collect_memory_tools()
        assert tools == []

    def test_partial_deps_returns_empty(self) -> None:
        flow = _make_flow(
            memory_embedding_provider=AsyncMock(),
            # missing session_factory and repo_factory
        )
        tools = flow._collect_memory_tools()
        assert tools == []

    def test_passes_config_to_create_memory_tools(self) -> None:
        """_collect_memory_tools should read half_life_days and mmr_lambda from memory_config."""
        mock_config = MagicMock()
        mock_config.memory.half_life_days = 60
        mock_config.memory.mmr_lambda = 0.3

        flow = _make_flow(
            agent_config=mock_config,
            memory_embedding_provider=AsyncMock(),
            memory_session_factory=MagicMock(),
            memory_repo_factory=MagicMock(),
        )

        with patch(
            "app.domain.services.tools.memory_tools.create_memory_tools",
        ) as mock_create:
            mock_create.return_value = [MagicMock(), MagicMock()]
            flow._collect_memory_tools()
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["half_life_days"] == 60
            assert call_kwargs["mmr_lambda"] == 0.3
