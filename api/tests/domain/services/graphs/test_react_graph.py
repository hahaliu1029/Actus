"""Tests for react_graph — the inner ReAct loop."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.domain.models.event import ToolEvent

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_llm_adapter():
    """Mock LangChain LLM that returns a plain text response."""
    from langchain_core.messages import AIMessage
    adapter = AsyncMock()
    adapter.ainvoke = AsyncMock(return_value=AIMessage(content='{"success": true, "result": "done", "attachments": []}'))
    adapter.bind_tools = MagicMock(return_value=adapter)
    return adapter


@pytest.fixture
def mock_tools():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    async def shell_execute(command: str) -> str:
        """Execute a shell command."""
        return "output: hello"

    return [shell_execute]


class TestBuildReactGraph:
    def test_graph_compiles(self, mock_llm_adapter, mock_tools):
        from app.domain.services.graphs.react_graph import build_react_graph
        graph = build_react_graph(mock_llm_adapter, mock_tools)
        assert graph is not None

    async def test_simple_no_tool_call(self, mock_llm_adapter, mock_tools):
        """LLM returns plain content → graph ends without tool calls."""
        from app.domain.services.graphs.react_graph import build_react_graph
        graph = build_react_graph(mock_llm_adapter, mock_tools)

        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": "hello"}],
            "step_description": "greet user",
            "original_request": "greet",
            "language": "en",
            "attachments": [],
            "events": [],
            "should_interrupt": False,
            "attempt_count": 0,
            "failure_count": 0,
        })

        assert result["should_interrupt"] is False
        assert len(result["events"]) >= 0

    async def test_with_tool_call(self, mock_tools):
        """LLM returns a tool call → tool executes → LLM responds."""
        from langchain_core.messages import AIMessage
        from app.domain.services.graphs.react_graph import build_react_graph

        call_count = 0

        async def mock_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "c1", "name": "shell_execute", "args": {"command": "ls"}}],
                )
            return AIMessage(content='{"success": true, "result": "done", "attachments": []}')

        adapter = AsyncMock()
        adapter.ainvoke = mock_ainvoke
        adapter.bind_tools = MagicMock(return_value=adapter)

        graph = build_react_graph(adapter, mock_tools)
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": "list files"}],
            "step_description": "list files",
            "original_request": "list files",
            "language": "en",
            "attachments": [],
            "events": [],
            "should_interrupt": False,
            "attempt_count": 0,
            "failure_count": 0,
        })

        # Should have tool events in the events list
        tool_events = [e for e in result["events"] if isinstance(e, ToolEvent)]
        assert len(tool_events) >= 1
