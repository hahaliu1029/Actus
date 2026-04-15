"""Tests for react_graph — the inner ReAct loop."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import ToolMessage
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
            "image_content_blocks": [],
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
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
            "image_content_blocks": [],
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
            "attempt_count": 0,
            "failure_count": 0,
        })

        # Should have tool events in the events list
        tool_events = [e for e in result["events"] if isinstance(e, ToolEvent)]
        assert len(tool_events) >= 1

    async def test_tool_failure_marks_error(self):
        """When a tool raises an exception, the tool message should be prefixed
        with [TOOL_ERROR] and the ToolEvent should have success=False."""
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph
        from app.domain.services.tools.tool_source_resolver import (
            annotate_and_register_tool_source,
        )

        @lc_tool
        async def failing_tool(query: str) -> str:
            """A tool that always fails."""
            raise RuntimeError("connection refused")

        annotate_and_register_tool_source(
            failing_tool, source="native", category="search"
        )

        call_count = 0

        async def mock_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "c1", "name": "failing_tool", "args": {"query": "test"}}],
                )
            return AIMessage(content='{"success": false, "result": "tool failed", "attachments": []}')

        adapter = AsyncMock()
        adapter.ainvoke = mock_ainvoke
        adapter.bind_tools = MagicMock(return_value=adapter)

        graph = build_react_graph(adapter, [failing_tool])
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": "search something"}],
            "step_description": "search",
            "original_request": "search",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
            "attempt_count": 0,
            "failure_count": 0,
        })

        # Verify ToolEvent has success=False
        called_events = [
            e for e in result["events"]
            if isinstance(e, ToolEvent) and e.function_result is not None
        ]
        assert any(not e.function_result.success for e in called_events), \
            "Expected at least one CALLED ToolEvent with success=False"

        # Verify tool message is prefixed with [TOOL_ERROR]
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert any("[TOOL_ERROR]" in m.content for m in tool_msgs), \
            "Expected tool message to contain [TOOL_ERROR] prefix"

    async def test_tool_result_success_false_detected(self):
        """When a tool raises an exception (from _unwrap on ToolResult.success=False),
        the system should detect it and mark the ToolEvent accordingly."""
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph

        @lc_tool
        async def search_web(query: str) -> str:
            """Search the web."""
            # Simulate what happens when _unwrap() encounters ToolResult(success=False)
            raise RuntimeError("Bing搜索出错: CAPTCHA blocked")

        call_count = 0

        async def mock_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "c1", "name": "search_web", "args": {"query": "AI news"}}],
                )
            return AIMessage(content='{"success": false, "result": "search failed", "attachments": []}')

        adapter = AsyncMock()
        adapter.ainvoke = mock_ainvoke
        adapter.bind_tools = MagicMock(return_value=adapter)

        graph = build_react_graph(adapter, [search_web])
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": "search AI news"}],
            "step_description": "search",
            "original_request": "search",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
            "attempt_count": 0,
            "failure_count": 0,
        })

        # ToolEvent should have success=False
        called_events = [
            e for e in result["events"]
            if isinstance(e, ToolEvent) and e.function_result is not None
        ]
        assert any(not e.function_result.success for e in called_events)

        # Tool message should be prefixed with [TOOL_ERROR]
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert any("[TOOL_ERROR]" in m.content for m in tool_msgs)


class TestToolNodeTruncation:
    """Verify Tier 1 truncation: tool results > tool_result_max_chars are truncated."""

    async def test_tool_node_truncates_large_result(self):
        """Tool returning > max_chars -> ToolMessage.content truncated with head+tail."""
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph
        from app.domain.services.tools.tool_source_resolver import (
            annotate_and_register_tool_source,
        )

        large_output = "X" * 500

        @lc_tool
        async def big_tool(query: str) -> str:
            """Returns large output."""
            return large_output

        annotate_and_register_tool_source(
            big_tool, source="native", category="search"
        )

        call_count = 0

        async def mock_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "c1", "name": "big_tool", "args": {"query": "test"}}],
                )
            return AIMessage(content="done")

        adapter = AsyncMock()
        adapter.ainvoke = mock_ainvoke
        adapter.bind_tools = MagicMock(return_value=adapter)

        graph = build_react_graph(adapter, [big_tool], tool_result_max_chars=100)
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": "run big tool"}],
            "step_description": "test",
            "original_request": "test",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
            "attempt_count": 0,
            "failure_count": 0,
        })

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "big_tool"]
        assert len(tool_msgs) >= 1
        assert len(tool_msgs[0].content) < 500
        assert "已截断" in tool_msgs[0].content

    async def test_tool_node_truncates_tool_event(self):
        """ToolEvent.function_result.message should also be truncated."""
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph
        from app.domain.services.tools.tool_source_resolver import (
            annotate_and_register_tool_source,
        )

        @lc_tool
        async def big_tool(query: str) -> str:
            """Returns large output."""
            return "Y" * 500

        annotate_and_register_tool_source(
            big_tool, source="native", category="search"
        )

        call_count = 0

        async def mock_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "c1", "name": "big_tool", "args": {"query": "test"}}],
                )
            return AIMessage(content="done")

        adapter = AsyncMock()
        adapter.ainvoke = mock_ainvoke
        adapter.bind_tools = MagicMock(return_value=adapter)

        graph = build_react_graph(adapter, [big_tool], tool_result_max_chars=100)
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": "run"}],
            "step_description": "test",
            "original_request": "test",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
            "attempt_count": 0,
            "failure_count": 0,
        })

        called_events = [
            e for e in result["events"]
            if isinstance(e, ToolEvent) and e.function_result is not None
        ]
        assert len(called_events) >= 1
        event_msg = called_events[0].function_result.message
        assert len(event_msg) < 500
        assert "已截断" in event_msg

    async def test_tool_result_max_chars_parameter(self):
        """Passing a small tool_result_max_chars triggers truncation at that threshold."""
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph
        from app.domain.services.tools.tool_source_resolver import (
            annotate_and_register_tool_source,
        )

        @lc_tool
        async def medium_tool(query: str) -> str:
            """Returns medium output."""
            return "Z" * 200

        annotate_and_register_tool_source(
            medium_tool, source="native", category="search"
        )

        call_count = 0

        async def mock_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "c1", "name": "medium_tool", "args": {"query": "test"}}],
                )
            return AIMessage(content="done")

        adapter = AsyncMock()
        adapter.ainvoke = mock_ainvoke
        adapter.bind_tools = MagicMock(return_value=adapter)

        graph = build_react_graph(adapter, [medium_tool], tool_result_max_chars=50)
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": "run"}],
            "step_description": "test",
            "original_request": "test",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
            "attempt_count": 0,
            "failure_count": 0,
        })

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "medium_tool"]
        assert len(tool_msgs) >= 1
        assert len(tool_msgs[0].content) < 200
        assert "已截断" in tool_msgs[0].content
