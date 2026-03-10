"""Tests for LangChain tool wrappers."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.tools import BaseTool as LCBaseTool
from app.domain.models.tool_result import ToolResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_sandbox():
    sandbox = AsyncMock()
    sandbox.read_file = AsyncMock(return_value=ToolResult(success=True, message="file content"))
    sandbox.write_file = AsyncMock(return_value=ToolResult(success=True, message="OK"))
    sandbox.exec_command = AsyncMock(return_value=ToolResult(success=True, message="hello"))
    sandbox.read_shell_output = AsyncMock(return_value=ToolResult(success=True, message="output"))
    sandbox.wait_process = AsyncMock(return_value=ToolResult(success=True, message="done"))
    sandbox.write_shell_input = AsyncMock(return_value=ToolResult(success=True, message="OK"))
    sandbox.kill_process = AsyncMock(return_value=ToolResult(success=True, message="killed"))
    sandbox.replace_in_file = AsyncMock(return_value=ToolResult(success=True, message="replaced"))
    sandbox.search_in_file = AsyncMock(return_value=ToolResult(success=True, message="found"))
    sandbox.find_files = AsyncMock(return_value=ToolResult(success=True, message="file.py"))
    sandbox.list_files = AsyncMock(return_value=ToolResult(success=True, message="dir listing"))
    return sandbox


@pytest.fixture
def mock_browser():
    browser = AsyncMock()
    browser.view_page = AsyncMock(return_value=ToolResult(success=True, message="page content"))
    browser.navigate = AsyncMock(return_value=ToolResult(success=True, message="navigated"))
    browser.click = AsyncMock(return_value=ToolResult(success=True, message="clicked"))
    browser.input = AsyncMock(return_value=ToolResult(success=True, message="typed"))
    browser.move_mouse = AsyncMock(return_value=ToolResult(success=True, message="moved"))
    browser.press_key = AsyncMock(return_value=ToolResult(success=True, message="pressed"))
    browser.select_option = AsyncMock(return_value=ToolResult(success=True, message="selected"))
    browser.scroll_up = AsyncMock(return_value=ToolResult(success=True, message="scrolled"))
    browser.scroll_down = AsyncMock(return_value=ToolResult(success=True, message="scrolled"))
    browser.console_exec = AsyncMock(return_value=ToolResult(success=True, message="result"))
    browser.console_view = AsyncMock(return_value=ToolResult(success=True, message="logs"))
    browser.restart = AsyncMock(return_value=ToolResult(success=True, message="restarted"))
    return browser


@pytest.fixture
def mock_search_engine():
    engine = AsyncMock()
    engine.invoke = AsyncMock(return_value=ToolResult(success=True, message="results"))
    return engine


class TestCreateNativeTools:
    def test_returns_list_of_langchain_tools(self, mock_sandbox, mock_browser, mock_search_engine):
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(
            sandbox=mock_sandbox,
            browser=mock_browser,
            search_engine=mock_search_engine,
        )
        assert isinstance(tools, list)
        assert all(isinstance(t, LCBaseTool) for t in tools)

    def test_expected_tool_names(self, mock_sandbox, mock_browser, mock_search_engine):
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(
            sandbox=mock_sandbox,
            browser=mock_browser,
            search_engine=mock_search_engine,
        )
        names = {t.name for t in tools}
        assert "message_notify_user" in names
        assert "message_ask_user" in names
        assert "file_read" in names
        assert "shell_execute" in names
        assert "browser_view" in names
        assert "search_web" in names


class TestMessageTools:
    async def test_message_notify_user(self, mock_sandbox, mock_browser, mock_search_engine):
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox=mock_sandbox, browser=mock_browser, search_engine=mock_search_engine)
        notify = next(t for t in tools if t.name == "message_notify_user")
        result = await notify.ainvoke({"text": "hello"})
        assert "Continue" in str(result)

    async def test_message_ask_user(self, mock_sandbox, mock_browser, mock_search_engine):
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox=mock_sandbox, browser=mock_browser, search_engine=mock_search_engine)
        ask = next(t for t in tools if t.name == "message_ask_user")
        result = await ask.ainvoke({"text": "confirm?"})
        assert result is not None
