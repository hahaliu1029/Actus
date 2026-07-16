"""Tests for LangChain tool wrappers."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.tools import BaseTool as LCBaseTool
from app.domain.models.tool_result import ToolResult
from app.application.services.sandbox_accessors import EagerBrowserAccessor, EagerSandboxAccessor

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
            sandbox_accessor=EagerSandboxAccessor(mock_sandbox),
            browser_accessor=EagerBrowserAccessor(mock_browser),
            search_engine=mock_search_engine,
        )
        assert isinstance(tools, list)
        assert all(isinstance(t, LCBaseTool) for t in tools)

    def test_expected_tool_names(self, mock_sandbox, mock_browser, mock_search_engine):
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(
            sandbox_accessor=EagerSandboxAccessor(mock_sandbox),
            browser_accessor=EagerBrowserAccessor(mock_browser),
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
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(mock_sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        notify = next(t for t in tools if t.name == "message_notify_user")
        result = await notify.ainvoke({"text": "hello"})
        assert "Continue" in str(result)

    async def test_message_ask_user(self, mock_sandbox, mock_browser, mock_search_engine):
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(mock_sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        ask = next(t for t in tools if t.name == "message_ask_user")
        result = await ask.ainvoke({"text": "confirm?"})
        assert result is not None


class TestShellExecuteStatusHandling:
    """Regression tests for the shell_execute wrapper status+timeout handling.

    The original bug: when the sandbox returned status="running" with
    output=None (e.g. ``apt-get update`` exceeding the 5s sync wait), the
    wrapper collapsed the payload to the literal string ``"None"``, so the
    LLM thought the command had finished with empty output and proceeded.
    """

    def _build_tools(self, sandbox):
        from app.domain.services.tools.langchain_tools import create_native_tools
        return create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(AsyncMock()), search_engine=AsyncMock())

    async def test_running_status_returns_poll_instructions(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                message="",
                data={
                    "session_id": "default",
                    "command": "apt-get update && apt-get install -y unzip",
                    "status": "running",
                    "returncode": None,
                    "output": None,
                },
            )
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        result = await shell.ainvoke({"command": "apt-get update && apt-get install -y unzip"})

        assert isinstance(result, str)
        assert result != "None"
        assert "still running" in result
        assert "shell_wait_process" in result
        assert "default" in result

    async def test_completed_status_returns_output(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"status": "completed", "returncode": 0, "output": "hello world"},
            )
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        result = await shell.ainvoke({"command": "echo hello world"})
        assert result == "hello world"

    async def test_completed_empty_output_reports_exit_code(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"status": "completed", "returncode": 0, "output": None},
            )
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        result = await shell.ainvoke({"command": "touch /tmp/x"})
        assert "None" not in result
        assert "exit code 0" in result

    async def test_completed_nonzero_exit_includes_code(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"status": "completed", "returncode": 1, "output": "not found"},
            )
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        result = await shell.ainvoke({"command": "ls /nope"})
        assert "not found" in result
        assert "exit code: 1" in result

    async def test_wait_seconds_is_forwarded_to_sandbox(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"status": "completed", "returncode": 0, "output": "ok"},
            )
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        await shell.ainvoke({"command": "sleep 10", "wait_seconds": 60})

        sandbox.exec_command.assert_awaited_once()
        call_kwargs = sandbox.exec_command.await_args.kwargs
        assert call_kwargs["wait_seconds"] == 60
        assert call_kwargs["command"] == "sleep 10"

    async def test_legacy_mock_without_data_still_works(self, mock_browser, mock_search_engine):
        # Backward compat: tests that mock exec_command with only ``message`` (no data dict).
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(return_value=ToolResult(success=True, message="hello"))
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        result = await shell.ainvoke({"command": "echo hello"})
        assert result == "hello"

    async def test_sandbox_failure_returns_error_content(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(success=False, message="sandbox unreachable", data=None)
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        result = await shell.ainvoke({"command": "ls"})
        assert result == "sandbox unreachable"

    async def test_wait_seconds_is_clamped_to_max(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"status": "completed", "returncode": 0, "output": "ok"},
            )
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        # LLM passes an absurd value — wrapper must clamp below the httpx 600s timeout.
        await shell.ainvoke({"command": "sleep 1", "wait_seconds": 99999})

        forwarded = sandbox.exec_command.await_args.kwargs["wait_seconds"]
        assert forwarded is not None
        assert 1 <= forwarded <= 599  # strictly below httpx timeout

    async def test_wait_seconds_nonpositive_falls_back_to_sandbox_default(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"status": "completed", "returncode": 0, "output": "ok"},
            )
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        await shell.ainvoke({"command": "ls", "wait_seconds": 0})
        assert sandbox.exec_command.await_args.kwargs["wait_seconds"] is None

        sandbox.exec_command.reset_mock()
        sandbox.exec_command.return_value = ToolResult(
            success=True,
            data={"status": "completed", "returncode": 0, "output": "ok"},
        )
        await shell.ainvoke({"command": "ls", "wait_seconds": -5})
        assert sandbox.exec_command.await_args.kwargs["wait_seconds"] is None

    async def test_running_status_fetches_partial_output(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={
                    "session_id": "default",
                    "status": "running",
                    "returncode": None,
                    "output": None,
                },
            )
        )
        sandbox.read_shell_output = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"session_id": "default", "output": "Reading package lists... 35%"},
            )
        )
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        result = await shell.ainvoke({"command": "apt-get update"})

        sandbox.read_shell_output.assert_awaited_once()
        assert "still running" in result
        assert "Reading package lists... 35%" in result
        assert "partial output" in result

    async def test_running_status_tolerates_peek_failure(self, mock_browser, mock_search_engine):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"session_id": "default", "status": "running", "output": None},
            )
        )
        sandbox.read_shell_output = AsyncMock(side_effect=RuntimeError("transient network blip"))
        from app.domain.services.tools.langchain_tools import create_native_tools
        tools = create_native_tools(sandbox_accessor=EagerSandboxAccessor(sandbox), browser_accessor=EagerBrowserAccessor(mock_browser), search_engine=mock_search_engine)
        shell = next(t for t in tools if t.name == "shell_execute")

        # Peek failure must not break the running-status message.
        result = await shell.ainvoke({"command": "apt-get update"})
        assert "still running" in result
        assert "None" not in result
