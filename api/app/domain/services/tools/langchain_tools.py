"""LangChain tool wrappers for Actus native tools.

Each function wraps the corresponding sandbox/browser/search method and returns
a string result (LangChain convention).

Usage:
    tools = create_native_tools(sandbox=sandbox, browser=browser, search_engine=engine)
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from langchain_core.tools import StructuredTool, tool as lc_tool

from app.domain.external.browser import Browser
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine


# --------------------------------------------------------------------------- #
# Message tools
# --------------------------------------------------------------------------- #


def _make_message_tools() -> list[StructuredTool]:
    """Create message tools (no external dependency needed)."""

    @lc_tool
    async def message_notify_user(text: str) -> str:
        """Send a notification to the user without waiting for a reply. Use for progress updates, confirmations, or status reports."""
        return "Continue"

    @lc_tool
    async def message_ask_user(
        text: str,
        attachments: Optional[Union[str, List[str]]] = None,
        suggest_user_takeover: Optional[Literal["none", "shell", "browser"]] = None,
    ) -> str:
        """Ask the user a question and wait for their reply. Use for clarification, confirmation, or requesting input."""
        return "WAITING_FOR_USER"

    return [message_notify_user, message_ask_user]


# --------------------------------------------------------------------------- #
# File tools
# --------------------------------------------------------------------------- #


def _make_file_tools(sandbox: Sandbox) -> list[StructuredTool]:
    """Create file tools that delegate to sandbox."""

    @lc_tool
    async def file_read(
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False,
        max_length: int = 2000,
    ) -> str:
        """Read file content from the sandbox filesystem."""
        result = await sandbox.read_file(filepath, start_line=start_line, end_line=end_line, sudo=sudo, max_length=max_length)
        return str(result)

    @lc_tool
    async def file_write(
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> str:
        """Write content to a file in the sandbox filesystem."""
        result = await sandbox.write_file(
            filepath, content, append=append,
            leading_newline=leading_newline, trailing_newline=trailing_newline, sudo=sudo,
        )
        return str(result) if result else "File written successfully"

    @lc_tool
    async def file_str_replace(filepath: str, old_str: str, new_str: str, sudo: bool = False) -> str:
        """Replace a string in a file."""
        result = await sandbox.replace_in_file(filepath, old_str, new_str, sudo=sudo)
        return str(result) if result else "Replacement done"

    @lc_tool
    async def file_find_in_content(filepath: str, regex: str, sudo: bool = False) -> str:
        """Search file content using regex."""
        result = await sandbox.search_in_file(filepath, regex, sudo=sudo)
        return str(result)

    @lc_tool
    async def file_find_by_name(dir_path: str, glob_pattern: str) -> str:
        """Find files by name pattern."""
        result = await sandbox.find_files(dir_path, glob_pattern)
        return str(result)

    @lc_tool
    async def file_list(dir_path: str) -> str:
        """List directory contents."""
        result = await sandbox.list_files(dir_path)
        return str(result)

    return [file_read, file_write, file_str_replace, file_find_in_content, file_find_by_name, file_list]


# --------------------------------------------------------------------------- #
# Shell tools
# --------------------------------------------------------------------------- #


def _make_shell_tools(sandbox: Sandbox) -> list[StructuredTool]:
    """Create shell tools that delegate to sandbox."""

    @lc_tool
    async def shell_execute(command: str, session_id: str = "default", exec_dir: str = "") -> str:
        """Execute a shell command in the sandbox."""
        result = await sandbox.exec_command(session_id=session_id, exec_dir=exec_dir, command=command)
        return str(result)

    shell_execute.metadata = {"require_confirmation": True}

    @lc_tool
    async def shell_read_output(session_id: str = "default") -> str:
        """Read the latest output from a shell session."""
        result = await sandbox.read_shell_output(session_id=session_id)
        return str(result)

    @lc_tool
    async def shell_wait_process(session_id: str = "default", seconds: int = 5) -> str:
        """Wait for a running process to produce output."""
        result = await sandbox.wait_process(session_id=session_id, seconds=seconds)
        return str(result)

    @lc_tool
    async def shell_write_input(input_text: str, session_id: str = "default", press_enter: bool = True) -> str:
        """Write input to a running shell process."""
        result = await sandbox.write_shell_input(session_id=session_id, input_text=input_text, press_enter=press_enter)
        return str(result)

    @lc_tool
    async def shell_kill_process(session_id: str = "default") -> str:
        """Kill a running process in a shell session."""
        result = await sandbox.kill_process(session_id=session_id)
        return str(result)

    return [shell_execute, shell_read_output, shell_wait_process, shell_write_input, shell_kill_process]


# --------------------------------------------------------------------------- #
# Browser tools
# --------------------------------------------------------------------------- #


def _make_browser_tools(browser: Browser) -> list[StructuredTool]:
    """Create browser tools that delegate to Browser."""

    @lc_tool
    async def browser_view() -> str:
        """Get a snapshot of the current browser page content and screenshot."""
        result = await browser.view_page()
        return str(result)

    @lc_tool
    async def browser_navigate(url: str) -> str:
        """Navigate the browser to a URL."""
        result = await browser.navigate(url)
        return str(result)

    @lc_tool
    async def browser_click(
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> str:
        """Click an element on the page by index or coordinates."""
        result = await browser.click(index=index, coordinate_x=coordinate_x, coordinate_y=coordinate_y)
        return str(result)

    @lc_tool
    async def browser_input(
        text: str,
        press_enter: bool = True,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> str:
        """Type text into an input field."""
        result = await browser.input(text, press_enter=press_enter, index=index, coordinate_x=coordinate_x, coordinate_y=coordinate_y)
        return str(result)

    @lc_tool
    async def browser_move_mouse(coordinate_x: float, coordinate_y: float) -> str:
        """Move the mouse cursor to specific coordinates."""
        result = await browser.move_mouse(coordinate_x=coordinate_x, coordinate_y=coordinate_y)
        return str(result)

    @lc_tool
    async def browser_press_key(key: str) -> str:
        """Press a keyboard key."""
        result = await browser.press_key(key)
        return str(result)

    @lc_tool
    async def browser_select_option(index: int, option: int) -> str:
        """Select an option from a dropdown."""
        result = await browser.select_option(index=index, option=option)
        return str(result)

    @lc_tool
    async def browser_scroll_up(to_top: bool = False) -> str:
        """Scroll the page up."""
        result = await browser.scroll_up(to_top=to_top)
        return str(result)

    @lc_tool
    async def browser_scroll_down(to_bottom: bool = False) -> str:
        """Scroll the page down."""
        result = await browser.scroll_down(to_down=to_bottom)
        return str(result)

    @lc_tool
    async def browser_console_exec(javascript: str) -> str:
        """Execute JavaScript in the browser console."""
        result = await browser.console_exec(javascript)
        return str(result)

    @lc_tool
    async def browser_console_view(max_lines: int = 50) -> str:
        """View the browser console output."""
        result = await browser.console_view(max_lines=max_lines)
        return str(result)

    @lc_tool
    async def browser_restart(url: str = "") -> str:
        """Restart the browser, optionally navigating to a URL."""
        result = await browser.restart(url=url)
        return str(result)

    return [
        browser_view, browser_navigate, browser_click, browser_input,
        browser_move_mouse, browser_press_key, browser_select_option,
        browser_scroll_up, browser_scroll_down, browser_console_exec,
        browser_console_view, browser_restart,
    ]


# --------------------------------------------------------------------------- #
# Search tools
# --------------------------------------------------------------------------- #


def _make_search_tools(search_engine: SearchEngine) -> list[StructuredTool]:
    """Create search tools."""

    @lc_tool
    async def search_web(query: str, date_range: Optional[str] = None) -> str:
        """Search the web for information."""
        result = await search_engine.invoke(query, date_range=date_range)
        return str(result)

    return [search_web]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def create_native_tools(
    sandbox: Sandbox,
    browser: Browser,
    search_engine: SearchEngine,
) -> list[StructuredTool]:
    """Create all native LangChain tools.

    Returns a flat list of tools ready to be bound to an LLM or added to a ToolNode.
    """
    tools: list[StructuredTool] = []
    tools.extend(_make_message_tools())
    tools.extend(_make_file_tools(sandbox))
    tools.extend(_make_shell_tools(sandbox))
    tools.extend(_make_browser_tools(browser))
    tools.extend(_make_search_tools(search_engine))
    return tools
