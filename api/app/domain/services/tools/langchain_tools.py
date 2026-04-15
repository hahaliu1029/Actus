"""LangChain tool wrappers for Actus native tools.

Each function wraps the corresponding sandbox/browser/search method and returns
a string result (LangChain convention).

Usage:
    tools = create_native_tools(sandbox=sandbox, browser=browser, search_engine=engine)
"""

from __future__ import annotations

import shlex
from typing import List, Literal, Optional, Union

from langchain_core.tools import StructuredTool, tool as lc_tool

from app.domain.external.browser import Browser
from app.domain.external.file_processor import FileProcessorLookup, FileProcessResult
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)


def _unwrap(result: object) -> str:
    """Extract business payload from a ToolResult, raising on failure.

    ToolResult has {success, message, data}. The model should see `data`
    (the actual tool output), not the Pydantic repr of the wrapper.

    If ``result`` has ``success=False``, raise so that the caller (ToolNode or
    react_graph tool_node) can handle the error structurally rather than relying
    on string pattern matching.
    """
    if hasattr(result, "success") and not result.success:
        raise RuntimeError(getattr(result, "message", None) or str(result))
    # Extract .data (the actual payload); fall back to .message then str()
    if hasattr(result, "data") and result.data is not None:
        data = result.data
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            # Common sandbox pattern: {"returncode": 0, "output": "..."}
            if "output" in data:
                # Guard: bare ``str(None)`` would leak the literal "None" to the
                # LLM and hide empty-output success cases.
                output = data["output"]
                return output if isinstance(output, str) else ("" if output is None else str(output))
            import json
            return json.dumps(data, ensure_ascii=False)
        return str(data)
    if hasattr(result, "message") and result.message:
        return result.message
    return str(result)


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
        """Ask the user a question and wait for their reply. Use for clarification, confirmation, or requesting input.

        NOTE: The system may return SOFT_HINT if it determines the agent should
        try to solve autonomously first. Only call again if user input is truly
        required.
        """
        # Actual SOFT_HINT / interrupt logic is handled by react_graph's tool_node.
        # This is the fallback return value.
        return "WAITING_FOR_USER"

    tools = [message_notify_user, message_ask_user]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="message")
    return tools


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
        return _unwrap(result)

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
        return _unwrap(result) if result else "File written successfully"

    file_write.metadata = {"risk_level": "medium"}

    @lc_tool
    async def file_str_replace(filepath: str, old_str: str, new_str: str, sudo: bool = False) -> str:
        """Replace a string in a file."""
        result = await sandbox.replace_in_file(filepath, old_str, new_str, sudo=sudo)
        return _unwrap(result) if result else "Replacement done"

    file_str_replace.metadata = {"risk_level": "medium"}

    @lc_tool
    async def file_find_in_content(filepath: str, regex: str, sudo: bool = False) -> str:
        """Search file content using regex."""
        result = await sandbox.search_in_file(filepath, regex, sudo=sudo)
        return _unwrap(result)

    @lc_tool
    async def file_find_by_name(dir_path: str, glob_pattern: str) -> str:
        """Find files by name pattern."""
        result = await sandbox.find_files(dir_path, glob_pattern)
        return _unwrap(result)

    @lc_tool
    async def file_list(dir_path: str) -> str:
        """List directory contents."""
        result = await sandbox.list_files(dir_path)
        return _unwrap(result)

    tools = [file_read, file_write, file_str_replace, file_find_in_content, file_find_by_name, file_list]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="file")
    return tools


# --------------------------------------------------------------------------- #
# Shell tools
# --------------------------------------------------------------------------- #


def _make_shell_tools(sandbox: Sandbox) -> list[StructuredTool]:
    """Create shell tools that delegate to sandbox."""

    _DEFAULT_WAIT_SECONDS = 5  # Matches sandbox service default, kept in sync intentionally.
    # Upper bound on the sync wait window. Stays under the httpx client timeout
    # (``DockerSandbox`` uses ``timeout=600``) so an LLM-supplied value can never
    # cause a ReadTimeout that would orphan the background command.
    _MAX_WAIT_SECONDS = 580

    @lc_tool
    async def shell_execute(
        command: str,
        session_id: str = "default",
        exec_dir: str = "",
        wait_seconds: Optional[int] = None,
    ) -> str:
        """Execute a shell command in the sandbox.

        The sandbox synchronously waits up to ``wait_seconds`` (default 5s) for
        the command to finish. If the command is still running when the wait
        window elapses, this returns a message describing how to poll for
        completion via ``shell_wait_process`` or ``shell_read_output`` — the
        command keeps running in the background on ``session_id``.

        For long-running commands (package installs, downloads, builds, etc.)
        pass a larger ``wait_seconds`` so the tool blocks until completion
        instead of returning early. Values are clamped to a safe ceiling that
        stays below the underlying HTTP client timeout.
        """
        # Clamp LLM-supplied wait_seconds so it cannot exceed the httpx client
        # timeout (which would orphan the command) or slip through as a non-positive.
        clamped_wait: Optional[int] = None
        if wait_seconds is not None and wait_seconds > 0:
            clamped_wait = min(wait_seconds, _MAX_WAIT_SECONDS)

        result = await sandbox.exec_command(
            session_id=session_id,
            exec_dir=exec_dir,
            command=command,
            wait_seconds=clamped_wait,
        )
        if hasattr(result, "success") and not result.success:
            raise RuntimeError(getattr(result, "message", None) or str(result))

        data = getattr(result, "data", None)
        # Legacy / mocked sandbox that doesn't return structured data — fall
        # through to the generic unwrap path.
        if not isinstance(data, dict):
            return _unwrap(result)

        status = data.get("status")
        output = data.get("output")
        if not isinstance(output, str):
            output = "" if output is None else str(output)
        returncode = data.get("returncode")

        if status == "running":
            effective_wait = clamped_wait or _DEFAULT_WAIT_SECONDS
            # Sandbox only populates ``output`` on completion; fetch whatever
            # has buffered so far via ``read_shell_output`` (best-effort) so the
            # LLM can see progress without an extra poll round-trip.
            partial_text = ""
            try:
                peek = await sandbox.read_shell_output(session_id=session_id)
                peek_data = getattr(peek, "data", None)
                if isinstance(peek_data, dict):
                    raw = peek_data.get("output")
                    if isinstance(raw, str) and raw.strip():
                        partial_text = f"\n--- partial output ---\n{raw}"
            except Exception:
                # Best-effort: the LLM can still call shell_read_output explicitly.
                pass
            return (
                f"[shell_execute] Command is still running on session '{session_id}' "
                f"after the {effective_wait}s sync wait window. The process keeps running "
                f"in the background.\n"
                f"Next step: call shell_wait_process(session_id='{session_id}', seconds=N) "
                f"to wait longer, or shell_read_output(session_id='{session_id}') to peek "
                f"current output. For long operations (apt/pip install, downloads, builds) "
                f"you can also re-invoke shell_execute with a larger wait_seconds."
                f"{partial_text}"
            )

        # status == "completed" (or unknown/legacy) — surface output + returncode.
        if output:
            if returncode is not None and returncode != 0:
                return f"{output}\n[shell_execute] exit code: {returncode}"
            return output
        if returncode == 0:
            return "[shell_execute] Command completed successfully with no output (exit code 0)."
        if returncode is not None:
            return f"[shell_execute] Command completed with no output (exit code {returncode})."
        return "[shell_execute] Command completed with no output."

    shell_execute.metadata = {"risk_level": "high"}

    @lc_tool
    async def shell_read_output(session_id: str = "default") -> str:
        """Read the latest output from a shell session."""
        result = await sandbox.read_shell_output(session_id=session_id)
        return _unwrap(result)

    @lc_tool
    async def shell_wait_process(session_id: str = "default", seconds: int = 5) -> str:
        """Wait for a running process to produce output."""
        result = await sandbox.wait_process(session_id=session_id, seconds=seconds)
        return _unwrap(result)

    @lc_tool
    async def shell_write_input(input_text: str, session_id: str = "default", press_enter: bool = True) -> str:
        """Write input to a running shell process."""
        result = await sandbox.write_shell_input(session_id=session_id, input_text=input_text, press_enter=press_enter)
        return _unwrap(result)

    @lc_tool
    async def shell_kill_process(session_id: str = "default") -> str:
        """Kill a running process in a shell session."""
        result = await sandbox.kill_process(session_id=session_id)
        return _unwrap(result)

    tools = [shell_execute, shell_read_output, shell_wait_process, shell_write_input, shell_kill_process]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="shell")
    return tools


# --------------------------------------------------------------------------- #
# Browser tools
# --------------------------------------------------------------------------- #


def _make_browser_tools(browser: Browser) -> list[StructuredTool]:
    """Create browser tools that delegate to Browser."""

    @lc_tool
    async def browser_view() -> str:
        """Get a snapshot of the current browser page content and screenshot."""
        result = await browser.view_page()
        return _unwrap(result)

    @lc_tool
    async def browser_navigate(url: str) -> str:
        """Navigate the browser to a URL."""
        result = await browser.navigate(url)
        return _unwrap(result)

    @lc_tool
    async def browser_click(
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> str:
        """Click an element on the page by index or coordinates."""
        result = await browser.click(index=index, coordinate_x=coordinate_x, coordinate_y=coordinate_y)
        return _unwrap(result)

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
        return _unwrap(result)

    @lc_tool
    async def browser_move_mouse(coordinate_x: float, coordinate_y: float) -> str:
        """Move the mouse cursor to specific coordinates."""
        result = await browser.move_mouse(coordinate_x=coordinate_x, coordinate_y=coordinate_y)
        return _unwrap(result)

    @lc_tool
    async def browser_press_key(key: str) -> str:
        """Press a keyboard key."""
        result = await browser.press_key(key)
        return _unwrap(result)

    @lc_tool
    async def browser_select_option(index: int, option: int) -> str:
        """Select an option from a dropdown."""
        result = await browser.select_option(index=index, option=option)
        return _unwrap(result)

    @lc_tool
    async def browser_scroll_up(to_top: bool = False) -> str:
        """Scroll the page up."""
        result = await browser.scroll_up(to_top=to_top)
        return _unwrap(result)

    @lc_tool
    async def browser_scroll_down(to_bottom: bool = False) -> str:
        """Scroll the page down."""
        result = await browser.scroll_down(to_down=to_bottom)
        return _unwrap(result)

    @lc_tool
    async def browser_console_exec(javascript: str) -> str:
        """Execute JavaScript in the browser console."""
        result = await browser.console_exec(javascript)
        return _unwrap(result)

    browser_console_exec.metadata = {"risk_level": "high"}

    @lc_tool
    async def browser_console_view(max_lines: int = 50) -> str:
        """View the browser console output."""
        result = await browser.console_view(max_lines=max_lines)
        return _unwrap(result)

    @lc_tool
    async def browser_restart(url: str = "") -> str:
        """Restart the browser, optionally navigating to a URL."""
        result = await browser.restart(url=url)
        return _unwrap(result)

    tools = [
        browser_view, browser_navigate, browser_click, browser_input,
        browser_move_mouse, browser_press_key, browser_select_option,
        browser_scroll_up, browser_scroll_down, browser_console_exec,
        browser_console_view, browser_restart,
    ]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="browser")
    return tools


# --------------------------------------------------------------------------- #
# Search tools
# --------------------------------------------------------------------------- #


def _make_search_tools(search_engine: SearchEngine) -> list[StructuredTool]:
    """Create search tools."""

    @lc_tool
    async def search_web(query: str, date_range: Optional[str] = None) -> str:
        """Search the web for information."""
        result = await search_engine.invoke(query, date_range=date_range)
        return _unwrap(result)

    tools = [search_web]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="search")
    return tools


# --------------------------------------------------------------------------- #
# File view tools (multimodal file understanding)
# --------------------------------------------------------------------------- #

# Extension → MIME fallback (when `file --mime-type` fails or returns generic type)
_EXT_MIME_MAP = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".flac": "audio/flac", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".webm": "video/webm", ".avi": "video/x-msvideo",
    ".mov": "video/quicktime", ".mkv": "video/x-matroska",
}


def _make_file_view_tools(
    sandbox: Sandbox,
    processor_lookup: FileProcessorLookup,
    supports_vision: bool,
    supports_pdf_input: bool = False,
) -> list[StructuredTool]:
    """Create file_view tool for multimodal file understanding."""

    @lc_tool
    async def file_view(filepath: str) -> FileProcessResult | str:
        """View and understand a file's content. Use this for images, PDFs,
        audio, and video files instead of file_read.
        Returns the file content in a format the model can understand."""

        # 1. Detect MIME type (sandbox `file` command + extension fallback)
        mime_result = await sandbox.exec_command(
            "default", "", f"file --mime-type -b {shlex.quote(filepath)}"
        )

        # Check for execution failure (path not found, permission denied, etc.)
        if hasattr(mime_result, "success") and not mime_result.success:
            raise RuntimeError(f"Cannot access file: {mime_result}")

        # Extract the actual command output from ToolResult.data
        # ToolResult.data is a dict with keys: returncode, output, etc.
        mime_type = ""
        if hasattr(mime_result, "data") and isinstance(mime_result.data, dict):
            returncode = mime_result.data.get("returncode", -1)
            output = mime_result.data.get("output") or ""
            if returncode == 0:
                mime_type = output.strip()
            elif returncode in {126, 127}:
                # `file` command not found / not executable — fall through to extension
                pass
            else:
                # Real command error (file not found, permission denied, etc.)
                raise RuntimeError(
                    f"Cannot detect file type: {output.strip() or mime_result}"
                )
        else:
            mime_type = str(mime_result).strip()

        # Fallback to extension when `file` unavailable or returns generic/empty type
        if not mime_type or mime_type == "application/octet-stream":
            ext = "." + filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
            mime_type = _EXT_MIME_MAP.get(ext, mime_type or "application/octet-stream")

        # 2. Find processor
        processor = processor_lookup.get_processor(mime_type)
        if processor is None:
            return f"Unsupported file type: {mime_type}. Use file_read for text files."

        # 3. Process file — tool_node splits: text → ToolMessage, image_blocks → HumanMessage
        filename = filepath.rsplit("/", 1)[-1]
        return await processor.process(
            sandbox_path=filepath,
            filename=filename,
            mime_type=mime_type,
            supports_vision=supports_vision,
            supports_pdf_input=supports_pdf_input,
        )

    tools = [file_view]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="file")
    return tools


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def create_native_tools(
    sandbox: Sandbox,
    browser: Browser,
    search_engine: SearchEngine,
    processor_lookup: FileProcessorLookup | None = None,
    supports_vision: bool = True,
    supports_pdf_input: bool = False,
) -> list[StructuredTool]:
    """Create all native LangChain tools.

    Returns a flat list of tools ready to be bound to an LLM or added to a ToolNode.
    """
    tools: list[StructuredTool] = []
    tools.extend(_make_message_tools())
    tools.extend(_make_file_tools(sandbox))
    if processor_lookup:
        tools.extend(_make_file_view_tools(sandbox, processor_lookup, supports_vision, supports_pdf_input))
    tools.extend(_make_shell_tools(sandbox))
    tools.extend(_make_browser_tools(browser))
    tools.extend(_make_search_tools(search_engine))
    return tools
