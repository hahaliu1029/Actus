"""LangChain tool wrappers for Actus native tools.

Each function wraps the corresponding sandbox/browser/search method and returns
a string result (LangChain convention).

Usage:
    tools = create_native_tools(
        sandbox_accessor=sandbox_accessor, browser_accessor=browser_accessor,
        search_engine=engine,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from typing import Any, Awaitable, Callable, List, Literal, Optional, Union

from langchain_core.tools import BaseTool, StructuredTool, tool as lc_tool
from pydantic import BaseModel

from app.domain.external.browser import BrowserAccessor
from app.domain.external.file_processor import FileProcessorLookup, FileProcessResult
from app.domain.external.sandbox import SandboxAccessor, SandboxHandle
from app.domain.external.search import SearchEngine
from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    DecisionReason,
    Denied,
    FileBlock,
    ImageUrlBlock,
    Passthrough,
    TextBlock,
    ToolOutcome,
    MultimodalPayload,
)
from app.domain.services.tools.memory_mount_scope import MemoryMountScope
from app.domain.services.tools._supervisor_tool_wrapper import (
    wrap_tool_list_for_supervisor,
)
from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)

logger = logging.getLogger(__name__)


def _coerce_result_content(result: object) -> str:
    """Extract the human-readable payload from a legacy ToolResult-like object."""
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
            return json.dumps(data, ensure_ascii=False)
        return str(data)
    if hasattr(result, "message") and result.message:
        return result.message
    return str(result)


def _wrap_result_outcome(
    result: object | None,
    *,
    failure_code: str = "native_tool_error",
    default_success_message: str = "",
) -> ToolOutcome:
    """Convert legacy ToolResult-like objects into typed ToolOutcome."""
    if result is None:
        return AllowSuccess(content=default_success_message)

    if hasattr(result, "success") and not result.success:
        message = getattr(result, "message", None) or str(result)
        return AllowError(
            content=message,
            reason=DecisionReason(
                type="exception",
                code=failure_code,
                message=message,
            ),
            retryable=False,
        )

    content = _coerce_result_content(result) or default_success_message
    # AllowSuccess.data is Optional[dict[str, Any]]; coerce BaseModel (e.g., SearchResults)
    # via model_dump so typed Pydantic producers don't silently drop their payload.
    raw_data = result.data if hasattr(result, "data") else None
    if isinstance(raw_data, dict):
        data = raw_data
    elif isinstance(raw_data, BaseModel):
        data = raw_data.model_dump(mode="json")
    else:
        data = None
    return AllowSuccess(content=content, data=data)


def _exception_outcome(tool_name: str, exc: Exception) -> AllowError:
    reason_type = "timeout" if isinstance(exc, asyncio.TimeoutError) else "exception"
    return AllowError(
        content=f"{tool_name} 异常: {exc}",
        reason=DecisionReason(
            type=reason_type,
            code=(
                f"{tool_name}_timeout"
                if isinstance(exc, asyncio.TimeoutError)
                else type(exc).__name__
            ),
            message=str(exc),
        ),
        retryable=isinstance(exc, asyncio.TimeoutError),
    )


async def _invoke_result_tool(
    tool_name: str,
    call: Awaitable[object | None],
    *,
    failure_code: str = "native_tool_error",
    default_success_message: str = "",
) -> tuple[str, ToolOutcome]:
    try:
        result = await call
    except Exception as exc:  # pragma: no cover - behavior verified via callers
        outcome = _exception_outcome(tool_name, exc)
        return outcome.content, outcome

    outcome = _wrap_result_outcome(
        result,
        failure_code=failure_code,
        default_success_message=default_success_message,
    )
    return outcome.content, outcome


def _multimodal_block_from_dict(block: dict[str, Any]):
    block_type = block.get("type")
    if block_type == "image_url":
        return ImageUrlBlock.model_validate(block)
    if block_type == "file":
        return FileBlock.model_validate(block)
    if block_type == "text":
        return TextBlock.model_validate(block)
    raise ValueError(f"Unsupported multimodal block type: {block_type!r}")


# --------------------------------------------------------------------------- #
# Message tools
# --------------------------------------------------------------------------- #


def _make_message_tools() -> list[StructuredTool]:
    """Create message tools (no external dependency needed)."""

    @lc_tool(response_format="content_and_artifact")
    async def message_notify_user(text: str) -> tuple[str, ToolOutcome]:
        """Send a notification to the user without waiting for a reply. Use for progress updates, confirmations, or status reports."""
        outcome = AllowSuccess(content="Continue")
        return outcome.content, outcome

    @lc_tool(response_format="content_and_artifact")
    async def message_ask_user(
        text: str,
        attachments: Optional[Union[str, List[str]]] = None,
        suggest_user_takeover: Optional[Literal["none", "shell", "browser"]] = None,
    ) -> tuple[str, ToolOutcome]:
        """Ask the user a question and wait for their reply. Use for clarification, confirmation, or requesting input.

        NOTE: The system may return SOFT_HINT if it determines the agent should
        try to solve autonomously first. Only call again if user input is truly
        required.
        """
        del attachments, suggest_user_takeover
        # Actual SOFT_HINT / interrupt logic is handled by react_graph's tool_node.
        # This is the fallback return value.
        outcome = AllowSuccess(content="WAITING_FOR_USER")
        return outcome.content, outcome

    tools = [message_notify_user, message_ask_user]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="message")
    return tools


# --------------------------------------------------------------------------- #
# File tools
# --------------------------------------------------------------------------- #


def _make_file_tools(
    sandbox_accessor: SandboxAccessor,
    *,
    memory_mount_scope: MemoryMountScope | None = None,
) -> list[StructuredTool]:
    """Create file tools that delegate to sandbox.

    PR-1b (SPM Task 9): the factory captures the ``SandboxAccessor``; every tool
    coroutine pulls the concrete handle lazily via ``await sandbox_accessor.get()``
    on each call (Eager accessor = zero-IO byte-equivalent ``always`` behavior).

    ``memory_mount_scope``（codex fix P0）：非空时，读类工具（file_read /
    file_str_replace / file_find_in_content）会先把 filepath 映射回 api
    容器侧实际路径，``lstat`` 判 symlink——是 symlink 就直接 SecurityError
    不发 sandbox HTTP。这是 design §724 Case C (iii) 的读侧 containment，
    防 host 预植 symlink 透过 bind mount 暴露容器内敏感文件。scope 为空时
    保持旧行为（兼容非 memory 场景 / 未 mount memory 的 sandbox）。
    """
    # 异常信息写在一处，避免 log grep 时三处文案漂移。
    _SYMLINK_ERR = (
        "refuse to follow symlink within memory mount scope "
        "(memory bind-mount read containment)"
    )

    async def _refuse_symlink_outcome(filepath: str) -> tuple[str, ToolOutcome]:
        """Client-side security gate：symlink → ``Denied`` outcome。

        ``Denied.reason.type = 'ast_validator'`` 契合"静态 pre-execution 拒绝"
        语义（CS2.4 允许的四类之一）。LangGraph react_graph 把 content 写入
        tool_result 暴露给 LLM，让 LLM 看到拒绝原因而非超时/network error。
        **不发 sandbox HTTP**——这是整个守护的核心不变式。
        """
        message = f"{_SYMLINK_ERR}: {filepath}"
        reason = DecisionReason(
            type="ast_validator",
            code="memory_mount_symlink_refused",
            message=message,
        )
        outcome = Denied(content=message, reason=reason)
        return outcome.content, outcome

    def _is_symlink_in_scope(filepath: str) -> bool:
        """scope 非空 + memory mount 内 + 任一祖先（含 user_id 根）是 symlink。

        codex fix P1 round-2：叶子 lstat 不够——攻击者把 ``{user_id}/user/``
        做成 symlink 指向 ``/etc``，叶子文件 ``secret.txt`` 自己不是 symlink
        但父目录是，整条路径仍被 kernel resolve 到 target。由
        ``MemoryMountScope.any_ancestor_is_symlink`` 逐段 lstat 检测。
        """
        if memory_mount_scope is None:
            return False
        return memory_mount_scope.any_ancestor_is_symlink(filepath)

    @lc_tool(response_format="content_and_artifact")
    async def file_read(
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False,
        max_length: int = 2000,
    ) -> tuple[str, ToolOutcome]:
        """Read file content from the sandbox filesystem."""
        sandbox = await sandbox_accessor.get()
        if _is_symlink_in_scope(filepath):
            return await _refuse_symlink_outcome(filepath)
        return await _invoke_result_tool(
            "file_read",
            sandbox.read_file(
                filepath,
                start_line=start_line,
                end_line=end_line,
                sudo=sudo,
                max_length=max_length,
            ),
            failure_code="sandbox_file_read_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def file_write(
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> tuple[str, ToolOutcome]:
        """Write content to a file in the sandbox filesystem."""
        sandbox = await sandbox_accessor.get()
        return await _invoke_result_tool(
            "file_write",
            sandbox.write_file(
                filepath,
                content,
                append=append,
                leading_newline=leading_newline,
                trailing_newline=trailing_newline,
                sudo=sudo,
            ),
            failure_code="sandbox_file_write_error",
            default_success_message="File written successfully",
        )

    file_write.metadata = {"risk_level": "medium"}

    @lc_tool(response_format="content_and_artifact")
    async def file_str_replace(
        filepath: str, old_str: str, new_str: str, sudo: bool = False
    ) -> tuple[str, ToolOutcome]:
        """Replace a string in a file."""
        sandbox = await sandbox_accessor.get()
        # memory mount 内读写都可能跟随 symlink；守护覆盖所有读类 file tools。
        if _is_symlink_in_scope(filepath):
            return await _refuse_symlink_outcome(filepath)
        return await _invoke_result_tool(
            "file_str_replace",
            sandbox.replace_in_file(filepath, old_str, new_str, sudo=sudo),
            failure_code="sandbox_file_replace_error",
            default_success_message="Replacement done",
        )

    file_str_replace.metadata = {"risk_level": "medium"}

    @lc_tool(response_format="content_and_artifact")
    async def file_find_in_content(
        filepath: str, regex: str, sudo: bool = False
    ) -> tuple[str, ToolOutcome]:
        """Search file content using regex."""
        sandbox = await sandbox_accessor.get()
        if _is_symlink_in_scope(filepath):
            return await _refuse_symlink_outcome(filepath)
        return await _invoke_result_tool(
            "file_find_in_content",
            sandbox.search_in_file(filepath, regex, sudo=sudo),
            failure_code="sandbox_file_search_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def file_find_by_name(
        dir_path: str, glob_pattern: str
    ) -> tuple[str, ToolOutcome]:
        """Find files by name pattern."""
        sandbox = await sandbox_accessor.get()
        return await _invoke_result_tool(
            "file_find_by_name",
            sandbox.find_files(dir_path, glob_pattern),
            failure_code="sandbox_file_find_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def file_list(dir_path: str) -> tuple[str, ToolOutcome]:
        """List directory contents."""
        sandbox = await sandbox_accessor.get()
        return await _invoke_result_tool(
            "file_list",
            sandbox.list_files(dir_path),
            failure_code="sandbox_file_list_error",
        )

    tools = [file_read, file_write, file_str_replace, file_find_in_content, file_find_by_name, file_list]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="file")
    return tools


# --------------------------------------------------------------------------- #
# Shell tools
# --------------------------------------------------------------------------- #


def _make_shell_tools(sandbox_accessor: SandboxAccessor) -> list[StructuredTool]:
    """Create shell tools that delegate to sandbox.

    PR-1b (SPM Task 9): each tool coroutine pulls the handle lazily via
    ``await sandbox_accessor.get()`` per call (Eager = zero-IO byte-equivalent).
    """

    _DEFAULT_WAIT_SECONDS = 5  # Matches sandbox service default, kept in sync intentionally.
    # Upper bound on the sync wait window. Stays under the httpx client timeout
    # (``DockerSandbox`` uses ``timeout=600``) so an LLM-supplied value can never
    # cause a ReadTimeout that would orphan the background command.
    _MAX_WAIT_SECONDS = 580

    @lc_tool(response_format="content_and_artifact")
    async def shell_execute(
        command: str,
        session_id: str = "default",
        exec_dir: str = "",
        wait_seconds: Optional[int] = None,
    ) -> tuple[str, ToolOutcome]:
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
        sandbox = await sandbox_accessor.get()
        # Clamp LLM-supplied wait_seconds so it cannot exceed the httpx client
        # timeout (which would orphan the command) or slip through as a non-positive.
        clamped_wait: Optional[int] = None
        if wait_seconds is not None and wait_seconds > 0:
            clamped_wait = min(wait_seconds, _MAX_WAIT_SECONDS)

        try:
            result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=exec_dir,
                command=command,
                wait_seconds=clamped_wait,
            )
        except Exception as exc:
            outcome = _exception_outcome("shell_execute", exc)
            return outcome.content, outcome
        if hasattr(result, "success") and not result.success:
            outcome = _wrap_result_outcome(result, failure_code="sandbox_shell_error")
            return outcome.content, outcome

        data = getattr(result, "data", None)
        # Legacy / mocked sandbox that doesn't return structured data — fall
        # through to the generic unwrap path.
        if not isinstance(data, dict):
            outcome = _wrap_result_outcome(result, failure_code="sandbox_shell_error")
            return outcome.content, outcome

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
            content = (
                f"[shell_execute] Command is still running on session '{session_id}' "
                f"after the {effective_wait}s sync wait window. The process keeps running "
                f"in the background.\n"
                f"Next step: call shell_wait_process(session_id='{session_id}', seconds=N) "
                f"to wait longer, or shell_read_output(session_id='{session_id}') to peek "
                f"current output. For long operations (apt/pip install, downloads, builds) "
                f"you can also re-invoke shell_execute with a larger wait_seconds."
                f"{partial_text}"
            )
            outcome = AllowSuccess(content=content, data=data)
            return outcome.content, outcome

        # status == "completed" (or unknown/legacy) — surface output + returncode.
        if output:
            if returncode is not None and returncode != 0:
                content = f"{output}\n[shell_execute] exit code: {returncode}"
            else:
                content = output
            outcome = AllowSuccess(content=content, data=data)
            return outcome.content, outcome
        if returncode == 0:
            content = "[shell_execute] Command completed successfully with no output (exit code 0)."
            outcome = AllowSuccess(content=content, data=data)
            return outcome.content, outcome
        if returncode is not None:
            content = f"[shell_execute] Command completed with no output (exit code {returncode})."
            outcome = AllowSuccess(content=content, data=data)
            return outcome.content, outcome
        outcome = AllowSuccess(
            content="[shell_execute] Command completed with no output.",
            data=data,
        )
        return outcome.content, outcome

    shell_execute.metadata = {"risk_level": "high"}

    @lc_tool(response_format="content_and_artifact")
    async def shell_read_output(
        session_id: str = "default",
    ) -> tuple[str, ToolOutcome]:
        """Read the latest output from a shell session."""
        sandbox = await sandbox_accessor.get()
        return await _invoke_result_tool(
            "shell_read_output",
            sandbox.read_shell_output(session_id=session_id),
            failure_code="sandbox_shell_read_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def shell_wait_process(
        session_id: str = "default", seconds: int = 5
    ) -> tuple[str, ToolOutcome]:
        """Wait for a running process to produce output."""
        sandbox = await sandbox_accessor.get()
        return await _invoke_result_tool(
            "shell_wait_process",
            sandbox.wait_process(session_id=session_id, seconds=seconds),
            failure_code="sandbox_wait_process_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def shell_write_input(
        input_text: str,
        session_id: str = "default",
        press_enter: bool = True,
    ) -> tuple[str, ToolOutcome]:
        """Write input to a running shell process."""
        sandbox = await sandbox_accessor.get()
        return await _invoke_result_tool(
            "shell_write_input",
            sandbox.write_shell_input(
                session_id=session_id,
                input_text=input_text,
                press_enter=press_enter,
            ),
            failure_code="sandbox_shell_input_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def shell_kill_process(
        session_id: str = "default",
    ) -> tuple[str, ToolOutcome]:
        """Kill a running process in a shell session."""
        sandbox = await sandbox_accessor.get()
        return await _invoke_result_tool(
            "shell_kill_process",
            sandbox.kill_process(session_id=session_id),
            failure_code="sandbox_shell_kill_error",
        )

    tools = [shell_execute, shell_read_output, shell_wait_process, shell_write_input, shell_kill_process]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="shell")
    return tools


# --------------------------------------------------------------------------- #
# Browser tools
# --------------------------------------------------------------------------- #


def _make_browser_tools(browser_accessor: BrowserAccessor) -> list[StructuredTool]:
    """Create browser tools that delegate to Browser.

    PR-1b (SPM Task 9): each tool coroutine pulls the browser lazily via
    ``await browser_accessor.get()`` per call (Eager = zero-IO byte-equivalent).
    """

    @lc_tool(response_format="content_and_artifact")
    async def browser_view() -> tuple[str, ToolOutcome]:
        """Get a snapshot of the current browser page content and screenshot."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_view",
            browser.view_page(),
            failure_code="browser_view_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_navigate(url: str) -> tuple[str, ToolOutcome]:
        """Navigate the browser to a URL."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_navigate",
            browser.navigate(url),
            failure_code="browser_navigate_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_click(
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> tuple[str, ToolOutcome]:
        """Click an element on the page by index or coordinates."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_click",
            browser.click(
                index=index,
                coordinate_x=coordinate_x,
                coordinate_y=coordinate_y,
            ),
            failure_code="browser_click_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_input(
        text: str,
        press_enter: bool = True,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> tuple[str, ToolOutcome]:
        """Type text into an input field."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_input",
            browser.input(
                text,
                press_enter=press_enter,
                index=index,
                coordinate_x=coordinate_x,
                coordinate_y=coordinate_y,
            ),
            failure_code="browser_input_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_move_mouse(
        coordinate_x: float, coordinate_y: float
    ) -> tuple[str, ToolOutcome]:
        """Move the mouse cursor to specific coordinates."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_move_mouse",
            browser.move_mouse(coordinate_x=coordinate_x, coordinate_y=coordinate_y),
            failure_code="browser_move_mouse_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_press_key(key: str) -> tuple[str, ToolOutcome]:
        """Press a keyboard key."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_press_key",
            browser.press_key(key),
            failure_code="browser_press_key_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_select_option(
        index: int, option: int
    ) -> tuple[str, ToolOutcome]:
        """Select an option from a dropdown."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_select_option",
            browser.select_option(index=index, option=option),
            failure_code="browser_select_option_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_scroll_up(to_top: bool = False) -> tuple[str, ToolOutcome]:
        """Scroll the page up."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_scroll_up",
            browser.scroll_up(to_top=to_top),
            failure_code="browser_scroll_up_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_scroll_down(
        to_bottom: bool = False
    ) -> tuple[str, ToolOutcome]:
        """Scroll the page down."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_scroll_down",
            browser.scroll_down(to_down=to_bottom),
            failure_code="browser_scroll_down_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_console_exec(
        javascript: str,
    ) -> tuple[str, ToolOutcome]:
        """Execute JavaScript in the browser console."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_console_exec",
            browser.console_exec(javascript),
            failure_code="browser_console_exec_error",
        )

    browser_console_exec.metadata = {"risk_level": "high"}

    @lc_tool(response_format="content_and_artifact")
    async def browser_console_view(
        max_lines: int = 50,
    ) -> tuple[str, ToolOutcome]:
        """View the browser console output."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_console_view",
            browser.console_view(max_lines=max_lines),
            failure_code="browser_console_view_error",
        )

    @lc_tool(response_format="content_and_artifact")
    async def browser_restart(url: str = "") -> tuple[str, ToolOutcome]:
        """Restart the browser, optionally navigating to a URL."""
        browser = await browser_accessor.get()
        return await _invoke_result_tool(
            "browser_restart",
            browser.restart(url=url),
            failure_code="browser_restart_error",
        )

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

    @lc_tool(response_format="content_and_artifact")
    async def search_web(
        query: str, date_range: Optional[str] = None
    ) -> tuple[str, ToolOutcome]:
        """Search the web for information."""
        return await _invoke_result_tool(
            "search_web",
            search_engine.invoke(query, date_range=date_range),
            failure_code="search_web_error",
        )

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


_FILE_VIEW_CACHE_VERSION = 1  # bump when processor output shape changes


async def _stat_file_for_cache(sandbox: SandboxHandle, filepath: str) -> dict | None:
    """B12 P3: argv-style os.stat（免 shell 注入 + ns 精度）。失败 → None → cache bypass。"""
    import json as _json

    script = (
        "import os,sys,json;"
        "s=os.stat(sys.argv[1]);"
        "print(json.dumps({"
        "'realpath':os.path.realpath(sys.argv[1]),"
        "'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,"
        "'dev':s.st_dev,'ino':s.st_ino}))"
    )
    try:
        result = await sandbox.exec_command(
            "default", "", f"python3 -c {shlex.quote(script)} {shlex.quote(filepath)}"
        )
        if (hasattr(result, "data") and isinstance(result.data, dict)
                and result.data.get("returncode") == 0):
            return _json.loads((result.data.get("output") or "").strip())
    except Exception:
        pass
    return None


def _make_file_view_tools(
    sandbox_accessor: SandboxAccessor,
    file_processor_lookup: FileProcessorLookup | None = None,
    file_processor_factory: Callable[[SandboxHandle], FileProcessorLookup] | None = None,
    supports_vision: bool = True,
    supports_pdf_input: bool = False,
    *,
    file_view_media_type_enabled: bool = False,
    file_view_image_cache_enabled: bool = False,
    document_preview_enabled: bool = False,
) -> list[StructuredTool]:
    """Create file_view tool for multimodal file understanding.

    PR-1b (SPM Task 9): the handle is pulled lazily per call via
    ``await sandbox_accessor.get()``.

    Two processor-supply paths (mutually exclusive — at most one non-None):
      * ``file_processor_lookup`` — a ready ``FileProcessorLookup`` (current
        behavior; used by every Eager call site today).
      * ``file_processor_factory`` — deferred construction for the PR-1c OnDemand
        world where the lookup can only be built once a handle exists. Inside the
        tool body the pulled handle is passed to ``factory(handle)``; the result
        is memoized in a per-run closure cache keyed by ``handle.generation`` so a
        poison/regeneration transparently rebuilds it.
    """
    assert not (
        file_processor_lookup is not None and file_processor_factory is not None
    ), "file_view: file_processor_lookup and file_processor_factory are mutually exclusive"
    # Factory-path per-run cache (per-tool lifetime); keyed by handle generation.
    _fp_cache: dict[int, FileProcessorLookup] = {}

    @lc_tool(response_format="content_and_artifact")
    async def file_view(filepath: str) -> tuple[str, ToolOutcome]:
        """View and understand a file's content. Use this for images, PDFs,
        audio, and video files instead of file_read.
        Returns the file content in a format the model can understand."""

        sandbox = await sandbox_accessor.get()
        if file_processor_factory is not None:
            _gen = sandbox.generation
            processor_lookup = _fp_cache.get(_gen)
            if processor_lookup is None:
                processor_lookup = file_processor_factory(sandbox)
                _fp_cache[_gen] = processor_lookup
        else:
            processor_lookup = file_processor_lookup

        # 1. Detect MIME type (sandbox `file` command + extension fallback)
        try:
            mime_result = await sandbox.exec_command(
                "default", "", f"file --mime-type -b {shlex.quote(filepath)}"
            )
        except Exception as exc:
            outcome = _exception_outcome("file_view", exc)
            return outcome.content, outcome

        # Check for execution failure (path not found, permission denied, etc.)
        if hasattr(mime_result, "success") and not mime_result.success:
            outcome = AllowError(
                content=f"Cannot access file: {mime_result}",
                reason=DecisionReason(
                    type="exception",
                    code="file_view_access_error",
                    message=str(mime_result),
                ),
            )
            return outcome.content, outcome

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
                outcome = AllowError(
                    content=f"Cannot detect file type: {output.strip() or mime_result}",
                    reason=DecisionReason(
                        type="exception",
                        code="file_view_mime_detect_error",
                        message=output.strip() or str(mime_result),
                    ),
                )
                return outcome.content, outcome
        else:
            mime_type = str(mime_result).strip()

        # Fallback to extension when `file` unavailable or returns generic/empty type
        if not mime_type or mime_type == "application/octet-stream":
            ext = "." + filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
            mime_type = _EXT_MIME_MAP.get(ext, mime_type or "application/octet-stream")

        # 2. Find processor
        processor = processor_lookup.get_processor(mime_type)
        if processor is None:
            outcome = AllowSuccess(
                content=f"Unsupported file type: {mime_type}. Use file_read for text files."
            )
            return outcome.content, outcome

        # 3. Process file (with optional B12 P3 session-scoped cache)
        # tool_node splits: text → ToolMessage, image_blocks → HumanMessage
        filename = filepath.rsplit("/", 1)[-1]
        _cache_key = None
        result = None
        if file_view_image_cache_enabled and hasattr(processor_lookup, "cache_get"):
            _stat = await _stat_file_for_cache(sandbox, filepath)
            if _stat is not None:
                _cache_key = (
                    getattr(sandbox, "generation", None),
                    _stat.get("realpath"), _stat.get("size"), _stat.get("mtime_ns"),
                    _stat.get("ctime_ns"),  # PR-3 codex R1#P2: ctime 无法被 os.utime 回拨 → 防 mtime-restore stale-hit
                    _stat.get("dev"), _stat.get("ino"), mime_type,
                    supports_vision, supports_pdf_input, _FILE_VIEW_CACHE_VERSION,
                )
                try:
                    result = processor_lookup.cache_get(_cache_key)  # None on miss
                except Exception:  # noqa: BLE001 — best-effort cache; a throwing backend (e.g. remote) must not break file_view
                    logger.warning("file_view cache_get failed; reprocessing", exc_info=True)
                    result = None
        if result is None:
            import time as _time
            _process_start = _time.time()
            try:
                result = await processor.process(
                    sandbox_path=filepath,
                    filename=filename,
                    mime_type=mime_type,
                    supports_vision=supports_vision,
                    supports_pdf_input=supports_pdf_input,
                )
            except Exception as exc:
                outcome = _exception_outcome("file_view", exc)
                return outcome.content, outcome
            if _cache_key is not None and hasattr(processor_lookup, "cache_put"):
                try:
                    processor_lookup.cache_put(_cache_key, result, _process_start)
                except Exception:  # noqa: BLE001 — best-effort cache write; result already computed, never fail the tool
                    logger.warning("file_view cache_put failed; result already returned", exc_info=True)

        typed_blocks = [
            _multimodal_block_from_dict(block)
            for block in (*result.image_blocks, *result.document_blocks)
        ]
        if typed_blocks:
            image_count = sum(
                1 for block in typed_blocks if isinstance(block, ImageUrlBlock)
            )
            summary = result.text or (
                f"[file_view: file_view — {image_count} image(s) loaded]"
            )
            outcome = Passthrough(
                content=summary,
                data=MultimodalPayload(
                    blocks=typed_blocks,
                    media_type=(
                        result.media_type if file_view_media_type_enabled else None
                    ),
                    document_preview=(
                        result.document_preview if document_preview_enabled else None
                    ),
                ),
            )
            return outcome.content, outcome

        outcome = AllowSuccess(content=result.text or "")
        return outcome.content, outcome

    tools = [file_view]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="file")
    return tools


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def create_native_tools(
    sandbox_accessor: SandboxAccessor | None,
    browser_accessor: BrowserAccessor | None,
    search_engine: SearchEngine,
    file_processor_lookup: FileProcessorLookup | None = None,
    supports_vision: bool = True,
    supports_pdf_input: bool = False,
    memory_mount_scope: MemoryMountScope | None = None,
    supervisor: Any | None = None,
    *,
    include_sandbox_tools: bool = True,
    file_processor_factory: Callable[[SandboxHandle], FileProcessorLookup] | None = None,
    file_view_media_type_enabled: bool = False,
    file_view_image_cache_enabled: bool = False,
    document_preview_enabled: bool = False,
) -> list[BaseTool]:
    """Create all native LangChain tools.

    PR-1b (SPM Task 9): sandbox/browser are supplied as ``SandboxAccessor`` /
    ``BrowserAccessor`` (typing-only contracts). The Eager accessors used by every
    call site today make ``get()`` a zero-IO pure return, so tool behavior is
    byte-equivalent to the old raw-handle wiring (INV-SPM-2).

    ``include_sandbox_tools``（off-assembly future-proofing）：True（默认）时构造
    文件/file_view/shell/browser 沙箱族——此时两个 accessor 必须非 None（误装配
    fail-fast，assert）。False 时整族跳过（off 档只装 message/search，永不解引用
    None accessor）。本任务无调用点传 False——仅冻结签名与门结构。

    ``file_processor_lookup`` / ``file_processor_factory`` 至多一个非 None（互斥由
    ``_make_file_view_tools`` assert）；两者皆 None 则不注册 file_view。

    ``memory_mount_scope``（codex fix P0）透传到 ``_make_file_tools`` 打开
    客户端侧 symlink 守护。agent runner / planner_react 构造 scope 后传入；
    没有 user_id 或 memory mount 未启用时保持 None（旧行为）。

    Returns a flat list of tools ready to be bound to an LLM or added to a ToolNode.
    """
    tools: list[StructuredTool] = []
    tools.extend(_make_message_tools())
    if include_sandbox_tools:
        assert sandbox_accessor is not None, (
            "create_native_tools(include_sandbox_tools=True) requires a sandbox_accessor"
        )
        assert browser_accessor is not None, (
            "create_native_tools(include_sandbox_tools=True) requires a browser_accessor"
        )
        tools.extend(_make_file_tools(sandbox_accessor, memory_mount_scope=memory_mount_scope))
        if file_processor_lookup is not None or file_processor_factory is not None:
            tools.extend(_make_file_view_tools(
                sandbox_accessor,
                file_processor_lookup=file_processor_lookup,
                file_processor_factory=file_processor_factory,
                supports_vision=supports_vision,
                supports_pdf_input=supports_pdf_input,
                file_view_media_type_enabled=file_view_media_type_enabled,
                file_view_image_cache_enabled=file_view_image_cache_enabled,
                document_preview_enabled=document_preview_enabled,
            ))
        tools.extend(_make_shell_tools(sandbox_accessor))
        tools.extend(_make_browser_tools(browser_accessor))
    tools.extend(_make_search_tools(search_engine))
    return wrap_tool_list_for_supervisor(tools, supervisor)
