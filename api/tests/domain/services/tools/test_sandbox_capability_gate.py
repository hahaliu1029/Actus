"""SPM Task 24 — off-mode tool-face contraction.

Covers the pure capability gate, the ``message_ask_user`` takeover-enum
contraction, ``create_native_tools(include_sandbox_tools=False)`` family skip,
and the MCP ``sandbox_files_enabled`` off refusal (+ always byte-parity).

off stays REJECTED at the config layer (ALLOWED={always,on_demand}); these
tests drive the runtime code paths directly with explicit booleans — no
settings monkeypatch needed here.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.tool_result import ToolResult
from app.domain.services.tools import langchain_tools as lt
from app.domain.services.tools.langchain_tools import (
    SANDBOX_FACE_TOOL_NAMES,
    apply_sandbox_capability_gate,
    create_native_tools,
)
from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


def _mk_named_tools(names: list[str]) -> list[_NamedTool]:
    return [_NamedTool(n) for n in names]


def _collect_factory_tool_names() -> set[str]:
    """Names produced by the four sandbox tool families (real factories)."""
    acc = MagicMock()
    browser = MagicMock()
    names: set[str] = set()
    for tool in lt._make_file_tools(acc):
        names.add(tool.name)
    for tool in lt._make_shell_tools(acc):
        names.add(tool.name)
    for tool in lt._make_browser_tools(browser):
        names.add(tool.name)
    for tool in lt._make_file_view_tools(acc, file_processor_lookup=MagicMock()):
        names.add(tool.name)
    return names


_SKILL_CREATION_NAMES = {"generate_skill", "install_skill", "brainstorm_skill"}


def _takeover_enum_tokens(tools: list) -> tuple[bool, bool, bool]:
    """(has_shell, has_browser, has_none) in message_ask_user args schema."""
    for tool in tools:
        if tool.name == "message_ask_user":
            blob = json.dumps(tool.args_schema.model_json_schema())
            return ("shell" in blob, "browser" in blob, "none" in blob)
    raise AssertionError("message_ask_user not found")


class TestSandboxCapabilityGate:
    def test_off_removes_exactly_the_sandbox_face(self) -> None:
        tools = _mk_named_tools(
            ["shell_execute", "message_notify_user", "memory_recall", "generate_skill"]
        )
        out = apply_sandbox_capability_gate(tools, sandbox_tools_enabled=False)
        assert {t.name for t in out} == {"message_notify_user", "memory_recall"}

    def test_on_is_identity(self) -> None:
        tools = _mk_named_tools(["shell_execute", "message_notify_user"])
        # byte-zero spot: same list object returned when enabled.
        assert apply_sandbox_capability_gate(tools, sandbox_tools_enabled=True) is tools

    def test_face_set_matches_factories(self) -> None:
        """Anchor: literal set == live factory output (a new sandbox tool that
        is not added here turns this red → forced human review)."""
        assert _collect_factory_tool_names() | _SKILL_CREATION_NAMES == set(
            SANDBOX_FACE_TOOL_NAMES
        )

    def test_face_set_count_frozen(self) -> None:
        # 6 file + 1 file_view + 5 shell + 12 browser + 3 skill-creation = 27
        assert len(SANDBOX_FACE_TOOL_NAMES) == 27


class TestMessageAskUserTakeoverEnum:
    def test_off_message_ask_user_no_sandbox_takeover_enum(self) -> None:
        tools = lt._make_message_tools(sandbox_tools_enabled=False)
        has_shell, has_browser, has_none = _takeover_enum_tokens(tools)
        assert not has_shell and not has_browser
        assert has_none

    def test_on_message_ask_user_full_takeover_enum(self) -> None:
        # byte-zero: default / always keeps the full takeover enum.
        tools = lt._make_message_tools()
        assert _takeover_enum_tokens(tools) == (True, True, True)
        tools_true = lt._make_message_tools(sandbox_tools_enabled=True)
        assert _takeover_enum_tokens(tools_true) == (True, True, True)


class TestCreateNativeToolsOff:
    def test_off_skips_sandbox_families_keeps_message_search(self) -> None:
        tools = create_native_tools(None, None, MagicMock(), include_sandbox_tools=False)
        names = {t.name for t in tools}
        assert names == {"message_notify_user", "message_ask_user", "search_web"}
        # off native message tool also contracts the takeover enum.
        assert _takeover_enum_tokens(tools) == (False, False, True)

    def test_off_native_tools_share_no_name_with_face(self) -> None:
        tools = create_native_tools(None, None, MagicMock(), include_sandbox_tools=False)
        assert not ({t.name for t in tools} & set(SANDBOX_FACE_TOOL_NAMES))


def _path_tool_mcp() -> Any:
    mcp = MagicMock()
    mcp.get_tools.return_value = [
        {
            "function": {
                "name": "read_doc",
                "description": "read a doc",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        }
    ]
    mcp.invoke = AsyncMock(
        return_value=ToolResult(success=True, message="ok", data={"r": 1})
    )
    mcp.tool_server_bindings.return_value = {"read_doc": "srv"}
    return mcp


class TestMcpSandboxFilesEnabled:
    async def test_off_sandbox_path_returns_disabled_no_invoke(self) -> None:
        mcp = _path_tool_mcp()
        tools = create_mcp_langchain_tools(mcp, sandbox_files_enabled=False)
        assert len(tools) == 1
        result = await tools[0].ainvoke({"path": "/home/ubuntu/report.pdf"})
        assert "未启用沙箱文件系统" in str(result)
        mcp.invoke.assert_not_called()

    async def test_off_non_sandbox_arg_still_invokes(self) -> None:
        mcp = _path_tool_mcp()
        tools = create_mcp_langchain_tools(mcp, sandbox_files_enabled=False)
        result = await tools[0].ainvoke({"path": "https://example.com/x.pdf"})
        assert "ok" in str(result)
        mcp.invoke.assert_awaited_once()

    async def test_always_sandbox_path_byte_identical_passthrough(self) -> None:
        """always (default True) + no uploader → path passes through unchanged."""
        mcp = _path_tool_mcp()
        tools = create_mcp_langchain_tools(mcp, sandbox_files_enabled=True)
        result = await tools[0].ainvoke({"path": "/home/ubuntu/report.pdf"})
        assert "ok" in str(result)
        mcp.invoke.assert_awaited_once_with("read_doc", path="/home/ubuntu/report.pdf")

    async def test_default_flag_is_always_behavior(self) -> None:
        # Omitting the flag entirely == always (byte-zero for existing callers).
        mcp = _path_tool_mcp()
        tools = create_mcp_langchain_tools(mcp)
        await tools[0].ainvoke({"path": "/home/ubuntu/report.pdf"})
        mcp.invoke.assert_awaited_once_with("read_doc", path="/home/ubuntu/report.pdf")
