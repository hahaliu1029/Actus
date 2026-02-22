from __future__ import annotations

import pytest

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.tool_result import ToolResult
from app.domain.services.tools.skill import SkillTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def exec_command(self, session_id: str, exec_dir: str, command: str) -> ToolResult:
        self.calls.append((session_id, exec_dir, command))
        return ToolResult(success=True, message="ok", data={"session_id": session_id})

    async def read_shell_output(self, session_id: str, console: bool = False) -> ToolResult:
        return ToolResult(success=True, data={"session_id": session_id, "output": "native-ok"})


class _FakeMCPTool:
    def __init__(self) -> None:
        self.called: tuple[str, dict] | None = None

    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        self.called = (tool_name, kwargs)
        return ToolResult(success=True, data="mcp-ok")


class _FakeA2ATool:
    def __init__(self) -> None:
        self.called: tuple[str, str] | None = None

    async def call_remote_agent(self, id: str, query: str) -> ToolResult:
        self.called = (id, query)
        return ToolResult(success=True, data="a2a-ok")


async def test_native_skill_executes_entry_command() -> None:
    sandbox = _FakeSandbox()
    skill_tool = SkillTool(sandbox=sandbox, mcp_tool=_FakeMCPTool(), a2a_tool=_FakeA2ATool())

    skill = Skill(
        slug="demo-native",
        name="Demo Native",
        source_type=SkillSourceType.GITHUB,
        source_ref="github:owner/repo",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "name": "Demo Native",
            "runtime_type": "native",
            "skill_md": "# Demo Native\nUse this skill when user asks for demo shell action.",
            "tools": [
                {
                    "name": "run_demo",
                    "description": "run",
                    "parameters": {"target": {"type": "string"}},
                    "required": ["target"],
                    "entry": {
                        "exec_dir": "/home/ubuntu/workspace",
                        "command": "echo demo",
                    },
                }
            ],
        },
        installed_by="admin-1",
    )

    await skill_tool.initialize([skill])
    tools = skill_tool.get_tools()
    assert tools[0]["function"]["name"] == "skill_demo_native_run_demo"
    assert "Skill guide:" in tools[0]["function"]["description"]

    result = await skill_tool.invoke("skill_demo_native_run_demo", target="hello")

    assert result.success is True
    assert sandbox.calls
    _, _, command = sandbox.calls[0]
    assert "echo demo" in command


async def test_mcp_skill_delegates_to_mcp_tool() -> None:
    mcp_tool = _FakeMCPTool()
    skill_tool = SkillTool(sandbox=_FakeSandbox(), mcp_tool=mcp_tool, a2a_tool=_FakeA2ATool())

    skill = Skill(
        slug="demo-mcp",
        name="Demo MCP",
        source_type=SkillSourceType.MCP_REGISTRY,
        source_ref="mcp:demo",
        runtime_type=SkillRuntimeType.MCP,
        manifest={
            "name": "Demo MCP",
            "runtime_type": "mcp",
            "tools": [
                {
                    "name": "route",
                    "description": "route",
                    "parameters": {"query": {"type": "string"}},
                    "required": ["query"],
                    "entry": {
                        "tool_name": "mcp_demo_route",
                    },
                }
            ],
        },
        installed_by="admin-1",
    )

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_mcp_route", query="q")

    assert result.success is True
    assert mcp_tool.called == ("mcp_demo_route", {"query": "q"})


async def test_a2a_skill_delegates_to_a2a_tool() -> None:
    a2a_tool = _FakeA2ATool()
    skill_tool = SkillTool(sandbox=_FakeSandbox(), mcp_tool=_FakeMCPTool(), a2a_tool=a2a_tool)

    skill = Skill(
        slug="demo-a2a",
        name="Demo A2A",
        source_type=SkillSourceType.GITHUB,
        source_ref="github:owner/a2a",
        runtime_type=SkillRuntimeType.A2A,
        manifest={
            "name": "Demo A2A",
            "runtime_type": "a2a",
            "tools": [
                {
                    "name": "delegate",
                    "description": "delegate",
                    "parameters": {"query": {"type": "string"}},
                    "required": ["query"],
                    "entry": {
                        "agent_id": "agent-1",
                    },
                }
            ],
        },
        installed_by="admin-1",
    )

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_a2a_delegate", query="hello")

    assert result.success is True
    assert a2a_tool.called == ("agent-1", "hello")


async def test_skill_tool_normalizes_and_shortens_function_name() -> None:
    skill_tool = SkillTool(sandbox=_FakeSandbox(), mcp_tool=_FakeMCPTool(), a2a_tool=_FakeA2ATool())

    skill = Skill(
        slug="pptx-skill",
        name="PPTX",
        source_type=SkillSourceType.GITHUB,
        source_ref="github:owner/pptx",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "name": "PPTX",
            "runtime_type": "native",
            "tools": [
                {
                    "name": "Render Slide/Deck (Very Long Name For External Skill Ecosystem)",
                    "description": "render",
                    "parameters": {},
                    "required": [],
                    "entry": {"exec_dir": "/home/ubuntu/workspace", "command": "echo pptx"},
                }
            ],
        },
        installed_by="admin-1",
    )

    await skill_tool.initialize([skill])
    function_name = skill_tool.get_tools()[0]["function"]["name"]
    assert len(function_name) <= 64
    assert "/" not in function_name
    assert "-" not in function_name
