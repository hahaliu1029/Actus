from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.domain.models.skill_creator import SkillCreationProgress, SkillCreationResult
from app.domain.services.tools.create_skill import CreateSkillTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.id = "sb-123"
    return sandbox


@pytest.fixture
def tool(mock_sandbox: MagicMock) -> CreateSkillTool:
    creator = MagicMock()
    return CreateSkillTool(skill_creator_service=creator, sandbox=mock_sandbox)


def test_get_tools_returns_schema(tool: CreateSkillTool) -> None:
    tools = tool.get_tools()
    assert len(tools) == 1
    fn_schema = tools[0]["function"]
    assert fn_schema["name"] == "create_skill"
    assert "description" in fn_schema["parameters"]["properties"]


async def test_invoke_calls_service_with_sandbox(
    tool: CreateSkillTool,
) -> None:
    async def fake_create(**kwargs):
        assert kwargs["sandbox"] is not None
        yield SkillCreationProgress(step="analyzing", message="...")
        yield SkillCreationResult(
            skill_id="s1",
            skill_name="test",
            tools=["run"],
            files_count=3,
            summary="Skill 'test' 创建成功",
        )

    tool._creator.create = fake_create
    result = await tool.invoke("create_skill", description="创建测试 skill")

    assert result.success
    assert "创建成功" in (result.message or "")


async def test_invoke_returns_error_on_failure(tool: CreateSkillTool) -> None:
    async def fake_create(**kwargs):
        del kwargs
        yield SkillCreationProgress(step="validating", message="验证失败: SyntaxError")

    tool._creator.create = fake_create
    result = await tool.invoke("create_skill", description="创建 skill")

    assert result.success is False
    assert "失败" in (result.message or "")
