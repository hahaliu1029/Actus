from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.skill_creator import (
    SkillCreationProgress,
    SkillCreationResult,
    SkillGeneratedFiles,
    ScriptFile,
)
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
def mock_creator() -> MagicMock:
    creator = MagicMock()
    creator.generate = MagicMock()
    creator.install = AsyncMock()
    return creator


@pytest.fixture
def tool(mock_creator: MagicMock, mock_sandbox: MagicMock) -> CreateSkillTool:
    return CreateSkillTool(
        skill_creator_service=mock_creator,
        sandbox=mock_sandbox,
        user_id="user-1",
    )


def test_get_tools_returns_both_schemas(tool: CreateSkillTool) -> None:
    tools = tool.get_tools()
    names = [t["function"]["name"] for t in tools]
    assert "generate_skill" in names
    assert "install_skill" in names
    assert len(tools) == 2


class TestGenerateSkill:
    async def test_generate_returns_success_with_skill_data(
        self, tool: CreateSkillTool, mock_creator: MagicMock,
    ) -> None:
        files = SkillGeneratedFiles(
            skill_md="---\nname: test\n---",
            manifest={"name": "test", "tools": [{"name": "run"}]},
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=["requests"],
        )

        async def fake_generate(**kwargs):
            yield SkillCreationProgress(step="researching", message="调研中")
            yield SkillCreationProgress(step="validating", message="验证通过")
            yield files

        mock_creator.generate = fake_generate
        result = await tool.invoke("generate_skill", description="创建测试 skill")

        assert result.success
        assert result.data is not None
        assert "skill_data" in result.data
        assert result.data["tools"] == ["run"]

    async def test_generate_returns_failure_on_validation_error(
        self, tool: CreateSkillTool, mock_creator: MagicMock,
    ) -> None:
        async def fake_generate(**kwargs):
            yield SkillCreationProgress(step="validating", message="沙箱验证失败", detail="SyntaxError")

        mock_creator.generate = fake_generate
        result = await tool.invoke("generate_skill", description="创建 skill")

        assert result.success is False
        assert "validation_errors" in (result.data or {})

    async def test_generate_with_blueprint_json(
        self, tool: CreateSkillTool, mock_creator: MagicMock,
    ) -> None:
        files = SkillGeneratedFiles(
            skill_md="---\nname: test\n---",
            manifest={"name": "test", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=[],
        )

        received_blueprint = {}

        async def fake_generate(**kwargs):
            received_blueprint.update({"bp": kwargs.get("blueprint")})
            yield files

        mock_creator.generate = fake_generate
        bp = json.dumps({"skill_name": "my-skill", "description": "测试", "tools": [], "search_keywords": [], "estimated_deps": []})
        result = await tool.invoke("generate_skill", description="测试", blueprint_json=bp)

        assert result.success
        assert received_blueprint["bp"] is not None
        assert received_blueprint["bp"].skill_name == "my-skill"

    async def test_generate_with_blueprint_object(
        self, tool: CreateSkillTool, mock_creator: MagicMock,
    ) -> None:
        files = SkillGeneratedFiles(
            skill_md="---\nname: test\n---",
            manifest={"name": "test", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=[],
        )

        received_blueprint = {}

        async def fake_generate(**kwargs):
            received_blueprint["bp"] = kwargs.get("blueprint")
            yield files

        mock_creator.generate = fake_generate
        result = await tool.invoke(
            "generate_skill",
            description="测试",
            blueprint={
                "skill_name": "my-skill",
                "description": "测试",
                "tools": [],
                "search_keywords": [],
                "estimated_deps": [],
            },
        )

        assert result.success
        assert received_blueprint["bp"] is not None
        assert received_blueprint["bp"].skill_name == "my-skill"


class TestInstallSkill:
    async def test_install_returns_success(
        self, tool: CreateSkillTool, mock_creator: MagicMock,
    ) -> None:
        mock_creator.install.return_value = SkillCreationResult(
            skill_id="s1",
            skill_name="test",
            tools=["run"],
            files_count=3,
            summary="Skill 'test' 创建成功",
        )

        files = SkillGeneratedFiles(
            skill_md="---\nname: test\n---",
            manifest={"name": "test", "tools": [{"name": "run"}]},
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=[],
        )
        skill_data = files.model_dump_json()

        result = await tool.invoke("install_skill", skill_data=skill_data)

        assert result.success
        assert "创建成功" in (result.message or "")

    async def test_install_returns_error_on_invalid_json(
        self, tool: CreateSkillTool,
    ) -> None:
        result = await tool.invoke("install_skill", skill_data="not-json")

        assert result.success is False
        assert "解析失败" in (result.message or "")
