from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.skill_creator import SkillBlueprint, ToolDef, ToolParamDef
from app.domain.services.tools.brainstorm_skill import BrainstormSkillTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_creator_service() -> MagicMock:
    service = MagicMock()
    service.analyze = AsyncMock()
    return service


@pytest.fixture
def tool(mock_creator_service: MagicMock) -> BrainstormSkillTool:
    return BrainstormSkillTool(skill_creator_service=mock_creator_service)


def test_get_tools_returns_brainstorm_schema(tool: BrainstormSkillTool) -> None:
    tools = tool.get_tools()
    assert len(tools) == 1
    fn_schema = tools[0]["function"]
    assert fn_schema["name"] == "brainstorm_skill"
    assert "description" in fn_schema["parameters"]["properties"]


async def test_brainstorm_returns_formatted_blueprint(
    tool: BrainstormSkillTool,
    mock_creator_service: MagicMock,
) -> None:
    mock_creator_service.analyze.return_value = SkillBlueprint(
        skill_name="cn-en-translator",
        description="中英互译工具",
        tools=[
            ToolDef(
                name="translate_text",
                description="翻译文本",
                parameters=[
                    ToolParamDef(name="text", type="string", description="待翻译文本", required=True),
                    ToolParamDef(name="direction", type="string", description="翻译方向", required=False),
                ],
            ),
        ],
        search_keywords=["translate python"],
        estimated_deps=["googletrans"],
    )

    result = await tool.invoke("brainstorm_skill", description="创建一个中英翻译工具")

    assert result.success
    assert "cn-en-translator" in (result.message or "")
    assert "translate_text" in (result.message or "")
    assert "googletrans" in (result.message or "")
    assert result.data is not None
    assert result.data["skill_name"] == "cn-en-translator"


async def test_brainstorm_returns_blueprint_and_json(
    tool: BrainstormSkillTool,
    mock_creator_service: MagicMock,
) -> None:
    mock_creator_service.analyze.return_value = SkillBlueprint(
        skill_name="meeting-audio-analyzer",
        description="会议音频分析工具",
        tools=[],
        search_keywords=["meeting audio analysis python"],
        estimated_deps=["openai-whisper"],
    )

    result = await tool.invoke("brainstorm_skill", description="创建一个会议音频分析 skill")

    assert result.success is True
    assert result.data is not None
    assert result.data["skill_name"] == "meeting-audio-analyzer"
    assert result.data["blueprint"]["skill_name"] == "meeting-audio-analyzer"
    assert "blueprint_json" in result.data


async def test_brainstorm_returns_error_on_exception(
    tool: BrainstormSkillTool,
    mock_creator_service: MagicMock,
) -> None:
    mock_creator_service.analyze.side_effect = Exception("LLM 调用失败")

    result = await tool.invoke("brainstorm_skill", description="创建工具")

    assert result.success is False
    assert "异常" in (result.message or "")
