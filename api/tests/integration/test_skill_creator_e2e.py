"""Skill Creator 端到端（mock）测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.skill_creator_service import SkillCreatorService
from app.domain.models.skill_creator import SkillCreationProgress, SkillCreationResult
from app.domain.models.tool_result import ToolResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_full_pipeline_with_mocked_llm_and_sandbox() -> None:
    mock_llm = AsyncMock()
    mock_github = MagicMock()
    mock_github.research_keywords = AsyncMock(return_value=[])
    mock_github.format_research_report = MagicMock(return_value="无参考")
    mock_skill_service = AsyncMock()

    mock_skill_service.install_skill = AsyncMock(
        return_value=SimpleNamespace(id="echo--abc12345", name="echo")
    )

    mock_sandbox = AsyncMock()
    mock_sandbox.write_file = AsyncMock(return_value=ToolResult(success=True, message="ok"))
    mock_sandbox.exec_command = AsyncMock(return_value=ToolResult(success=True, message="ok"))

    analyze_resp = {
        "role": "assistant",
        "content": json.dumps(
            {
                "skill_name": "echo",
                "description": "Echo back input",
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo",
                        "parameters": [
                            {
                                "name": "text",
                                "type": "string",
                                "description": "Input",
                                "required": True,
                            }
                        ],
                    }
                ],
                "search_keywords": ["echo python"],
                "estimated_deps": [],
            }
        ),
    }
    generate_resp = {
        "role": "assistant",
        "content": json.dumps(
            {
                "skill_md": "---\nname: echo\nruntime_type: native\n---\n# Echo",
                "manifest": {
                    "name": "echo",
                    "slug": "echo",
                    "version": "0.1.0",
                    "description": "Echo",
                    "runtime_type": "native",
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo",
                            "parameters": {"text": {"type": "string"}},
                            "required": ["text"],
                            "entry": {"command": "python bundle/echo.py"},
                        }
                    ],
                },
                "scripts": [
                    {
                        "path": "bundle/echo.py",
                        "content": "import sys, json\nargs = json.loads(sys.argv[1]) if len(sys.argv)>1 else {}\nprint(json.dumps({'result': args.get('text', '')}))",
                    }
                ],
                "dependencies": [],
            }
        ),
    }
    mock_llm.invoke.side_effect = [analyze_resp, generate_resp]

    service = SkillCreatorService(
        llm=mock_llm,
        github_client=mock_github,
        skill_service=mock_skill_service,
    )

    events = []
    async for event in service.create(
        description="创建一个 echo skill",
        sandbox=mock_sandbox,
        installed_by="test-user",
    ):
        events.append(event)

    progress_events = [event for event in events if isinstance(event, SkillCreationProgress)]
    result_events = [event for event in events if isinstance(event, SkillCreationResult)]

    assert len(result_events) == 1
    assert result_events[0].skill_name == "echo"
    assert "echo" in result_events[0].tools

    steps = [event.step for event in progress_events]
    assert "analyzing" in steps
    assert "researching" in steps
    assert "generating" in steps
    assert "validating" in steps
    assert "installing" in steps

    mock_skill_service.install_skill.assert_called_once()
