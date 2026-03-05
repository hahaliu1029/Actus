from __future__ import annotations

import ast
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.skill_creator_service import SkillCreatorService
from app.domain.models.skill_creator import (
    SkillBlueprint,
    SkillCreationProgress,
    SkillCreationResult,
    SkillGeneratedFiles,
    ScriptFile,
    ToolDef,
)
from app.domain.models.tool_result import ToolResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_github_client() -> MagicMock:
    client = MagicMock()
    client.research_keywords = AsyncMock(return_value=[])
    client.format_research_report = MagicMock(return_value="未找到相关参考。")
    return client


@pytest.fixture
def mock_skill_service() -> AsyncMock:
    service = AsyncMock()
    service.install_skill = AsyncMock(
        return_value=SimpleNamespace(id="test-skill--abc12345", name="test-skill")
    )
    return service


@pytest.fixture
def mock_sandbox() -> AsyncMock:
    sandbox = AsyncMock()
    sandbox.id = "sandbox-123"
    sandbox.exec_command = AsyncMock(return_value=ToolResult(success=True, message="OK"))
    sandbox.write_file = AsyncMock(return_value=ToolResult(success=True, message="OK"))
    return sandbox


@pytest.fixture
def service(
    mock_llm: AsyncMock,
    mock_github_client: MagicMock,
    mock_skill_service: AsyncMock,
) -> SkillCreatorService:
    return SkillCreatorService(
        llm=mock_llm,
        github_client=mock_github_client,
        skill_service=mock_skill_service,
    )


class TestAnalyzeRequirement:
    async def test_analyze_extracts_blueprint(
        self,
        service: SkillCreatorService,
        mock_llm: AsyncMock,
    ) -> None:
        mock_llm.invoke.return_value = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "skill_name": "bilibili-whisper-summary",
                    "description": "下载B站视频并用whisper转录后总结",
                    "tools": [
                        {
                            "name": "download_and_summarize",
                            "description": "下载视频并总结",
                            "parameters": [
                                {
                                    "name": "url",
                                    "type": "string",
                                    "description": "视频URL",
                                    "required": True,
                                }
                            ],
                        }
                    ],
                    "search_keywords": ["bilibili download python", "whisper transcribe"],
                    "estimated_deps": ["yt-dlp", "openai-whisper"],
                }
            ),
        }

        blueprint = await service._analyze_requirement("帮我创建一个下载B站视频的 skill")

        assert blueprint.skill_name == "bilibili-whisper-summary"
        assert len(blueprint.tools) == 1
        assert "yt-dlp" in blueprint.estimated_deps


class TestGenerateFiles:
    async def test_generate_produces_valid_files(
        self,
        service: SkillCreatorService,
        mock_llm: AsyncMock,
    ) -> None:
        blueprint = SkillBlueprint(
            skill_name="test-skill",
            description="测试用",
            tools=[ToolDef(name="run", description="运行", parameters=[])],
            search_keywords=[],
            estimated_deps=["requests"],
        )

        mock_llm.invoke.return_value = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "skill_md": "---\nname: test-skill\n---\n# Test Skill",
                    "manifest": {
                        "name": "test-skill",
                        "slug": "test-skill",
                        "version": "0.1.0",
                        "description": "测试用",
                        "runtime_type": "native",
                        "tools": [
                            {
                                "name": "run",
                                "description": "运行",
                                "parameters": {},
                                "required": [],
                                "entry": {"command": "python bundle/run.py"},
                            }
                        ],
                    },
                    "scripts": [
                        {"path": "bundle/run.py", "content": "import sys\nprint('hello')"}
                    ],
                    "dependencies": ["requests"],
                }
            ),
        }

        files = await service._generate_files(blueprint, "调研报告内容")

        assert "name: test-skill" in files.skill_md
        assert files.manifest["runtime_type"] == "native"
        assert len(files.scripts) == 1
        ast.parse(files.scripts[0].content)


class TestValidateInSandbox:
    async def test_validate_passes_on_valid_script(
        self,
        service: SkillCreatorService,
        mock_sandbox: AsyncMock,
    ) -> None:
        files = SkillGeneratedFiles(
            skill_md="---\nname: t\n---\n# T",
            manifest={"name": "t", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="print('ok')")],
            dependencies=["requests"],
        )

        errors = await service._validate_in_sandbox(files, mock_sandbox)

        assert errors == []

    async def test_validate_catches_syntax_error(
        self,
        service: SkillCreatorService,
        mock_sandbox: AsyncMock,
    ) -> None:
        files = SkillGeneratedFiles(
            skill_md="---\nname: t\n---\n# T",
            manifest={"name": "t", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="def f(\n  broken")],
            dependencies=[],
        )

        errors = await service._validate_in_sandbox(files, mock_sandbox)

        assert errors
        assert "语法错误" in errors[0] or "SyntaxError" in errors[0]


class TestFullPipeline:
    async def test_create_yields_progress_events(
        self,
        service: SkillCreatorService,
        mock_llm: AsyncMock,
        mock_sandbox: AsyncMock,
    ) -> None:
        analyze_response = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "skill_name": "test-skill",
                    "description": "测试",
                    "tools": [{"name": "run", "description": "执行", "parameters": []}],
                    "search_keywords": ["test python"],
                    "estimated_deps": [],
                }
            ),
        }
        generate_response = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "skill_md": "---\nname: test-skill\n---\n# Test",
                    "manifest": {
                        "name": "test-skill",
                        "slug": "test-skill",
                        "version": "0.1.0",
                        "description": "测试",
                        "runtime_type": "native",
                        "tools": [
                            {
                                "name": "run",
                                "description": "执行",
                                "parameters": {},
                                "required": [],
                                "entry": {"command": "python bundle/run.py"},
                            }
                        ],
                    },
                    "scripts": [{"path": "bundle/run.py", "content": "print('ok')"}],
                    "dependencies": [],
                }
            ),
        }
        mock_llm.invoke.side_effect = [analyze_response, generate_response]

        events = []
        async for event in service.create(
            description="创建一个测试 skill",
            sandbox=mock_sandbox,
            installed_by="user-1",
        ):
            events.append(event)

        progress_steps = [
            event.step
            for event in events
            if isinstance(event, SkillCreationProgress)
        ]
        results = [
            event
            for event in events
            if isinstance(event, SkillCreationResult)
        ]

        assert "analyzing" in progress_steps
        assert "researching" in progress_steps
        assert "generating" in progress_steps
        assert "validating" in progress_steps
        assert "installing" in progress_steps
        assert len(results) == 1
        assert results[0].skill_name == "test-skill"
