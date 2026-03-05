from app.domain.models.skill_creator import (
    ScriptFile,
    SkillBlueprint,
    SkillCreationProgress,
    SkillGeneratedFiles,
    ToolDef,
    ToolParamDef,
)


class TestSkillBlueprint:
    def test_create_blueprint_with_required_fields(self) -> None:
        tool = ToolDef(
            name="download_video",
            description="下载B站视频",
            parameters=[
                ToolParamDef(
                    name="url",
                    type="string",
                    description="视频URL",
                    required=True,
                )
            ],
        )
        blueprint = SkillBlueprint(
            skill_name="bilibili-downloader",
            description="下载B站视频并转录",
            tools=[tool],
            search_keywords=["bilibili download python", "yt-dlp"],
            estimated_deps=["yt-dlp"],
        )

        assert blueprint.skill_name == "bilibili-downloader"
        assert len(blueprint.tools) == 1
        assert blueprint.tools[0].parameters[0].required is True

    def test_blueprint_slug_normalization(self) -> None:
        blueprint = SkillBlueprint(
            skill_name="My Cool Skill!",
            description="test",
            tools=[],
            search_keywords=[],
            estimated_deps=[],
        )
        assert blueprint.normalized_slug == "my-cool-skill"

    def test_blueprint_accepts_parameters_dict_and_required_list(self) -> None:
        blueprint = SkillBlueprint.model_validate(
            {
                "skill_name": "video-summary",
                "description": "下载并总结视频",
                "tools": [
                    {
                        "name": "download_video",
                        "description": "下载视频",
                        "parameters": {
                            "url": {
                                "type": "string",
                                "description": "视频地址",
                            },
                            "output_dir": {
                                "type": "string",
                                "description": "输出目录",
                                "default": "./videos",
                            },
                        },
                        "required": ["url"],
                    }
                ],
                "search_keywords": ["yt-dlp python"],
                "estimated_deps": ["yt-dlp"],
            }
        )

        assert len(blueprint.tools) == 1
        assert len(blueprint.tools[0].parameters) == 2
        assert blueprint.tools[0].parameters[0].name == "url"
        assert blueprint.tools[0].parameters[0].required is True


class TestSkillGeneratedFiles:
    def test_generated_files_structure(self) -> None:
        files = SkillGeneratedFiles(
            skill_md="---\nname: test\n---\n# Test",
            manifest={"name": "test", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="print('hello')")],
            dependencies=["requests"],
        )

        assert len(files.scripts) == 1
        assert files.dependencies == ["requests"]


class TestSkillCreationProgress:
    def test_progress_serialization(self) -> None:
        progress = SkillCreationProgress(step="analyzing", message="正在分析需求...")
        data = progress.model_dump()
        assert data["step"] == "analyzing"
        assert "message" in data
