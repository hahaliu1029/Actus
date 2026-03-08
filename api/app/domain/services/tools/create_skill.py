"""Skill 生成与安装工具。"""

from __future__ import annotations

import json
import logging

from app.domain.external.sandbox import Sandbox
from app.domain.models.skill_creator import (
    SkillBlueprint,
    SkillCreationProgress,
    SkillGeneratedFiles,
)
from app.domain.models.tool_result import ToolResult

from .base import BaseTool, tool

logger = logging.getLogger(__name__)


class CreateSkillTool(BaseTool):
    """Skill 生成（generate_skill）与安装（install_skill）工具集。"""

    name: str = "skill_creator"

    def __init__(self, skill_creator_service, sandbox: Sandbox, user_id: str = "") -> None:
        super().__init__()
        self._creator = skill_creator_service
        self._sandbox = sandbox
        self._user_id = user_id

    @tool(
        name="generate_skill",
        description=(
            "生成 Skill 代码并在沙箱验证。返回生成结果和验证状态，不自动安装。"
            "用户确认后再调用 install_skill 完成安装。"
        ),
        parameters={
            "description": {
                "type": "string",
                "description": "用户对目标 Skill 的完整需求描述",
            },
            "blueprint": {
                "type": "object",
                "description": "（可选）brainstorm_skill 返回的 data.blueprint 对象，传入后跳过分析步骤",
            },
            "blueprint_json": {
                "type": "string",
                "description": "（可选）brainstorm_skill 返回的蓝图 JSON 字符串，传入后跳过分析步骤。优先使用 blueprint 参数",
            },
        },
        required=["description"],
    )
    async def generate_skill(
        self,
        description: str,
        blueprint: dict | None = None,
        blueprint_json: str = "",
    ) -> ToolResult:
        progress_log: list[str] = []
        generated_files: SkillGeneratedFiles | None = None
        last_error: str | None = None

        try:
            parsed_blueprint: SkillBlueprint | None = None
            if blueprint is not None:
                parsed_blueprint = SkillBlueprint.model_validate(blueprint)
            elif blueprint_json:
                try:
                    parsed_blueprint = SkillBlueprint.model_validate_json(blueprint_json)
                except Exception:
                    parsed_blueprint = SkillBlueprint.model_validate(
                        json.loads(blueprint_json)
                    )

            async for event in self._creator.generate(
                description=description,
                sandbox=self._sandbox,
                blueprint=parsed_blueprint,
            ):
                if isinstance(event, SkillGeneratedFiles):
                    generated_files = event
                elif isinstance(event, SkillCreationProgress):
                    progress_log.append(f"[{event.step}] {event.message}")
                    if event.detail:
                        last_error = event.detail
                    logger.info("generate_skill 进度: step=%s message=%s", event.step, event.message)

            if generated_files is not None:
                tool_names = [
                    str(t.get("name") or "")
                    for t in generated_files.manifest.get("tools", [])
                    if isinstance(t, dict)
                ]
                return ToolResult(
                    success=True,
                    message=f"Skill 生成并验证通过，包含工具: {', '.join(tool_names)}",
                    data={
                        "tools": tool_names,
                        "dependencies": generated_files.dependencies,
                        "scripts_count": len(generated_files.scripts),
                        "skill_data": generated_files.model_dump_json(),
                        "progress_log": progress_log,
                    },
                )

            return ToolResult(
                success=False,
                message=f"Skill 生成失败: {progress_log[-1] if progress_log else '未知错误'}",
                data={
                    "validation_errors": last_error or (progress_log[-1] if progress_log else ""),
                    "progress_log": progress_log,
                },
            )
        except Exception as exc:
            logger.exception("generate_skill 异常: %s", exc)
            return ToolResult(success=False, message=f"Skill 生成异常: {exc}")

    @tool(
        name="install_skill",
        description="安装已生成并验证通过的 Skill。传入 generate_skill 返回的 skill_data。",
        parameters={
            "skill_data": {
                "type": "string",
                "description": "generate_skill 返回的 data.skill_data JSON 字符串",
            },
        },
        required=["skill_data"],
    )
    async def install_skill(self, skill_data: str) -> ToolResult:
        try:
            files = SkillGeneratedFiles.model_validate_json(skill_data)
        except Exception as exc:
            return ToolResult(success=False, message=f"skill_data 解析失败: {exc}")

        try:
            result = await self._creator.install(files, installed_by=self._user_id)
            return ToolResult(
                success=True,
                message=result.summary,
                data=result.model_dump(),
            )
        except Exception as exc:
            logger.exception("install_skill 异常: %s", exc)
            return ToolResult(success=False, message=f"Skill 安装异常: {exc}")
