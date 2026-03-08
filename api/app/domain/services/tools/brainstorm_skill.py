"""Brainstorm Skill 工具 — 生成 Skill 蓝图预览。"""

from __future__ import annotations

import json
import logging

from app.domain.models.skill_creator import SkillBlueprint
from app.domain.models.tool_result import ToolResult

from .base import BaseTool, tool

logger = logging.getLogger(__name__)


def _format_blueprint(bp: SkillBlueprint) -> str:
    """将 SkillBlueprint 格式化为用户友好的预览文本。"""
    lines = [
        "Skill 蓝图预览",
        "",
        f"名称: {bp.skill_name}",
        f"描述: {bp.description}",
        "",
        "工具列表:",
    ]
    for i, t in enumerate(bp.tools, 1):
        params = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else " = ?")
            for p in t.parameters
        )
        lines.append(f"  {i}. {t.name}({params}) — {t.description}")

    if bp.estimated_deps:
        lines.append("")
        lines.append(f"预计依赖: {', '.join(bp.estimated_deps)}")

    return "\n".join(lines)


class BrainstormSkillTool(BaseTool):
    """生成 Skill 蓝图预览，供用户确认后再正式创建。"""

    name: str = "skill_brainstormer"

    def __init__(self, skill_creator_service) -> None:
        super().__init__()
        self._creator = skill_creator_service

    @tool(
        name="brainstorm_skill",
        description=(
            "根据需求描述生成 Skill 蓝图预览（名称、工具列表、参数、依赖），"
            "供用户确认后再正式创建。"
        ),
        parameters={
            "description": {
                "type": "string",
                "description": "经过澄清后的完整需求描述",
            }
        },
        required=["description"],
    )
    async def brainstorm_skill(self, description: str) -> ToolResult:
        try:
            blueprint = await self._creator.analyze(description)
            preview = _format_blueprint(blueprint)
            payload = blueprint.model_dump(mode="json")
            return ToolResult(
                success=True,
                message=preview,
                data={
                    **payload,
                    "blueprint": payload,
                    "blueprint_json": json.dumps(payload, ensure_ascii=False),
                },
            )
        except Exception as exc:
            logger.exception("蓝图生成异常: %s", exc)
            return ToolResult(success=False, message=f"蓝图生成异常: {exc}")
