"""Create Skill 工具。"""

from __future__ import annotations

import logging

from app.domain.external.sandbox import Sandbox
from app.domain.models.skill_creator import SkillCreationProgress, SkillCreationResult
from app.domain.models.tool_result import ToolResult

from .base import BaseTool, tool

logger = logging.getLogger(__name__)


class CreateSkillTool(BaseTool):
    """在 Agent 对话中触发 Skill Creator 流水线。"""

    name: str = "skill_creator"

    def __init__(self, skill_creator_service, sandbox: Sandbox, user_id: str = "") -> None:
        super().__init__()
        self._creator = skill_creator_service
        self._sandbox = sandbox
        self._user_id = user_id

    @tool(
        name="create_skill",
        description=(
            "根据自然语言描述自动创建 Native Skill。"
            "当用户请求创建/制作/开发 Skill 时调用。"
        ),
        parameters={
            "description": {
                "type": "string",
                "description": "用户对目标 Skill 的自然语言描述",
            }
        },
        required=["description"],
    )
    async def create_skill(self, description: str) -> ToolResult:
        progress_messages: list[str] = []
        final_result: SkillCreationResult | None = None

        try:
            async for event in self._creator.create(
                description=description,
                sandbox=self._sandbox,
                installed_by=self._user_id,
            ):
                if isinstance(event, SkillCreationResult):
                    final_result = event
                elif isinstance(event, SkillCreationProgress):
                    progress_messages.append(f"[{event.step}] {event.message}")
                    logger.info(
                        "CreateSkillTool 进度: step=%s message=%s",
                        event.step,
                        event.message,
                    )

            if final_result:
                return ToolResult(
                    success=True,
                    message=final_result.summary,
                    data=final_result.model_dump(),
                )

            last_message = progress_messages[-1] if progress_messages else "未知错误"
            return ToolResult(success=False, message=f"Skill 创建失败: {last_message}")
        except Exception as exc:
            logger.exception("Skill 创建异常: %s", exc)
            return ToolResult(success=False, message=f"Skill 创建异常: {exc}")
