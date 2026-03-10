"""LangChain tool wrappers for Skill creation tools.

Wraps BrainstormSkillTool and CreateSkillTool (legacy BaseTool instances)
as LangChain StructuredTool functions so they can be used in the LangGraph
react_graph.

Usage:
    tools = create_skill_langchain_tools(
        brainstorm_skill_tool=brainstorm_tool,
        create_skill_tool=create_tool,
    )
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import StructuredTool, tool as lc_tool

from app.domain.services.tools.base import BaseTool


def create_skill_langchain_tools(
    brainstorm_skill_tool: BaseTool | None = None,
    create_skill_tool: BaseTool | None = None,
) -> list[StructuredTool]:
    """Create LangChain wrappers for skill creation tools.

    Returns an empty list if the corresponding BaseTool instance is None.
    """
    tools: list[StructuredTool] = []

    if brainstorm_skill_tool is not None:

        @lc_tool
        async def brainstorm_skill(description: str) -> str:
            """根据需求描述生成 Skill 蓝图预览（名称、工具列表、参数、依赖），供用户确认后再正式创建。"""
            result = await brainstorm_skill_tool.invoke(
                "brainstorm_skill", description=description,
            )
            return result.model_dump_json()

        tools.append(brainstorm_skill)

    if create_skill_tool is not None:

        @lc_tool
        async def generate_skill(
            description: str,
            blueprint: Optional[dict] = None,
            blueprint_json: Optional[str] = "",
        ) -> str:
            """生成 Skill 代码并在沙箱验证。返回生成结果和验证状态，不自动安装。用户确认后再调用 install_skill 完成安装。"""
            kwargs: dict = {"description": description}
            if blueprint is not None:
                kwargs["blueprint"] = blueprint
            if blueprint_json:
                kwargs["blueprint_json"] = blueprint_json
            result = await create_skill_tool.invoke("generate_skill", **kwargs)
            return result.model_dump_json()

        @lc_tool
        async def install_skill(skill_data: str) -> str:
            """安装已生成并验证通过的 Skill。传入 generate_skill 返回的 data.skill_data JSON 字符串。"""
            result = await create_skill_tool.invoke(
                "install_skill", skill_data=skill_data,
            )
            return result.model_dump_json()

        tools.append(generate_skill)
        tools.append(install_skill)

    return tools
