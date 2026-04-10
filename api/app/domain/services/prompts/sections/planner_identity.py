"""planner_identity section — planner agent identity (5-step plan loop).

B5 C6: ported from ``prompts/planner.py:PLANNER_SYSTEM_PROMPT`` (ZH) and
``prompts/en/planner.py:PLANNER_SYSTEM_PROMPT`` (EN). The legacy constants
stay in place until C7.5 deletes them.

Used by both the planner registry (plan creation) and the updater
registry (plan update) because both main_graph nodes (planner_node and
updater_node) currently use the same system prompt.

priority=10, cacheable=True, in MINIMAL_MODE_ALLOWLIST (planner must
identify itself even in minimal mode for child-agent scenarios).
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


_ZH_TEXT = """你是一个任务规划智能体 (Task Planner Agent), 你需要为任务创建或更新计划:
1. 分析用户的消息并理解用户的需求;
2. 确定完成任务需要使用哪些工具;
3. 根据用户的消息确定工作语言;
4. 生成计划的目标和步骤;"""


_EN_TEXT = """You are a task planner agent, and you need to create or update a plan for the task:
1. Analyze the user's message and understand the user's needs
2. Determine what tools you need to use to complete the task
3. Determine the working language based on the user's message
4. Generate the plan's goal and steps"""


def _render(ctx: RenderContext) -> SectionOutput:
    """Return the language-appropriate planner identity prompt."""
    text = _EN_TEXT if ctx.lang == "en" else _ZH_TEXT
    return SectionOutput(text=text)


planner_identity_section = Section(
    id="planner_identity",
    priority=10,
    cacheable=True,
    dynamic=False,
    render=_render,
)
