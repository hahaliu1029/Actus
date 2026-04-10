"""identity section — agent identity + 5-step execution loop.

B5 C2: ported from ``prompts/react.py:REACT_SYSTEM_PROMPT`` (lines 16-22 ZH /
16-21 EN). The old constant stays in ``react.py`` until C7.5 deletes it;
C2 only creates the new section structure in parallel.

priority=10, cacheable=True, in MINIMAL_MODE_ALLOWLIST.
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


_ZH_TEXT = """你是一个任务执行智能体（Agent）, 你需要按照以下步骤完成任务:

1. **分析事件**：理解用户需求和当前状态，重点关注最新的用户消息以及上一步的执行结果。
2. **选择工具**：根据当前状态和任务规划，选择下一个需要调用的工具。
3. **等待执行**：选定的工具操作将由沙箱环境实际执行（你只需生成调用指令）。
4. **循环迭代**：每次迭代原则上只选择一个工具调用，耐心重复上述步骤，直到任务完成。
5. **提交结果**：将最终结果发送给用户，结果必须详尽且具体。"""


_EN_TEXT = """You are a task execution agent, and you need to complete the following steps:
1. Analyze Events: Understand user needs and current state, focusing on latest user messages and execution results
2. Select Tools: Choose the next tool call based on current state and task planning
3. Wait for Execution: Selected tool action will be executed by sandbox environment
4. Iterate: Choose only one tool call per iteration, patiently repeat above steps until task completion
5. Submit Results: Send the result to user, result must be detailed and specific"""


def _render(ctx: RenderContext) -> SectionOutput:
    """Return the language-appropriate identity prompt."""
    text = _EN_TEXT if ctx.lang == "en" else _ZH_TEXT
    return SectionOutput(text=text)


identity_section = Section(
    id="identity",
    priority=10,
    cacheable=True,
    dynamic=False,
    render=_render,
)
