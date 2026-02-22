import logging
from typing import Any, AsyncGenerator, Optional

from app.domain.models.event import BaseEvent, MessageEvent, PlanEvent, PlanEventStatus
from app.domain.models.message import Message
from app.domain.models.plan import Plan, Step
from app.domain.services.prompts.planner import (
    CREATE_PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    UPDATE_PLAN_PROMPT,
)
from app.domain.services.prompts.system import SYSTEM_PROMPT

from .base import BaseAgent

"""
多Agent系统/flow=PlannerAgent+ReActAgent

顺序:
1. PlannerAgent生成规划;
2. 循环取出规划中的子步骤，让ReActAgent执行，依次迭代;
3. ReActAgent执行完每一个子步骤之后，需要将子步骤结果+Plan传递给PlannerAgent让其更新计划/Plan；
4. 循环取出规划中的子步骤，让ReActAgent执行，依次迭代;
5. ...
6. 直到所有子任务/步骤都完成，这时候将子步骤的所有结果汇总进行总结(ReActAgent);

PlannerAgent:
- 功能: 将用户的需求拆解成多个子任务+根据已完成的子任务更新规划
- 提示词: 创建规划的prompt、更新规划的prompt

ReActAgent:
- 功能: 迭代执行完每一个子任务、汇总所有的子任务进行总结
- 提示词: 执行任务的prompt、汇总总结prompt
"""

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """规划Agent，用于将用户的任务/需求拆解成多个子步骤"""

    name: str = "planner"
    _system_prompt: str = SYSTEM_PROMPT + PLANNER_SYSTEM_PROMPT
    _format: Optional[str] = "json_object"
    _tool_choice: Optional[str] = "none"

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def _build_fallback_plan(self, message: Message, raw_response: str) -> Plan:
        """在规划结构化解析失败时提供最小可执行计划，避免任务直接中断。"""
        user_message = (message.message or "").strip()
        has_attachments = bool(message.attachments)

        if has_attachments:
            step_description = "读取并分析用户上传文件，提取关键信息。"
            if user_message:
                step_description += f" 基于需求“{user_message}”输出可执行工作计划。"
            else:
                step_description += " 输出可执行工作计划。"
        else:
            step_description = user_message or "梳理需求并输出可执行工作计划。"

        fallback_message = (
            (raw_response or "").strip() or "已收到任务，我将先整理需求并给出可执行工作计划。"
        )
        language_probe = f"{user_message}\n{fallback_message}"

        return Plan(
            title="工作计划",
            goal=user_message or "输出可执行工作计划",
            language="zh" if self._contains_cjk(language_probe) else "en",
            message=fallback_message,
            steps=[Step(description=step_description)],
        )

    def _parse_plan_or_fallback(
        self,
        parsed_obj: Any,
        message: Message,
        raw_response: str,
    ) -> Plan:
        """尝试解析结构化计划，失败时降级为兜底计划。"""
        if isinstance(parsed_obj, dict):
            try:
                return Plan.model_validate(parsed_obj)
            except Exception as parse_error:
                logger.warning(
                    "PlannerAgent计划结构化解析失败，降级使用兜底计划: %s",
                    parse_error,
                )
        else:
            logger.warning(
                "PlannerAgent计划结果并非字典，降级使用兜底计划: %r",
                parsed_obj,
            )

        return self._build_fallback_plan(message, raw_response)

    async def create_plan(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """根据用户传递的消息创建计划/规划，迭代返回对应的事件"""
        # 1.根据用户传递的消息生成创建plan的提示词
        query = CREATE_PLAN_PROMPT.format(
            message=message.message,
            attachments="\n".join(message.attachments),
        )

        # 2.调用invoke函数返回迭代事件
        async for event in self.invoke(query):
            # 3.规划智能体因为使用json_object，正常情况下会返回MessageEvent
            if isinstance(event, MessageEvent):
                # 4.记录日志并使用json解析器解析得到对应的数据
                logger.info(f"PlannerAgent生成消息: {event.message}")
                parsed_obj = await self._json_parser.invoke(event.message)

                # 5.将解析对象转换成Plan计划（解析失败时自动降级）
                plan = self._parse_plan_or_fallback(
                    parsed_obj=parsed_obj,
                    message=message,
                    raw_response=event.message,
                )

                # 6.返回PlanEvent表示规划创建成功
                yield PlanEvent(plan=plan, status=PlanEventStatus.CREATED)
            else:
                # 返回不是消息事件的事件
                yield event

    async def update_plan(
        self, plan: Plan, step: Step
    ) -> AsyncGenerator[BaseEvent, None]:
        """根据传递的原始规划+子步骤更新事件"""
        # 1.使用plan+step创建更新Plan提示词
        query = UPDATE_PLAN_PROMPT.format(
            plan=plan.model_dump_json(),
            step=step.model_dump_json(),
        )

        # 2.调用invoke获取对应的事件
        async for event in self.invoke(query):
            # 3.判断规划Agent生成的事件是不是消息事件
            if isinstance(event, MessageEvent):
                # 4.记录日志并解析json
                logger.info(f"PlannerAgent生成消息: {event.message}")
                parsed_obj = await self._json_parser.invoke(event.message)

                # 5.将解析对象转换成Plan（解析失败时沿用当前plan继续执行）
                if isinstance(parsed_obj, dict):
                    try:
                        updated_plan = Plan.model_validate(parsed_obj)
                    except Exception as parse_error:
                        logger.warning(
                            "PlannerAgent更新计划解析失败，沿用原计划继续执行: %s",
                            parse_error,
                        )
                        yield PlanEvent(plan=plan, status=PlanEventStatus.UPDATED)
                        continue
                else:
                    logger.warning(
                        "PlannerAgent更新计划结果并非字典，沿用原计划继续执行: %r",
                        parsed_obj,
                    )
                    yield PlanEvent(plan=plan, status=PlanEventStatus.UPDATED)
                    continue

                # 6.拷贝更新计划中的steps，避免造成数据污染
                new_steps = [Step.model_validate(step) for step in updated_plan.steps]

                # 7.查询旧计划中第一个未完成的计划
                first_pending_index = None
                for idx, step in enumerate(plan.steps):
                    if not step.done:
                        first_pending_index = idx
                        break

                # 8.判断是否有未完成的步骤，如果有则执行更新
                if first_pending_index is not None:
                    # 9.获取历史已完成的子步骤并更新
                    updated_steps = plan.steps[:first_pending_index]
                    updated_steps.extend(new_steps)

                    # 10.更新plan规划
                    plan.steps = updated_steps

                # 11.返回规划更新事件
                yield PlanEvent(plan=plan, status=PlanEventStatus.UPDATED)
            else:
                # 其他事件则直接返回
                yield event
