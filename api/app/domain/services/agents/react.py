import logging
from typing import Any, AsyncGenerator, Dict

from app.domain.models.event import (
    BaseEvent,
    ControlAction,
    ControlEvent,
    ControlScope,
    ControlSource,
    ErrorEvent,
    MessageEvent,
    StepEvent,
    StepEventStatus,
    ToolEvent,
    ToolEventStatus,
    WaitEvent,
)
from app.domain.models.file import File
from app.domain.models.message import Message
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.models.tool_result import ToolResult
from app.domain.services.prompts.react import (
    EXECUTION_PROMPT,
    REACT_SYSTEM_PROMPT,
    SUMMARIZE_PROMPT,
)
from app.domain.services.prompts.system import SYSTEM_PROMPT

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """基于ReAct架构的执行Agent"""

    name: str = "react"
    _system_prompt: str = SYSTEM_PROMPT + REACT_SYSTEM_PROMPT
    _format: str = (
        "json_object"  # format控制的是content、工具调用控制的是tool_calls两者不冲突
    )
    _step_tool_attempt_rounds: int = 0
    _step_failed_tool_calls: int = 0

    def _intercept_tool_call(
        self, function_name: str, function_args: Dict[str, Any]
    ) -> ToolResult | None:
        if function_name != "message_ask_user":
            return None

        required_attempts = (
            self._agent_config.skill_selection.ask_user_min_attempt_rounds_per_step
        )
        effective_attempt_score = max(
            self._step_tool_attempt_rounds,
            self._step_failed_tool_calls,
        )
        if effective_attempt_score >= required_attempts:
            return None

        return ToolResult(
            success=False,
            message="ASK_USER_BLOCKED_BY_POLICY",
            data={
                "code": "ASK_USER_BLOCKED_BY_POLICY",
                "tool_attempt_rounds": self._step_tool_attempt_rounds,
                "failed_tool_calls": self._step_failed_tool_calls,
                "effective_attempt_score": effective_attempt_score,
                "required_attempts": required_attempts,
                "remaining_attempts": max(
                    0, required_attempts - effective_attempt_score
                ),
            },
        )

    def _on_tool_result(self, function_name: str, result: ToolResult) -> None:
        # message_ask_user 不计入尝试/失败计数，避免通过连续被拦截提问绕过门控。
        if function_name == "message_ask_user":
            return
        # 计数粒度：按每个 tool_call 计数（而非每轮LLM回合）；unknown-tool 同样会走这里。
        self._step_tool_attempt_rounds += 1
        if not result.success:
            self._step_failed_tool_calls += 1

    async def execute_step(
        self, plan: Plan, step: Step, message: Message
    ) -> AsyncGenerator[BaseEvent, None]:
        """根据传递的消息+规划+子步骤，执行相应的子步骤"""
        # execute_step 与 Planner step 一一对应；无 Planner 时由 Runner 构造虚拟 step 并管理锁定。
        self._step_tool_attempt_rounds = 0
        self._step_failed_tool_calls = 0

        # 1.根据传递的内容生成执行消息
        query = EXECUTION_PROMPT.format(
            message=message.message,
            attachments="\n".join(message.attachments),
            language=plan.language,
            step=step.description,
        )

        # 2.更新步骤的执行状态为运行中并返回Step事件
        step.status = ExecutionStatus.RUNNING
        yield StepEvent(step=step, status=StepEventStatus.STARTED)

        # 3.调用invoke获取agent返回的事件内容
        async for event in self.invoke(query):
            # 4.判断事件类型执行不同操作
            if isinstance(event, ToolEvent):
                # 5.工具事件需要判断工具的名称是否为message_ask_user
                if event.function_name == "message_ask_user":
                    if event.status == ToolEventStatus.CALLED:
                        result_data = (
                            event.function_result.data
                            if event.function_result and event.function_result.data
                            else {}
                        )
                        blocked_by_policy = (
                            isinstance(result_data, dict)
                            and result_data.get("code") == "ASK_USER_BLOCKED_BY_POLICY"
                        )
                        if blocked_by_policy:
                            logger.info(
                                "message_ask_user 在本 step 内被门控拦截，继续自动执行。"
                            )
                            continue

                        yield MessageEvent(
                            role="assistant",
                            message=event.function_args.get("text", ""),
                        )
                        # 7.根据接管建议分流：none 走等待，shell/browser 走控制请求
                        suggest_takeover = str(
                            event.function_args.get("suggest_user_takeover", "none")
                        ).strip().lower()
                        if suggest_takeover in {
                            ControlScope.SHELL.value,
                            ControlScope.BROWSER.value,
                        }:
                            yield ControlEvent(
                                action=ControlAction.REQUESTED,
                                scope=ControlScope(suggest_takeover),
                                source=ControlSource.AGENT,
                            )
                        else:
                            if suggest_takeover not in {"", "none"}:
                                logger.warning(
                                    "ReActAgent检测到非法suggest_user_takeover值，降级为WaitEvent: %r",
                                    suggest_takeover,
                                )
                            yield WaitEvent()
                        return
                    continue
            elif isinstance(event, MessageEvent):
                # 8.返回消息事件，意味着content有内容，content有内容则代表执行Agent已运行完毕
                step.status = ExecutionStatus.COMPLETED

                # 9.message中输出的数据结构为json，需要提取并解析
                parsed_obj = await self._json_parser.invoke(event.message)
                new_step = None
                if isinstance(parsed_obj, dict):
                    try:
                        new_step = Step.model_validate(parsed_obj)
                    except Exception as parse_error:
                        logger.warning(
                            "ReActAgent步骤结果结构化解析失败，降级使用原始文本: %s",
                            parse_error,
                        )
                else:
                    logger.warning(
                        "ReActAgent步骤结果并非字典，降级使用原始文本: %r",
                        parsed_obj,
                    )

                # 10.更新子步骤的数据（解析失败时降级为原始文本，避免中断整条任务）
                if new_step:
                    step.success = new_step.success
                    step.result = new_step.result
                    step.attachments = new_step.attachments
                else:
                    step.success = bool((event.message or "").strip())
                    step.result = event.message
                    step.attachments = []

                # 11.返回步骤完成事件
                yield StepEvent(step=step, status=StepEventStatus.COMPLETED)

                # 12.如果子步骤拿到了结果，还需要返回一段消息给用户(将结果返回给用户)
                if step.result:
                    yield MessageEvent(role="assistant", message=step.result)
                continue
            elif isinstance(event, ErrorEvent):
                # 13.错误事件更新步骤的状态
                step.status = ExecutionStatus.FAILED
                step.error = event.error

                # 14.返回子步骤对应事件
                yield StepEvent(step=step, status=StepEventStatus.FAILED)
                # 15.本步骤执行失败，直接结束当前步骤，避免后续被误标记为completed
                return

            # 15.其他场景将事件直接返回
            yield event

        # 16.循环迭代完成后，只有未失败的步骤才标记完成
        if step.status != ExecutionStatus.FAILED:
            step.status = ExecutionStatus.COMPLETED

    async def summarize(self) -> AsyncGenerator[BaseEvent, None]:
        """调用Agent汇总历史的消息并生成最终回复+附件"""
        # 1.构建请求query
        query = SUMMARIZE_PROMPT

        # 2.调用invoke方法获取Agent生成的事件
        async for event in self.invoke(query):
            # 3.判断事件类型是否为消息事件，如果是则表示Agent结构化生成汇总内容
            if isinstance(event, MessageEvent):
                # 4.记录日志并解析输出内容
                logger.info(f"执行Agent生成汇总内容: {event.message}")
                parsed_obj = await self._json_parser.invoke(event.message)

                # 5.将解析数据转换为Message对象
                message = Message.model_validate(parsed_obj)

                # 6.提取消息中的附件信息
                attachments = [
                    File(filepath=filepath) for filepath in message.attachments
                ]

                # 7.返回消息事件并将消息+附件进行相应
                yield MessageEvent(
                    role="assistant",
                    message=message.message,
                    attachments=attachments,
                )
            else:
                # 8.其他事件则直接返回
                yield event
