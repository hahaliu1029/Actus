import logging
import re
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
from app.domain.models.skill_creation_state import SkillCreationState
from app.domain.models.tool_result import ToolResult
from app.domain.services.prompts.react import (
    EXECUTION_PROMPT,
    REACT_SYSTEM_PROMPT,
    SUMMARIZE_PROMPT,
)
from app.domain.services.prompts.system import SYSTEM_PROMPT

from .base import BaseAgent

logger = logging.getLogger(__name__)

_SKIP_BLUEPRINT_CONFIRM_PATTERN = re.compile(
    r"(开始吧|直接创建|直接生成)", re.IGNORECASE
)


class ReActAgent(BaseAgent):
    """基于ReAct架构的执行Agent"""

    name: str = "react"
    _system_prompt: str = SYSTEM_PROMPT + REACT_SYSTEM_PROMPT
    _format: str = (
        "json_object"  # format控制的是content、工具调用控制的是tool_calls两者不冲突
    )
    _step_tool_attempt_rounds: int = 0
    _step_failed_tool_calls: int = 0
    _step_ask_user_soft_hint_count: int = 0
    _confirmed_tool_names: set = set()  # 已被用户确认的工具，跳过后续确认拦截

    def _get_skill_creation_resume_allowed_tools(self) -> set[str] | None:
        if "install" in self._skill_creation_approved_actions:
            return {"install_skill"}
        if "generate" in self._skill_creation_approved_actions:
            return {"generate_skill"}
        return None

    def _filter_tools_for_skill_creation_resume(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        allowed_tools = self._get_skill_creation_resume_allowed_tools()
        if not allowed_tools:
            return tools

        filtered_tools: list[dict[str, Any]] = []
        for schema in tools:
            function_info = schema.get("function", {})
            function_name = function_info.get("name")
            if not isinstance(function_name, str) or not function_name:
                continue
            if function_name.startswith("message_") or function_name in allowed_tools:
                filtered_tools.append(schema)
        return filtered_tools

    def _get_available_tools(self) -> list[dict[str, Any]]:
        return self._filter_tools_for_skill_creation_resume(
            super()._get_available_tools()
        )

    def _build_effective_system_prompt(self) -> str:
        prompt = super()._build_effective_system_prompt()
        allowed_tools = self._get_skill_creation_resume_allowed_tools()
        if not allowed_tools:
            return prompt

        stage = (
            "安装确认后的恢复阶段"
            if "install" in self._skill_creation_approved_actions
            else "蓝图确认后的恢复阶段"
        )
        tool_names = ", ".join(
            ["message_notify_user", "message_ask_user", *sorted(allowed_tools)]
        )
        gate_notice = (
            "\n\n## Skill Creation Tool Gate\n"
            f"- 当前处于{stage}。\n"
            f"- 当前仅可调用工具: {tool_names}\n"
            "- 其他 Skill Creator 工具当前不可调用，即使在通用工具摘要中出现也不要调用。"
        )
        return prompt + gate_notice

    def _get_tool_choice(self) -> str | None:
        if self._get_skill_creation_resume_allowed_tools():
            return "required"
        return super()._get_tool_choice()

    def _build_execution_query(self, plan: Plan, step: Step, message: Message) -> str:
        query = EXECUTION_PROMPT.format(
            message=message.message,
            attachments="\n".join(message.attachments),
            language=plan.language,
            step=step.description,
        )

        if "install" in self._skill_creation_approved_actions:
            query += (
                "\n\n恢复提示：用户刚刚已经确认安装。"
                "不要重新调用 brainstorm_skill 或 generate_skill，"
                "请直接调用 install_skill，并优先使用上一轮 generate_skill "
                "已返回并保存在上下文中的 skill_data。"
            )
        elif "generate" in self._skill_creation_approved_actions:
            query += (
                "\n\n恢复提示：用户刚刚已经确认蓝图。"
                "不要再次调用 brainstorm_skill，"
                "请基于上一轮已确认并保存在上下文中的 blueprint/blueprint_json 继续，"
                "优先调用 generate_skill。"
            )

        return query

    def _intercept_tool_call(
        self, function_name: str, function_args: Dict[str, Any]
    ) -> ToolResult | None:
        # 第一层：工具级强制确认（已确认的工具跳过）
        if self._is_tool_confirmation_required(function_name):
            if function_name in self._confirmed_tool_names:
                self._confirmed_tool_names.discard(function_name)
                return None
            return ToolResult(
                success=False,
                message="TOOL_CONFIRMATION_REQUIRED",
                data={
                    "code": "TOOL_CONFIRMATION_REQUIRED",
                    "tool_name": function_name,
                    "function_args": function_args,
                },
            )

        # Skill Creator 专用确认门控
        # 放行令牌仅在此处做"是否放行"判定，不主动消费；
        # 令牌会在 execute_step 中工具成功后统一 discard，
        # 从而保证工具失败重试期间 _filter_tools_for_skill_creation_resume 持续生效。
        if (
            function_name == "generate_skill"
            and "generate" in self._skill_creation_approved_actions
        ):
            return None
        if (
            function_name == "install_skill"
            and "install" in self._skill_creation_approved_actions
        ):
            return None
        if (
            function_name == "generate_skill"
            and self._skill_creation_state is not None
            and self._skill_creation_state.pending_action == "generate"
        ):
            return ToolResult(
                success=False,
                message="SKILL_CONFIRMATION_REQUIRED",
                data={
                    "code": "SKILL_CONFIRMATION_REQUIRED",
                    "pending_action": "generate",
                    "tool_name": function_name,
                },
            )
        if function_name == "install_skill":
            pending_action = (
                self._skill_creation_state.pending_action
                if self._skill_creation_state is not None
                else "install"
            )
            return ToolResult(
                success=False,
                message="SKILL_CONFIRMATION_REQUIRED",
                data={
                    "code": "SKILL_CONFIRMATION_REQUIRED",
                    "pending_action": pending_action,
                    "tool_name": function_name,
                },
            )

        # 第二层：message_ask_user 软引导门控
        if function_name != "message_ask_user":
            return None

        # 如果已经收到过一次 SOFT_HINT，第二次直接放行
        if self._step_ask_user_soft_hint_count >= 1:
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

        # 软引导：返回 success=True + SOFT_HINT
        self._step_ask_user_soft_hint_count += 1
        return ToolResult(
            success=True,
            message="SOFT_HINT: 建议先尝试使用工具自动解决。如确实需要用户介入，请再次调用 message_ask_user。",
            data={
                "code": "ASK_USER_SOFT_HINT",
                "tool_attempt_rounds": self._step_tool_attempt_rounds,
                "effective_attempt_score": effective_attempt_score,
                "required_attempts": required_attempts,
            },
        )

    def _on_tool_result(self, function_name: str, result: ToolResult) -> None:
        # message_ask_user 不计入尝试/失败计数，避免通过连续被拦截提问绕过门控。
        if function_name == "message_ask_user":
            return
        # require_confirmation 拦截的工具不计入
        if (
            isinstance(result.data, dict)
            and result.data.get("code") == "TOOL_CONFIRMATION_REQUIRED"
        ):
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
        self._step_ask_user_soft_hint_count = 0
        await self._ensure_skill_creation_state()

        # 1.根据传递的内容生成执行消息
        query = self._build_execution_query(plan, step, message)

        # 2.更新步骤的执行状态为运行中并返回Step事件
        step.status = ExecutionStatus.RUNNING
        yield StepEvent(step=step, status=StepEventStatus.STARTED)

        # 3.调用invoke获取agent返回的事件内容
        async for event in self.invoke(query):
            # 4.判断事件类型执行不同操作
            if isinstance(event, ToolEvent):
                # 5.处理工具级强制确认和 message_ask_user
                if event.status == ToolEventStatus.CALLED:
                    result_data = (
                        event.function_result.data
                        if event.function_result and event.function_result.data
                        else {}
                    )
                    # 5a.工具级强制确认：向用户展示确认提示并暂停
                    if (
                        isinstance(result_data, dict)
                        and result_data.get("code") == "TOOL_CONFIRMATION_REQUIRED"
                    ):
                        tool_name = result_data.get("tool_name", event.function_name)
                        func_args = result_data.get("function_args", {})
                        args_summary = ", ".join(
                            f"{k}={v!r}" for k, v in func_args.items()
                        )
                        yield MessageEvent(
                            role="assistant",
                            message=f"即将执行 `{tool_name}({args_summary})`，是否继续？",
                        )
                        yield WaitEvent()
                        return
                    if (
                        isinstance(result_data, dict)
                        and result_data.get("code") == "SKILL_CONFIRMATION_REQUIRED"
                    ):
                        pending_action = result_data.get("pending_action", "generate")
                        if pending_action == "install":
                            prompt = "请先确认是否安装生成好的 Skill。"
                        else:
                            prompt = "请先确认蓝图是否符合预期，再继续生成。"
                        # 持久化等待状态，确保 roll_back 能正确处理用户确认
                        self._skill_creation_state = SkillCreationState(
                            pending_action=pending_action,
                            approval_status="pending",
                            last_tool_name=event.function_name,
                            last_tool_call_id=event.tool_call_id,
                            saved_tool_result_json=event.function_result.model_dump_json(),
                            skill_data=(
                                event.function_args.get("skill_data", "")
                                if pending_action == "install"
                                and isinstance(event.function_args, dict)
                                else ""
                            ),
                        )
                        await self._persist_skill_creation_state()
                        logger.info(
                            "Skill 创建门控拦截: action=%s tool=%s",
                            pending_action,
                            event.function_name,
                        )
                        yield MessageEvent(role="assistant", message=prompt)
                        yield WaitEvent()
                        return

                    if (
                        event.function_name == "brainstorm_skill"
                        and event.function_result
                        and event.function_result.success
                        and not self._should_skip_blueprint_confirm(message.message)
                    ):
                        preview = event.function_result.message or ""
                        self._skill_creation_state = SkillCreationState(
                            pending_action="generate",
                            approval_status="pending",
                            last_tool_name="brainstorm_skill",
                            last_tool_call_id=event.tool_call_id,
                            saved_tool_result_json=event.function_result.model_dump_json(),
                            blueprint=(
                                result_data.get("blueprint")
                                if isinstance(result_data, dict)
                                else None
                            ),
                            blueprint_json=(
                                result_data.get("blueprint_json", "")
                                if isinstance(result_data, dict)
                                else ""
                            ),
                        )
                        await self._persist_skill_creation_state()
                        yield MessageEvent(
                            role="assistant",
                            message=f"{preview}\n\n请确认蓝图是否符合预期。",
                        )
                        yield WaitEvent()
                        return

                    if (
                        event.function_name == "generate_skill"
                        and event.function_result
                    ):
                        if not event.function_result.success:
                            logger.warning(
                                "generate_skill 执行失败，跳过安装确认门控: %s",
                                event.function_result.message,
                            )
                    if (
                        event.function_name == "generate_skill"
                        and event.function_result
                        and event.function_result.success
                    ):
                        # generate 成功，正式消费放行令牌
                        self._skill_creation_approved_actions.discard("generate")
                        logger.info(
                            "generate_skill 成功，进入安装确认等待: call_id=%s",
                            event.tool_call_id,
                        )
                        self._skill_creation_state = SkillCreationState(
                            pending_action="install",
                            approval_status="pending",
                            last_tool_name="generate_skill",
                            last_tool_call_id=event.tool_call_id,
                            saved_tool_result_json=event.function_result.model_dump_json(),
                            skill_data=(
                                result_data.get("skill_data", "")
                                if isinstance(result_data, dict)
                                else ""
                            ),
                        )
                        await self._persist_skill_creation_state()
                        yield MessageEvent(
                            role="assistant",
                            message="Skill 代码生成并验证通过，是否确认安装？",
                        )
                        yield WaitEvent()
                        return

                    if (
                        event.function_name == "install_skill"
                        and event.function_result
                        and event.function_result.success
                    ):
                        if self._skill_creation_state is not None:
                            await self._clear_skill_creation_state()
                        # 消费残留的 install 令牌（防止跨轮泄漏）
                        self._skill_creation_approved_actions.discard("install")

                # 6.message_ask_user 专用处理
                if event.function_name == "message_ask_user":
                    if event.status == ToolEventStatus.CALLED:
                        result_data = (
                            event.function_result.data
                            if event.function_result and event.function_result.data
                            else {}
                        )
                        # 6a.软引导提示：LLM 继续自动执行
                        soft_hint = (
                            isinstance(result_data, dict)
                            and result_data.get("code") == "ASK_USER_SOFT_HINT"
                        )
                        if soft_hint:
                            logger.info(
                                "message_ask_user 收到软引导提示，LLM 将继续自动执行。"
                            )
                            continue

                        yield MessageEvent(
                            role="assistant",
                            message=event.function_args.get("text", ""),
                        )
                        # 7.根据接管建议分流：none 走等待，shell/browser 走控制请求
                        suggest_takeover = (
                            str(
                                event.function_args.get("suggest_user_takeover", "none")
                            )
                            .strip()
                            .lower()
                        )
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

    @staticmethod
    def _should_skip_blueprint_confirm(user_message: str) -> bool:
        """用户明确要求直接创建时，允许跳过蓝图确认。"""
        normalized = BaseAgent.normalize_skill_creation_reply(user_message)
        return bool(_SKIP_BLUEPRINT_CONFIRM_PATTERN.search(normalized))

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
