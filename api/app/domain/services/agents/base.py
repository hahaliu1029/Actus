import asyncio
import logging
import re
import uuid
from abc import ABC
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.models.app_config import AgentConfig
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.event import (
    BaseEvent,
    ErrorEvent,
    MessageEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.skill_creation_state import SkillCreationState
from app.domain.models.tool_result import ToolResult

# from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.tools.base import BaseTool

logger = logging.getLogger(__name__)

_SKILL_CREATION_TEXT_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
_SKILL_CREATION_AFFIRMATIVE_REPLIES = frozenset(
    {
        "好的",
        "安装吧",
        "嗯",
        "行",
        "ok",
        "yes",
        "对",
        "是的",
        "确定",
        "确认",
        "没问题",
        "可以",
        "继续",
        "开始吧",
        "安装",
        "同意",
        "通过",
        "好",
        "好的 继续生成吧",
        "可以 安装",
        "嗯 开始吧",
    }
)
_SKILL_CREATION_NEGATIVE_REPLIES = frozenset(
    {
        "不安装",
        "取消",
        "不",
        "no",
        "cancel",
        "否",
        "拒绝",
        "不要",
        "不用",
        "先别安装",
        "不要继续",
    }
)
_SKILL_CREATION_AFFIRMATIVE_PATTERNS = (
    "继续生成",
    "开始生成",
    "开始生产",
    "确认安装",
    "确认蓝图",
    "按这个蓝图",
    "就按这个",
    "就这样",
    "符合预期",
    "没问题",
    "可以生成",
    "可以安装",
    "开始安装",
    "继续安装",
    "确认生成",
    "同意蓝图",
    "蓝图没问题",
)


class BaseAgent(ABC):
    """基础Agent智能体"""

    _ANCHOR_PREFIX = "[上下文回顾]"
    name: str = ""  # 智能体名字
    _system_prompt: str = ""  # 系统预设prompt
    _format: Optional[str] = None  # Agent的响应格式
    _retry_interval: float = 1.0  # 重试间隔
    _tool_choice: Optional[str] = None  # 强制选择工具

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        session_id: str,  # 会话id
        # session_repository: SessionRepository,  # 会话数据仓库
        agent_config: AgentConfig,  # Agent配置
        llm: LLM,  # 语言模型协议
        json_parser: JSONParser,  # JSON输出解析器
        tools: List[BaseTool],  # 工具列表
        overflow_config: ContextOverflowConfig | None = None,  # 上下文超限治理配置
    ) -> None:
        """构造函数，完成Agent的初始化"""
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._session_id = session_id
        # self._session_repository = session_repository
        self._agent_config = agent_config
        self._llm = llm
        self._memory: Optional[Memory] = None
        self._json_parser = json_parser
        self._tools = tools
        self._overflow_config = overflow_config or ContextOverflowConfig()
        self._runtime_system_context: str = ""
        self._conversation_summaries: list = []
        self._skill_creation_state: SkillCreationState | None = None
        self._skill_creation_approved_actions: set[str] = set()

    def set_runtime_system_context(self, context: str) -> None:
        """设置运行时系统上下文（如动态激活的 Skill 指南）"""
        self._runtime_system_context = (context or "").strip()

    def set_conversation_summaries(self, summaries: list) -> None:
        """设置历史对话摘要，将在 system prompt 中注入。"""
        self._conversation_summaries = summaries or []

    def _build_effective_system_prompt(self) -> str:
        """构建当前生效的系统提示词（静态 + 运行时上下文 + 历史摘要）"""
        parts = [self._system_prompt]
        if self._runtime_system_context:
            parts.append(self._runtime_system_context)
        if self._conversation_summaries:
            summary_lines = ["## 历史对话摘要"]
            for summary in self._conversation_summaries:
                summary_lines.append(summary.to_prompt_text())
            parts.append("\n".join(summary_lines))
        return "\n\n".join(parts)

    def _ensure_system_message(self) -> None:
        """确保记忆中的首条 system 消息与当前系统提示词一致"""
        expected_system_prompt = self._build_effective_system_prompt()

        if self._memory.empty:
            self._memory.add_message(
                {
                    "role": "system",
                    "content": expected_system_prompt,
                }
            )
            return

        first_message = self._memory.messages[0]
        if first_message.get("role") != "system":
            self._memory.messages.insert(
                0,
                {
                    "role": "system",
                    "content": expected_system_prompt,
                },
            )
            return

        if first_message.get("content") != expected_system_prompt:
            first_message["content"] = expected_system_prompt

    def get_latest_assistant_content(self, max_chars: int = 500) -> str:
        """从 memory 中提取最近一条 assistant 消息的 content。"""
        if self._memory is None:
            return ""

        for message in reversed(self._memory.messages):
            if message.get("role") == "assistant" and message.get("content"):
                content = message["content"]
                return content[:max_chars] if len(content) > max_chars else content
        return ""

    def inject_context_anchor(
        self,
        session_status: str,
        user_message: str,
        original_request: str = "",
        completed_steps: list[str] | None = None,
    ) -> None:
        """注入上下文锚点消息，帮助 LLM 维持多轮对话连贯性。"""
        if self._memory is None:
            return

        for message in self._memory.messages:
            if (
                message.get("role") == "user"
                and message.get("content", "").startswith(self._ANCHOR_PREFIX)
            ):
                return

        if session_status == "waiting":
            content = f"{self._ANCHOR_PREFIX}\n用户回复了你的提问: {user_message}"
        else:
            lines = [self._ANCHOR_PREFIX]
            if original_request:
                lines.append(f"- 用户原始需求：{original_request}")
            if completed_steps:
                lines.append(f"- 已完成步骤：{'；'.join(completed_steps)}")
            lines.append(f"- 当前状态：{session_status}")
            lines.append(f"- 用户新消息：{user_message}")
            content = "\n".join(lines)

        self._memory.add_message({"role": "user", "content": content})

    async def _ensure_memory(self) -> None:
        """确保智能体记忆是存在的"""
        if self._memory is None:
            async with self._uow:
                self._memory = await self._uow.session.get_memory(
                    self._session_id, self.name
                )

    async def _ensure_skill_creation_state(self) -> None:
        """确保 Skill 创建等待状态已加载到运行时缓存。"""
        if self._skill_creation_state is not None:
            return

        async with self._uow:
            get_state = getattr(self._uow.session, "get_skill_creation_state", None)
            if get_state is None:
                self._skill_creation_state = None
                return
            self._skill_creation_state = await get_state(self._session_id)
            if self._skill_creation_state is None:
                logger.warning(
                    "Skill 创建状态加载为 None: agent=%s session=%s",
                    self.name, self._session_id,
                )
            else:
                logger.info(
                    "Skill 创建状态加载成功: agent=%s pending=%s approval=%s skill_data_len=%d",
                    self.name,
                    self._skill_creation_state.pending_action,
                    self._skill_creation_state.approval_status,
                    len(self._skill_creation_state.skill_data),
                )

    async def _persist_skill_creation_state(self) -> None:
        """持久化 Skill 创建等待状态。"""
        if self._skill_creation_state is None:
            return

        async with self._uow:
            save_state = getattr(self._uow.session, "save_skill_creation_state", None)
            if save_state is None:
                return
            await save_state(self._session_id, self._skill_creation_state)

    async def _clear_skill_creation_state(self) -> None:
        """清理 Skill 创建等待状态。"""
        self._skill_creation_state = None

        async with self._uow:
            clear_state = getattr(self._uow.session, "clear_skill_creation_state", None)
            if clear_state is None:
                return
            await clear_state(self._session_id)

    @staticmethod
    def normalize_skill_creation_reply(text: str) -> str:
        """归一化用户确认文本，便于确定性判定。"""
        normalized = _SKILL_CREATION_TEXT_RE.sub(" ", (text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _classify_skill_creation_reply(cls, text: str) -> str:
        normalized = cls.normalize_skill_creation_reply(text)
        if normalized in _SKILL_CREATION_NEGATIVE_REPLIES:
            return "negative"
        if normalized in _SKILL_CREATION_AFFIRMATIVE_REPLIES:
            return "affirmative"
        if any(pattern in normalized for pattern in _SKILL_CREATION_AFFIRMATIVE_PATTERNS):
            return "affirmative"
        return "revise"

    @staticmethod
    def _classify_skill_creation_structured_action(
        pending_action: str | None, action: str | None
    ) -> str | None:
        if not action:
            return None
        if action == "cancel":
            return "negative"
        if action == "revise":
            return "revise"
        if pending_action == "generate" and action == "generate":
            return "affirmative"
        if pending_action == "install" and action == "install":
            return "affirmative"
        return "revise"

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取Agent所有可用的工具列表参数声明/Schema"""
        available_tools = []
        for tool in self._tools:
            available_tools.extend(tool.get_tools())
        return available_tools

    def _get_tool(self, tool_name: str) -> BaseTool:
        """获取对应工具所在的工具集/包"""
        # 1.循环遍历所有工具包
        for tool in self._tools:
            # 2.判断工具包中是否存在该工具
            if tool.has_tool(tool_name):
                return tool

        raise ValueError(f"未知工具: {tool_name}")

    def _is_tool_confirmation_required(self, tool_name: str) -> bool:
        """检查指定工具是否需要用户确认"""
        for tool in self._tools:
            if tool.has_tool(tool_name):
                return tool.get_tool_confirmation_required(tool_name)
        return False

    def _build_unknown_tool_result(self, function_name: str) -> ToolResult:
        """构建未知工具结果并回灌给 LLM，避免直接中断执行。"""
        available_names: list[str] = []
        for tool_schema in self._get_available_tools():
            if not isinstance(tool_schema, dict):
                continue
            function_info = tool_schema.get("function")
            if not isinstance(function_info, dict):
                continue
            tool_name = function_info.get("name")
            if isinstance(tool_name, str) and tool_name:
                available_names.append(tool_name)

        candidate_limit = self._agent_config.skill_selection.unknown_tool_candidate_limit
        candidates = list(dict.fromkeys(available_names))[:candidate_limit]

        return ToolResult(
            success=False,
            message=f"UNKNOWN_TOOL: {function_name}",
            data={
                "code": "UNKNOWN_TOOL",
                "tool_name": function_name,
                "candidates": candidates,
            },
        )

    def _get_tool_choice(self) -> Optional[str]:
        """返回当前轮调用LLM时应使用的 tool_choice。"""
        return self._tool_choice

    def _intercept_tool_call(
        self, function_name: str, function_args: Dict[str, Any]
    ) -> ToolResult | None:
        """工具调用拦截器，默认不拦截。"""
        return None

    def _on_tool_result(self, function_name: str, result: ToolResult) -> None:
        """工具结果回调，子类可覆写统计每 step 的调用状态。"""
        return None

    async def _invoke_llm(
        self, messages: List[Dict[str, Any]], format: Optional[str] = None
    ) -> Dict[str, Any]:
        """调用语言模型并处理记忆内容"""
        # 1.将消息添加到记忆中
        await self._add_to_memory(messages)

        # 2.组装语言模型的响应格式
        response_format = {"type": format} if format else None

        # 3.循环向LLM发起提问直到最大重试次数
        last_error = "未知错误"
        for _ in range(self._agent_config.max_retries):
            try:
                # 4.调用语言模型获取响应内容
                message = await self._llm.invoke(
                    messages=self._memory.get_messages(),
                    tools=self._get_available_tools(),
                    response_format=response_format,
                    tool_choice=self._get_tool_choice(),
                )
                if not isinstance(message, dict):
                    raise ValueError("LLM返回的消息格式非法")

                # 5.处理AI响应内容避免空回复
                if message.get("role") == "assistant":
                    if not message.get("content") and not message.get("tool_calls"):
                        logger.warning(f"LLM回复了空内容，执行重试")
                        await self._add_to_memory(
                            [
                                {"role": "assistant", "content": ""},
                                {"role": "user", "content": "AI无响应内容，请继续。"},
                            ]
                        )
                        await asyncio.sleep(self._retry_interval)
                        continue

                    # 6.取出非空消息并处理工具调用(兼容DeepSeek思考模型的写法)
                    filtered_message = {
                        "role": "assistant",
                        "content": message.get("content"),
                    }
                    if message.get("reasoning_content"):
                        filtered_message["reasoning_content"] = message.get(
                            "reasoning_content"
                        )
                    if message.get("tool_calls"):
                        # 7.取出工具调用的数据，限制LLM一次只能调用工具
                        filtered_message["tool_calls"] = message.get("tool_calls")[:1]
                else:
                    # 8.非AI消息则记录日志并存储message
                    logger.warning(
                        f"LLM响应内容无法确认消息角色: {message.get('role')}"
                    )
                    filtered_message = message

                # 9.将消息添加到记忆中
                await self._add_to_memory([filtered_message])
                return filtered_message
            except Exception as e:
                # 10.记录日志并睡眠指定的时间
                last_error = str(e)
                logger.error(f"调用语言模型发生错误: {str(e)}")
                await asyncio.sleep(self._retry_interval)
                continue

        raise RuntimeError(f"调用语言模型失败: {last_error}")

    async def _invoke_tool(
        self, tool: BaseTool, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """传递工具包+工具名字+对应参数调用指定工具"""
        # 1.执行循环调用工具获取结果
        err = ""
        for _ in range(self._agent_config.max_retries):
            try:
                return await tool.invoke(tool_name, **arguments)
            except Exception as e:
                err = str(e)
                logger.exception(f"调用工具[{tool_name}]出错, 错误: {str(e)}")
                await asyncio.sleep(self._retry_interval)
                continue

        # 2.循环最大重试次数后没有结果则将错误作为工具的执行结果，让LLM自行处理
        return ToolResult(success=False, message=err)

    async def _add_to_memory(self, messages: List[Dict[str, Any]]) -> None:
        """将对应的信息添加到记忆中"""
        # 1.先检查确保记忆是存在的
        await self._ensure_memory()

        # 2.确保首条系统消息和当前上下文一致
        self._ensure_system_message()

        # 3.将正常消息添加到记忆中
        self._memory.add_messages(messages)

        # 4.将记忆持久化到数据仓库中
        async with self._uow:
            await self._uow.session.save_memory(
                self._session_id, self.name, self._memory
            )

    async def compact_memory(self, keep_summary: bool = False) -> None:
        """压缩Agent的记忆"""
        await self._ensure_memory()
        self._memory.compact(keep_summary=keep_summary)
        async with self._uow:
            await self._uow.session.save_memory(
                self._session_id, self.name, self._memory
            )

    async def roll_back(self, message: Message) -> None:
        """Agent的状态回滚，该函数用于确保Agent的消息列表状态是正确，用于发送新消息、暂停/停止任务、通知用户"""
        await self._ensure_skill_creation_state()
        # 1.取出记忆中的最后一条消息，检查是否是工具调用
        await self._ensure_memory()
        if self._skill_creation_state and self._skill_creation_state.pending_action:
            state = self._skill_creation_state
            decision = self._classify_skill_creation_structured_action(
                state.pending_action,
                message.skill_confirmation_action,
            ) or self._classify_skill_creation_reply(message.message or "")
            logger.info(
                "Skill 创建确认回滚: pending=%s decision=%s action=%s reply=%r",
                state.pending_action,
                decision,
                message.skill_confirmation_action,
                (message.message or "")[:50],
            )

            if decision in {"affirmative", "revise"}:
                # 仅当记忆中尚无该工具结果时才补充（execute_step 可能已写入）
                last_msg = self._memory.get_last_message() if self._memory else None
                already_has_tool_result = (
                    last_msg
                    and last_msg.get("role") == "tool"
                    and last_msg.get("tool_call_id") == state.last_tool_call_id
                )
                if not already_has_tool_result:
                    self._memory.add_message(
                        {
                            "role": "tool",
                            "tool_call_id": state.last_tool_call_id,
                            "function_name": state.last_tool_name,
                            "content": state.saved_tool_result_json,
                        }
                    )

            if decision == "affirmative":
                if state.pending_action:
                    self._skill_creation_approved_actions.add(state.pending_action)
                # 保留状态并标记为已批准，供跨轮恢复时自动补齐关键 payload（如 skill_data）。
                state.approval_status = "approved"
                self._skill_creation_state = state
                await self._persist_skill_creation_state()
            elif decision == "negative":
                await self._clear_skill_creation_state()
                last_message = self._memory.get_last_message()
                if last_message and last_message.get("tool_calls"):
                    self._memory.roll_back()

            async with self._uow:
                await self._uow.session.save_memory(
                    self._session_id, self.name, self._memory
                )
            return

        last_message = self._memory.get_last_message()
        if (
            not last_message
            or not last_message.get("tool_calls")
            or len(last_message.get("tool_calls")) == 0
        ):
            return

        # 2.取出消息中的工具调用参数
        tool_call = last_message.get("tool_calls")[0]

        # 3.提取工具名字、id
        function_name = tool_call.get("function", {}).get("name")
        tool_call_id = tool_call.get("id")

        # 4.判断下当前的工具是不是通知用户(message_ask_user)或需要确认的工具
        if function_name == "message_ask_user":
            self._memory.add_message(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "function_name": function_name,
                    "content": message.model_dump_json(),
                }
            )
        elif self._is_tool_confirmation_required(function_name):
            # 5.需要确认的工具：用户确认后注入工具结果，保留上下文
            user_text = (message.message or "").strip().lower()
            refused = user_text in {"否", "取消", "no", "cancel", "拒绝", "不"}
            if refused:
                result_content = ToolResult(
                    success=False,
                    message="用户已拒绝该操作",
                ).model_dump_json()
            else:
                result_content = ToolResult(
                    success=True,
                    message="CONFIRMATION_GRANTED",
                    data={"code": "CONFIRMATION_GRANTED"},
                ).model_dump_json()
                # 标记该工具已确认，下次拦截时跳过
                if hasattr(self, "_confirmed_tool_names"):
                    self._confirmed_tool_names.add(function_name)
            self._memory.add_message(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "function_name": function_name,
                    "content": result_content,
                }
            )
        else:
            # 6.否则直接删除最后一条消息
            self._memory.roll_back()

        # 6.将记忆持久化
        async with self._uow:
            await self._uow.session.save_memory(
                self._session_id, self.name, self._memory
            )

    async def invoke(
        self, query: str, format: Optional[str] = None
    ) -> AsyncGenerator[BaseEvent, None]:
        """传递消息+响应格式调用程序生成异步迭代内容"""
        # 1.需要判断下是否传递了format
        format = format if format else self._format

        # 2.调用语言模型获取响应内容
        try:
            message = await self._invoke_llm(
                [{"role": "user", "content": query}],
                format,
            )
        except Exception as e:
            yield ErrorEvent(error=str(e))
            return

        if not isinstance(message, dict):
            yield ErrorEvent(error="LLM返回了非法消息格式")
            return

        # 3.循环遍历直到最大迭代次数
        for _ in range(self._agent_config.max_iterations):
            # 4.如果响应内容无工具调用则表示LLM生成了文本回答，这时候就是最终答案
            if not message.get("tool_calls"):
                break

            # 5.循环遍历工具参数并执行
            tool_messages = []
            for tool_call in message["tool_calls"]:
                if not tool_call.get("function"):
                    continue

                # 6.取出调用工具id、名字、参数信息
                tool_call_id = tool_call.get("id") or str(uuid.uuid4())
                function_name = tool_call["function"]["name"]
                function_args = await self._json_parser.invoke(
                    tool_call["function"].get("arguments", "{}")
                )
                if not isinstance(function_args, dict):
                    function_args = {}

                # 7.取出Agent中对应的工具（未知工具降级为可恢复结果，避免直接中断）
                tool: BaseTool | None = None
                tool_name = "unknown"
                unknown_tool_result: ToolResult | None = None
                try:
                    tool = self._get_tool(function_name)
                    tool_name = tool.name
                except ValueError:
                    unknown_tool_result = self._build_unknown_tool_result(function_name)
                    logger.warning("检测到未知工具调用，降级回灌给LLM: %s", function_name)

                # 8.返回工具即将调用事件，其中tool_content比较特殊，需要在具体业务中进行实现，这里留空即可
                yield ToolEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    function_name=function_name,
                    function_args=function_args,
                    status=ToolEventStatus.CALLING,
                )

                # 9.调用工具并获取结果
                if unknown_tool_result is not None:
                    result = unknown_tool_result
                else:
                    intercepted_result = self._intercept_tool_call(
                        function_name, function_args
                    )
                    if intercepted_result is not None:
                        result = intercepted_result
                    else:
                        # mypy: unknown_tool_result is None 时 tool 必存在
                        assert tool is not None
                        result = await self._invoke_tool(tool, function_name, function_args)
                self._on_tool_result(function_name, result)

                # 10.返回工具调用结果，其中tool_content比较特殊，需要在业务中进行实现
                yield ToolEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    function_name=function_name,
                    function_args=function_args,
                    function_result=result,
                    status=ToolEventStatus.CALLED,
                )

                # 11.组装工具响应
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "function_name": function_name,
                        "content": result.model_dump_json(),
                    }
                )

            # 12.所有工具都执行完成后，调用LLM获取汇总消息二次提供
            try:
                message = await self._invoke_llm(tool_messages)
            except Exception as e:
                yield ErrorEvent(error=str(e))
                return
            if not isinstance(message, dict):
                yield ErrorEvent(error="LLM返回了非法消息格式")
                return
        else:
            # 13.超过最大迭代次数后，则抛出错误
            yield ErrorEvent(
                error=f"Agent迭代超过最大迭代次数: {self._agent_config.max_iterations}, 任务处理失败"
            )

        # 14.在指定步骤内完成了迭代则返回消息事件
        yield MessageEvent(message=str(message.get("content", "")))
