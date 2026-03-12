"""Planner+ReAct Flow — now delegates to LangGraph main_graph + react_graph.

This module preserves the same public interface (constructor, invoke, done)
so that AgentTaskRunner requires minimal changes.
"""

import logging
from typing import AsyncGenerator, Callable, Optional

from app.domain.external.browser import Browser
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.models.app_config import AgentConfig
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.conversation_summary import ConversationSummary
from app.domain.models.event import BaseEvent, DoneEvent, WaitEvent
from app.domain.models.interrupt_state import InterruptState
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import ExecutionStatus, Plan
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.graphs.event_bridge import GraphEventBridge
from app.domain.services.graphs.main_graph import build_main_graph
from app.domain.services.graphs.message_utils import dicts_to_messages, messages_to_dicts
from app.domain.services.graphs.react_graph import build_react_graph
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools
from app.domain.services.tools.langchain_skill_tools import create_skill_langchain_tools
from app.domain.services.tools.langchain_tools import create_native_tools
from app.domain.services.tools.mcp import MCPTool
from app.domain.services.tools.skill import SkillTool
from app.infrastructure.external.llm.langchain_adapter import LLMAdapter

from .base import BaseFlow, FlowStatus
from .skill_creation_graph import SkillCreationGraph
from .skill_graph_canary import is_skill_graph_enabled

logger = logging.getLogger(__name__)


class PlannerReActFlow(BaseFlow):
    """Planner+ReAct orchestration flow backed by LangGraph."""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: LLM,
        agent_config: AgentConfig,
        session_id: str,
        json_parser: JSONParser,
        browser: Browser,
        sandbox: Sandbox,
        search_engine: SearchEngine,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
        skill_tool: SkillTool,
        create_skill_tool: BaseTool | None = None,
        brainstorm_skill_tool: BaseTool | None = None,
        overflow_config: ContextOverflowConfig | None = None,
        summary_llm: LLM | None = None,
        user_id: str = "",
        skill_graph_canary_percent: int = 0,
    ) -> None:
        self._uow_factory = uow_factory
        self._session_id = session_id
        self._json_parser = json_parser
        self._summary_llm = summary_llm or llm
        self.status = FlowStatus.IDLE
        self.plan: Optional[Plan] = None
        self._memory_config = agent_config.memory
        self._overflow_config = overflow_config
        self._skill_context = ""

        # Skill creation subgraph
        self._user_id = user_id
        self._skill_graph_canary_percent = skill_graph_canary_percent
        self._brainstorm_skill_tool = brainstorm_skill_tool
        self._create_skill_tool = create_skill_tool

        # 中断恢复上下文：保存中断时的步骤、消息历史、原始请求
        self._saved_current_step = None
        self._saved_messages: list = []
        self._saved_original_request: str = ""

        # 延迟绑定：保存依赖引用，在 invoke() 时构建工具和图
        # MCP/A2A 在 AgentTaskRunner.run() 中异步初始化，构造时尚未就绪
        self._llm = llm
        self._agent_config = agent_config
        self._sandbox = sandbox
        self._browser = browser
        self._search_engine = search_engine
        self._mcp_tool = mcp_tool
        self._a2a_tool = a2a_tool
        self._llm_adapter = LLMAdapter(llm=llm)
        self._graphs_built = False
        self._react_graph = None
        self._main_graph = None

    def set_skill_context(self, skill_context: str) -> None:
        """Set activated skill context for this round."""
        self._skill_context = skill_context

    def _ensure_graphs(self) -> None:
        """延迟构建工具列表和 LangGraph 图。

        在 invoke() 首次调用时执行，此时 MCP/A2A 已完成异步初始化，
        能正确获取到所有可用工具。后续调用会重新构建以反映工具变化。
        """
        from app.domain.services.tools.langchain_a2a import create_a2a_langchain_tools

        lc_tools = create_native_tools(
            sandbox=self._sandbox, browser=self._browser,
            search_engine=self._search_engine,
        )
        lc_tools.extend(create_mcp_langchain_tools(self._mcp_tool))
        lc_tools.extend(create_a2a_langchain_tools(self._a2a_tool))
        lc_tools.extend(create_skill_langchain_tools(
            brainstorm_skill_tool=self._brainstorm_skill_tool,
            create_skill_tool=self._create_skill_tool,
        ))

        tool_names = [t.name for t in lc_tools]
        logger.info("延迟绑定工具列表 (%d tools): %s", len(lc_tools), tool_names)

        self._react_graph = build_react_graph(
            llm=self._llm_adapter, tools=lc_tools, agent_config=self._agent_config,
        )
        self._main_graph = build_main_graph(
            planner_llm=self._llm,
            react_graph=self._react_graph,
            json_parser=self._json_parser,
            summary_llm=self._summary_llm,
            uow_factory=self._uow_factory,
            session_id=self._session_id,
            agent_config=self._agent_config,
        )
        self._graphs_built = True

    def _build_context_anchor(self, message: Message) -> str:
        """构建上下文锚点，注入到 Memory 中帮助 LLM 保持多轮连贯性。"""
        parts = ["[上下文回顾]"]
        if self.plan:
            parts.append(f"- 原始需求：{self.plan.goal}")
            completed = [s.description for s in self.plan.steps
                         if s.status == ExecutionStatus.COMPLETED]
            pending = [s.description for s in self.plan.steps
                       if s.status != ExecutionStatus.COMPLETED]
            if completed:
                parts.append(f"- 已完成：{'；'.join(completed)}")
            if pending:
                parts.append(f"- 待完成：{'；'.join(pending)}")
        parts.append(f"- 当前消息：{message.message}")
        return "\n".join(parts)

    async def _generate_summary(
        self, existing: list[ConversationSummary], plan: Plan,
    ) -> ConversationSummary:
        """调用 LLM 生成结构化对话摘要。"""
        from app.domain.services.prompts.summary import GENERATE_SUMMARY_PROMPT
        steps_summary = "\n".join(
            f"- {s.description}: {'完成' if s.status == ExecutionStatus.COMPLETED else '未完成'}"
            for s in plan.steps
        )
        prompt = GENERATE_SUMMARY_PROMPT.format(
            round_number=len(existing) + 1,
            plan_goal=plan.goal,
            steps_summary=steps_summary,
        )
        response = await self._summary_llm.invoke(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        parsed = await self._json_parser.invoke(response.get("content", ""))
        if not isinstance(parsed, dict):
            parsed = {}
        return ConversationSummary(
            round_number=len(existing) + 1,
            user_intent=parsed.get("user_intent", plan.goal),
            plan_summary=parsed.get("plan_summary", ""),
            execution_results=parsed.get("execution_results", []),
            decisions=parsed.get("decisions", []),
            unresolved=parsed.get("unresolved", []),
        )

    async def _check_overflow(self, memory: Memory) -> None:
        """检测上下文溢出，超过硬阈值时做激进压缩。"""
        if not self._overflow_config or not self._overflow_config.context_overflow_guard_enabled:
            return
        from app.domain.services.context.model_context_window import resolve_context_window
        # 使用字符估算 token（粗略：1 token ≈ 3-4 字符中英混合）
        total_chars = sum(len(str(m.get("content", ""))) for m in memory.messages)
        estimated_tokens = int(total_chars / 3 * self._overflow_config.token_safety_factor)
        window = resolve_context_window("", self._overflow_config)
        hard_limit = int(window * self._overflow_config.hard_trigger_ratio)
        if estimated_tokens > hard_limit:
            logger.warning(f"上下文溢出: ~{estimated_tokens} tokens > hard_limit {hard_limit}, 执行硬压缩")
            memory.compact(keep_summary=False)
            # 保留系统消息 + 最近 N 条
            if len(memory.messages) > 20:
                memory.messages = memory.messages[:1] + memory.messages[-19:]
            async with self._uow_factory() as uow:
                await uow.session.save_memory(self._session_id, "react", memory)

    def _is_skill_graph_active(self) -> bool:
        return is_skill_graph_enabled(self._user_id, self._skill_graph_canary_percent)

    async def _try_drive_skill_graph(
        self, message: Message,
    ) -> AsyncGenerator[BaseEvent, None] | None:
        """Try to drive skill creation subgraph. Returns async generator or None."""
        if not self._is_skill_graph_active():
            return None
        if not self._brainstorm_skill_tool or not self._create_skill_tool:
            return None

        action = message.skill_confirmation_action

        async with self._uow_factory() as uow:
            graph_state = await uow.session.get_skill_graph_state(self._session_id)

        if graph_state is None and action is None:
            return None
        if graph_state is not None and graph_state.is_terminal:
            return None

        async def _drive() -> AsyncGenerator[BaseEvent, None]:
            graph = SkillCreationGraph(
                brainstorm_tool=self._brainstorm_skill_tool,
                create_skill_tool=self._create_skill_tool,
            )
            async for event in graph.run(
                state=graph_state,
                action=action,
                original_request=getattr(graph_state, "original_request", ""),
            ):
                yield event

            new_state = graph.state
            if new_state is not None:
                async with self._uow_factory() as uow:
                    if new_state.is_terminal:
                        await uow.session.clear_skill_graph_state(self._session_id)
                    else:
                        await uow.session.save_skill_graph_state(
                            self._session_id, new_state,
                        )

        return _drive()

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """Run the flow — delegates to LangGraph main_graph."""
        # Try skill creation subgraph first
        subgraph_gen = await self._try_drive_skill_graph(message)
        if subgraph_gen is not None:
            async for event in subgraph_gen:
                yield event
            return

        # 延迟绑定：每次 invoke 重新构建工具和图，确保 MCP/A2A 已初始化
        self._ensure_graphs()

        # Build input state for main_graph
        # 如果上次因中断（接管）暂停，恢复保存的上下文
        # 优先使用内存中的状态（同一 runner 内恢复），否则尝试从 DB 恢复（跨 runner 恢复）
        if self.status == FlowStatus.IDLE and self.plan is None:
            try:
                async with self._uow_factory() as uow:
                    persisted = await uow.session.get_interrupt_state(self._session_id)
                if persisted is not None and persisted.plan is not None:
                    self.plan = persisted.plan
                    self._saved_current_step = persisted.current_step
                    self._saved_messages = persisted.messages
                    self._saved_original_request = persisted.original_request
                    self.status = FlowStatus.EXECUTING
                    logger.info(
                        "从数据库恢复中断状态: session=%s plan=%s step=%s messages=%d",
                        self._session_id,
                        persisted.plan.title,
                        persisted.current_step.description if persisted.current_step else "<none>",
                        len(persisted.messages),
                    )
                else:
                    logger.info(
                        "未找到中断状态（persisted=%s），将作为新任务处理: session=%s",
                        "None" if persisted is None else f"plan={persisted.plan}",
                        self._session_id,
                    )
            except Exception as e:
                logger.warning(f"加载中断状态失败，将作为新任务处理: {e}", exc_info=True)

        is_resuming = self.status == FlowStatus.EXECUTING and self.plan is not None

        # === Before Graph: 加载记忆和摘要 ===
        async with self._uow_factory() as uow:
            memory = await uow.session.get_memory(self._session_id, "react")
            summaries = await uow.session.get_summary(self._session_id)

        # 上下文锚点：非首轮 + 配置开启时注入
        if self._memory_config.context_anchor_enabled and not memory.empty:
            anchor = self._build_context_anchor(message)
            memory.add_message({"role": "user", "content": anchor})

        # 摘要文本
        recent_summaries = summaries[-self._memory_config.summary_max_rounds:]
        summary_texts = [s.to_prompt_text() for s in recent_summaries]

        # 将 Memory 中的 dict 消息转换为 LangChain BaseMessage
        if is_resuming:
            raw_messages = self._saved_messages
        else:
            raw_messages = memory.get_messages()
        lc_messages = dicts_to_messages(raw_messages) if raw_messages else []

        input_state = {
            "message": message.message,
            "language": getattr(message, "language", "zh"),
            "attachments": getattr(message, "attachments", []),
            "plan": self.plan,
            "current_step": self._saved_current_step if is_resuming else None,
            "messages": lc_messages,
            "execution_summary": "",
            "events": [],
            "flow_status": FlowStatus.EXECUTING.value if is_resuming else (
                self.status.value if hasattr(self.status, "value") else FlowStatus.IDLE.value
            ),
            "session_id": self._session_id,
            "should_interrupt": False,
            "is_resuming": is_resuming,
            "original_request": self._saved_original_request if is_resuming else (
                self.plan.goal if self.plan else ""
            ),
            "skill_context": self._skill_context,
            "conversation_summaries": summary_texts,
        }

        bridge = GraphEventBridge()
        try:
            async for event in bridge.run(self._main_graph, input_state):
                yield event
        finally:
            # 使用 try/finally 确保持久化逻辑始终执行，即使消费方提前退出
            # （例如 WaitEvent 触发 agent_task_runner.run() 的 return，
            #  导致本 async generator 被 aclose()、GeneratorExit 抛入 yield 处）。
            # bridge.run() 的 finally 会 await 图任务完成，
            # 因此此处 bridge.final_state 已包含完整的图输出。
            await self._persist_after_graph(bridge.final_state, summaries)

    async def _persist_after_graph(
        self, final: dict, summaries: list[ConversationSummary],
    ) -> None:
        """Post-graph persistence: save memory, interrupt state, summaries.

        CRITICAL: 该方法在 invoke() 的 finally 块中调用（async generator 被 aclose
        时通过 GeneratorExit → finally 触发）。如果此方法抛出异常，异常会传播到
        agent_task_runner 的 except Exception 处理器，导致会话状态被设为 COMPLETED
        而非 WAITING，InterruptState 永远无法保存——恢复时上下文完全丢失。
        因此整个方法用 try/except 包裹，确保永不向上抛出异常。
        """
        try:
            await self._persist_after_graph_inner(final, summaries)
        except Exception as exc:
            logger.exception(
                "持久化后处理异常（已抑制，避免破坏 generator 清理链）: %s", exc
            )

    async def _persist_after_graph_inner(
        self, final: dict, summaries: list[ConversationSummary],
    ) -> None:
        """Inner implementation of post-graph persistence."""
        self.plan = final.get("plan")

        # 将 LangChain BaseMessage 转回 dict 用于 Memory/InterruptState 持久化
        raw_messages = final.get("messages", [])
        try:
            dict_messages = messages_to_dicts(raw_messages) if raw_messages else []
        except Exception as exc:
            logger.warning("messages_to_dicts 转换失败，使用空消息列表: %s", exc)
            dict_messages = []

        if final.get("should_interrupt"):
            # 中断（接管）：保存上下文以便恢复（内存 + DB 双写）
            self.status = FlowStatus.EXECUTING
            self._saved_current_step = final.get("current_step")
            self._saved_messages = dict_messages
            self._saved_original_request = final.get("original_request", "")

            logger.info(
                "中断持久化: session=%s plan=%s step=%s messages=%d",
                self._session_id,
                self.plan.title if self.plan else "<none>",
                self._saved_current_step.description if self._saved_current_step else "<none>",
                len(dict_messages),
            )

            # 持久化 Memory 到 DB，确保跨 runner 恢复时可加载
            interrupt_memory = Memory(messages=list(dict_messages))
            interrupt_memory.compact(keep_summary=self._memory_config.compact_keep_summary)
            try:
                async with self._uow_factory() as uow:
                    await uow.session.save_memory(self._session_id, "react", interrupt_memory)
            except Exception as e:
                logger.warning(f"中断时保存 Memory 失败: {e}")

            # 持久化中断状态到 DB，确保新 AgentTaskRunner 实例可恢复
            interrupt_state = InterruptState(
                plan=self.plan,
                current_step=self._saved_current_step,
                messages=self._saved_messages,
                original_request=self._saved_original_request,
            )
            try:
                async with self._uow_factory() as uow:
                    await uow.session.save_interrupt_state(self._session_id, interrupt_state)
                logger.info("InterruptState 已持久化到 DB: session=%s", self._session_id)
            except Exception as e:
                logger.warning(f"中断时保存 InterruptState 失败: {e}")
        else:
            # === After Graph: 记忆压缩、保存、摘要生成 ===
            memory = Memory(messages=list(dict_messages))
            memory.compact(keep_summary=self._memory_config.compact_keep_summary)

            try:
                async with self._uow_factory() as uow:
                    await uow.session.save_memory(self._session_id, "react", memory)
            except Exception as e:
                logger.warning(f"保存 Memory 失败: {e}")

            # ConversationSummary 生成（容错，不阻塞）
            plan = final.get("plan") or self.plan
            if (self._memory_config.summary_enabled
                    and plan
                    and len(plan.steps) >= self._memory_config.summary_min_steps):
                try:
                    new_summary = await self._generate_summary(summaries, plan)
                    all_summaries = (summaries + [new_summary])[
                        -self._memory_config.summary_max_rounds:
                    ]
                    async with self._uow_factory() as uow:
                        await uow.session.save_summary(self._session_id, all_summaries)
                except Exception as e:
                    logger.warning(f"生成对话摘要失败，不阻塞: {e}")

            # 上下文溢出检测
            try:
                await self._check_overflow(memory)
            except Exception as e:
                logger.warning(f"上下文溢出检测失败: {e}")

            # 清理已消费的中断状态
            try:
                async with self._uow_factory() as uow:
                    await uow.session.clear_interrupt_state(self._session_id)
            except Exception as e:
                logger.warning(f"清理 InterruptState 失败: {e}")

            # 正常完成：重置状态
            self.status = FlowStatus.IDLE
            self._saved_current_step = None
            self._saved_messages = []
            self._saved_original_request = ""

    @property
    def done(self) -> bool:
        return self.status == FlowStatus.IDLE
