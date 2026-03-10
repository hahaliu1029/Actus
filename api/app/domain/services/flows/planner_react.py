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
from app.domain.models.event import BaseEvent, DoneEvent, WaitEvent
from app.domain.models.message import Message
from app.domain.models.plan import Plan
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.graphs.event_bridge import GraphEventBridge
from app.domain.services.graphs.main_graph import build_main_graph
from app.domain.services.graphs.react_graph import build_react_graph
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools
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
        self._skill_context = ""

        # Skill creation subgraph
        self._user_id = user_id
        self._skill_graph_canary_percent = skill_graph_canary_percent
        self._brainstorm_skill_tool = brainstorm_skill_tool
        self._create_skill_tool = create_skill_tool

        # Build LangChain tools
        lc_tools = create_native_tools(
            sandbox=sandbox, browser=browser, search_engine=search_engine,
        )
        lc_tools.extend(create_mcp_langchain_tools(mcp_tool))
        # TODO: Add A2A tools, skill tools, brainstorm/create skill tools

        # 中断恢复上下文：保存中断时的步骤、消息历史、原始请求
        self._saved_current_step = None
        self._saved_messages: list = []
        self._saved_original_request: str = ""

        # Build LLM adapter
        self._llm_adapter = LLMAdapter(llm=llm)

        # Build graphs
        self._react_graph = build_react_graph(
            llm=self._llm_adapter, tools=lc_tools, agent_config=agent_config,
        )
        self._main_graph = build_main_graph(
            planner_llm=llm,
            react_graph=self._react_graph,
            json_parser=json_parser,
            summary_llm=self._summary_llm,
            uow_factory=uow_factory,
            session_id=session_id,
            agent_config=agent_config,
        )

    def set_skill_context(self, skill_context: str) -> None:
        """Set activated skill context for this round."""
        self._skill_context = skill_context

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

        # Build input state for main_graph
        # 如果上次因中断（接管）暂停，恢复保存的上下文
        is_resuming = self.status == FlowStatus.EXECUTING and self.plan is not None
        input_state = {
            "message": message.message,
            "language": getattr(message, "language", "en"),
            "attachments": getattr(message, "attachments", []),
            "plan": self.plan,
            "current_step": self._saved_current_step if is_resuming else None,
            "messages": self._saved_messages if is_resuming else [],
            "execution_summary": "",
            "events": [],
            "flow_status": "executing" if is_resuming else (
                self.status.value if hasattr(self.status, "value") else "idle"
            ),
            "session_id": self._session_id,
            "should_interrupt": False,
            "original_request": self._saved_original_request if is_resuming else "",
            "skill_context": self._skill_context,
        }

        bridge = GraphEventBridge()
        async for event in bridge.run(self._main_graph, input_state):
            yield event

        # Update internal state from graph result
        final = bridge.final_state
        self.plan = final.get("plan")
        flow_status = final.get("flow_status", "idle")

        if final.get("should_interrupt"):
            # 中断（接管）：保存上下文以便恢复
            self.status = FlowStatus.EXECUTING
            self._saved_current_step = final.get("current_step")
            self._saved_messages = final.get("messages", [])
            self._saved_original_request = final.get("original_request", "")
        else:
            # 正常完成：重置状态
            self.status = FlowStatus.IDLE
            self._saved_current_step = None
            self._saved_messages = []
            self._saved_original_request = ""

    @property
    def done(self) -> bool:
        return self.status == FlowStatus.IDLE
