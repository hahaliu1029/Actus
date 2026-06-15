"""Planner+ReAct Flow — now delegates to LangGraph main_graph + react_graph.

This module preserves the same public interface (constructor, invoke, done)
so that AgentTaskRunner requires minimal changes.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Optional,
    Sequence,
)

if TYPE_CHECKING:
    from app.domain.models.app_config import ToolRuntimeConfig
    from app.domain.services.permission.engine import PermissionEngine
    from app.domain.services.prompts.assembler import PromptAssembler
    from app.domain.services.prompts.memory_snapshot import MemorySnapshot
    from app.domain.services.provider_profiles._base import ProviderProfile
    from app.domain.services.session.session_state_machine import SessionStateMachine


from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.application.services.coordinator_runtime_deps import (
    _CoordinatorRuntimeDeps,
    _NullCoordinatorRuntimeDeps,
)
from app.domain.external.browser import Browser
from app.domain.external.sandbox import SandboxHandle
from app.domain.external.search import SearchEngine
from app.domain.models.app_config import AgentConfig
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.conversation_summary import ConversationSummary
from app.domain.models.event import (
    BaseEvent,
    DoneEvent,
    MessageEvent,
    PlanEvent,
    PlanEventStatus,
    TitleEvent,
    WaitEvent,
)
from app.domain.models.llm_responses import ConversationSummaryResponse, PlanResponse, StepDef
from app.domain.models.memory import Memory
from app.domain.models.memory_chunk import FlushBatch, RawChunk, memory_content_hash
from app.domain.models.message import Message
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.repositories.uow import IUnitOfWork
from langgraph.types import Command

from app.domain.services.context.model_context_window import resolve_context_window
from app.domain.services.graphs.event_bridge import GraphEventBridge
from app.domain.services.graphs.main_graph import build_main_graph
from app.domain.services.graphs.message_utils import (
    dicts_to_messages,
    format_attachments_text,
    messages_to_dicts,
)
from app.domain.services.graphs.react_graph import build_react_graph
from app.domain.services.graphs.compaction import CompactionResult, GradualCompactor
from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.planner_guardrails import salvage_empty_memory_recall_plan
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.base import BaseTool
from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools
from app.domain.services.tools.langchain_skill_tools import create_skill_langchain_tools
from app.domain.services.tools.langchain_tools import create_native_tools
from app.domain.services.tools._supervisor_tool_wrapper import (
    wrap_tool_list_for_supervisor,
)
from app.domain.services.tools.mcp import MCPTool
from app.domain.services.tools.skill import SkillTool

from .base import BaseFlow, FlowStatus


def _iso_or_none(dt: datetime | None) -> str | None:
    """Serialize optional datetime → ISO-8601 string for JSON payloads."""
    return dt.isoformat() if dt is not None else None
from .skill_creation_graph import SkillCreationGraph
from .skill_graph_canary import is_skill_graph_enabled

logger = logging.getLogger(__name__)


def _apply_plan_update(existing: Plan, response: PlanResponse) -> Plan:
    """[C2 PR-1 §4.2 r7 P0-2] Apply a planner update preserving Step.id.

    When the LLM's new ``StepDef.id`` matches an existing Step, that
    Step's runtime state (status / result / error / success / attachments)
    is carried over so replan does not lose progress. When the LLM omits
    the id or invents a new one, a deterministic fallback id rooted on
    ``existing.id`` is assigned via :func:`_assign_fallback_step_id` so
    cross-step references stay stable across replan rounds.
    """
    from app.domain.services.graphs.main_graph import _assign_fallback_step_id

    by_id = {s.id: s for s in existing.steps}
    new_steps: list[Step] = []
    for i, sd in enumerate(response.steps):
        sid = sd.id or _assign_fallback_step_id(existing.id, i)
        prior = by_id.get(sid)
        new_steps.append(
            Step(
                id=sid,
                description=sd.description,
                status=prior.status if prior is not None else ExecutionStatus.PENDING,
                result=prior.result if prior is not None else None,
                error=prior.error if prior is not None else None,
                success=prior.success if prior is not None else False,
                attachments=list(prior.attachments) if prior is not None else [],
                parallel_work_units=sd.parallel_work_units,
            )
        )
    return Plan(steps=new_steps)


class PlannerReActFlow(BaseFlow):
    """Planner+ReAct orchestration flow backed by LangGraph."""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: BaseChatModel,
        agent_config: AgentConfig,
        session_id: str,
        browser: Browser,
        sandbox: SandboxHandle,
        search_engine: SearchEngine,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
        skill_tool: SkillTool,
        create_skill_tool: BaseTool | None = None,
        brainstorm_skill_tool: BaseTool | None = None,
        overflow_config: ContextOverflowConfig | None = None,
        summary_llm: BaseChatModel | None = None,
        user_id: str = "",
        skill_graph_canary_percent: int = 0,
        checkpointer_pool: object | None = None,
        checkpointer: Any = None,
        supports_vision: bool = True,
        supports_pdf_input: bool = False,
        file_processor_lookup: Any = None,  # FileProcessorLookup | None
        memory_embedding_provider=None,
        memory_session_factory=None,
        memory_repo_factory=None,
        memory_write_service=None,  # PR-3: memory_save routes writes here
        memory_session_redis=None,  # PR-3: per-session save counter
        memory_session_save_cap: int = 20,  # PR-3
        approval_state_reader: Any = None,  # R5b-2: ApprovalStateReader | None（读路径 single source）
        confirmation_manager: Any = None,  # ConfirmationManager | None
        prompt_assembler: "PromptAssembler | None" = None,  # B5 C5b
        _allow_default_prompt_assembler: bool = False,  # B5 post-audit: test-only escape hatch
        tool_runtime: "ToolRuntimeConfig | None" = None,  # R2 CS2
        # B2 PR-1: typed LLM Recovery chain — when supplied, _ensure_graphs
        # wraps self._llm with wrap_with_recovery once (idempotent guard via
        # _recovery_wrapped). None = legacy path, no wrap.
        profile: "ProviderProfile | None" = None,
        # M1 PR-4+8 LLM quality gate (all optional — None = gate disabled,
        # falls back to legacy size-only path)
        memory_gate_llm: BaseChatModel | None = None,
        memory_gate_breaker: Any = None,  # MemoryGateBreaker | None
        memory_gate_daily_cap: Any = None,  # MemoryGateDailyCap | None
        memory_gate_threshold: float = 0.7,
        memory_gate_batch_cap: int = 20,
        memory_notification_emitter: Any = None,  # MemoryNotificationEmitter | None
        # B4 M0: session-scoped CostCallbackHandler. When set, attached to
        # every LangGraph invoke so LLM calls generate CostRecord rows.
        cost_callback_handler: Any = None,
        execution_supervisor: Any = None,
        permission_engine: "PermissionEngine | None" = None,
        session_state_machine: "SessionStateMachine | None" = None,
        # PR-9b-A A5: lifespan-scoped coordinator runtime deps. Default is the
        # frozen NullCoordinatorRuntimeDeps sentinel so legacy callers (tests
        # + non-coordinator paths) inject zero coord cfg keys; production
        # construction wires a real ``_CoordinatorRuntimeDeps`` aggregate.
        _coord_deps: _CoordinatorRuntimeDeps | _NullCoordinatorRuntimeDeps = _NullCoordinatorRuntimeDeps(),
    ) -> None:
        self._cost_callback_handler = cost_callback_handler
        self._execution_supervisor = execution_supervisor
        self._permission_engine = permission_engine
        self._session_state_machine = session_state_machine
        self._supports_vision = supports_vision
        self._supports_pdf_input = supports_pdf_input
        self._file_processor_lookup = file_processor_lookup
        self._uow_factory = uow_factory
        self._session_id = session_id
        self._summary_llm = summary_llm or llm
        self._deferred_final_state: dict | None = None
        self._deferred_summaries: list | None = None
        self.status = FlowStatus.IDLE
        self.plan: Optional[Plan] = None
        self._memory_config = agent_config.memory
        self._overflow_config = overflow_config
        self._tool_runtime = tool_runtime
        self._token_estimator = TokenEstimator(
            strategy=self._overflow_config.token_estimator,
            model_name=self._overflow_config.model_name,
        ) if self._overflow_config else TokenEstimator()
        self._compactor = GradualCompactor(
            token_estimator=self._token_estimator,
            soft_trigger_ratio=self._overflow_config.soft_trigger_ratio,
            hard_trigger_ratio=self._overflow_config.hard_trigger_ratio,
            target_ratio=self._overflow_config.target_ratio,
            summary_max_chars=self._overflow_config.summary_max_chars,
            token_safety_factor=self._overflow_config.token_safety_factor,
        ) if self._overflow_config else None
        self._last_compaction_result: CompactionResult | None = None

        # Skill creation subgraph
        self._user_id = user_id
        self._skill_graph_canary_percent = skill_graph_canary_percent
        self._brainstorm_skill_tool = brainstorm_skill_tool
        self._create_skill_tool = create_skill_tool
        self._skill_tool = skill_tool

        # 延迟绑定：保存依赖引用，在 invoke() 时构建工具和图
        # MCP/A2A 在 AgentTaskRunner.run() 中异步初始化，构造时尚未就绪
        self._llm = llm
        self._agent_config = agent_config
        self._sandbox = sandbox
        self._browser = browser
        self._search_engine = search_engine
        self._mcp_tool = mcp_tool
        self._a2a_tool = a2a_tool
        self._graphs_built = False
        self._react_graph = None
        self._main_graph = None

        # B2 PR-1: typed LLM Recovery chain. When profile is provided,
        # _ensure_graphs wraps self._llm with wrap_with_recovery on first
        # call. The flag prevents nested wrapping on subsequent calls.
        self._profile = profile
        self._recovery_wrapped: bool = False

        # LangGraph checkpointer — 跨 graph 重建复用，支持 interrupt/resume
        self._checkpointer = checkpointer  # None = lazy-init AsyncPostgresSaver
        self._checkpointer_pool = checkpointer_pool
        self._assembler = None
        # B5 C5b: PromptAssembler instance for section-based system prompt
        # assembly. Consumed by main_graph executor/planner/updater nodes
        # and by ``_run_planner_for_detection``. Constructed by
        # ``AgentTaskRunner`` and passed in at flow construction time.
        self._prompt_assembler = prompt_assembler
        # B5 post-audit (HIGH #1): test-only escape hatch matching the
        # ``build_main_graph`` contract. When True AND
        # ``prompt_assembler is None``, both the graph construction path
        # and the detection-path planner silently build a minimal default
        # assembler; production callers (``AgentTaskRunner``) always pass
        # a configured instance and leave this False.
        self._allow_default_prompt_assembler = _allow_default_prompt_assembler

        # Phase 3: 动态 skill 切换回调（由 AgentTaskRunner 在 invoke 前设置）
        self._skill_context_refresher = None
        self._react_graph_provider = None
        self._skill_guide_injector = None
        # TODO #30: provider returning runner's current _last_skill_context.
        # Replaces the ``self._skill_context`` instance field (clock 2) that
        # bypass paths used to set via ``set_skill_context``. Read points are:
        # (a) ``_run_planner_for_detection`` detection_state
        # (b) ``input_for_graph["skill_context"]`` at invoke time (two branches)
        # Wired by ``AgentTaskRunner`` in the invoke main loop L2590 block.
        # Returns ``""`` when not wired (test harness / pre-wiring).
        #
        # NOTE: this is a narrow replacement for clock 2 field only. The LangGraph
        # ``state.skill_context`` field + ``_skill_context_refresher`` fallback
        # path + ``updater_node`` refresher block are NOT retired in this spec —
        # they stay alive because the resume path relies on them.
        # See spec §1.4 / §6.3.
        self._skill_context_provider: Callable[[], str] | None = None
        # B5 post-audit LOW #1: optional language_callback set by
        # AgentTaskRunner. main_graph.planner_node calls it via
        # configurable["language_callback"](plan.language) after parsing
        # the planner output so downstream telemetry picks up the real
        # session language. None = feature disabled (test paths).
        self._language_callback: "Callable[[str], None] | None" = None

        # 会话 Skill 池 getter（由 AgentTaskRunner 在 run() 中设置），
        # 用于 get_skill_guide 工具按需加载完整 SKILL.md。
        # 使用 callable 而非直接列表引用，避免 runner 重新赋值后 stale。
        self._skill_pool_getter: Callable[[], list] | None = None
        self._file_listings_getter: Callable[[], dict[str, list[str]]] | None = None
        self._sandbox_skill_root: str = "/home/ubuntu/workspace/.skills"

        # MCP progressive loading (set by AgentTaskRunner in run())
        self._mcp_tool_ref: Callable | None = None
        self._activated_mcp_tools_ref: Callable[[], set[str]] | None = None
        self._mcp_always_bind_names: set[str] = set()

        # Flush scheduling: cursor + pending batch
        self._flush_cursor: int = 0
        self._pending_flush_batch: FlushBatch | None = None

        # Memory tools dependencies (C6)
        self._memory_embedding_provider = memory_embedding_provider
        self._memory_session_factory = memory_session_factory
        self._memory_repo_factory = memory_repo_factory
        self._memory_write_service = memory_write_service
        self._memory_session_redis = memory_session_redis
        self._memory_session_save_cap = memory_session_save_cap
        self._has_memory_tools = False  # set by _collect_all_tools

        # M1 PR-4+8 gate state — all None when gate disabled, in which
        # case _evaluate_flush_gate skips the LLM branch entirely.
        self._memory_gate_llm = memory_gate_llm
        self._memory_gate_breaker = memory_gate_breaker
        self._memory_gate_daily_cap = memory_gate_daily_cap
        self._memory_gate_threshold = memory_gate_threshold
        self._memory_gate_batch_cap = memory_gate_batch_cap
        self._memory_notification_emitter = memory_notification_emitter

        # R5b-2 Reader 接入；ApprovalCache 已于 R5b-4 移除。Legacy writer slot
        # retired in PE-4c (PE owns grant writes via build_permission_engine(writer=...)).
        self._approval_state_reader = approval_state_reader
        self._confirmation_manager = confirmation_manager

        # D5: Execution health monitoring — persist across invoke/resume
        self._execution_config = agent_config.execution
        from app.domain.services.tools.tool_failure_tracker import ToolFailureTracker
        from app.domain.services.execution_metrics import ExecutionMetrics
        self._tool_failure_tracker = ToolFailureTracker(
            max_same_failures=self._execution_config.max_same_tool_failures,
        )
        self._execution_metrics = ExecutionMetrics()

        # PR-9b-A A5: process-scoped coordinator runtime deps + per-run
        # cancel_event. The cancel_event is initialized to None here; the
        # owning ``AgentTaskRunner.invoke`` writes a fresh ``asyncio.Event``
        # onto this attribute at the start of each task — see
        # ``AgentTaskRunner._prime_planner_cancel_event_for_coord_deps``
        # (PR-9b-A audit round-1 P1 fix for INV-A6 — "each non-None"). Tests
        # set it directly on the instance to assert the cfg-building contract.
        self._coord_deps = _coord_deps
        self._cancel_event: Optional[asyncio.Event] = None
        # [C2 finish-core §5.1.1] Set True by set_cancel_event so a coord
        # (parent) re-invoke's prime cannot clobber an adapter-injected event.
        self._cancel_event_externally_injected: bool = False
        # [C2b §4.1] Coordinator-child scope context. None for root + non-child
        # flows; set by AgentTaskRunner.set_coordinator_child_permission_context
        # via the invoke-adapter. _build_config injects it (+ the SSM) into the
        # child graph cfg so react_graph's tool_node child-scope guard fires.
        self._child_permission_context = None
        # [C2b budget §3-5] Child-only BudgetEnforcementCallback. None for
        # root/parent flows; set by AgentTaskRunner.set_budget_callback via the
        # starter→adapter→runner chain. _build_config appends it to
        # cfg["callbacks"] when non-None — the append condition IS this
        # None-check (no extra child gate needed: the setter's only production
        # call chain originates in the coordinator starter, which only ever
        # holds child instances).
        self._budget_callback = None

    def set_cancel_event(self, event: "asyncio.Event") -> None:
        """[C2 finish-core §5.1.1 G1a] External seam: the coordinator invoke-
        adapter injects the per-work-unit cancel_event into a CHILD flow
        (coord_deps Null) so react_graph cancel checkpoints observe it."""
        self._cancel_event = event
        self._cancel_event_externally_injected = True

    def set_child_permission_context(self, cpc) -> None:
        """[C2b §4.1] External seam (mirror set_cancel_event): the coordinator
        invoke-adapter injects the child's ChildPermissionContext so
        _build_config threads it (+ the SSM) into the child graph cfg for the
        tool_node child-scope guard."""
        self._child_permission_context = cpc

    def set_budget_callback(self, cb) -> None:
        """[C2b budget §3-5] External seam (mirror set_cancel_event): the
        coordinator starter late-injects the child's BudgetEnforcementCallback
        (via adapter → AgentTaskRunner → here) so _build_config appends it to
        the graph callbacks list alongside the cost handler."""
        self._budget_callback = cb

    @property
    def summary_llm(self):
        return self._summary_llm

    async def _get_checkpointer(self):
        """Lazy-initialize checkpointer.

        Returns injected checkpointer (test via MemorySaver) or creates
        AsyncPostgresSaver backed by the shared connection pool (prod).

        Note: AsyncPostgresSaver(pool) must be called in an async context
        because its __init__ calls asyncio.get_running_loop().
        """
        if self._checkpointer is not None:
            return self._checkpointer
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        self._checkpointer = AsyncPostgresSaver(self._checkpointer_pool)
        return self._checkpointer

    async def close(self) -> None:
        """Release checkpointer reference. Pool connections are managed by the pool."""
        self._checkpointer = None

    def _get_skill_context_seed(self) -> str:
        """Return the current skill context string from the runner callback.

        Used by ``_run_planner_for_detection`` and ``invoke()``'s
        ``input_for_graph`` construction to seed the skill_context value.
        Returns ``""`` when the provider is not wired (test scenarios
        that construct ``PlannerReActFlow`` without a runner).
        """
        if self._skill_context_provider is None:
            return ""
        return self._skill_context_provider()

    # -- Tool collection sub-methods ------------------------------------------

    def _collect_native_tools(self) -> list:
        """Collect sandbox/browser/search tools."""
        return create_native_tools(
            sandbox=self._sandbox, browser=self._browser,
            search_engine=self._search_engine,
            processor_lookup=self._file_processor_lookup,
            supports_vision=self._supports_vision,
            supports_pdf_input=self._supports_pdf_input,
            memory_mount_scope=self._build_memory_mount_scope(),
            supervisor=self._execution_supervisor,
        )

    def _build_memory_mount_scope(self):
        """Build MemoryMountScope for client-side symlink guard (codex fix P0).

        走 ``memory_mount_scope.build_memory_mount_scope_from_settings``
        shared factory——同一语义要同时应用于 ``AgentTaskRunner._build_lc_tools_full``
        的 step graph 绑定路径。settings 拿不到（测试）时 factory 收 None 返 None。
        """
        try:
            from core.config import get_settings

            settings = get_settings()
        except Exception:
            return None
        from app.domain.services.tools.memory_mount_scope import (
            build_memory_mount_scope_from_settings,
        )

        return build_memory_mount_scope_from_settings(self._user_id, settings)

    async def _collect_mcp_tools(self) -> list:
        """Collect MCP tools with progressive loading.

        Small tool set (<=15): bind all directly.
        Large tool set (>15): only always_bind + discovery tools.
        """
        if not self._mcp_tool:
            return []
        MCP_AUTO_BIND_THRESHOLD = 15
        all_mcp_tools = self._mcp_tool.get_tools()
        tools: list = []
        if len(all_mcp_tools) <= MCP_AUTO_BIND_THRESHOLD:
            tools.extend(create_mcp_langchain_tools(self._mcp_tool, tool_names=None))
        else:
            if self._mcp_always_bind_names:
                tools.extend(create_mcp_langchain_tools(
                    self._mcp_tool, tool_names=self._mcp_always_bind_names,
                ))
            if self._mcp_tool_ref is not None and self._activated_mcp_tools_ref is not None:
                from app.domain.services.tools.langchain_mcp_discovery import create_mcp_discovery_tools
                tools.extend(create_mcp_discovery_tools(
                    mcp_tool_ref=self._mcp_tool_ref,
                    activated_tools_ref=self._activated_mcp_tools_ref,
                ))
        return tools

    def _collect_a2a_tools(self) -> list:
        """Collect A2A remote agent tools."""
        from app.domain.services.tools.langchain_a2a import create_a2a_langchain_tools
        return create_a2a_langchain_tools(self._a2a_tool)

    def _collect_skill_creation_tools(self) -> list:
        """Collect skill creation tools + conditional get_skill_guide."""
        tools = create_skill_langchain_tools(
            brainstorm_skill_tool=self._brainstorm_skill_tool,
            create_skill_tool=self._create_skill_tool,
        )
        if self._skill_pool_getter is not None:
            from app.domain.services.tools.langchain_skill_tools import create_skill_guide_tool
            tools.append(create_skill_guide_tool(
                skill_pool_ref=self._skill_pool_getter,
                file_listings_ref=self._file_listings_getter,
                sandbox_skill_root=self._sandbox_skill_root,
            ))
        return tools

    def _collect_memory_tools(self) -> list:
        """Create memory search/get (+save if fully wired) tools."""
        if not (self._memory_session_factory and self._memory_repo_factory):
            return []
        from app.domain.services.tools.memory_tools import create_memory_tools
        return create_memory_tools(
            embedding_provider=self._memory_embedding_provider,
            session_factory=self._memory_session_factory,
            repo_factory=self._memory_repo_factory,
            user_id=self._user_id,
            session_id=self._session_id,
            memory_write_service=self._memory_write_service,
            session_redis=self._memory_session_redis,
            session_save_cap=self._memory_session_save_cap,
            half_life_days=self._memory_config.half_life_days,
            mmr_lambda=self._memory_config.mmr_lambda,
        )

    async def _collect_all_tools(self) -> list:
        """Aggregate all tool categories for initial graph build.

        Order: native -> MCP -> A2A -> skill creation -> memory.
        Dynamic Skill tools are NOT included here — they are injected
        per-step by react_graph_provider.
        """
        tools: list = []
        tools.extend(self._collect_native_tools())
        tools.extend(await self._collect_mcp_tools())
        tools.extend(self._collect_a2a_tools())
        tools.extend(self._collect_skill_creation_tools())
        tools.extend(self._collect_memory_tools())
        self._has_memory_tools = any(
            t.name in ("memory_search", "memory_get", "memory_save") for t in tools
        )
        return wrap_tool_list_for_supervisor(tools, self._execution_supervisor)

    # -- Graph construction ---------------------------------------------------

    async def _ensure_graphs(self) -> None:
        """延迟构建工具列表和 LangGraph 图。

        在 invoke() 首次调用时执行，此时 MCP/A2A 已完成异步初始化，
        能正确获取到所有可用工具。后续调用会重新构建以反映工具变化。

        注意：此处只绑定 native + MCP + A2A + skill_creation 工具。
        动态 Skill 工具（来自 SkillTool）不在此绑定，而是由
        react_graph_provider 按步骤渐进式注入，避免一次性全量绑定。
        Planner 和 Executor 通过 skill_context（名称+描述）了解可用 Skill。
        """
        checkpointer = await self._get_checkpointer()
        lc_tools = await self._collect_all_tools()

        has_guide_tool = self._skill_pool_getter is not None
        tool_names = [t.name for t in lc_tools]
        logger.info(
            "基础工具列表 (%d tools, get_skill_guide=%s, 不含动态Skill): %s",
            len(lc_tools), has_guide_tool, tool_names,
        )

        # Context assembler (B2)
        from app.domain.services.graphs.context_assembler import ContextAssembler
        from app.domain.services.prompts.budget import compute_effective_window

        assembler = None
        if self._overflow_config:
            total_context_window = resolve_context_window(
                self._overflow_config.model_name, self._overflow_config,
            )
            # B5 C9: compute the effective history window by subtracting the
            # system prompt budget and the reserved output allocation. Both
            # ``ContextAssembler`` (here) and ``GradualCompactor.try_compact``
            # (in ``_check_overflow``) read this via the same helper so their
            # budgets stay in sync.
            effective_window = compute_effective_window(
                total_context_window=total_context_window,
                system_prompt_max_tokens=self._overflow_config.system_prompt_max_tokens,
                reserved_output_tokens=self._overflow_config.reserved_output_tokens,
            )
            # P1.2 fix: use composition-layer helper so domain code does NOT
            # import infrastructure directly. Same DI pattern as
            # _build_configurable() line ~1232.
            from app.application.composition import build_decision_recorder as _bdr
            assembler = ContextAssembler(
                estimator=TokenEstimator(
                    strategy=self._overflow_config.token_estimator,
                    model_name=self._overflow_config.model_name,
                ),
                effective_window=effective_window,
                # B5 C9: reserved_output_tokens is NOT passed here in the
                # new-API path — compute_effective_window already subtracted
                # it when deriving effective_window.
                safety_factor=self._overflow_config.token_safety_factor,
                tool_compress_trigger_ratio=self._overflow_config.tool_compress_trigger_ratio,
                decision_recorder=_bdr(),
            )
        self._assembler = assembler

        # B2 Recovery wrap — idempotent; safe to re-enter _ensure_graphs.
        if self._profile is not None and not self._recovery_wrapped:
            from app.infrastructure.external.llm.actus_recovery_chat_model import (
                wrap_with_recovery,
            )
            callback = None
            if self._compactor is not None and self._overflow_config is not None:
                callback = self._build_on_context_overflow_callback()
            self._llm = wrap_with_recovery(
                self._llm, profile=self._profile, on_context_overflow=callback,
            )
            self._recovery_wrapped = True

        self._react_graph = build_react_graph(
            llm=self._llm, tools=lc_tools, agent_config=self._agent_config,
            tool_result_max_chars=(
                self._overflow_config.tool_result_max_chars
                if self._overflow_config else 8000
            ),
            assembler=assembler,
            tool_runtime_config=self._tool_runtime,
        )
        memory_snapshot_provider = self._build_memory_snapshot_provider()

        # B5 PR-S2-2: hand the OTel-backed traced_node decorator to the
        # graph builder so each registered LangGraph node emits a
        # ``graph.node.<name>`` span. Composition layer is the only
        # site that imports both the OTel SDK and the domain graph.
        from app.application.composition import build_traced_node_decorator

        self._main_graph = build_main_graph(
            planner_llm=self._llm,
            react_graph=self._react_graph,
            summary_llm=self._summary_llm,
            uow_factory=self._uow_factory,
            session_id=self._session_id,
            agent_config=self._agent_config,
            checkpointer=checkpointer,
            assembler=assembler,
            prompt_assembler=self._prompt_assembler,
            supports_vision=self._supports_vision,
            memory_snapshot_provider=memory_snapshot_provider,
            node_decorator=build_traced_node_decorator(),
            _allow_default_prompt_assembler=self._allow_default_prompt_assembler,
        )
        self._graphs_built = True

    def _build_memory_snapshot_provider(
        self,
    ) -> "Callable[[], Awaitable[MemorySnapshot | None]] | None":
        """Return an async closure that fetches a ``MemorySnapshot`` per call.

        The provider is handed to ``build_main_graph`` and invoked by each
        node (planner / executor / updater) before it builds its
        ``RenderContext``. We keep the DI concerns here (session factory,
        repo factory, user_id, exception handling) and hand ``main_graph``
        a single, already-safe async callable.

        Returns ``None`` when any of the required dependencies is missing
        (test harnesses without memory wiring, or anonymous sessions with
        ``user_id == ""``). ``build_main_graph`` interprets ``None`` as
        "keep memory sections inert".

        On every invocation we open a fresh async session so each render
        gets an up-to-date view (the agent loop may have written new
        chunks between steps). Any DB / network / unexpected error is
        swallowed here and logged — memory is a nice-to-have; an
        unreachable DB must not abort the plan.
        """
        if not (self._user_id and self._memory_session_factory and self._memory_repo_factory):
            return None

        from app.domain.services.prompts.memory_snapshot import build_memory_snapshot

        session_factory = self._memory_session_factory
        repo_factory = self._memory_repo_factory
        user_id = self._user_id

        async def _provider() -> "MemorySnapshot | None":
            try:
                async with session_factory() as session:
                    repo = repo_factory(session)
                    return await build_memory_snapshot(repo, user_id)
            except Exception as exc:
                logger.warning(
                    "build_memory_snapshot failed (swallowed, user_id=%s): %s",
                    user_id, exc,
                )
                return None

        return _provider

    def _build_previous_plan_context(self) -> str:
        """构建前一轮计划的上下文摘要，包含步骤结果和生成的文件路径。

        当用户发送新消息时（如「输出为ppt」），Planner 需要知道前一轮做了什么、
        生成了哪些文件，才能正确理解用户意图并制定关联计划。
        """
        if not self.plan or not self.plan.steps:
            return ""
        lines = [f"### 前一轮任务回顾"]
        lines.append(f"- 目标：{self.plan.goal}")
        for s in self.plan.steps:
            status = "已完成" if s.done else "未完成"
            line = f"- 步骤「{s.description}」: {status}"
            if s.result:
                # result 通常是 JSON 字符串，包含 success/attachments/result 字段
                # 截取前 300 字符以控制 token 消耗
                line += f"\n  执行结果: {s.result[:300]}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _build_fallback_context_from_memory(raw_messages: list[dict]) -> str:
        """从 Memory 的原始消息中提取简要上下文，作为 summary 不可用时的兜底。

        提取逻辑：取第一条 user 消息（原始需求）和最后一条 assistant 消息
        （最终结果），拼接为简要回顾。
        """
        first_user = ""
        last_assistant = ""
        # 收集 tool 消息中提到的文件路径
        file_paths: list[str] = []

        for msg in raw_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if role == "user" and not first_user:
                first_user = content[:300]
            elif role == "assistant" and content.strip():
                last_assistant = content[:300]
            elif role == "tool" and content:
                # 从工具结果中提取文件路径（启发式，容忍少量误报）
                for token in content.split():
                    cleaned = token.rstrip(",.;:)]}\"'")
                    if cleaned.startswith("/home/") and "." in cleaned.rsplit("/", 1)[-1]:
                        file_paths.append(cleaned[:200])

        if not first_user:
            return ""

        lines = ["### 前一轮上下文（从记忆恢复）"]
        lines.append(f"- 用户需求：{first_user}")
        if last_assistant:
            lines.append(f"- 最终结果：{last_assistant}")
        if file_paths:
            unique = list(dict.fromkeys(file_paths))[:10]
            lines.append(f"- 涉及文件：{', '.join(unique)}")
        return "\n".join(lines)

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
        from app.domain.services.prompts import get_prompt_bundle

        bundle = get_prompt_bundle(getattr(plan, "language", "zh"))
        steps_summary = "\n".join(
            f"- {s.description}: {'完成' if s.status == ExecutionStatus.COMPLETED else '未完成'}"
            + (f"\n  结果: {s.result[:200]}" if s.result else "")
            for s in plan.steps
        )
        prompt = bundle.GENERATE_SUMMARY_PROMPT.format(
            round_number=len(existing) + 1,
            plan_goal=plan.goal,
            steps_summary=steps_summary,
        )
        messages = [HumanMessage(content=prompt)]
        structured = self._summary_llm.with_structured_output(ConversationSummaryResponse)
        # B4 M0: graph-external LLM call — must thread the session cost
        # callback explicitly (LangGraph context doesn't apply here).
        ainvoke_kwargs: dict[str, object] = {}
        if self._cost_callback_handler is not None:
            ainvoke_kwargs["config"] = {
                "callbacks": [self._cost_callback_handler],
                "metadata": {
                    "langgraph_node": "conversation_summary",
                    "langgraph_step": 0,
                },
            }
        parsed: ConversationSummaryResponse | None = await structured.ainvoke(
            messages, **ainvoke_kwargs
        )
        if parsed is None:
            parsed = ConversationSummaryResponse()
        return ConversationSummary(
            round_number=len(existing) + 1,
            user_intent=parsed.user_intent or plan.goal,
            plan_summary=parsed.plan_summary,
            execution_results=parsed.execution_results,
            decisions=parsed.decisions,
            unresolved=parsed.unresolved,
        )

    async def _check_overflow(self, memory: Memory) -> CompactionResult | None:
        """检测上下文溢出，根据水位触发渐进压缩。"""
        if not self._overflow_config or not self._overflow_config.context_overflow_guard_enabled:
            return None
        # _compactor is always non-None when _overflow_config is non-None (see __init__)
        msgs = dicts_to_messages(memory.messages)
        total_window = resolve_context_window(
            self._overflow_config.model_name, self._overflow_config
        )
        # B5 C9: pass the TOTAL context window to GradualCompactor, not the
        # effective window. The compactor's soft/hard trigger ratios (0.85 /
        # 0.95) are calibrated against the full model context — feeding it
        # the effective_window would shift thresholds earlier and cause
        # premature hard-compaction in the 0.85 - 0.95 utilization band.
        # The assembler still uses effective_window (see _build_graphs) —
        # both layers derive from the same config, but each uses the
        # appropriate input for its internal math.
        # B4 M0: graph-external — pass cost callback so Level-2 compaction's
        # summary LLM call lands in the cost ledger with
        # node_name="context_compaction". ``getattr`` guards against
        # bypass-init tests; production __init__ always sets the attribute.
        _cost_handler = getattr(self, "_cost_callback_handler", None)
        compact_config: dict | None = None
        if _cost_handler is not None:
            compact_config = {
                "callbacks": [_cost_handler],
                "metadata": {
                    "langgraph_node": "context_compaction",
                    "langgraph_step": 0,
                },
            }
        # Pre-compute messages_input_hash BEFORE try_compact (it may mutate messages reference)
        from app.domain.services.graphs.compaction import compute_messages_input_hash
        messages_input_hash = compute_messages_input_hash(msgs)

        result = await self._compactor.try_compact(
            messages=msgs,
            context_window=total_window,
            summary_llm=self._summary_llm,
            config=compact_config,
        )

        if result.level_applied > 0:
            memory.messages = messages_to_dicts(result.messages)
            from app.application.services.compaction_recorder import record_compaction
            from dataclasses import replace
            async with self._uow_factory() as uow:
                await uow.session.save_memory(self._session_id, "react", memory)
                compaction_id = await record_compaction(
                    uow=uow,
                    session_id=self._session_id,
                    result=result,
                    messages_input_hash=messages_input_hash,
                )
                await uow.db_session.commit()  # [R2-P1-3] explicit commit-with-raise

            # [VOK rec from CXR1] Use dataclasses.replace (codebase idiom) instead
            # of object.__setattr__.
            result = replace(result, compaction_id=compaction_id)

            # [CXR1-P2-6] Path A OTel — fire AFTER successful commit so we don't
            # emit an OTel event for a record that rolled back. NO attrs:
            # CANONICAL_ATTRIBUTES is FROZEN per spec § Section 2 / [R2-P1-5];
            # passing compaction_id here would be silently dropped.
            try:
                # Use composition-layer helper rather than direct infrastructure
                # import, following the same DI pattern as _build_configurable()
                # line ~1224. Domain code MUST NOT import infrastructure directly.
                from app.application.composition import build_decision_recorder
                _recorder = build_decision_recorder()
                primary_kind = result.operations[-1]["kind"] if result.operations else "hard_truncate"
                _recorder("compaction", outcome=primary_kind)
            except Exception as exc:  # pragma: no cover — record_decision swallows OTel emit failures internally; this guards only programmer errors
                logger.debug("OTel record_decision('compaction') unexpectedly raised: %s", exc)

            logger.info(
                "compaction level=%d, %d→%d tokens, removed %d msgs, id=%s",
                result.level_applied, result.tokens_before,
                result.tokens_after, result.messages_removed, compaction_id,
            )

        self._last_compaction_result = result
        return result

    def _build_on_context_overflow_callback(self):
        """Session-scoped OnContextOverflow closure consumed by ActusRecoveryChatModel
        when R4 (TriggerRecompact) fires on a CONTEXT_OVERFLOW classification.

        Mirrors `_check_overflow` exactly so the proactive (post-tool) and emergency
        (provider-400) compact paths share contract:
          - Audit Round 9 P2 #2: respect `context_overflow_guard_enabled`. When the
            guard is disabled the proactive path is off; the emergency path must
            obey the same switch so users see consistent behavior.
          - Audit Round 9 P1 #1: forward the same `cost_handler` + `langgraph_node=
            "context_compaction"` metadata so Level-2 summary LLM calls land in
            the B4 cost ledger.
          - Audit Round 11 P1 #1: B2 hard-depends on `GradualCompactor.try_compact`
            threading `config=` through to `summary_llm.ainvoke`. That contract
            shipped in B4 M0 (`compaction.py:153-160` / `:411-417`); the
            regression-lock test in Step 5b protects it.
        """
        compactor = self._compactor
        overflow_config = self._overflow_config
        summary_llm = self._summary_llm
        cost_handler = getattr(self, "_cost_callback_handler", None)

        async def callback(messages, kwargs):
            if not overflow_config.context_overflow_guard_enabled:
                return None
            # Module-scope binding (see Step 1) so monkeypatch.setattr(
            # planner_react, "resolve_context_window", ...) takes effect here.
            total_window = resolve_context_window(overflow_config.model_name, overflow_config)
            compact_config: dict | None = None
            if cost_handler is not None:
                compact_config = {
                    "callbacks": [cost_handler],
                    "metadata": {
                        "langgraph_node": "context_compaction",
                        "langgraph_step": 0,
                    },
                }
            result = await compactor.try_compact(
                messages=messages,
                context_window=total_window,
                summary_llm=summary_llm,
                config=compact_config,
            )
            if result.tokens_after >= result.tokens_before:
                return None
            if result.level_applied == 0:
                return None
            # [R4-P2-3] Path B is fully SSE-silent + DB-silent. Only OTel decision
            # fires so ops dashboards can count recovery-triggered compactions.
            try:
                # Use composition-layer helper rather than direct infrastructure
                # import (same DI pattern as _build_configurable() line ~1224).
                from app.application.composition import build_decision_recorder
                _recorder = build_decision_recorder()
                primary_kind = (
                    result.operations[-1]["kind"] if result.operations else "hard_truncate"
                )
                _recorder(
                    "compaction_recovery_callback",
                    outcome=primary_kind,
                )
            except Exception:  # noqa: BLE001
                pass  # OTel hiccup must never abort the compaction callback
            return list(result.messages)

        return callback

    def _is_skill_graph_active(self) -> bool:
        return is_skill_graph_enabled(self._user_id, self._skill_graph_canary_percent)

    async def _try_drive_skill_graph(
        self, message: Message,
    ) -> AsyncGenerator[BaseEvent, None] | None:
        """Drive skill creation subgraph for continuation messages.

        Only handles cases where there's an existing graph state (from a
        previous brainstorm/generate step). Initial request detection is
        done via planner output in invoke().
        """
        sg_active = self._is_skill_graph_active()
        has_tools = bool(self._brainstorm_skill_tool and self._create_skill_tool)
        if not sg_active:
            return None
        if not has_tools:
            return None

        action = message.skill_confirmation_action

        async with self._uow_factory() as uow:
            graph_state = await uow.session.get_skill_graph_state(self._session_id)

        # 无已有子图状态 → 不是续接消息，交由 invoke() 的 planner 路由处理
        if graph_state is None:
            return None
        if graph_state.is_terminal:
            return None

        # 防御性校验：检测持久化恢复后状态是否完整
        # 已知场景：CancelledError/断连导致状态部分写入或未写入，
        # DB 中仅剩 status="wait_generate"（默认值）而其余字段为空。
        if not graph_state.original_request:
            logger.warning(
                "Skill 子图状态不完整（original_request 为空），"
                "可能是持久化失败导致的残留状态，清除后走正常流程: "
                "session=%s status=%s",
                self._session_id, graph_state.status,
            )
            try:
                async with self._uow_factory() as uow:
                    await uow.session.clear_skill_graph_state(self._session_id)
            except Exception as exc:
                logger.warning("清除残留 Skill 子图状态失败: %s", exc)
            return None

        async def _drive() -> AsyncGenerator[BaseEvent, None]:
            graph = SkillCreationGraph(
                brainstorm_tool=self._brainstorm_skill_tool,
                create_skill_tool=self._create_skill_tool,
            )
            try:
                new_state, events = await graph.run(
                    state=graph_state,
                    action=action,
                    original_request=graph_state.original_request,
                )
            except Exception as exc:
                # graph.run() 抛出未捕获异常时，保留原始 graph_state 不清除，
                # 确保下次重试时仍能恢复 blueprint/original_request
                logger.error(
                    "Skill 子图执行异常（原始状态已保留，可重试）: %s", exc,
                    exc_info=True,
                )
                yield MessageEvent(
                    role="assistant",
                    message="Skill 创建遇到错误，请重试或取消。",
                )
                return
            # 关键：先持久化状态，再 yield 事件。
            ok = await self._persist_skill_graph_state(new_state)
            if not ok:
                yield MessageEvent(
                    role="assistant",
                    message="Skill 状态保存失败，请重试。",
                )
                return
            for event in events:
                yield event

        return _drive()

    async def _persist_skill_graph_state(self, state: Any) -> bool:
        """持久化 SkillCreationGraph 的状态（终态清除，非终态保存）。

        Returns
        -------
        bool : 持久化是否成功。调用方应检查返回值，
               持久化失败时不应继续展示蓝图等事件。
        """
        if state is None:
            return True
        try:
            async with self._uow_factory() as uow:
                if state.is_terminal:
                    await uow.session.clear_skill_graph_state(self._session_id)
                else:
                    await uow.session.save_skill_graph_state(
                        self._session_id, state,
                    )
            return True
        except BaseException as exc:
            logger.error(
                "Skill 子图状态持久化失败: session=%s %s",
                self._session_id, exc, exc_info=True,
            )
            return False

    async def _start_skill_subgraph(
        self, message: Message,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Start the skill creation subgraph for an initial request.

        Called after planner detects skill creation intent. Drives
        brainstorm_skill → WaitEvent, then persists state for continuation.
        """
        graph = SkillCreationGraph(
            brainstorm_tool=self._brainstorm_skill_tool,
            create_skill_tool=self._create_skill_tool,
        )
        try:
            new_state, events = await graph.run(
                state=None,
                action=None,
                original_request=message.message,
            )
        except Exception as exc:
            logger.error("Skill 初始子图执行异常: %s", exc, exc_info=True)
            yield MessageEvent(
                role="assistant",
                message="Skill 蓝图生成失败，请重新发起创建请求。",
            )
            return
        # 关键：先持久化状态，再 yield 事件。
        # 如果持久化失败，不展示蓝图（避免用户确认后找不到状态）。
        ok = await self._persist_skill_graph_state(new_state)
        if not ok:
            yield MessageEvent(
                role="assistant",
                message="Skill 蓝图生成成功但状态保存失败，请重试。",
            )
            return
        for event in events:
            yield event

    async def _run_planner_for_detection(
        self, message: Message, summary_texts: list[str],
    ) -> tuple[Plan, list[BaseEvent]]:
        """Run the planner LLM and return (Plan, plan_events).

        Used for intent detection before deciding whether to route to skill
        creation subgraph or normal main_graph flow. The plan is reused by
        main_graph (skipping planner_node) to avoid double LLM calls.
        """
        from app.domain.services.prompts import (
            get_prompt_bundle,
            get_prompt_section_bundle,
        )
        from app.domain.services.prompts.render_context import build_render_context
        from app.domain.services.prompts.section import PromptMode

        bundle = get_prompt_bundle(message.language)
        lang = message.language

        attachments = getattr(message, "attachments", [])
        image_blocks = getattr(message, "image_content_blocks", [])
        prompt = bundle.CREATE_PLAN_PROMPT.format(
            message=message.message,
            attachments=format_attachments_text(
                attachments, has_image_blocks=bool(image_blocks), for_planner=True,
                supports_vision=self._supports_vision,
            ),
        )

        # B5 C7.5 + post-audit MEDIUM #2: PromptAssembler is the single code
        # path. This call site is OUTSIDE the main graph (pre-graph language
        # detection), so we build a minimal state dict from local fields.
        # The DI contract mirrors ``build_main_graph``: missing assembler is
        # a production misconfiguration, test-only paths opt in via
        # ``_allow_default_prompt_assembler``.
        if self._prompt_assembler is None:
            if not self._allow_default_prompt_assembler:
                raise RuntimeError(
                    "_run_planner_for_detection requires a PromptAssembler "
                    "instance. In production, AgentTaskRunner._build_prompt_assembler "
                    "constructs one and passes it via PlannerReActFlow. If "
                    "this is a test that needs the default, construct the "
                    "flow with _allow_default_prompt_assembler=True."
                )
            from app.domain.services.graphs.token_estimator import TokenEstimator
            from app.domain.services.prompts.assembler import (
                PromptAssembler as _PromptAssemblerImpl,
            )
            from app.domain.services.prompts.budget import SystemPromptBudget

            logger.warning(
                "PlannerReActFlow._run_planner_for_detection: "
                "prompt_assembler is None and _allow_default_prompt_assembler=True — "
                "constructing a minimal default. This path is intended for "
                "tests only; production should always inject a configured "
                "PromptAssembler via AgentTaskRunner."
            )
            self._prompt_assembler = _PromptAssemblerImpl(
                budget=SystemPromptBudget(max_tokens=10000),
                token_estimator=TokenEstimator(strategy="hybrid"),
                telemetry=None,
            )
        section_bundle = get_prompt_section_bundle(lang)
        detection_state = {
            "language": lang,
            "skill_context": self._get_skill_context_seed(),
            "conversation_summaries": list(summary_texts),
        }
        detection_config = {"configurable": {}}
        ctx = build_render_context(
            detection_state, detection_config, self._agent_config
        )
        result = self._prompt_assembler.assemble(
            section_bundle.planner,
            ctx,
            PromptMode.FULL,
            # Detection-path planner has no react_graph_provider by design.
            fallback_used=False,
        )
        system_content = result.text

        # Planner 不传图片（同 main_graph.planner_node），避免幻觉图片内容
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt),
        ]
        structured = self._llm.with_structured_output(PlanResponse)
        # B4 M0: graph-external LLM call. When skill tools route through the
        # detection planner, the produced plan is reused and the in-graph
        # planner_node is skipped (see _try_drive_skill_graph reuse path),
        # so without this the entire planner cost vanishes from the ledger.
        # Attribute to "planner_node" so by_node aggregation matches the
        # in-graph planner bucket.
        # ``getattr`` guards against bypass-init tests (object.__new__ /
        # MagicMock); production __init__ always sets the attribute.
        _cost_handler = getattr(self, "_cost_callback_handler", None)
        ainvoke_kwargs: dict[str, object] = {}
        if _cost_handler is not None:
            ainvoke_kwargs["config"] = {
                "callbacks": [_cost_handler],
                "metadata": {
                    "langgraph_node": "planner_node",
                    "langgraph_step": 0,
                },
            }
        try:
            parsed: PlanResponse | None = await structured.ainvoke(
                messages, **ainvoke_kwargs
            )
        except Exception:
            logger.warning("Planner structured output failed in detection, using fallback plan")
            parsed = None

        if parsed is None:
            parsed = PlanResponse(
                title="Task",
                goal=message.message,
                language=message.language,
                steps=[StepDef(description=message.message)],
                message="好的，我来帮你处理。",
            )
        parsed = salvage_empty_memory_recall_plan(
            parsed,
            user_message=message.message,
            fallback_language=message.language,
            has_memory_tools=self._has_memory_tools,
            skill_context=self._get_skill_context_seed(),
        )

        # [C2 PR-1 §4.2 r7 P0-2] Build steps via deterministic helper so
        # StepDef.id is propagated directly; when omitted, fall back to
        # _assign_fallback_step_id (no random UUIDs).
        from app.domain.services.graphs.main_graph import (
            _assign_fallback_step_id,
            _build_plan_from_response,
        )

        steps = _build_plan_from_response(parsed).steps
        if not steps:
            steps = [
                Step(
                    id=_assign_fallback_step_id("default", 0),
                    description=message.message,
                )
            ]
        plan = Plan(
            title=parsed.title or "Task",
            goal=parsed.goal or message.message,
            language=parsed.language or message.language,
            steps=steps,
            message=parsed.message or "",
            status=ExecutionStatus.RUNNING,
        )

        events: list[BaseEvent] = [
            TitleEvent(title=plan.title),
            MessageEvent(role="assistant", message=plan.message),
            PlanEvent(plan=plan, status=PlanEventStatus.CREATED),
        ]

        return plan, events

    @staticmethod
    def _plan_uses_skill_creation(plan: Plan) -> bool:
        """Check if the planner's plan references skill creation tools.

        The planner prompt instructs it to output step descriptions containing
        'brainstorm_skill → generate_skill → install_skill' for skill creation
        requests. However, LLM 可能把关键词放在 message/goal 中而非 step
        description（甚至返回空 steps），因此需要同时检查多个字段。
        """
        parts: list[str] = [s.description for s in plan.steps]
        parts.append(plan.message or "")
        parts.append(plan.goal or "")
        text = " ".join(parts).lower()
        return "brainstorm_skill" in text or "generate_skill" in text

    def _build_config(self) -> dict:
        """Build the LangGraph config dict shared by invoke() and resume()."""
        # TODO(PR-9 integration): When ACTUS_C2_COORDINATOR_ENABLED=true,
        # this method MUST also wire the C2 coordinator dispatch path into
        # ``configurable`` so the parallel-execution subgraph and child
        # runners have everything they need.
        #
        # Subgraph / orchestration deps (consumed by
        # ``parallel_execution_subgraph.py`` ``dispatch_node`` /
        # ``worker_node`` / ``reducer_node`` + ``main_graph._run_parallel_backend``):
        #
        #   - "parallel_execution_subgraph": built via
        #     ``build_parallel_execution_subgraph()`` (PR-6, isolated).
        #   - "session_service": existing UoW-managed write API used by
        #     ``dispatch_node`` to ``create_session_with_parent`` and by
        #     ``_rehydrate_dispatch`` to ``bump_coordinator_attempt`` /
        #     fetch the child session map.
        #   - "rehydrate_service": pod-restart rehydrate path
        #     (``parallel_execution_subgraph.py:214``).
        #   - "child_runner_starter": runner_starter adapter (PR-5) that
        #     wraps ``AgentTaskRunner`` in the
        #     ``CoordinatorChildInnerRunner`` Protocol.
        #   - "mailbox_publisher" / "mailbox_subscriber": the live Redis
        #     publisher and subscriber adapters (consumed at lines
        #     ``parallel_execution_subgraph.py:269`` and ``:468``).
        #   - "envelope_factory": ``CoordinatorEnvelopeFactory`` —
        #     defaultable but the explicit DI wire avoids per-call
        #     ``CoordinatorEnvelopeFactory()`` ctor.
        #   - "orchestrator_factory" / "terminal_waiter" / "probe_quota":
        #     DI from interfaces/service_dependencies — PR-6 ships the
        #     classes; PR-9 wires the singletons.
        #   - "coordinator_limits": ``load_coordinator_limits_from_env()``
        #     (PR-6 §14.3 constants + per-env override validation).
        #   - "session_repository": already exists in
        #     service_dependencies; thread for the descendants cap.
        #   - "cancel_event": parent cancel ``asyncio.Event``; observed by
        #     the orchestrator + worker_node waiter. Children get their OWN
        #     per-work-unit events (dispatch creates them — C2b budget D9);
        #     user cancel reaches children via CANCEL_REQUEST envelope
        #     fan-out, not via this shared object.
        #   - "user_id": already wired at line ~1322 (
        #     ``self._user_id``); read by ``_run_parallel_backend`` (
        #     ``main_graph.py:96``) + ``dispatch_node`` for the user-scoped
        #     concurrency cap / descendants count. Listed here for
        #     completeness — no new wiring needed in PR-9, just confirm
        #     ``self._user_id`` continues to be populated for the
        #     coordinator-enabled call site.
        #
        # Apply path deps (consumed by ``main_graph._run_parallel_backend``
        # post-reducer):
        #
        #   - "patch_reducer_service": invoked from ``reducer_node`` to
        #     deduplicate + finalize PatchApplyPlan.
        #   - "patch_applier": invoked by main_graph after reducer when
        #     ``apply_plan.is_apply_required is True``.
        #   - "parent_sandbox": needed by ``dispatch_node`` for
        #     ``compute_digest`` / ``read_file`` (base_digest +
        #     seed_content_ref enrichment) and by ``reducer_node`` for
        #     §9.3 step-5 drift detection.
        #   - "artifact_storage": MinIO put for spawn manifests + seed
        #     content (``put_content_addressed_bytes``).
        #
        # All PR-6 components are isolated/standalone and DO NOT require
        # this wiring to ship in PR-6; they activate only when PR-9
        # flips the feature flag and dispatch starts driving the
        # parallel subgraph. See ``coordinator_child_runner.py:235``
        # for the matching wiring TODO on the child side
        # (BudgetEnforcementCallback + CoordinatorChildWallclockWatchdog
        # — both shipped in PR-6, both dead-coded until PR-9 binds them
        # to the inner_runner's LLM callbacks list).
        from app.domain.services.execution_watchdog import ExecutionControl, ExecutionWatchdog

        # Read tool confirmation settings from AgentConfig (config.yaml, user-editable)
        tc = getattr(self._agent_config, "tool_confirmation", None)
        tc_enabled = getattr(tc, "enabled", True) if tc else True
        tc_timeout = getattr(tc, "timeout_seconds", 300) if tc else 300
        tc_smart_approve = getattr(tc, "smart_approve_enabled", False) if tc else False
        tc_smart_approve_medium_only = getattr(tc, "smart_approve_medium_only", False) if tc else False

        # D5: Create fresh watchdog + control per invoke/resume (timer resets).
        # Tracker + metrics persist on self (survive across invoke/resume).
        ec = self._execution_config
        watchdog = ExecutionWatchdog(
            total_timeout_seconds=ec.total_timeout_seconds,
            idle_timeout_seconds=ec.idle_timeout_seconds,
        )
        control = ExecutionControl()

        cfg: dict = {
            "configurable": {
                "thread_id": self._session_id,
                "skill_context_refresher": self._skill_context_refresher,
                "react_graph_provider": self._react_graph_provider,
                "skill_guide_injector": self._skill_guide_injector,
                "has_file_view": self._file_processor_lookup is not None,
                "has_memory_tools": self._has_memory_tools,
                "approval_state_reader": self._approval_state_reader,
                "skill_tool": self._skill_tool,  # PE SkillSource metadata build + fail-closed skill guard
                "confirmation_manager": self._confirmation_manager,
                "user_id": self._user_id,
                "session_id": self._session_id,
                "tool_confirmation_enabled": tc_enabled,
                # PE-4c: per-source flags retired. The gate helper inside
                # react_graph reads the master ``enabled`` switch off the
                # tool_confirmation config object itself.
                "tool_confirmation_config": tc,
                "smart_approve_enabled": tc_smart_approve,
                "smart_approve_medium_only": tc_smart_approve_medium_only,
                "summary_llm": self._summary_llm if hasattr(self, "_summary_llm") else None,
                "tool_confirmation_timeout_seconds": tc_timeout,
                # D5: Execution health monitoring
                "execution_watchdog": watchdog,
                "execution_control": control,
                "tool_failure_tracker": self._tool_failure_tracker,
                "execution_metrics": self._execution_metrics,
                # B5 C5a: make the LLM adapter and AgentConfig available to
                # executor_node so PromptAssembler consumers (C5b) can read
                # provider name / model details without re-injecting via state.
                # bound_tool_names is NOT injected here — it is per-step and
                # gets written by react_graph_provider on each call (C5a Phase 3).
                "llm": self._llm,
                "agent_config": self._agent_config,
                # B5 post-audit LOW #1: optional callback for main_graph
                # nodes to notify the runner of session language changes.
                # Injected by ``AgentTaskRunner`` before each session start.
                "language_callback": self._language_callback,
            }
        }
        # B5 PR-S3-2 reviewer round-2 P3: thread the OTel-backed
        # ``decision_recorder`` through configurable so domain decision
        # points (``SmartApprove`` constructed inside ``react_graph``)
        # can call it without importing infrastructure directly.
        # Composition layer owns the OTel wiring; domain receives only
        # an opaque callable. Recovery records decisions inside its
        # infrastructure wrapper and bypasses this hook.
        from app.application.composition import build_decision_recorder

        cfg["configurable"]["decision_recorder"] = build_decision_recorder()
        # PE-0 Phase 7: inject PermissionEngine + SessionStateMachine into
        # LangGraph configurable. ``tc`` is already resolved above from
        # self._agent_config. Both PE and SSM must be non-None AND the master
        # switch must be True for injection — missing either means fall back
        # to legacy path. (PE-4c: per-source flags retired.)
        # P2#6: respect the tool_confirmation.enabled master switch.
        # When tc.enabled is False the operator intends dangerous tools to execute
        # without any confirmation gate (legacy semantics).  Injecting PE here while
        # tc.enabled=False would route tool calls through PE's Stage S/P chain,
        # which may enqueue confirmation requests even though the operator disabled
        # the feature.  Guard: only inject PE when tc is None (no config, default on)
        # OR tc.enabled is True.
        tc_master_enabled = (
            bool(getattr(tc, "enabled", True))
            if tc is not None
            else True
        )
        # PE-1 §2.5: _create_task already gated the PE/SSM build by
        # is_pe_enabled_for_source for every supported source. Here we only
        # need to inject IF both objects exist AND the master switch is on.
        # Per-source gating is performed inside react_graph._pe_dispatch
        # via is_pe_enabled_for_source on the per-call tool_source.
        if (
            self._permission_engine is not None
            and self._session_state_machine is not None
            and tc_master_enabled
        ):
            cfg["configurable"]["permission_engine"] = self._permission_engine
            cfg["configurable"]["session_state_machine"] = self._session_state_machine
        # B4 M0: attach the session-scoped CostCallbackHandler so every LLM
        # call inside the graph (planner, executor, updater, summarizer)
        # fires on_chat_model_start → on_llm_end and writes a CostRecord.
        # LangGraph merges ``callbacks`` with the adapter's own callback
        # list so telemetry + cost stacking is additive.
        #
        # B5 PR-S2-2: append the OTel-backed tool span handler so each
        # tool invocation produces a ``tool.<name>`` span with
        # ``tool_args_hash`` (sha256[:16]) + ``tool_args_size`` and the
        # canonical join keys (trace_id / step_id from the contextvar
        # ``traced_node`` binds). Raw args are NEVER on the span.
        from app.application.composition import build_observability_callbacks

        callbacks: list[Any] = []
        if self._cost_callback_handler is not None:
            callbacks.append(self._cost_callback_handler)
        callbacks.extend(build_observability_callbacks())
        # [C2b budget §3-5] Budget enforcement rides the SAME list as the cost
        # handler — independent handlers, additive dispatch (INV-B5). Child
        # flows only (see set_budget_callback).
        if self._budget_callback is not None:
            callbacks.append(self._budget_callback)
        if callbacks:
            cfg["callbacks"] = callbacks

        # PR-9b-A A5: parallel-subgraph runtime deps (always-live wiring;
        # flag at main_graph.py:695 gates ENTRY to _run_parallel_backend,
        # not WIRING). When _coord_deps is the NullCoordinatorRuntimeDeps
        # sentinel (legacy tests + non-coordinator paths), the 19 cfg keys
        # are SKIPPED — INV-A10 guarantees zero side-effect ctors fire on
        # construction in that branch.
        if not isinstance(self._coord_deps, _NullCoordinatorRuntimeDeps):
            cd = self._coord_deps
            cfg["configurable"].update({
                "parallel_execution_subgraph": cd.parallel_execution_subgraph,
                "session_service": cd.session_service,
                "rehydrate_service": cd.rehydrate_service,
                "child_runner_starter": cd.child_runner_starter,
                "mailbox_publisher": cd.mailbox_publisher,
                "mailbox_subscriber": cd.mailbox_subscriber,
                "envelope_factory": cd.envelope_factory,
                "orchestrator_factory": cd.orchestrator_factory,
                "terminal_waiter": cd.terminal_waiter,
                "probe_quota": cd.probe_quota,
                "coordinator_limits": cd.coordinator_limits,
                "session_repository": cd.session_repository,
                "cancel_event": self._cancel_event,  # per-run, not in _coord_deps
                "patch_reducer_service": cd.patch_reducer_service,
                "patch_applier_deps": cd.patch_applier_deps,
                "parent_sandbox": self._coord_deps.parent_sandbox_adapter_factory(
                    self._sandbox
                ),  # [finish-core §5.2 G2] Port, not raw handle (domain stays infra-free)
                "artifact_storage": cd.artifact_storage,
                "cost_rollup_service": cd.cost_rollup_service,
                "coordinator_metrics_recorder": cd.coordinator_metrics_recorder,
            })
            # event_queue is intentionally NOT here — GraphEventBridge merges
            # {"event_queue": q} into configurable AT INVOCATION TIME
            # (event_bridge.py:74-79); main_graph.py:123 then passes the
            # merged cfg into parallel_execution_subgraph invocation.
        elif self._cancel_event is not None:
            # [C2 finish-core §5.1.1 INV-F1.11] Coordinator CHILD path:
            # coord_deps is Null (child must not be a nested coordinator) but
            # the invoke-adapter called set_cancel_event(...). Inject ONLY the
            # cancel_event so react_graph._should_cancel trips on parent cancel.
            cfg["configurable"]["cancel_event"] = self._cancel_event
            # [C2b §4.1] Child-scope guard wiring. Inject the ChildPermissionContext
            # + the SSM so react_graph's tool_node child-scope guard can enforce the
            # manifest/lease. This is INTENTIONALLY independent of the PE master
            # switch: a coordinator child runs with tool_confirmation.enabled=False
            # so the PE/SSM injection block above is skipped — but the guard still
            # needs the SSM for the live revision read. Injecting the SSM alone does
            # NOT enable PE dispatch (tool_node requires BOTH permission_engine AND
            # session_state_machine; the child has no permission_engine).
            if self._child_permission_context is not None:
                cfg["configurable"]["child_permission_context"] = (
                    self._child_permission_context
                )
                cfg["configurable"]["session_state_machine"] = (
                    self._session_state_machine
                )
        return cfg

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """Run the flow — delegates to LangGraph main_graph."""
        # 1. Continuation: existing skill graph state → drive subgraph
        subgraph_gen = await self._try_drive_skill_graph(message)
        if subgraph_gen is not None:
            async for event in subgraph_gen:
                yield event
            return

        # 延迟绑定：每次 invoke 重新构建工具和图，确保 MCP/A2A 已初始化
        await self._ensure_graphs()

        config = self._build_config()

        # === Before Graph: load summaries ===
        async with self._uow_factory() as uow:
            summaries = await uow.session.get_summary(self._session_id)

        # Load flush_cursor from persisted Memory (needed for both resume and new task)
        try:
            async with self._uow_factory() as uow:
                _memory_for_cursor = await uow.session.get_memory(self._session_id, "react")
                self._flush_cursor = _memory_for_cursor.flush_cursor
        except Exception:
            self._flush_cursor = 0

        # Detect pending interrupt via checkpointer state
        is_resume = False
        try:
            graph_state = await self._main_graph.aget_state(config)
            if graph_state and graph_state.next:
                is_resume = True
        except Exception:
            pass

        if is_resume:
            # Resume path: checkpointer has saved full state, pass user response.
            # IMPORTANT: If the interrupt was from a tool_confirmation (react_graph
            # tool_node), the resume value must be a structured dict like
            # {"action": "approve", "scope": "session"}. Plain text from the user
            # should NOT be routed here — tool confirmations use the dedicated
            # _resume_tool_confirmation() path in agent_service.
            # This path only handles message_ask_user interrupts (plain text is fine).
            input_for_graph = Command(resume=message.message)
            logger.info(
                "通过 checkpointer 恢复中断: session=%s, resume=%s",
                self._session_id, message.message[:50],
            )
        else:
            # New task path
            async with self._uow_factory() as uow:
                memory = await uow.session.get_memory(self._session_id, "react")

            # Context anchor injection
            if self._memory_config.context_anchor_enabled and not memory.empty:
                anchor = self._build_context_anchor(message)
                memory.add_message({"role": "user", "content": anchor})

            # Summary texts
            recent_summaries = summaries[-self._memory_config.summary_max_rounds:]
            summary_texts = [s.to_prompt_text() for s in recent_summaries]

            # 注入前一轮计划上下文（防御 conversation_summaries 为空的情况）
            # 即使 summary 生成失败，self.plan 仍然保留了上一轮的完整计划和步骤结果
            previous_plan_context = self._build_previous_plan_context()
            if previous_plan_context:
                summary_texts.insert(0, previous_plan_context)

            # Convert Memory dict messages to LangChain BaseMessage
            raw_messages = memory.get_messages()
            lc_messages = dicts_to_messages(raw_messages) if raw_messages else []

            # Fallback: 当 summary_texts 为空但 Memory 中有历史消息时，
            # 从 raw messages 中提取简要上下文。这处理以下场景：
            # - 新任务创建了新的 PlannerReActFlow（self.plan=None）
            # - 且 DB summary 生成失败（例如 LLM 404）
            # 此时 raw_messages 是唯一的历史上下文来源。
            if not summary_texts and raw_messages:
                fallback = self._build_fallback_context_from_memory(raw_messages)
                if fallback:
                    summary_texts.append(fallback)

            # 2. Planner-first routing: run planner, check for skill creation intent
            _skill_tools_available = (
                self._is_skill_graph_active()
                and self._brainstorm_skill_tool is not None
                and self._create_skill_tool is not None
            )
            if _skill_tools_available:
                plan, plan_events = await self._run_planner_for_detection(
                    message, summary_texts,
                )
                is_skill = self._plan_uses_skill_creation(plan)
                if is_skill:
                    # Emit plan events (title, message, plan) then start subgraph
                    for event in plan_events:
                        yield event
                    async for event in self._start_skill_subgraph(message):
                        yield event
                    return

                # Not skill creation: reuse pre-computed plan, skip planner_node
                # by setting flow_status=executing
                input_for_graph = {
                    "message": message.message,
                    "language": message.language,
                    "attachments": getattr(message, "attachments", []),
                    "image_content_blocks": getattr(message, "image_content_blocks", []),
                    "plan": plan,
                    "current_step": plan.get_next_step(),
                    "messages": lc_messages,
                    "execution_summary": "",
                    "events": [],
                    "flow_status": FlowStatus.EXECUTING.value,
                    "session_id": self._session_id,
                    "should_interrupt": False,
                    "resume_value": None,
                    "original_request": plan.goal,
                    "skill_context": self._get_skill_context_seed(),
                    "conversation_summaries": summary_texts,
                }
                # Emit pre-computed plan events before bridge
                for event in plan_events:
                    yield event
            else:
                # No skill creation tools → normal flow with planner_node
                input_for_graph = {
                    "message": message.message,
                    "language": message.language,
                    "attachments": getattr(message, "attachments", []),
                    "image_content_blocks": getattr(message, "image_content_blocks", []),
                    "plan": self.plan,
                    "current_step": None,
                    "messages": lc_messages,
                    "execution_summary": "",
                    "events": [],
                    "flow_status": self.status.value if hasattr(self.status, "value") else FlowStatus.IDLE.value,
                    "session_id": self._session_id,
                    "should_interrupt": False,
                    "resume_value": None,
                    "original_request": self.plan.goal if self.plan else "",
                    "skill_context": self._get_skill_context_seed(),
                    "conversation_summaries": summary_texts,
                }

        bridge = GraphEventBridge()
        try:
            async for event in bridge.run(self._main_graph, input_for_graph, config=config):
                yield event
        finally:
            # 使用 try/finally 确保持久化逻辑始终执行，即使消费方提前退出
            # （例如 WaitEvent 触发 agent_task_runner.run() 的 return，
            #  导致本 async generator 被 aclose()、GeneratorExit 抛入 yield 处）。
            # bridge.run() 的 finally 会 await 图任务完成，
            # 因此此处 bridge.final_state 已包含完整的图输出。
            if bridge.final_state.get("should_interrupt"):
                # 中断路径：同步持久化（保留原逻辑）
                await self._persist_after_graph(bridge.final_state, summaries)
            else:
                # 正常完成路径：延迟到 invoke() 的 FINISHING 阶段
                self._deferred_final_state = bridge.final_state
                self._deferred_summaries = summaries

    async def resume(self, command: Any) -> AsyncGenerator[BaseEvent, None]:
        """Resume the graph from a pending interrupt using a LangGraph Command.

        The command (e.g. Command(resume=value)) is passed directly as input
        to the main_graph. The same checkpointer/thread_id config is used so
        that LangGraph can restore the interrupted state and continue.
        """
        await self._ensure_graphs()

        config = self._build_config()

        async with self._uow_factory() as uow:
            summaries = await uow.session.get_summary(self._session_id)

        bridge = GraphEventBridge()
        try:
            async for event in bridge.run(self._main_graph, command, config=config):
                yield event
        finally:
            if bridge.final_state.get("should_interrupt"):
                await self._persist_after_graph(bridge.final_state, summaries)
            else:
                self._deferred_final_state = bridge.final_state
                self._deferred_summaries = summaries

    async def _persist_after_graph(
        self, final: dict, summaries: list[ConversationSummary],
    ) -> None:
        """Post-graph persistence: save memory and summaries.

        CRITICAL: 该方法在 invoke() 的 finally 块中调用（async generator 被 aclose
        时通过 GeneratorExit → finally 触发）。如果此方法抛出异常，异常会传播到
        agent_task_runner 的 except Exception 处理器。
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
        """Inner implementation of post-graph persistence.

        中断路径：checkpointer 已自动保存完整图状态，无需手动序列化 messages/step。
        仅保存 Memory（用于上下文锚点）和更新 flow 状态。
        """
        self.plan = final.get("plan")

        # 将 LangChain BaseMessage 转回 dict 用于 Memory 持久化
        raw_messages = final.get("messages", [])
        try:
            dict_messages = messages_to_dicts(raw_messages) if raw_messages else []
        except Exception as exc:
            logger.warning("messages_to_dicts 转换失败，使用空消息列表: %s", exc)
            dict_messages = []

        # Evaluate flush gate BEFORE Memory construction (uses raw_messages)
        try:
            await self._evaluate_flush_gate(
                raw_messages, final.get("plan") or self.plan,
            )
        except Exception as exc:
            logger.warning("flush gate 评估失败: %s", exc)

        if final.get("should_interrupt"):
            # Checkpointer has automatically saved full graph state for Command(resume=...).
            # Only persist Memory (for context anchors) and update flow status.
            self.status = FlowStatus.EXECUTING
            self.plan = final.get("plan")

            logger.info(
                "中断持久化 (checkpointer): session=%s plan=%s",
                self._session_id,
                self.plan.title if self.plan else "<none>",
            )

            memory = Memory(messages=list(dict_messages), flush_cursor=self._flush_cursor)
            memory.compact(keep_summary=self._memory_config.compact_keep_summary)
            try:
                async with self._uow_factory() as uow:
                    await uow.session.save_memory(self._session_id, "react", memory)
            except Exception as e:
                logger.warning(f"中断时保存 Memory 失败: {e}")
        else:
            # === After Graph: 记忆压缩、保存、摘要生成 ===
            memory = Memory(messages=list(dict_messages), flush_cursor=self._flush_cursor)
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

            # 正常完成：重置状态
            self.status = FlowStatus.IDLE

    # ── Flush scheduling: gate + chunking ────────────────────────────────────

    async def _evaluate_flush_gate(
        self,
        raw_messages: Sequence[BaseMessage],
        plan: Any,
    ) -> None:
        """Evaluate whether to produce a FlushBatch for background flushing.

        Two-stage gate:
        1. **Size gate** (legacy)：completed steps + new token count 必须
           达标，否则直接返回。把显然没实质内容的小 flush 挡在 LLM 费用
           产生之前。
        2. **LLM quality gate** (M1 PR-4+8)：启用时（``memory_gate_llm is
           not None``），每个 chunk 过 classifier 打分；threshold + daily
           cap + 进程级 circuit breaker 共同决定最终保留集。禁用时直接
           透传全部 chunk 与历史 size-only 行为一致。

        Side-effects:
        - Always clears ``_pending_flush_batch`` at entry.
        - Sets ``_pending_flush_batch`` to kept chunks when any pass.
        - May reset ``_flush_cursor`` on compaction shrink.
        """
        self._pending_flush_batch = None

        if not self._memory_config.flush_enabled:
            return

        current_len = len(raw_messages)

        # Compaction shrink: context was compacted and now shorter than cursor
        if current_len < self._flush_cursor:
            self._flush_cursor = current_len
            return

        # No new messages since last flush
        if current_len == self._flush_cursor:
            return

        # Count completed steps
        steps_completed = 0
        if plan is not None and hasattr(plan, "steps"):
            steps_completed = sum(
                1 for s in plan.steps
                if s.status == ExecutionStatus.COMPLETED
            )

        if steps_completed < self._memory_config.flush_min_steps:
            return

        # Estimate new tokens since last cursor
        new_messages = raw_messages[self._flush_cursor:]
        new_token_count = self._token_estimator.estimate_messages(new_messages)

        if new_token_count < self._memory_config.flush_min_new_tokens:
            return

        # Size gate passed — build chunks
        chunks = self._chunk_messages(list(new_messages), plan=plan)
        if not chunks:
            return

        # LLM quality gate
        kept_chunks = await self._apply_llm_gate(chunks)
        if not kept_chunks:
            # 推进 cursor，即使全部 drop——否则下一次 _evaluate_flush_gate
            # 会用同一段 new_messages 再跑一次 LLM gate，既浪费钱又会把
            # 同一批内容在审计视图里重复评估。
            self._flush_cursor = current_len
            return

        self._pending_flush_batch = FlushBatch(
            session_id=self._session_id,
            user_id=self._user_id,
            from_cursor=self._flush_cursor,
            target_cursor=current_len,
            chunks=tuple(kept_chunks),
        )
        self._flush_cursor = current_len  # C5.1: 乐观推进，在 persist 之前生效

    async def _emit_memory_notification(
        self,
        *,
        event_type: str,
        payload: dict,
    ) -> None:
        """Fire-and-forget notification write. Swallows emitter errors—
        a failed notification must never break the flush pipeline.
        No user_id → no-op (test paths that don't wire the emitter)."""
        if self._memory_notification_emitter is None:
            return
        if not self._user_id:
            return
        try:
            await self._memory_notification_emitter.emit(
                user_id=self._user_id,
                event_type=event_type,
                payload=payload,
            )
        except Exception as exc:
            logger.warning(
                "memory notification emission failed: event=%s err=%s",
                event_type, exc,
            )

    async def _apply_llm_gate(
        self,
        chunks: "list[RawChunk]",
    ) -> "list[RawChunk]":
        """Run the LLM quality gate and return kept chunks.

        Gate 禁用（``_memory_gate_llm is None``）时不做任何过滤——legacy
        size-only 路径，全部 chunk 透传，``category`` 和 ``auto_promoted_at``
        保持 None（入库后 ``MemoryChunk.category = NULL``，等同 M1 前语义）。

        Gate 启用时每个 drop 的 chunk 不会入 memory_chunks / fs。breaker
        OPEN / daily cap 打满 / LLM 异常这三类情况都走 "drop 全部" 路径
        （degraded fail-closed）；调用方下一轮 flush 继续尝试。
        """
        # Gate off → legacy passthrough
        if self._memory_gate_llm is None:
            return list(chunks)

        # Empty user_id + gate on = construction error. The daily cap
        # Redis key would collapse to ``memory:auto_promote_daily::<date>``,
        # pooling every anonymous user's quota into one bucket; the
        # notification emitter also refuses to fire. Safer to drop the
        # batch silently than to poison shared state.
        if not self._user_id:
            logger.warning(
                "memory_gate enabled but user_id is empty — dropping batch; "
                "wire a real user_id or disable the gate for this session",
            )
            return []

        # Circuit breaker OPEN → drop everything for this flush without
        # calling LLM or Redis. Rising-edge notification happens inside the
        # except branch below via ``record_failure()``; the OPEN-on-entry
        # path re-uses the existing notification (no re-emit to avoid tray
        # flooding when LLM is down for many consecutive flushes).
        if self._memory_gate_breaker is not None and self._memory_gate_breaker.is_open():
            logger.info(
                "memory_gate circuit breaker OPEN — dropping %d chunks "
                "without LLM call", len(chunks),
            )
            return []

        # Truncate to per-batch cost cap before anything else. Chunks past
        # the cap don't silently fall through to storage—they are simply
        # not offered to the LLM and never get ``auto_promoted_at`` set,
        # so they stay dropped (not retried with a fresh cursor, because
        # _evaluate_flush_gate advances the cursor past the full window).
        if len(chunks) > self._memory_gate_batch_cap:
            logger.info(
                "memory_gate batch cap %d < %d chunks; truncating tail",
                self._memory_gate_batch_cap, len(chunks),
            )
            chunks = chunks[: self._memory_gate_batch_cap]

        # Classify via LLM
        from app.domain.services.memory_gate import (
            MemoryGateClassifier,
            MemoryGateInput,
            filter_kept_decisions,
        )
        classifier = MemoryGateClassifier(self._memory_gate_llm)
        try:
            inputs = [
                MemoryGateInput(chunk_index=i, text=c.content)
                for i, c in enumerate(chunks)
            ]
            # B4 M0: graph-external — thread cost callback so gate tokens
            # reach the ledger with node_name="memory_gate".
            gate_config: dict | None = None
            if self._cost_callback_handler is not None:
                gate_config = {
                    "callbacks": [self._cost_callback_handler],
                    "metadata": {
                        "langgraph_node": "memory_gate",
                        "langgraph_step": 0,
                    },
                }
            decisions = await classifier.classify(inputs, config=gate_config)
        except Exception as exc:
            just_opened = False
            if self._memory_gate_breaker is not None:
                just_opened = self._memory_gate_breaker.record_failure()
            logger.warning("memory_gate LLM classify failed: %s", exc)
            # Only emit on the rising edge (counter just crossed threshold)
            # so the user's tray doesn't flood with identical notifications
            # when LLM is down for many flushes in a row.
            if just_opened:
                await self._emit_memory_notification(
                    event_type="memory_gate_paused",
                    payload={
                        "consecutive_failures": (
                            self._memory_gate_breaker.consecutive_failures
                        ),
                        "last_error": str(exc)[:500],
                        "cooldown_until": _iso_or_none(
                            self._memory_gate_breaker.cooldown_until()
                        ),
                    },
                )
            return []
        if self._memory_gate_breaker is not None:
            self._memory_gate_breaker.record_success()

        kept_decisions = filter_kept_decisions(
            decisions, threshold=self._memory_gate_threshold,
        )
        if not kept_decisions:
            return []

        # Daily cap — reserve N slots; if oversubscribed, drop all kept
        # chunks this flush. "Partial-grant" (keep top-K, drop rest) is
        # tempting but picks a semantic: whose top-K? By confidence?
        # By order? For M1 we choose "all or nothing per flush" to keep
        # the failure mode simple; M2 eval can revisit.
        if self._memory_gate_daily_cap is not None:
            granted, _remaining = await self._memory_gate_daily_cap.try_reserve(
                self._user_id, len(kept_decisions),
            )
            if not granted:
                logger.info(
                    "memory_gate daily cap exhausted for user=%s; "
                    "dropping %d kept chunks",
                    self._user_id, len(kept_decisions),
                )
                # Unlike breaker rising-edge, daily cap notification fires
                # at most once per user per day because downstream flushes
                # short-circuit to `granted=False` too; dedup-by-day in the
                # store is not added yet (accepted M1 noise ceiling — UI
                # polling coalesces identical event_types visually).
                await self._emit_memory_notification(
                    event_type="quota_exceeded",
                    payload={
                        "attempted": len(kept_decisions),
                        "cap": getattr(
                            self._memory_gate_daily_cap, "cap", None,
                        ),
                    },
                )
                return []

        # Enrich kept chunks with gate-assigned category + promotion ts.
        # Decisions came back keyed by chunk_index (== position in the
        # truncated ``chunks`` list); we re-resolve via a dict so LLM
        # output order doesn't matter.
        now = datetime.now(timezone.utc)
        by_index = {d.chunk_index: d for d in kept_decisions}
        kept_chunks: list[RawChunk] = []
        for i, c in enumerate(chunks):
            d = by_index.get(i)
            if d is None:
                continue
            kept_chunks.append(
                RawChunk(
                    content=c.content,
                    session_id=c.session_id,
                    user_id=c.user_id,
                    source=c.source,
                    metadata=c.metadata,
                    content_hash=c.content_hash,
                    category=d.category,
                    auto_promoted_at=now,
                )
            )
        return kept_chunks

    def _chunk_messages(
        self,
        messages: list[BaseMessage],
        plan: Any = None,
    ) -> list[RawChunk]:
        """Convert a sequence of LangChain messages into RawChunks.

        Phases:
        1. Group messages (skip SystemMessage, merge AIMessage+ToolMessage groups)
        2. Merge short text groups (<50 tokens)
        3. Split oversized groups (>500 tokens) with paragraph overlap
        4. Build RawChunks with metadata
        """
        # Phase 1: Group messages
        groups: list[dict[str, Any]] = []
        current_group: dict[str, Any] | None = None

        for idx, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                continue

            text = self._message_to_text(msg)
            msg_type = type(msg).__name__

            if isinstance(msg, ToolMessage):
                # Merge ToolMessage into the preceding AIMessage group
                if current_group is not None:
                    current_group["texts"].append(text)
                    current_group["message_types"].add(msg_type)
                    tool_name = getattr(msg, "name", "") or ""
                    if tool_name:
                        current_group["tool_names"].add(tool_name)
                else:
                    # Orphaned ToolMessage — start a new group
                    current_group = {
                        "texts": [text],
                        "message_types": {msg_type},
                        "tool_names": {getattr(msg, "name", "") or ""},
                        "turn_index": idx,
                    }
                    groups.append(current_group)
                continue

            if isinstance(msg, AIMessage) and current_group is not None:
                # Check if this AI message has tool_calls — extend the group
                tool_calls = msg.additional_kwargs.get("tool_calls", [])
                if tool_calls:
                    current_group["texts"].append(text)
                    current_group["message_types"].add(msg_type)
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        if name:
                            current_group["tool_names"].add(name)
                    continue

            # Start a new group (HumanMessage or AIMessage without pending tool_calls)
            if current_group is not None:
                pass  # Already appended
            current_group = {
                "texts": [text],
                "message_types": {msg_type},
                "tool_names": set(),
                "turn_index": idx,
            }
            groups.append(current_group)

        if not groups:
            return []

        # Phase 2: Merge short groups (<50 tokens)
        merged_groups: list[dict[str, Any]] = []
        buffer: dict[str, Any] | None = None

        for group in groups:
            group_text = "\n".join(group["texts"])
            token_count = self._token_estimator.estimate(group_text)

            if buffer is None:
                buffer = {
                    "texts": list(group["texts"]),
                    "message_types": set(group["message_types"]),
                    "tool_names": set(group["tool_names"]),
                    "turn_index": group["turn_index"],
                    "token_count": token_count,
                }
            elif buffer["token_count"] < 50:
                # Buffer is short (<50 tokens) — merge with current group
                buffer["texts"].extend(group["texts"])
                buffer["message_types"].update(group["message_types"])
                buffer["tool_names"].update(group["tool_names"])
                buffer["token_count"] += token_count
            else:
                merged_groups.append(buffer)
                buffer = {
                    "texts": list(group["texts"]),
                    "message_types": set(group["message_types"]),
                    "tool_names": set(group["tool_names"]),
                    "turn_index": group["turn_index"],
                    "token_count": token_count,
                }

        if buffer is not None:
            merged_groups.append(buffer)

        # Phase 3 & 4: Split oversized + build RawChunks
        chunks: list[RawChunk] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        # Extract step title from plan if available
        step_title = ""
        if plan is not None and hasattr(plan, "steps") and plan.steps:
            # Use the first completed step's description as context
            for s in plan.steps:
                if s.status == ExecutionStatus.COMPLETED:
                    step_title = s.description
                    break

        for group in merged_groups:
            content = "\n".join(group["texts"])
            token_count = group.get("token_count") or self._token_estimator.estimate(content)

            if token_count > 500:
                # Split oversized
                parts = self._split_with_overlap(content, target_tokens=250)
            else:
                parts = [content]

            for part in parts:
                if not part.strip():
                    continue
                content_hash = memory_content_hash(part)
                meta: dict[str, Any] = {
                    "turn_index": group["turn_index"],
                    "message_types": sorted(group["message_types"]),
                    "created_at": now_iso,
                }
                if step_title:
                    meta["step_title"] = step_title
                tool_names = group.get("tool_names", set())
                if tool_names:
                    meta["tool_names"] = sorted(tool_names)

                chunks.append(RawChunk(
                    content=part,
                    session_id=self._session_id,
                    user_id=self._user_id,
                    source="session_flush",
                    metadata=meta,
                    content_hash=content_hash,
                ))

        return chunks

    def _split_with_overlap(self, text: str, target_tokens: int = 250) -> list[str]:
        """Split text on paragraph boundaries with ~50 token overlap.

        Returns a list of text segments. Short texts are returned as-is.
        """
        estimated_tokens = self._token_estimator.estimate(text)
        if estimated_tokens <= target_tokens:
            return [text]

        paragraphs = text.split("\n\n")
        if len(paragraphs) <= 1:
            # No paragraph boundaries — fall back to character-level splitting
            return self._split_by_chars(text, target_tokens)

        OVERLAP_TOKENS = 50
        segments: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._token_estimator.estimate(para)

            # If a single paragraph exceeds target, sub-split it first
            if para_tokens > target_tokens:
                # Flush current buffer before sub-splitting
                if current_parts:
                    segments.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                # Sub-split the oversized paragraph by characters
                sub_parts = self._split_by_chars(para, target_tokens)
                segments.extend(sub_parts)
                continue

            if current_tokens + para_tokens > target_tokens and current_parts:
                segments.append("\n\n".join(current_parts))
                # Overlap: keep last paragraph(s) worth ~50 tokens
                overlap_parts: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_parts):
                    p_tok = self._token_estimator.estimate(p)
                    if overlap_tokens + p_tok > OVERLAP_TOKENS:
                        break
                    overlap_parts.insert(0, p)
                    overlap_tokens += p_tok
                current_parts = overlap_parts
                current_tokens = overlap_tokens

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            segments.append("\n\n".join(current_parts))

        return segments if segments else [text]

    def _split_by_chars(self, text: str, target_tokens: int = 250) -> list[str]:
        """Fallback split for text without paragraph boundaries.

        Splits on newline or space boundaries, with ~50 token overlap.
        """
        # Estimate chars per target — rough heuristic
        total_tokens = self._token_estimator.estimate(text)
        if total_tokens <= target_tokens:
            return [text]

        chars_per_token = len(text) / max(total_tokens, 1)
        target_chars = int(target_tokens * chars_per_token)
        overlap_chars = int(50 * chars_per_token)

        segments: list[str] = []
        start = 0
        while start < len(text):
            end = start + target_chars
            if end >= len(text):
                segments.append(text[start:])
                break

            # Try to break at newline or space
            break_at = text.rfind("\n", start + target_chars // 2, end)
            if break_at == -1:
                break_at = text.rfind(" ", start + target_chars // 2, end)
            if break_at == -1:
                break_at = end

            segments.append(text[start:break_at].rstrip())
            start = max(break_at - overlap_chars, start + 1)

        return segments if segments else [text]

    @staticmethod
    def _message_to_text(msg: BaseMessage) -> str:
        """Extract text content from a LangChain BaseMessage."""
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts)
        return ""

    @property
    def done(self) -> bool:
        return self.status == FlowStatus.IDLE
