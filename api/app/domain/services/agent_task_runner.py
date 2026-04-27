import asyncio
import base64
import hashlib
import io
import logging
import mimetypes
import re
import time
import unicodedata
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, BinaryIO, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from app.domain.services.prompts.assembler import PromptAssembler
    from app.domain.services.provider_profiles import ProviderProfile  # A7 Task 2.7

from langchain_core.language_models import BaseChatModel

from app.application.services.continuation_intent_classifier import (
    ContinuationIntentClassifier,
)
from app.domain.external.browser import Browser
from app.domain.external.file_storage import FileStorage
from app.domain.external.memory_flusher import MemoryFlusher
from app.domain.external.sandbox import Sandbox, SandboxHandle
from app.domain.external.search import SearchEngine
from app.domain.external.task import Task, TaskRunner
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    MCPConfig,
    SkillRiskPolicy,
    ToolRuntimeConfig,
)
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.event import (
    A2AToolContent,
    BaseEvent,
    BrowserToolContent,
    CompactionEvent,
    ContextStatusEvent,
    ControlAction,
    ControlEvent,
    DoneEvent,
    ErrorEvent,
    Event,
    FileToolContent,
    FinishingEvent,
    HealthEvent,
    HealthStatus,
    MCPToolContent,
    MessageEvent,
    SearchToolContent,
    ShellToolContent,
    SkillToolContent,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    ToolConfirmationEvent,
    ToolEvent,
    ToolEventStatus,
    WaitEvent,
)
from app.domain.models.file import File
from app.domain.models.message import Message
from app.domain.models.search import SearchResults
from app.domain.models.session import SessionStatus
from app.domain.models.skill import Skill
from app.domain.models.tool_result import ToolResult
from app.domain.models.user_tool_enablement import ToolType

# from app.domain.repositories.file_repository import FileRepository
# from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.application.services.skill_index_service import SkillIndexService
from app.application.services.skill_selector import SkillSelectionMeta, SkillSelector
from app.domain.services.flows.planner_react import PlannerReActFlow
from app.domain.services.graphs.background_summary import run_background_summary
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.brainstorm_skill import BrainstormSkillTool
from app.domain.services.tools.create_skill import CreateSkillTool
from app.domain.services.tools.mcp import MCPTool
from app.domain.services.tools.skill import SkillTool
from app.domain.services.tools.skill_bundle_sync import SkillBundleSyncManager
from app.domain.services.tools.tool_source_resolver import (
    resolve_tool_source_from_tool,
)
from app.infrastructure.repositories.db_user_tool_enablement_repository import (
    DBUserToolEnablementRepository,
)
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.storage.postgres import get_postgres
from core.config import get_settings
from fastapi import UploadFile
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)

MESSAGE_STREAM_CHUNK_SIZE = 24
MESSAGE_STREAM_CHUNK_DELAY_SEC = 0.03
SKILL_CONTEXT_MAX_SKILLS = 6
SKILL_CONTEXT_MAX_TOTAL_CHARS = 8000
SKILL_CONTEXT_MAX_SNIPPET_CHARS = 1200
TOOL_SUMMARY_MAX_ITEMS_PER_GROUP = 6


def _compute_has_positive_match(scores: list[float]) -> bool:
    """基于相对排名判断是否有正向匹配（模型无关的阈值逻辑）。"""
    if not scores or scores[0] < 0.15:
        return False
    if len(scores) < 2:
        return True
    gap = scores[0] - scores[1]
    return gap > 0.05 or scores[0] > 0.4


@dataclass(slots=True)
class SelectionDebugMeta:
    selection_source: str
    continuation_decision: bool
    continuation_decision_source: str
    llm_invoked: bool
    llm_cache_hit: bool
    llm_latency_ms: int | None
    message_len: int
    message_digest: str
    max_score: int
    second_score: int
    effective_threshold: int
    token_count: int
    has_positive_match: bool


@dataclass(slots=True)
class StepSkillActivationState:
    """当前 step 的 skill 锁定状态（仅内存态）。"""

    step_id: str
    user_message: str
    locked_skills: list[Skill] = field(default_factory=list)
    consecutive_unknown_tool_calls: int = 0
    reselect_count: int = 0


class SkillGuideInjector:
    """按需注入 Tier 2 skill guide 到 tool result 中。"""

    def __init__(self, skills: list, preloaded_ids: set[str]):
        from typing import Any

        self._tool_to_skill: dict[str, Any] = {}
        self._injected: set[str] = set()
        self._preloaded_ids = preloaded_ids
        for skill in skills:
            for tool in (skill.manifest or {}).get("tools", []):
                if isinstance(tool, dict) and tool.get("name"):
                    # 保留第一个映射（高分 skill 优先，假设 skills 已按 score 排序）
                    if tool["name"] not in self._tool_to_skill:
                        self._tool_to_skill[tool["name"]] = skill

    def __call__(self, tool_name: str) -> str | None:
        skill = self._tool_to_skill.get(tool_name)
        if not skill:
            return None
        if skill.id in self._preloaded_ids or skill.id in self._injected:
            return None
        self._injected.add(skill.id)
        manifest = skill.manifest or {}
        body = str(manifest.get("context_blob") or manifest.get("skill_md") or "")
        return body[:1200] if body else None


class AgentTaskRunner(TaskRunner):
    """基于Agent智能体的任务运行器"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: BaseChatModel,  # 大语言模型
        agent_config: AgentConfig,  # 智能体配置
        mcp_config: MCPConfig,  # mcp配置
        a2a_config: A2AConfig,  # a2a配置
        session_id: str,  # 会话id
        user_id: str | None,  # 用户id
        # session_repository: SessionRepository,  # 会话仓库
        file_storage: FileStorage,  # 文件存储桶
        # file_repository: FileRepository,  # 文件数据仓库
        browser: Browser,  # 浏览器
        search_engine: SearchEngine,  # 搜索引擎
        sandbox: SandboxHandle | Sandbox,  # 沙箱（优先 SandboxHandle）
        skill_creator_service=None,  # skill创建服务
        skill_risk_policy: SkillRiskPolicy | None = None,  # skill风险策略
        overflow_config: ContextOverflowConfig | None = None,  # 上下文治理配置
        summary_llm: BaseChatModel | None = None,  # 摘要生成模型
        checkpointer_pool: object | None = None,  # checkpointer 连接池
        supports_vision: bool = True,  # 模型是否支持视觉/多模态
        supports_pdf_input: bool = False,  # 是否支持原生 PDF 文件输入
        file_processor_lookup: object | None = None,  # FileProcessorLookup, file_view 工具的处理器
        memory_flusher: MemoryFlusher | None = None,  # 记忆刷写调度器
        memory_embedding_provider=None,  # C6: 记忆向量化 provider
        memory_session_factory=None,  # C6: 记忆 DB session 工厂
        memory_repo_factory=None,  # C6: 记忆仓库工厂
        memory_write_service=None,  # PR-3: MemoryManagementService for memory_save tool
        memory_session_redis=None,  # PR-3: Redis client for per-session save counter
        memory_session_save_cap: int = 20,  # PR-3: per-session memory_save hard cap
        memory_gate_llm: BaseChatModel | None = None,  # PR-4+8: LLM quality gate
        memory_gate_breaker=None,  # PR-4+8: MemoryGateBreaker | None
        memory_gate_daily_cap=None,  # PR-4+8: MemoryGateDailyCap | None
        memory_gate_threshold: float = 0.7,
        memory_gate_batch_cap: int = 20,
        memory_notification_emitter=None,  # PR-4+8: MemoryNotificationEmitter
        approval_state_reader=None,  # R5b-2: ApprovalStateReader | None（读路径 single source）
        approval_state_writer=None,  # R5b-3: ApprovalStateWriter | None（写路径 single writer）
        confirmation_manager=None,  # Task 17: ConfirmationManager | None
        initial_language: str = "zh",  # B5 #29: bootstrap hint from AgentService._create_task
        tool_runtime: ToolRuntimeConfig | None = None,  # R2 CS2: wrapper cap + smart-approve timeout
        on_session_complete=None,  # Callback: async (session_id) -> None, called after COMPLETED/TIMED_OUT
        profile: "ProviderProfile | None" = None,  # A7 Task 2.7: provider capability profile (accepts_image_url, image_max_bytes, ...)
        # B4 M0: session-scoped CostCallbackHandler. Built by agent_service
        # via ``build_cost_callback_handler`` and forwarded into the flow so
        # every LLM call inside the graph emits a CostRecord. None = disabled
        # (legacy path / tests that bypass cost tracking).
        cost_callback_handler: Any = None,
    ) -> None:
        """构造函数，完成Agent任务运行器的创建"""
        # A7 Task 2.7: provider capability profile. None = legacy behavior
        # (accepts_image_url defaults to True via pathway — see _build_image_blocks).
        self.profile = profile
        self._cost_callback_handler = cost_callback_handler
        self._on_session_complete = on_session_complete
        self._approval_state_reader = approval_state_reader
        self._approval_state_writer = approval_state_writer
        self._confirmation_manager = confirmation_manager
        self._memory_flusher = memory_flusher
        self._memory_embedding_provider = memory_embedding_provider
        self._memory_session_factory = memory_session_factory
        self._memory_repo_factory = memory_repo_factory
        self._memory_write_service = memory_write_service
        self._memory_session_redis = memory_session_redis
        self._memory_session_save_cap = memory_session_save_cap
        self._memory_gate_llm = memory_gate_llm
        self._memory_gate_breaker = memory_gate_breaker
        self._memory_gate_daily_cap = memory_gate_daily_cap
        self._memory_gate_threshold = memory_gate_threshold
        self._memory_gate_batch_cap = memory_gate_batch_cap
        self._memory_notification_emitter = memory_notification_emitter
        self._file_processor_lookup = file_processor_lookup
        self._agent_config = agent_config
        self._tool_runtime = tool_runtime or ToolRuntimeConfig()
        self._llm = llm
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._session_id = session_id
        self._user_id = user_id
        # self._session_repository = session_repository
        self._sandbox = sandbox
        self._mcp_config = mcp_config
        self._mcp_tool = MCPTool()
        self._a2a_config = a2a_config
        self._a2a_tool = A2ATool()
        settings = get_settings()
        self._skill_repository = FileSkillRepository(settings.skills_root_dir)
        self._skill_bundle_sync = SkillBundleSyncManager(
            sandbox=sandbox,
            skills_root_dir=settings.skills_root_dir,
            sandbox_skill_root=settings.skill_sandbox_bundle_root,
        )
        self._skill_tool = SkillTool(
            sandbox=sandbox,
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
            risk_mode=(skill_risk_policy or SkillRiskPolicy()).mode.value,
            bundle_sync_manager=self._skill_bundle_sync,
            skill_sandbox_bundle_root=settings.skill_sandbox_bundle_root,
        )
        self._create_skill_tool = (
            CreateSkillTool(
                skill_creator_service=skill_creator_service,
                sandbox=sandbox,
                user_id=user_id or "",
            )
            if skill_creator_service is not None
            else None
        )
        self._brainstorm_skill_tool = (
            BrainstormSkillTool(
                skill_creator_service=skill_creator_service,
            )
            if skill_creator_service is not None
            else None
        )
        self._skill_index_service = SkillIndexService(
            skill_repository=self._skill_repository,
            skills_root=settings.skills_root_dir,
        )
        self._skill_selection_policy = agent_config.skill_selection
        self._skill_selector = SkillSelector(
            default_top_k=12,
            base_threshold=self._skill_selection_policy.base_threshold,
        )
        self._continuation_classifier = ContinuationIntentClassifier(
            llm=llm,
            timeout_seconds=self._skill_selection_policy.continuation_llm_timeout_seconds,
        )
        self._continuation_phrases = {
            self._normalize_continuation_text(item)
            for item in self._skill_selection_policy.continuation_phrases
            if self._normalize_continuation_text(item)
        }
        self._continuation_patterns = [
            re.compile(item)
            for item in self._skill_selection_policy.continuation_patterns
            if item
        ]
        self._continuation_decision_cache: OrderedDict[str, bool] = OrderedDict()
        self._last_effective_selected_skills: list[Skill] = []
        self._last_substantive_user_message: str = ""
        self._session_skill_pool: list[Skill] = []
        self._step_skill_state: StepSkillActivationState | None = None
        self._current_message_selected_skills: list[Skill] = []
        self._current_message_text: str = ""
        self._last_initialized_skill_ids: tuple[str, ...] = ()
        self._last_skill_risk_fp: tuple[str, ...] = ()  # R3: risk fingerprint for re-init detection
        # #27 hotfix: parallel to _last_initialized_skill_ids, this holds the
        # actual Skill objects that were last passed to SkillTool.initialize.
        # Required by _build_step_react_graph's outer rollback path so it can
        # actively call await self._skill_tool.initialize(snapshot_skills) to
        # reset SkillTool internal state when Phase 2 or Phase 3 fails after
        # Phase 1 already mutated SkillTool. Snapshot of IDs alone is not
        # enough because initialize() needs Skill objects.
        self._last_initialized_skills: list[Skill] = []
        self._last_virtual_step_id: str = ""
        self._embedding_available: bool = False
        # D5: Watchdog termination flag (set when HealthEvent TERMINATING is emitted)
        self._was_timed_out: bool = False
        self._embedding_index = None  # SkillEmbeddingIndex | None
        self._current_embedding_scores: list[float] | None = None
        self._tier2_preloaded_skill_ids: set[str] = set()
        self._last_skill_context: str = ""
        # B5 C5a: per-step authoritative metadata persisted across calls.
        # These three fields — together with self._last_initialized_skill_ids —
        # are updated atomically by _apply_refreshed_skills + Phase 3 of
        # _build_step_react_graph. Any partial update must be rolled back to
        # the pre-call snapshot (see atomicity tests in test_refresh_skills_atomicity.py).
        self._last_skill_ids: tuple[str, ...] = ()
        self._last_bound_tool_names: frozenset[str] = frozenset()
        # B5 C5a: per-step lc_tools cache keyed by (mode, skill_ids_tuple,
        # activated_mcp_tools_frozenset). Never cleaned (bounded by the set
        # of skill_id combinations actually used in this session). Cache is
        # valid because lc_tools construction is a pure function of the key.
        self._lc_tools_cache: dict[tuple, list[Any]] = {}
        self._activated_mcp_tools: set[str] = set()
        self._image_url_map: dict[str, str] = {}  # sandbox filepath → presigned URL
        self._supports_vision = supports_vision
        self._supports_pdf_input = supports_pdf_input
        self._file_storage = file_storage
        self._overflow_config = overflow_config or ContextOverflowConfig()
        # self._file_repository = file_repository
        self._browser = browser
        self._search_engine = search_engine

        # B5 post-audit MEDIUM #3: eagerly import the ZH/EN bundles at
        # session startup. The import triggers ``SectionRegistry.__post_init__``
        # first-use validation (render each section against ``_FIXTURE_CTX``
        # and scan for dangling skill tool refs). By doing this in the
        # ``AgentTaskRunner.__init__`` — which runs once per session — we
        # get practical startup fail-fast semantics without forcing the
        # lightweight ``get_prompt_bundle`` path to pay the validation
        # cost on every ``import app.domain.services.prompts`` call.
        from app.domain.services.prompts.bundles import EN_BUNDLE, ZH_BUNDLE  # noqa: F401

        # B5 C11: construct a JsonlPromptTelemetry instance and attach it
        # to both (a) the PromptAssembler (assembly events) and (b) the
        # LLM adapters (per-invocation events). All writes are
        # fire-and-forget — failures never propagate to the main path.
        self._prompt_telemetry = self._build_prompt_telemetry()

        # B5 C5b: construct PromptAssembler for section-based system prompt
        # assembly. Consumed by executor_node, planner_node, updater_node,
        # and planner_react._run_planner_for_detection.
        # B5 C9: budget routes through ``ContextOverflowConfig.system_prompt_max_tokens``.
        # B5 C7.5: no longer gated by a feature flag — always constructed.
        # B5 C11: telemetry is the JsonlPromptTelemetry built above.
        prompt_assembler = self._build_prompt_assembler()

        # B5 C11 + post-audit LOW #1: track the current session language
        # and wire telemetry into all LLM adapters with that language.
        # The initial value is ``"zh"`` because the actual language is
        # only known after ``planner_node`` runs and parses the user
        # message. ``set_language`` re-attaches telemetry with the new
        # language once main_graph notifies us via the
        # ``language_callback`` injected into ``configurable``.
        # Note: ``self._llm`` was already assigned at line ~212; here we
        # just stash ``summary_llm`` (which isn't stored elsewhere) and
        # track the language.
        # B5 #29: seed current language from application layer.
        # ``initial_language`` is computed by ``AgentService._create_task``
        # from the already-hydrated ``session.get_latest_plan()``.
        # Falls back to "zh" for brand-new sessions (no plan history yet).
        self._current_language: str = initial_language
        self._summary_llm_for_telemetry = summary_llm
        self._attach_telemetry_to_llms(self._current_language)

        self._flow = PlannerReActFlow(
            uow_factory=uow_factory,
            llm=llm,
            agent_config=agent_config,
            session_id=session_id,
            browser=browser,
            sandbox=sandbox,
            search_engine=search_engine,
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
            skill_tool=self._skill_tool,
            create_skill_tool=self._create_skill_tool,
            brainstorm_skill_tool=self._brainstorm_skill_tool,
            overflow_config=self._overflow_config,
            summary_llm=summary_llm,
            user_id=self._user_id or "",
            skill_graph_canary_percent=settings.skill_graph_canary_percent,
            checkpointer_pool=checkpointer_pool,
            supports_vision=supports_vision,
            supports_pdf_input=supports_pdf_input,
            file_processor_lookup=file_processor_lookup,
            memory_embedding_provider=self._memory_embedding_provider,
            memory_session_factory=self._memory_session_factory,
            memory_repo_factory=self._memory_repo_factory,
            memory_write_service=self._memory_write_service,
            memory_session_redis=self._memory_session_redis,
            memory_session_save_cap=self._memory_session_save_cap,
            memory_gate_llm=self._memory_gate_llm,
            memory_gate_breaker=self._memory_gate_breaker,
            memory_gate_daily_cap=self._memory_gate_daily_cap,
            memory_gate_threshold=self._memory_gate_threshold,
            memory_gate_batch_cap=self._memory_gate_batch_cap,
            memory_notification_emitter=self._memory_notification_emitter,
            approval_state_reader=self._approval_state_reader,
            approval_state_writer=self._approval_state_writer,
            confirmation_manager=self._confirmation_manager,
            prompt_assembler=prompt_assembler,
            tool_runtime=self._tool_runtime,
            # B4 M0: session-scoped cost callback attached into every invoke.
            cost_callback_handler=self._cost_callback_handler,
        )

    def _build_prompt_telemetry(self) -> Any:
        """Construct the ``JsonlPromptTelemetry`` instance used by
        the ``PromptAssembler`` and LLM adapter hooks.

        Log directory comes from ``settings.prompt_telemetry_log_dir``.
        If the directory can't be created, ``JsonlPromptTelemetry`` logs
        a warning at construction time and continues to accept writes
        (each write is also try/except-guarded — see C1 implementation).

        Returns a plain ``JsonlPromptTelemetry`` instance; can also be
        a ``None``-compatible stub for tests if the config is missing,
        but at the moment the default is always populated by settings.
        """
        from pathlib import Path

        from app.infrastructure.telemetry.prompt_telemetry import (
            JsonlPromptTelemetry,
        )

        log_dir = Path(get_settings().prompt_telemetry_log_dir)
        return JsonlPromptTelemetry(log_dir=log_dir)

    def _attach_telemetry_to_llms(self, lang: str) -> None:
        """Attach ``self._prompt_telemetry`` to the primary and summary LLMs.

        B5 post-audit LOW #1: called once at ``__init__`` with the default
        language (``"zh"``), then again from ``set_language`` after
        ``planner_node`` detects the real session language. The method is
        idempotent — calling with the same lang is a no-op in effect.

        Swallows ``AttributeError`` for test mocks that don't implement
        ``attach_telemetry``.
        """
        llm = getattr(self, "_llm", None)
        if llm is not None and hasattr(llm, "attach_telemetry"):
            llm.attach_telemetry(self._prompt_telemetry, lang=lang)
        summary_llm = getattr(self, "_summary_llm_for_telemetry", None)
        if (
            summary_llm is not None
            and summary_llm is not llm
            and hasattr(summary_llm, "attach_telemetry")
        ):
            summary_llm.attach_telemetry(self._prompt_telemetry, lang=lang)

    def set_language(self, lang: str) -> None:
        """Update the session language and re-attach telemetry.

        B5 post-audit LOW #1: called from ``main_graph.planner_node``
        after parsing ``plan.language`` so the invocation telemetry
        records the correct language for every subsequent LLM call
        in this session.

        No-op if ``lang`` equals the current language. Non-blocking —
        the underlying ``attach_telemetry`` swallows any adapter error.
        """
        if not lang or lang == self._current_language:
            return
        logger.info(
            "[Telemetry] session language updated: %s → %s",
            self._current_language,
            lang,
        )
        self._current_language = lang
        try:
            self._attach_telemetry_to_llms(lang)
        except Exception as exc:
            logger.warning(
                "[Telemetry] set_language failed to re-attach telemetry: %s",
                exc,
            )

    def _build_prompt_assembler(self) -> "PromptAssembler":
        """Construct the ``PromptAssembler`` used for all system-prompt assembly.

        Post-C7.5: this method always returns a real instance — there is no
        feature flag to gate it anymore. Consumers in ``main_graph`` and
        ``planner_react`` require a non-None assembler and raise
        ``RuntimeError`` if DI is misconfigured.

        Budget: ``ContextOverflowConfig.system_prompt_max_tokens`` (canonical
        source since B5 C9). Falls back to the canonical default of 10000
        (post M2-PR0) when no overflow_config is attached (test fixtures).
        Telemetry: ``self._prompt_telemetry`` populated by
        ``_build_prompt_telemetry`` (B5 C11). ``None`` when the runner
        skipped telemetry setup (test path that bypasses ``__init__``).
        """
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from app.domain.services.prompts.assembler import (
            PromptAssembler as _PromptAssembler,
        )
        from app.domain.services.prompts.budget import SystemPromptBudget

        # B5 C9 / M2-PR0: budget comes from ``ContextOverflowConfig.system_prompt_max_tokens``
        # (canonical config source, default 10000 post M2-PR0). When no
        # overflow_config is attached (e.g. in tests), fall back to the same
        # canonical default so fallback behavior matches production defaults.
        max_tokens = (
            self._overflow_config.system_prompt_max_tokens
            if self._overflow_config
            else 10000
        )
        budget = SystemPromptBudget(max_tokens=max_tokens)
        strategy = (
            self._overflow_config.token_estimator
            if self._overflow_config
            else "hybrid"
        )
        model_name = (
            self._overflow_config.model_name
            if self._overflow_config
            else ""
        )
        estimator = TokenEstimator(strategy=strategy, model_name=model_name)
        return _PromptAssembler(
            budget=budget,
            token_estimator=estimator,
            telemetry=getattr(self, "_prompt_telemetry", None),
        )

    async def _put_and_add_event(
        self, task: Task, event: Event, persist: bool = True
    ) -> None:
        """往指定任务的消息队列中添加事件"""
        # 1.往任务的输出消息队列中新增事件
        event_id = await task.output_stream.put(event.model_dump_json())
        event.id = event_id

        # 2.按需将事件添加到会话中（流式中间片段不落库）
        if persist:
            async with self._uow:
                await self._uow.session.add_event(self._session_id, event)

    async def _stream_assistant_message_event(
        self, event: MessageEvent
    ) -> AsyncGenerator[MessageEvent, None]:
        """将助手消息切片成可增量渲染的事件流，提升前端流式观感"""
        # 已携带 stream_id（如来自 summarizer 流式推送）→ 直接透传，不重新切片
        if event.stream_id:
            yield event
            return
        text = event.message or ""
        if not text or len(text) <= MESSAGE_STREAM_CHUNK_SIZE:
            event.stream_id = event.stream_id or str(uuid.uuid4())
            event.partial = False
            yield event
            return

        stream_id = event.stream_id or str(uuid.uuid4())
        total = len(text)
        for end in range(MESSAGE_STREAM_CHUNK_SIZE, total + MESSAGE_STREAM_CHUNK_SIZE, MESSAGE_STREAM_CHUNK_SIZE):
            current_end = min(end, total)
            is_final = current_end >= total

            yield MessageEvent(
                role=event.role,
                message=text[:current_end],
                stream_id=stream_id,
                partial=not is_final,
                attachments=event.attachments if is_final else [],
                created_at=event.created_at,
            )

            if not is_final:
                await asyncio.sleep(MESSAGE_STREAM_CHUNK_DELAY_SEC)

    @classmethod
    async def _pop_event(cls, task: Task) -> Event:
        """从任务的输入流中获取事件信息"""
        # 1.从任务task中读取数据
        event_id, event_str = await task.input_stream.pop()
        if event_str is None:
            logger.warning(f"AgentTaskRunner接收到空消息")
            return

        # 2.使用pydantic+type类型将字符串转换成事件
        event = TypeAdapter(Event).validate_json(event_str)
        event.id = event_id

        return event

    async def _sync_file_to_sandbox(self, file_id: str) -> File:
        """根据文件id将文件同步到沙箱中"""
        try:
            # 1.调用文件存储下载文件信息
            file_data, file = await self._file_storage.download_file(file_id)

            # 2.组装沙箱文件路径
            filepath = f"/home/ubuntu/upload/{file.filename}"

            # 3.调用沙箱将文件上传至沙箱
            tool_result = await self._sandbox.upload_file(
                file_data=file_data, filepath=filepath, filename=file.filename
            )

            # 4.判断是否上传成功
            if tool_result.success:
                file.filepath = filepath
                async with self._uow:
                    await self._uow.file.save(file)  # 可以更新也可以不更新
                return file
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步文件[{file_id}]失败: {str(e)}")

    async def _sync_message_attachments_to_sandbox(self, event: MessageEvent) -> None:
        """将消息事件中的附件同步到沙箱中"""
        # 1.定义附件列表
        attachments: List[str] = []

        try:
            # 2.判断消息中是否存在附件
            if event.attachments:
                # 3.循环遍历所有的消息附件
                for attachment in event.attachments:
                    # 4.根据同步文件的id将数据同步到沙箱中
                    file = await self._sync_file_to_sandbox(attachment.id)

                    # 5.文件是否同步成功
                    if file:
                        attachments.append(file)
                        async with self._uow:
                            await self._uow.session.add_file(self._session_id, file)

            # 6.更新消息事件中的attachments
            event.attachments = attachments
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到沙箱失败: {str(e)}")

    # 支持多模态识图的 MIME 类型前缀
    _IMAGE_MIME_PREFIXES = ("image/png", "image/jpeg", "image/gif", "image/webp")
    # base64 fallback 大小阈值（对齐 5MB API 硬限，3.75MB raw ≈ 5MB base64）
    _IMAGE_TARGET_RAW_SIZE = 3_932_160  # 3.75 MB

    async def _get_image_presigned_url(self, file: File) -> str | None:
        """Generate presigned URL via FileStorage protocol."""
        return await self._file_storage.get_presigned_url(file)

    async def _upload_sandbox_file_for_mcp(self, sandbox_path: str) -> str | None:
        """Download a file from sandbox and upload to storage, returning a presigned URL.

        Used by MCP tool path resolver for agent-generated files (e.g. extracted from zip)
        that don't have presigned URLs in _image_url_map.
        """
        import os
        from fastapi import UploadFile
        from io import BytesIO

        try:
            filename = os.path.basename(sandbox_path)
            # Read file from sandbox
            result = await self._sandbox.download_file(sandbox_path)
            if not result or not hasattr(result, "read"):
                logger.warning("Failed to download sandbox file %s", sandbox_path)
                return None
            file_bytes = result.read() if hasattr(result, "read") else result

            # Upload to storage as UploadFile
            upload = UploadFile(
                file=BytesIO(file_bytes),
                filename=filename,
                size=len(file_bytes),
            )
            file_obj = await self._file_storage.upload_file(upload)

            # Get presigned URL
            url = await self._file_storage.get_presigned_url(file_obj)
            if url:
                logger.info("Uploaded sandbox file %s → %s", sandbox_path, url[:80])
            return url
        except Exception as e:
            logger.warning("Failed to upload sandbox file %s for MCP: %s", sandbox_path, e)
            return None

    async def _build_image_blocks(self, attachments: list) -> list[dict]:
        """为图片附件构建 OpenAI multimodal content blocks + 元数据注入。

        A7 profile-aware: honors ``profile.accepts_image_url``.
        - accepts_image_url=True (OpenAI / generic) → use presigned URL when available
        - accepts_image_url=False (Kimi) → force base64 data: URL
        - On base64 I/O failure: degrade to ``text`` placeholder (§4.3b contract);
          NEVER fall back to an unsupported URL.

        Vision mode (supports_vision=True): 嵌入图片 blocks + 元数据
        Tool mode (supports_vision=False): 不嵌入图片 blocks，只记录 URL 映射供 MCP 工具使用
        """
        blocks: list[dict] = []
        self._image_url_map.clear()

        for attachment in attachments:
            # Three-state multimodal eligibility check
            if attachment.multimodal_eligible is False:
                continue
            elif attachment.multimodal_eligible is None:
                # None: non-image OR pre-migration image → fallback to mime_type
                mime = attachment.mime_type or ""
                if not any(mime.startswith(p) for p in self._IMAGE_MIME_PREFIXES):
                    continue

            try:
                # Always capture URL mapping (needed for MCP path resolution in both modes).
                # Wrap in its own try so a storage misconfig doesn't blow up the
                # entire image-block build (esp. on Kimi, where the URL is not
                # used for the LLM payload anyway).
                presigned_url = None
                if hasattr(self, "_get_image_presigned_url"):
                    try:
                        presigned_url = await self._get_image_presigned_url(attachment)
                    except Exception as _url_exc:
                        logger.debug(
                            "[A7] presigned URL fetch failed for attachment_id=%s: %s",
                            getattr(attachment, "id", "?"), _url_exc,
                        )
                        presigned_url = None
                if presigned_url and attachment.filepath:
                    self._image_url_map[attachment.filepath] = presigned_url

                # Tool mode (non-vision model): skip image blocks, only keep URL mapping
                if not self._supports_vision:
                    continue

                # A7 P1: profile.supports_vision acts as a hard ceiling. Even if
                # the user's LLMConfig says supports_vision=True, a profile that
                # declares no vision (e.g. DeepSeek Reasoner) must NOT embed
                # image blocks end-to-end.
                profile = getattr(self, "profile", None)
                if profile is not None and not profile.supports_vision:
                    continue

                # Vision mode: embed image blocks
                w, h = attachment.width, attachment.height
                detail = "low" if (w and h and w <= 512 and h <= 512) else "high"

                use_url = (
                    presigned_url and profile.accepts_image_url
                    if profile is not None
                    else bool(presigned_url)
                )

                if not use_url and profile is not None and not profile.accepts_image_base64:
                    # A7 P1: profile forbids both URL and base64 → no viable
                    # embedding path. Emit text placeholder directly without
                    # attempting download+encode.
                    filename = (
                        getattr(attachment, "filename", None)
                        or attachment.id
                        or "unknown"
                    )
                    blocks.append({
                        "type": "text",
                        "text": f"[image unavailable: {filename}]",
                    })
                    logger.warning(
                        "[A7] profile %s forbids both image URL and base64; "
                        "emitted text placeholder for attachment_id=%s",
                        profile.provider_id, attachment.id,
                    )
                    continue

                if use_url:
                    blocks.append({
                        "type": "image_url",
                        "image_url": {"url": presigned_url, "detail": detail},
                    })
                else:
                    # profile forbids URL (or no presigned URL) → base64 or degrade to text
                    try:
                        file_data, _ = await self._file_storage.download_file(
                            attachment.id
                        )
                        with file_data:
                            raw_bytes = file_data.read()
                        # Profile-level size guard (A7) supersedes the hard-coded
                        # _IMAGE_TARGET_RAW_SIZE when a profile is present.
                        max_bytes = (
                            profile.image_max_bytes
                            if profile is not None
                            else self._IMAGE_TARGET_RAW_SIZE
                        )
                        if len(raw_bytes) > max_bytes:
                            raise ValueError(
                                f"image {len(raw_bytes)}B exceeds "
                                f"image_max_bytes={max_bytes}"
                            )
                        b64_data = base64.b64encode(raw_bytes).decode("ascii")
                        mime = attachment.mime_type or "image/png"
                        data_url = f"data:{mime};base64,{b64_data}"
                        blocks.append({
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": detail},
                        })
                    except Exception as e:
                        # A7 §4.3b degrade contract: do NOT fall back to URL;
                        # emit text placeholder instead.
                        filename = (
                            getattr(attachment, "filename", None)
                            or attachment.id
                            or "unknown"
                        )
                        blocks.append({
                            "type": "text",
                            "text": f"[image unavailable: {filename}]",
                        })
                        logger.warning(
                            "[A7] image base64 encode failed for attachment_id=%s "
                            "filename=%s reason=%s; emitted text placeholder instead",
                            attachment.id, filename, repr(e),
                        )
                        # Skip metadata injection for degraded path.
                        continue

                # Metadata injection (skip for pre-migration images without dimensions)
                if w and h:
                    ow = getattr(attachment, "original_width", None) or w
                    oh = getattr(attachment, "original_height", None) or h
                    source = attachment.filepath or getattr(attachment, "filename", None)
                    if ow != w or oh != h:
                        scale = round(ow / w, 2)
                        meta_text = (
                            f"[Image: source: {source}, original {ow}x{oh}, "
                            f"displayed at {w}x{h}. "
                            f"Multiply coordinates by {scale:.2f} to map to original image.]"
                        )
                    else:
                        meta_text = f"[Image: source: {source}, {w}x{h}]"
                    blocks.append({"type": "text", "text": meta_text})

            except Exception as outer:
                logger.error(
                    "[A7] _build_image_blocks unexpected error for attachment_id=%s: %s",
                    getattr(attachment, "id", "?"), outer,
                )
                continue

        return blocks

    @classmethod
    def _get_stream_size(cls, f: BinaryIO) -> int:
        """根据传递的文件流，获取计算文件的大小"""
        # 1.记录当前文件指针位置
        current_pos = f.tell()

        # 2.将指针移动到文件末尾, seek，0: 偏移量、2: 相对文件末尾
        f.seek(0, 2)

        # 3.获取当前位置，也就是文件大小
        size = f.tell()

        # 4.恢复指针到原始位置
        f.seek(current_pos)

        return size

    async def _sync_file_to_storage(self, filepath: str) -> File:
        """将沙箱中指定的文件路径数据同步到存储桶中"""
        try:
            # 1.根据文件路径从会话中查找文件数据
            async with self._uow:
                file = await self._uow.session.get_file_by_path(
                    self._session_id, filepath
                )

            # 2.从沙箱中下载文件
            file_data = await self._sandbox.download_file(filepath)

            # 3.判断会话中的文件是否存在
            if file:
                async with self._uow:
                    await self._uow.session.remove_file(self._session_id, file.filepath)

            # 4.提取文件名字、文件信息并更新文件路径
            filename = filepath.split("/")[-1]
            content_type, _ = mimetypes.guess_type(filename)
            upload_file = UploadFile(
                file=file_data,
                filename=filename,
                size=self._get_stream_size(file_data),
                headers={"content-type": content_type or "application/octet-stream"},
            )

            # 5.上传文件到文件存储桶
            file = await self._file_storage.upload_file(upload_file)
            file.filepath = filepath

            # 6.往会话中新增一个文件信息
            async with self._uow:
                await self._uow.session.add_file(self._session_id, file)
            return file
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到文件存储桶失败: {str(e)}")

    # 需要同步到会话文件列表的输出文件扩展名
    _OUTPUT_FILE_EXTENSIONS = {
        ".pdf", ".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls",
        ".csv", ".md", ".txt", ".html", ".htm",
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
        ".mp3", ".mp4", ".wav",
        ".json", ".xml", ".yaml", ".yml",
    }

    async def _sync_generated_files(self, exec_dir: str) -> None:
        """扫描 exec_dir 中的输出文件并同步到会话文件列表。

        仅同步已知输出类型的文件，跳过 .py 等源代码文件
        （源代码通过 file_write 工具已自动同步）。
        """
        try:
            list_result = await self._sandbox.list_files(exec_dir)
            if not list_result.success:
                return

            files_data = (list_result.data or {}).get("files", [])
            for file_info in files_data:
                if file_info.get("is_dir"):
                    continue
                filepath = file_info.get("path", "")
                if not filepath:
                    continue

                # 仅同步已知输出文件类型
                ext = filepath.rsplit(".", 1)[-1] if "." in filepath else ""
                if f".{ext}" not in self._OUTPUT_FILE_EXTENSIONS:
                    continue

                # 检查是否已存在于会话文件列表
                async with self._uow:
                    existing = await self._uow.session.get_file_by_path(
                        self._session_id, filepath
                    )
                if existing:
                    continue

                logger.info(f"自动同步 shell/skill 输出文件: {filepath}")
                await self._sync_file_to_storage(filepath)
        except Exception as e:
            logger.warning(f"扫描输出文件失败（不影响主流程）: {e}")

    async def _sync_message_attachments_to_storage(self, event: MessageEvent) -> None:
        """将消息事件的附件同步到文件存储桶中"""
        # 1.定义附件列表存储数据
        attachments: List[File] = []

        try:
            # 2.判断消息中是否存在附件
            if event.attachments:
                # 3.循环遍历所有附件
                for attachment in event.attachments:
                    # 4.根据文件路径将数据同步到文件存储桶
                    file = await self._sync_file_to_storage(attachment.filepath)
                    if file:
                        attachments.append(file)

            # 5.更新时间中的附件列表资源
            event.attachments = attachments
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到存储桶失败: {str(e)}")

    async def _get_browser_screenshot(self) -> str:
        """获取浏览器截图并返回截图文件对应的在线URL"""
        # 1.调用浏览器完成截图
        screenshot = await self._browser.screenshot()

        # 2.将浏览器截图上传到文件存储中
        file = await self._file_storage.upload_file(
            UploadFile(
                file=io.BytesIO(screenshot),
                filename=f"{str(uuid.uuid4())}.png",
                size=self._get_stream_size(io.BytesIO(screenshot)),
            )
        )
        try:
            async with self._uow:
                await self._uow.session.add_file(self._session_id, file)
        except Exception as e:
            logger.warning(f"保存截图文件到会话失败: {str(e)}")

        # 3.优先返回预签名URL，避免私有桶直链403
        try:
            minio_store = getattr(self._file_storage, "minio_store", None)
            bucket = getattr(self._file_storage, "bucket", None)
            if minio_store and bucket and file.key:
                return await minio_store.presigned_get_url(
                    bucket_name=bucket,
                    object_name=file.key,
                    expiry_seconds=24 * 60 * 60,
                )
        except Exception as e:
            logger.warning(f"生成截图预签名URL失败，回退为原始链接: {str(e)}")

        # 4.预签名失败则回退为文件原始路径
        return file.filepath

    async def _load_enabled_skills(self) -> list[Skill]:
        """加载启用中的 Skill 列表。"""
        try:
            return await self._skill_index_service.list_enabled_skills()
        except Exception as e:
            logger.warning(f"加载Skill列表失败，降级为空列表: {str(e)}")
            return []

    async def _load_selected_skills(
        self,
        user_message: str,
        preference_map: dict[str, bool],
    ) -> list[Skill]:
        skills = await self._load_enabled_skills()
        filtered = self._filter_skills_by_user_preferences(skills, preference_map)
        selected_skills, _ = await self._select_skills_for_message(filtered, user_message)
        return selected_skills

    def _select_skills_from_pool(self, skill_pool: list[Skill], user_message: str) -> list[Skill]:
        """从给定技能池中选择本轮激活的技能子集。"""
        if not skill_pool:
            return []
        return self._skill_selector.select(skill_pool, user_message)

    @staticmethod
    def _normalize_continuation_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "").lower()
        normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _is_low_info_continuation_by_rule(self, message: str) -> bool:
        normalized = self._normalize_continuation_text(message)
        if not normalized:
            return False
        if normalized in self._continuation_phrases:
            return True
        return any(pattern.fullmatch(normalized) for pattern in self._continuation_patterns)

    def _should_invoke_continuation_llm(
        self,
        meta: SkillSelectionMeta,
        message: str,
    ) -> bool:
        if not self._skill_selection_policy.continuation_llm_enabled:
            return False
        if not self._last_substantive_user_message.strip():
            return False
        normalized = self._normalize_continuation_text(message)
        if not normalized:
            return False
        if len(normalized) > self._skill_selection_policy.short_message_max_chars:
            return False
        if meta.token_count > self._skill_selection_policy.llm_trigger_token_count:
            return False
        if meta.max_score > meta.effective_threshold:
            return False
        return True

    def _build_continuation_cache_key(self, current_message: str) -> str:
        current = self._normalize_continuation_text(current_message)
        previous = self._normalize_continuation_text(self._last_substantive_user_message)
        payload = f"{current}|{previous}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached_continuation_decision(self, key: str) -> bool | None:
        if key not in self._continuation_decision_cache:
            return None
        value = self._continuation_decision_cache.pop(key)
        self._continuation_decision_cache[key] = value
        return value

    def _set_cached_continuation_decision(self, key: str, value: bool) -> None:
        self._continuation_decision_cache[key] = value
        if len(self._continuation_decision_cache) <= self._skill_selection_policy.continuation_llm_cache_size:
            return
        self._continuation_decision_cache.popitem(last=False)

    async def _decide_continuation(
        self,
        message: str,
        meta: SkillSelectionMeta,
    ) -> tuple[bool, str, bool, bool, int | None]:
        if self._is_low_info_continuation_by_rule(message):
            return True, "rule", False, False, None
        if not self._should_invoke_continuation_llm(meta, message):
            return False, "fallback", False, False, None

        cache_key = self._build_continuation_cache_key(message)
        cached = self._get_cached_continuation_decision(cache_key)
        if cached is not None:
            return cached, "llm", False, True, 0

        # B4 M0: thread session-scoped cost callback into the continuation
        # classifier (graph-external LLM call). ``getattr`` guards
        # bypass-init test fixtures.
        _cost_handler = getattr(self, "_cost_callback_handler", None)
        classifier_config: dict | None = None
        if _cost_handler is not None:
            classifier_config = {
                "callbacks": [_cost_handler],
                "metadata": {
                    "langgraph_node": "continuation_classifier",
                    "langgraph_step": 0,
                },
            }

        started = time.perf_counter()
        decision = await self._continuation_classifier.classify(
            current_message=message,
            previous_substantive_message=self._last_substantive_user_message,
            config=classifier_config,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        self._set_cached_continuation_decision(cache_key, decision)
        return decision, "llm", True, False, latency_ms

    async def _select_skills_for_message(
        self,
        skill_pool: list[Skill],
        user_message: str,
    ) -> tuple[list[Skill], SelectionDebugMeta]:
        if not skill_pool:
            debug = SelectionDebugMeta(
                selection_source="fallback",
                continuation_decision=False,
                continuation_decision_source="fallback",
                llm_invoked=False,
                llm_cache_hit=False,
                llm_latency_ms=None,
                message_len=len(user_message or ""),
                message_digest=hashlib.sha256((user_message or "").encode("utf-8")).hexdigest(),
                max_score=0,
                second_score=0,
                effective_threshold=1,
                token_count=0,
                has_positive_match=False,
            )
            return [], debug

        # Phase 1: 优先使用 embedding
        embedding_results = None
        if self._embedding_available and self._embedding_index is not None:
            try:
                embedding_results = await self._embedding_index.query(user_message, top_k=12)
            except Exception:
                logger.warning("Embedding 检索失败，降级为 token-overlap")

        # 始终运行 token-overlap（shadow mode + fallback）
        meta = self._skill_selector.select_with_meta(skill_pool, user_message)

        if embedding_results:
            id_to_skill = {s.id: s for s in skill_pool}
            selected = [id_to_skill[sid] for sid, _ in embedding_results if sid in id_to_skill]
            scores = [score for sid, score in embedding_results if sid in id_to_skill]

            if selected:
                # Shadow mode 日志
                emb_ids = [s.id for s in selected[:6]]
                tok_ids = [s.id for s in meta.selected_skills[:6]]
                overlap = len(set(emb_ids) & set(tok_ids))
                logger.info("Skill选择对比 overlap=%d/6 emb=%s tok=%s", overlap, emb_ids, tok_ids)

                # 映射为 SkillSelectionMeta
                max_score = int(scores[0] * 100) if scores else 0
                second_score = int(scores[1] * 100) if len(scores) > 1 else 0
                has_positive = _compute_has_positive_match(scores)
                effective_threshold = min(max_score, 1) if has_positive else max_score + 1
                meta = SkillSelectionMeta(
                    selected_skills=selected,
                    max_score=max_score,
                    second_score=second_score,
                    token_count=len(user_message.split()),
                    effective_threshold=effective_threshold,
                )
                self._current_embedding_scores = scores
            else:
                self._current_embedding_scores = None
        else:
            self._current_embedding_scores = None

        (
            is_continuation,
            continuation_source,
            llm_invoked,
            llm_cache_hit,
            llm_latency_ms,
        ) = await self._decide_continuation(user_message, meta)

        if is_continuation and self._last_effective_selected_skills:
            selected_skills = list(self._last_effective_selected_skills)
            selection_source = "carry_over"
        else:
            selected_skills = list(meta.selected_skills)
            selection_source = "current" if meta.has_positive_match else "fallback"
            if not is_continuation and user_message.strip():
                self._last_effective_selected_skills = list(selected_skills)
                self._last_substantive_user_message = user_message

        debug = SelectionDebugMeta(
            selection_source=selection_source,
            continuation_decision=is_continuation,
            continuation_decision_source=continuation_source,
            llm_invoked=llm_invoked,
            llm_cache_hit=llm_cache_hit,
            llm_latency_ms=llm_latency_ms,
            message_len=len(user_message or ""),
            message_digest=hashlib.sha256((user_message or "").encode("utf-8")).hexdigest(),
            max_score=meta.max_score,
            second_score=meta.second_score,
            effective_threshold=meta.effective_threshold,
            token_count=meta.token_count,
            has_positive_match=meta.has_positive_match,
        )
        logger.info(
            "Skill选择 source=%s continuation=%s continuation_source=%s max=%s second=%s threshold=%s "
            "token_count=%s selected=%s llm_invoked=%s llm_cache_hit=%s llm_latency_ms=%s message_len=%s message_digest=%s",
            debug.selection_source,
            debug.continuation_decision,
            debug.continuation_decision_source,
            debug.max_score,
            debug.second_score,
            debug.effective_threshold,
            debug.token_count,
            [skill.id for skill in selected_skills],
            debug.llm_invoked,
            debug.llm_cache_hit,
            debug.llm_latency_ms,
            debug.message_len,
            debug.message_digest,
        )
        return selected_skills, debug

    @staticmethod
    def _strip_skill_frontmatter(skill_md: str) -> str:
        """移除 SKILL.md 的 YAML frontmatter，仅保留正文。"""
        raw = (skill_md or "").strip()
        if not raw.startswith("---"):
            return raw

        lines = raw.splitlines()
        if len(lines) < 3:
            return raw

        if lines[0].strip() != "---":
            return raw

        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :]).strip()
        return raw

    def _get_skill_guide_body(self, manifest: dict, skill) -> str:
        """提取 Skill 的完整 guide 内容（context_blob 或 skill_md）。"""
        context_blob = str(manifest.get("context_blob") or "").strip()
        if context_blob:
            body = context_blob
        else:
            skill_md = str(manifest.get("skill_md") or "").strip()
            body = self._strip_skill_frontmatter(skill_md)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if len(body) > SKILL_CONTEXT_MAX_SNIPPET_CHARS:
            body = body[:SKILL_CONTEXT_MAX_SNIPPET_CHARS].rstrip() + "\n...(truncated)"
        return body or (skill.description or "").strip() or "No additional guide content."

    def _build_skill_context_prompt(self, skills: list[Skill], scores: list[float] | None = None) -> str:
        """将已选中的 Skill 构建为运行时系统上下文（两级：Tier 2 完整指南 / Tier 1 轻量卡片）。"""
        TIER2_MAX_COUNT = 2
        TIER2_SCORE_THRESHOLD = 0.5

        if not skills:
            return ""

        sections: list[str] = [
            "## Active Skills",
            "Follow these selected SKILL.md guides when they are relevant to the current task.",
        ]

        self._tier2_preloaded_skill_ids = set()
        tier2_count = 0
        total_chars = sum(len(item) for item in sections)
        for i, skill in enumerate(skills[:SKILL_CONTEXT_MAX_SKILLS]):
            manifest = skill.manifest if isinstance(skill.manifest, dict) else {}

            # 决定是否使用 Tier 2（完整指南）
            if scores is not None:
                score = scores[i] if i < len(scores) else 0.0
                use_tier2 = score >= TIER2_SCORE_THRESHOLD and tier2_count < TIER2_MAX_COUNT
            else:
                use_tier2 = i == 0  # 无分数时，top-1 使用 Tier 2

            if use_tier2:
                body = self._get_skill_guide_body(manifest, skill)
                block = f"### {skill.name} ({skill.slug})\n{body}"
                self._tier2_preloaded_skill_ids.add(skill.id)
                tier2_count += 1
            else:
                # Tier 1: 轻量卡片，仅包含名称、描述和工具名
                desc = (skill.description or "").strip() or "No description."
                tools_list = manifest.get("tools") or []
                tool_names = [t["name"] for t in tools_list if isinstance(t, dict) and "name" in t]
                tools_line = f"Tools: {', '.join(tool_names)}" if tool_names else "Tools: (none)"
                block = f"### {skill.name} ({skill.slug})\n{desc}\n{tools_line}"

            if total_chars + len(block) > SKILL_CONTEXT_MAX_TOTAL_CHARS:
                break

            sections.append(block)
            total_chars += len(block)

        return "\n\n".join(sections)

    def _get_native_tool_names_by_category(self) -> dict[str, list[str]]:
        """从 create_native_tools 动态派生原生工具名，确保摘要与实际绑定一致。

        tool name 仅受 ``create_native_tools`` 的输入影响；memory_mount_scope
        只改变 wrapper 的运行时行为不改变 ``tool.name``，这里可以不传。但为
        了与 ``_build_lc_tools_full`` 保持参数一致避免未来漂移（比如新增
        依赖 scope 的工具），仍透传 scope。
        """
        if hasattr(self, "_cached_native_tool_names"):
            return self._cached_native_tool_names
        from app.domain.services.tools.langchain_tools import create_native_tools

        tools = create_native_tools(
            sandbox=self._sandbox,
            browser=self._browser,
            search_engine=self._search_engine,
            processor_lookup=self._file_processor_lookup,
            supports_vision=self._supports_vision,
            supports_pdf_input=self._supports_pdf_input,
            memory_mount_scope=self._build_memory_mount_scope(),
        )
        groups: dict[str, list[str]] = {}
        for tool in tools:
            prefix = tool.name.split("_")[0]
            groups.setdefault(prefix, []).append(tool.name)
        self._cached_native_tool_names = groups
        return groups

    def _build_memory_mount_scope(self):
        """Shared scope factory for both _get_native_tool_names_by_category
        和 _build_lc_tools_full（codex fix P0 round-2）。

        走 ``memory_mount_scope.build_memory_mount_scope_from_settings`` —
        与 PlannerReActFlow._build_memory_mount_scope 同源实现，确保 step graph
        每次重建 tool set 时都带上客户端 symlink 守卫，不只 planner 阶段。
        settings 拿不到（测试环境绕过 lifespan）时 factory 返 None → 旧行为。
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

    def _build_available_tool_summary(self) -> str:
        """构建可用工具摘要，减少模型对工具可用性的错觉。"""
        budget_tokens = self._skill_selection_policy.available_tool_summary_token_budget
        char_budget = max(400, budget_tokens * 4)

        # 从实际工具注册表动态获取，防止硬编码名称与实际绑定漂移
        native_groups = self._get_native_tool_names_by_category()

        skill_tools: list[str] = []
        try:
            for schema in self._skill_tool.get_tools():
                if not isinstance(schema, dict):
                    continue
                function_info = schema.get("function")
                if not isinstance(function_info, dict):
                    continue
                tool_name = function_info.get("name")
                if isinstance(tool_name, str) and tool_name:
                    skill_tools.append(tool_name)
        except Exception as exc:
            logger.debug("读取Skill工具摘要失败，降级为空: %s", exc)
            skill_tools = []
        creator_tools: list[str] = []
        if self._create_skill_tool is not None:
            try:
                for schema in self._create_skill_tool.get_tools():
                    if not isinstance(schema, dict):
                        continue
                    function_info = schema.get("function")
                    if not isinstance(function_info, dict):
                        continue
                    tool_name = function_info.get("name")
                    if isinstance(tool_name, str) and tool_name:
                        creator_tools.append(tool_name)
            except Exception as exc:
                logger.debug("读取Skill Creator工具摘要失败，降级为空: %s", exc)
                creator_tools = []

        if self._brainstorm_skill_tool is not None:
            try:
                for schema in self._brainstorm_skill_tool.get_tools():
                    if not isinstance(schema, dict):
                        continue
                    function_info = schema.get("function")
                    if not isinstance(function_info, dict):
                        continue
                    tool_name = function_info.get("name")
                    if isinstance(tool_name, str) and tool_name:
                        creator_tools.append(tool_name)
            except Exception as exc:
                logger.debug("读取Brainstorm工具摘要失败，降级为空: %s", exc)

        lines = ["## Available Tool Summary"]
        # 按固定顺序输出原生工具分组，名称从实际注册表派生
        # native tools 数量固定且总量小（~531 chars），不截断，保证与实际绑定完全一致
        for category in ("shell", "file", "browser", "message", "search"):
            names = native_groups.get(category, [])
            if names:
                lines.append(f"- {category}: {', '.join(names)}")
        if skill_tools:
            lines.append(
                "- active skill tools: "
                + ", ".join(skill_tools[:TOOL_SUMMARY_MAX_ITEMS_PER_GROUP])
            )
        else:
            lines.append(
                "- active skill tools: (none, use get_skill_guide to load full guide, "
                "tools will be bound at step execution)"
            )
        if creator_tools:
            lines.append(
                "- skill creator tools: "
                + ", ".join(creator_tools[:TOOL_SUMMARY_MAX_ITEMS_PER_GROUP])
            )

        # MCP 工具
        MCP_AUTO_BIND_THRESHOLD = 15
        mcp_all_names: list[str] = []
        try:
            for schema in self._mcp_tool.get_tools():
                if not isinstance(schema, dict):
                    continue
                fn = schema.get("function")
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name", "")
                if name:
                    mcp_all_names.append(name)
        except Exception:
            pass
        if mcp_all_names:
            if len(mcp_all_names) <= MCP_AUTO_BIND_THRESHOLD:
                # Small set: all tools directly bound — group by server for clarity
                # Parse server name from tool name: mcp_{server}_{tool}
                server_tools: dict[str, list[str]] = {}
                for name in mcp_all_names:
                    parts = name.split("_", 2)  # ["mcp", server, tool...]
                    server = parts[1] if len(parts) >= 3 else "unknown"
                    server_tools.setdefault(server, []).append(name)
                for server, tools in server_tools.items():
                    lines.append(f"- mcp ({server}): {', '.join(tools)}")
            else:
                # Large set: discovery mode
                always_bind_names = list(self._get_always_bind_tool_names())
                mcp_discovery_names = [n for n in mcp_all_names if n not in set(always_bind_names)]
                lines.append("- mcp discovery: list_mcp_tools, get_mcp_tool")
                if always_bind_names:
                    lines.append(
                        "- mcp always-bind: "
                        + ", ".join(always_bind_names[:TOOL_SUMMARY_MAX_ITEMS_PER_GROUP])
                    )
                if mcp_discovery_names:
                    lines.append(
                        "- mcp available (use get_mcp_tool to activate): "
                        + ", ".join(mcp_discovery_names[:TOOL_SUMMARY_MAX_ITEMS_PER_GROUP])
                    )

        # A2A 工具（仅在 manager 存在时才有 LangChain 工具绑定到 LLM）
        a2a_tools: list[str] = []
        if getattr(self._a2a_tool, "manager", None) is not None:
            try:
                for schema in self._a2a_tool.get_tools():
                    if not isinstance(schema, dict):
                        continue
                    function_info = schema.get("function")
                    if not isinstance(function_info, dict):
                        continue
                    tool_name = function_info.get("name")
                    if isinstance(tool_name, str) and tool_name:
                        a2a_tools.append(tool_name)
            except Exception as exc:
                logger.debug("读取A2A工具摘要失败，降级为空: %s", exc)
                a2a_tools = []
        if a2a_tools:
            lines.append(
                "- a2a tools: "
                + ", ".join(a2a_tools[:TOOL_SUMMARY_MAX_ITEMS_PER_GROUP])
            )

        # Memory tools（C6: memory_search + memory_get；PR-3: +memory_save）
        # 直接从 runner 已知依赖判断（_build_available_tool_summary 在 flow.invoke 前执行）
        if self._memory_session_factory and self._memory_repo_factory:
            memory_names = ["memory_search", "memory_get"]
            if (
                self._memory_write_service is not None
                and self._memory_session_redis is not None
            ):
                memory_names.append("memory_save")
            lines.append("- memory: " + ", ".join(memory_names))

        summary = "\n".join(lines).strip()
        if len(summary) > char_budget:
            summary = summary[:char_budget].rstrip() + "\n...(truncated)"
        return summary

    def _build_runtime_system_context(self, skills: list[Skill], scores: list[float] | None = None) -> str:
        """组装运行时上下文（Skill指南 + 可用工具摘要）。"""
        sections: list[str] = []
        skill_context = self._build_skill_context_prompt(skills, scores=scores)
        if skill_context:
            sections.append(skill_context)
        sections.append(self._build_available_tool_summary())
        return "\n\n".join(section for section in sections if section).strip()

    async def _refresh_skill_context_for_step(self, step_description: str) -> str:
        """Legacy entry point — kept for ``skill_context_refresher`` injection.

        B5 C5a: the implementation is now a thin wrapper around the pure
        ``_compute_refreshed_skills`` + atomic ``_apply_refreshed_skills``
        pair. ``updater_node`` still calls this to refresh the two-clock
        fallback (``state.skill_context``). New C5a call sites in
        ``_build_step_react_graph`` use the split pair directly.
        """
        result = await self._compute_refreshed_skills(step_description)
        if result is None:
            # Sticky: embedding top-1 score too low — keep previous selection.
            return self._last_skill_context
        await self._apply_refreshed_skills(result)
        return self._last_skill_context

    async def _compute_refreshed_skills(
        self, step_description: str
    ) -> "RefreshedSkillsResult | None":
        """Pure compute: select skills + build context, no side effects.

        Returns ``None`` when the embedding top-1 score is too low — the
        caller should keep ``self._last_*`` unchanged (sticky behavior).
        Otherwise returns a frozen ``RefreshedSkillsResult`` that
        ``_apply_refreshed_skills`` will apply atomically.

        **Must not mutate any ``self._*`` field.** Every state mutation
        happens in ``_apply_refreshed_skills``.
        """
        from app.domain.services.graphs.step_metadata import RefreshedSkillsResult

        query_parts = [step_description]
        if self._current_message_text:
            query_parts.append(self._current_message_text)
        query = "\n".join(query_parts)[:2000]

        if self._embedding_available and self._embedding_index is not None:
            try:
                results = await self._embedding_index.query(query, top_k=12)
                if results and results[0][1] < 0.2:
                    return None  # sticky
                id_to_skill = {s.id: s for s in self._session_skill_pool}
                skills = [id_to_skill[sid] for sid, _ in results if sid in id_to_skill]
                scores = [score for sid, score in results if sid in id_to_skill]
            except Exception:
                logger.warning("Step-level embedding 检索失败，使用 token-overlap")
                skills = self._skill_selector.select(self._session_skill_pool, query)
                scores = None
        else:
            skills = self._skill_selector.select(self._session_skill_pool, query)
            scores = None

        context = self._build_runtime_system_context(skills, scores=scores)
        return RefreshedSkillsResult(
            skills=tuple(skills),
            context=context,
            skill_ids=tuple(s.id for s in skills),
            scores=tuple(scores) if scores is not None else None,
        )

    async def _apply_refreshed_skills(
        self, result: "RefreshedSkillsResult"
    ) -> None:
        """Atomic mutation point for the refreshed skill selection (#27).

        Atomicity is guaranteed by assignment order: the two ``_last_*`` writes
        happen AFTER ``_initialize_skill_tool_if_needed`` returns successfully.
        If initialize raises, those two assignments are skipped, and
        ``_last_initialized_skill_ids`` is similarly guarded inside
        ``_initialize_skill_tool_if_needed`` (its own assignment is after the
        await). Since #27 makes ``SkillTool.initialize()`` atomic, the internal
        rollback layer that was here under B5 is no longer needed.

        See design: docs/superpowers/specs/2026-04-13-skill-tool-initialize-atomicity-design.md
        """
        if result.skills:
            await self._initialize_skill_tool_if_needed(list(result.skills))
        self._last_skill_context = result.context
        self._last_skill_ids = result.skill_ids

    async def _apply_preselected_skills(
        self,
        skills: list[Skill],
        scores: list[float] | None = None,
    ) -> None:
        """Apply a pre-selected skill list atomically.

        Used by the four non-``_build_step_react_graph`` code paths that
        already know which skills to activate and skip the embedding-based
        ``_compute_refreshed_skills`` query:

        1. Session bootstrap (before the main event loop starts).
        2. New-message boundary (right before ``flow.invoke``).
        3. Step-lock activation on step START / virtual step.
        4. Unknown-tool emergency reselection inside a running step.

        **Semantics**: "apply exactly this selection". An empty ``skills``
        list is an EXPLICIT CLEAR of the active skill selection — it calls
        ``_initialize_skill_tool_if_needed([])`` which advances
        ``_last_initialized_skill_ids`` to ``()`` and resets the skill
        portion of ``SkillTool`` internal state. ``_last_skill_ids`` is
        written to ``()``. ``_last_skill_context`` is written to whatever
        ``_build_runtime_system_context([], scores=None)`` produces —
        which is NOT an empty string: ``_build_runtime_system_context``
        always appends ``_build_available_tool_summary()`` covering
        native / MCP / A2A / memory tools, so empty-skills context is
        "tool summary without the Active Skills section".

        This intentionally differs from ``_apply_refreshed_skills`` (which
        gates init on non-empty ``result.skills`` because
        ``_compute_refreshed_skills`` uses empty tuples ambiguously).

        **Dedup**: the helper deduplicates ``skills`` by ``skill.id``,
        preserving first-occurrence order. Callers may pass lists with
        duplicates (e.g., from set-union of multiple selectors) and the
        helper will normalize before initializing ``SkillTool``. This
        prevents spurious re-init on the next call with deduplicated
        input.

        **Ordering**: ``_initialize_skill_tool_if_needed`` runs FIRST so
        that ``_build_runtime_system_context`` reads the fresh
        ``SkillTool`` internal state when assembling the available-tool
        summary. Reversing the order would produce a "new skill guide +
        stale tool summary" mixed context.

        **Atomicity**: single-coroutine runner, await-free critical section
        between ``_initialize_skill_tool_if_needed`` returning and the two
        final assignments. Failure modes:

        - ``_initialize_skill_tool_if_needed`` raises → post-#27 atomic
          guarantee keeps ``_last_initialized_skill_ids`` at its pre-call
          value; the two assignments below do not run; all three fields
          stay at their pre-call values.
        - ``_build_runtime_system_context`` raises AFTER init succeeds →
          ``_last_initialized_skill_ids`` has already advanced but the two
          final assignments do not run, producing a small split-brain
          window. This matches the old bypass path's risk level (it had
          the same failure mode relative to clock 2). Documented as a
          residual risk; no restore shell added.
        - Synchronous assignments at the tail cannot raise.

        ``_last_bound_tool_names`` is intentionally NOT touched — it
        represents "tools actually bound on the currently compiled
        step_react graph" and is owned exclusively by
        ``_build_step_react_graph`` Phase 3.

        See design: docs/superpowers/specs/2026-04-13-activate-step-skills-atomicity-design.md
        """
        # Dedup by skill.id, preserving first-occurrence order. Prevents
        # spurious re-init when a caller passes duplicates: without dedup,
        # _last_skill_ids would hold duplicate-containing tuples and the
        # next call with already-deduplicated input would not match,
        # triggering an unnecessary SkillTool.initialize. See spec §3.1
        # "Dedup contract".
        seen_ids: set[str] = set()
        deduped_skills: list[Skill] = []
        for skill in skills:
            if skill.id not in seen_ids:
                seen_ids.add(skill.id)
                deduped_skills.append(skill)

        await self._initialize_skill_tool_if_needed(deduped_skills)
        context = self._build_runtime_system_context(deduped_skills, scores=scores)
        self._last_skill_context = context
        self._last_skill_ids = tuple(skill.id for skill in deduped_skills)

    def _get_always_bind_tool_names(self) -> set[str]:
        """从 MCPConfig 提取所有 always_bind 工具名，组装完整前缀名。"""
        if not self._mcp_config or not self._mcp_config.mcpServers:
            return set()
        names: set[str] = set()
        for server_name, config in self._mcp_config.mcpServers.items():
            if not config.enabled or not config.always_bind:
                continue
            prefix = server_name if server_name.startswith("mcp_") else f"mcp_{server_name}"
            for tool_short_name in config.always_bind:
                names.add(f"{prefix}_{tool_short_name}")
        return names

    def _build_lc_tools_full(self) -> list[Any]:
        """Build the full lc_tools set for a step.

        Post-#27: all tool sources (native / mcp_auto / a2a / skill_static /
        dynamic_skill / skill_guide / memory) are always included. The
        category-selection mechanism (``_ALL_CATEGORIES`` / ``_MINIMAL_CATEGORIES``)
        was retired alongside the B5 graceful degradation path once
        ``SkillTool.initialize()`` became atomic. See design:
        docs/superpowers/specs/2026-04-13-skill-tool-initialize-atomicity-design.md
        """
        from app.domain.services.tools.langchain_tools import create_native_tools
        from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools
        from app.domain.services.tools.langchain_a2a import create_a2a_langchain_tools
        from app.domain.services.tools.langchain_skill_tools import (
            create_skill_guide_tool,
            create_skill_langchain_tools,
        )
        from app.domain.services.tools.langchain_dynamic_skill_tools import (
            create_dynamic_skill_langchain_tools,
        )

        lc_tools: list[Any] = []

        lc_tools.extend(
            create_native_tools(
                sandbox=self._sandbox,
                browser=self._browser,
                search_engine=self._search_engine,
                processor_lookup=self._file_processor_lookup,
                supports_vision=self._supports_vision,
                supports_pdf_input=self._supports_pdf_input,
                # codex fix P0 round-2：step graph 每次 rebuild 都要带守卫，
                # 否则 planner 阶段守护拦了，实际 tool call 路径还是裸透传。
                memory_mount_scope=self._build_memory_mount_scope(),
            )
        )

        # MCP: progressive auto-bind with discovery tools above the threshold.
        MCP_AUTO_BIND_THRESHOLD = 15
        all_mcp_tools = self._mcp_tool.get_tools()
        _url_map_ref = lambda: self._image_url_map
        _sandbox_uploader = self._upload_sandbox_file_for_mcp
        if len(all_mcp_tools) <= MCP_AUTO_BIND_THRESHOLD:
            lc_tools.extend(
                create_mcp_langchain_tools(
                    self._mcp_tool,
                    tool_names=None,
                    url_map_ref=_url_map_ref,
                    sandbox_file_uploader=_sandbox_uploader,
                )
            )
        else:
            mcp_bind_names = (
                self._get_always_bind_tool_names() | self._activated_mcp_tools
            )
            lc_tools.extend(
                create_mcp_langchain_tools(
                    self._mcp_tool,
                    tool_names=mcp_bind_names,
                    url_map_ref=_url_map_ref,
                    sandbox_file_uploader=_sandbox_uploader,
                )
            )
            from app.domain.services.tools.langchain_mcp_discovery import (
                create_mcp_discovery_tools,
            )
            lc_tools.extend(
                create_mcp_discovery_tools(
                    mcp_tool_ref=lambda: self._mcp_tool,
                    activated_tools_ref=lambda: self._activated_mcp_tools,
                )
            )

        lc_tools.extend(create_a2a_langchain_tools(self._a2a_tool))

        lc_tools.extend(
            create_skill_langchain_tools(
                brainstorm_skill_tool=self._brainstorm_skill_tool,
                create_skill_tool=self._create_skill_tool,
            )
        )

        lc_tools.extend(
            create_dynamic_skill_langchain_tools(self._skill_tool)
        )

        lc_tools.append(
            create_skill_guide_tool(
                skill_pool_ref=lambda: self._session_skill_pool,
                file_listings_ref=lambda: self._skill_bundle_sync.get_file_listing_all(),
                sandbox_skill_root=self._skill_bundle_sync.sandbox_skill_root,
            )
        )

        if self._memory_session_factory and self._memory_repo_factory:
            from app.domain.services.tools.memory_tools import create_memory_tools
            memory_config = self._flow._memory_config
            lc_tools.extend(
                create_memory_tools(
                    embedding_provider=self._memory_embedding_provider,
                    session_factory=self._memory_session_factory,
                    repo_factory=self._memory_repo_factory,
                    user_id=self._user_id,
                    session_id=self._session_id,
                    memory_write_service=self._memory_write_service,
                    session_redis=self._memory_session_redis,
                    session_save_cap=self._memory_session_save_cap,
                    half_life_days=memory_config.half_life_days,
                    mmr_lambda=memory_config.mmr_lambda,
                )
            )

        return lc_tools

    def _build_lc_tools_for_step(self) -> list[Any]:
        """Normal-path lc_tools construction with per-step cache.

        Cache key is ``(skill_ids, activated_mcp_tools)``. These are
        the fields that change BETWEEN STEPS within a single user message.
        ``_build_lc_tools_full`` also reads several instance
        attributes that are stable WITHIN a session but could vary across
        sessions or messages — namely ``self._skill_tool`` internal state,
        ``self._memory_session_factory`` / ``_memory_repo_factory``,
        ``self._user_id``, and ``self._flow._memory_config``. These are
        NOT in the cache key because they do not change mid-message in
        the current architecture.

        The cache is cleared at each message boundary (alongside
        ``_activated_mcp_tools.clear()``) to eliminate cross-message
        staleness risk. Within a message, re-using cached entries is safe
        because the implicit dependencies listed above are session-stable.
        """
        cache_key = (
            self._last_skill_ids,
            frozenset(self._activated_mcp_tools),
        )
        cached = self._lc_tools_cache.get(cache_key)
        if cached is not None:
            return cached
        tools = self._build_lc_tools_full()
        self._lc_tools_cache[cache_key] = tools
        return tools

    async def _build_step_react_graph(
        self, step_description: str = ""
    ) -> "tuple[Any, StepMetadata]":
        """Phase 3: progressive react_graph build + per-step metadata.

        B5 C5a: returns a 2-tuple ``(CompiledStateGraph, StepMetadata)``.
        The metadata is the authoritative per-step truth about tools bound
        and skill context in scope. See ``step_metadata.py`` for field
        semantics.

        **Atomicity contract (post-audit HIGH #1)**: the 4 bookkeeping
        fields ``_last_skill_context``, ``_last_skill_ids``,
        ``_last_bound_tool_names``, ``_last_initialized_skill_ids`` must
        be updated as a group. The entire 3-phase sequence is wrapped in
        a snapshot/restore guard so that if ANY phase raises (including
        the final ``build_react_graph`` call), all 4 fields are rolled
        back to their pre-call values and the exception propagates to
        the caller. Earlier partial-commit versions of this method
        violated the contract when Phase 1 succeeded and Phase 3
        raised — see test_refresh_skills_atomicity.py scenario 10.

        3 phases (any can fail independently):
        1. **Refresh skills** — ``_compute_refreshed_skills`` + atomic
           ``_apply_refreshed_skills``. On exception: rollback + caller
           observes the exception.
        2. **Build lc_tools** — factory errors propagate to the outer
           whole-function rollback (per #27, no graceful degradation).
        3. **Build react_graph** — if this raises, the whole function's
           snapshot/restore restores all 4 fields and re-raises.
        """
        from app.domain.services.graphs.react_graph import build_react_graph
        from app.domain.services.graphs.step_metadata import StepMetadata

        # Snapshot all 4 bookkeeping fields at function entry. Any
        # exception from any phase below restores them and re-raises so
        # the atomicity contract holds across the entire method.
        snapshot_skill_context = self._last_skill_context
        snapshot_skill_ids = self._last_skill_ids
        snapshot_bound_tool_names = self._last_bound_tool_names
        snapshot_initialized_skill_ids = self._last_initialized_skill_ids
        snapshot_risk_fp = self._last_skill_risk_fp
        # #27 hotfix: snapshot the Skill object list too. The 4 _last_* field
        # rollback alone is not enough — SkillTool internal state was already
        # mutated by Phase 1's _initialize_skill_tool_if_needed and stays on
        # the new skills if Phase 2 or Phase 3 fails. The outer except will
        # actively call await self._skill_tool.initialize(snapshot_skills) to
        # reset SkillTool, which requires the Skill objects (not just IDs).
        snapshot_initialized_skills = list(self._last_initialized_skills)

        try:
            # Phase 1: try refresh skills (pure compute + atomic apply).
            # ``_apply_refreshed_skills`` has its own internal rollback for
            # ``_initialize_skill_tool_if_needed`` failures and re-raises;
            # we catch here to mark the step as degraded and keep going
            # with the sticky previous selection rather than aborting.
            if step_description:
                try:
                    refreshed = await self._compute_refreshed_skills(step_description)
                    if refreshed is not None:
                        await self._apply_refreshed_skills(refreshed)
                    # refreshed is None → sticky: self._last_* unchanged
                except Exception as exc:
                    logger.warning(
                        "[ProgressiveSkillLoad] 刷新失败，继续使用上次选择: %s", exc
                    )
                    # self._last_* already rolled back by _apply_refreshed_skills

            # Phase 2: build lc_tools (factory errors propagate to outer
            # whole-function rollback, per #27).
            lc_tools = self._build_lc_tools_for_step()
            fresh_bound_tool_names = frozenset(t.name for t in lc_tools)

            # StepMetadata.bound_tool_names always advances to the fresh
            # set. The actual commit to self._last_bound_tool_names happens
            # AFTER build_react_graph succeeds so the rollback path covers
            # it too.
            stable_bound_tool_names = fresh_bound_tool_names

            # R1: filter by canonical category — excludes "skill creator" /
            # "skill guide" subcategories, matching pre-R1 behavior where the
            # "skill_" prefix check only caught skill_{slug}_{tool} dynamic
            # wrappers and not brainstorm_skill / generate_skill / install_skill
            # / get_skill_guide (their names do not start with "skill_").
            dynamic_tool_names = [
                t.name for t in lc_tools
                if resolve_tool_source_from_tool(t).category == "skill"
            ]
            logger.info(
                "[ProgressiveSkillLoad] step='%s' → 动态Skill工具 %d 个: %s, get_skill_guide=%s",
                step_description[:80] if step_description else "(无步骤描述)",
                len(dynamic_tool_names),
                dynamic_tool_names,
                bool(self._session_skill_pool),
            )

            # Phase 3: build the react_graph. If this raises, the outer
            # try/except restores all 4 snapshot fields before re-raising.
            step_react = build_react_graph(
                llm=self._llm,
                tools=lc_tools,
                agent_config=self._agent_config,
                tool_result_max_chars=(
                    self._flow._overflow_config.tool_result_max_chars
                    if self._flow._overflow_config
                    else 8000
                ),
                assembler=getattr(self._flow, "_assembler", None),
                tool_runtime_config=self._tool_runtime,
            )

            # Post-build atomic commit: only now do we advance
            # _last_bound_tool_names. All 4 fields are now consistent with
            # a successful step build.
            self._last_bound_tool_names = fresh_bound_tool_names

            metadata = StepMetadata(
                bound_tool_names=stable_bound_tool_names,
                skill_context=self._last_skill_context,
                skill_ids=self._last_skill_ids,
            )
            return step_react, metadata
        except Exception:
            # Whole-function rollback: any uncaught exception (including
            # build_react_graph failure) restores the 4 bookkeeping fields
            # to pre-call state before propagating. This guarantees the
            # "4 fields atomic" contract from the C5a design doc.
            self._last_skill_context = snapshot_skill_context
            self._last_skill_ids = snapshot_skill_ids
            self._last_bound_tool_names = snapshot_bound_tool_names
            self._last_initialized_skill_ids = snapshot_initialized_skill_ids
            self._last_skill_risk_fp = snapshot_risk_fp
            self._last_initialized_skills = snapshot_initialized_skills
            # #27 hotfix: SkillTool internal state was already mutated by
            # Phase 1's _initialize_skill_tool_if_needed before Phase 2 or
            # Phase 3 raised. The 4-field bookkeeping rollback above does not
            # touch self._skill_tool; without active restoration, sticky
            # follow-up calls would leak the failed step's tools (cache miss
            # + stale self._skill_tool in _build_lc_tools_full). Actively
            # re-initialize SkillTool to the snapshot state so reader paths
            # see the correct tools. SkillTool.initialize is itself atomic
            # (#27 design), so restore is success-or-no-op.
            try:
                await self._skill_tool.initialize(snapshot_initialized_skills)
            except Exception as restore_exc:
                # Restore failed — log and continue raising the original
                # exception. Subsequent calls may still see stale state,
                # but at least we tried. The dominant exception (from
                # Phase 2 or Phase 3) is more informative than this one.
                logger.error(
                    "[#27 hotfix] SkillTool state restoration failed during "
                    "_build_step_react_graph rollback: %s. Subsequent step "
                    "metadata may be inconsistent until next successful "
                    "_initialize_skill_tool_if_needed call.",
                    restore_exc,
                )
            # Also clear lc_tools cache — entries built during the failed
            # step (with skill_2 tools) must not be reused under the
            # rolled-back skill_ids key.
            self._lc_tools_cache.clear()
            raise

    @staticmethod
    def _skill_risk_fingerprint(skills: list["Skill"]) -> tuple[str, ...]:
        """Cache key that includes id + risk-relevant metadata.

        If scan_report or trust_origin changes (e.g. bundle sync rescan
        or startup backfill), the fingerprint changes and forces re-init
        so StructuredTool.metadata.risk_level picks up the new value.
        """
        parts: list[str] = []
        for s in skills:
            sr = s.scan_report or {}
            parts.append(
                f"{s.id}:{s.trust_origin}:{sr.get('content_hash', '')}:{sr.get('verdict', '')}"
            )
        return tuple(parts)

    async def _initialize_skill_tool_if_needed(self, skills: list[Skill]) -> None:
        """仅在技能集合变化时重新初始化 SkillTool，避免同 step 内抖动。

        R3: also checks risk fingerprint (trust_origin + scan verdict + hash)
        so risk changes mid-session force re-init of StructuredTool metadata.
        """
        skill_ids = tuple(skill.id for skill in skills)
        risk_fp = self._skill_risk_fingerprint(skills)
        if skill_ids == self._last_initialized_skill_ids and risk_fp == self._last_skill_risk_fp:
            logger.debug(
                "[ProgressiveSkillLoad] SkillTool 未变化，跳过重新初始化 (skills=%d)",
                len(skills),
            )
            return
        prev_ids = self._last_initialized_skill_ids
        await self._skill_tool.initialize(skills)
        self._last_initialized_skill_ids = skill_ids
        self._last_skill_risk_fp = risk_fp
        # #27 hotfix: keep object list in sync so outer rollback can restore.
        self._last_initialized_skills = list(skills)
        logger.info(
            "[ProgressiveSkillLoad] SkillTool 已更新: %s → %s",
            list(prev_ids) if prev_ids else "[]",
            list(skill_ids),
        )

    @staticmethod
    def _is_unknown_tool_event(event: ToolEvent) -> bool:
        """判断 ToolEvent 是否为 unknown-tool 降级结果.

        R4 路径: react_graph 合成 AllowError(reason.code="unknown_tool") 后,
        _translate_outcome 把 payload 存到 event.artifact dict 而非 function_result.data.
        优先读 artifact.outcome.reason.code, legacy fallback 保留用于
        pre-R4 事件日志回放 (function_result.data["code"] == "UNKNOWN_TOOL").
        """
        if event.status != ToolEventStatus.CALLED:
            return False

        # R4 path: artifact dict 保留 outcome.reason.code (小写 "unknown_tool")
        if isinstance(event.artifact, dict):
            outcome = event.artifact.get("outcome")
            if isinstance(outcome, dict):
                reason = outcome.get("reason")
                if isinstance(reason, dict) and reason.get("code") == "unknown_tool":
                    return True

        # Legacy path: pre-R4 事件日志回放 (大写 "UNKNOWN_TOOL")
        result = event.function_result
        if not result or result.success:
            return False
        data = result.data if result.data is not None else {}
        return isinstance(data, dict) and data.get("code") == "UNKNOWN_TOOL"

    def _build_virtual_step_id(self, event: BaseEvent) -> str:
        event_id = getattr(event, "id", "") or str(uuid.uuid4())
        return f"react-cycle-{event_id}"

    async def _activate_step_skills(
        self,
        *,
        step_id: str,
        user_message: str,
        selected_skills: list[Skill] | None = None,
        is_virtual: bool = False,
    ) -> None:
        """按 step 锁定技能集并更新运行时上下文（C5a 原子对路径）。

        Routes through ``_apply_preselected_skills`` so that
        ``_last_initialized_skill_ids`` / ``_last_skill_context`` /
        ``_last_skill_ids`` advance atomically together. If
        ``_initialize_skill_tool_if_needed`` raises, all three fields
        stay at their pre-call values (post-#27 atomic guarantee +
        await-free critical section in the helper). See design:
        docs/superpowers/specs/2026-04-13-activate-step-skills-atomicity-design.md
        §3.2 Site 3.
        """
        if (
            self._step_skill_state
            and self._step_skill_state.step_id == step_id
            and self._step_skill_state.locked_skills
        ):
            return

        target_skills = list(selected_skills or [])
        if not target_skills:
            target_skills, _ = await self._select_skills_for_message(
                self._session_skill_pool,
                user_message,
            )

        await self._apply_preselected_skills(target_skills)
        self._step_skill_state = StepSkillActivationState(
            step_id=step_id,
            user_message=user_message,
            locked_skills=list(target_skills),
        )
        if is_virtual:
            self._last_virtual_step_id = step_id

    async def _handle_step_skill_lock(self, event: BaseEvent, user_message: str) -> None:
        """处理 step 级 skill 锁定、unknown-tool 紧急重选与无 Planner 降级路径。"""
        if not self._skill_selection_policy.step_skill_lock_enabled:
            return

        if isinstance(event, StepEvent):
            step_id = event.step.id or self._build_virtual_step_id(event)
            if event.status == StepEventStatus.STARTED:
                await self._activate_step_skills(
                    step_id=step_id,
                    user_message=user_message,
                    selected_skills=self._current_message_selected_skills,
                )
                return
            if event.status in {StepEventStatus.COMPLETED, StepEventStatus.FAILED}:
                if self._step_skill_state and self._step_skill_state.step_id == step_id:
                    self._step_skill_state = None
                return

        if self._step_skill_state is None:
            virtual_step_id = self._build_virtual_step_id(event)
            await self._activate_step_skills(
                step_id=virtual_step_id,
                user_message=user_message,
                selected_skills=self._current_message_selected_skills,
                is_virtual=True,
            )

        if not isinstance(event, ToolEvent) or self._step_skill_state is None:
            return

        state = self._step_skill_state
        if not self._is_unknown_tool_event(event):
            state.consecutive_unknown_tool_calls = 0
            return

        state.consecutive_unknown_tool_calls += 1
        threshold = self._skill_selection_policy.step_skill_reselect_unknown_tool_threshold
        max_reselect = self._skill_selection_policy.step_skill_reselect_max_per_step
        if state.consecutive_unknown_tool_calls < threshold:
            return
        if state.reselect_count >= max_reselect:
            return

        selected_skills, _ = await self._select_skills_for_message(
            self._session_skill_pool,
            user_message,
        )
        # Field split (TODO #30 spec §3.2 Site 4):
        # - Attempt counters (reselect_count, consecutive_unknown_tool_calls)
        #   advance BEFORE apply so max_reselect cap is preserved even if
        #   _apply_preselected_skills raises.
        # - Success commit (locked_skills) advances AFTER apply, only when
        #   apply succeeds.
        state.reselect_count += 1
        state.consecutive_unknown_tool_calls = 0
        await self._apply_preselected_skills(selected_skills)
        state.locked_skills = list(selected_skills)
        logger.warning(
            "step内连续unknown-tool达到阈值，触发一次技能重选(step_id=%s, reselect_count=%s)",
            state.step_id,
            state.reselect_count,
        )

    async def _load_user_preferences_map(self, tool_type: ToolType) -> dict[str, bool]:
        """加载用户在指定工具类型下的偏好映射。"""
        if not self._user_id:
            return {}

        try:
            postgres = get_postgres()
            async with postgres.session_factory() as session:
                pref_repository = DBUserToolEnablementRepository(session)
                preferences = await pref_repository.get_by_user_id(self._user_id, tool_type)
                return {preference.tool_id: preference.enabled for preference in preferences}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "加载用户工具偏好失败，降级为默认启用(user_id=%s, tool_type=%s): %s",
                self._user_id,
                tool_type.value,
                str(e),
            )
            return {}

    @staticmethod
    def _apply_user_preferences_to_mcp_config(
        mcp_config: MCPConfig, preference_map: dict[str, bool]
    ) -> MCPConfig:
        """将用户偏好应用到 MCP 配置并返回副本。"""
        return MCPConfig(
            mcpServers={
                server_name: server_config.model_copy(
                    deep=True,
                    update={
                        "enabled": bool(
                            server_config.enabled
                            and preference_map.get(server_name, True)
                        )
                    },
                )
                for server_name, server_config in mcp_config.mcpServers.items()
            }
        )

    @staticmethod
    def _apply_user_preferences_to_a2a_config(
        a2a_config: A2AConfig, preference_map: dict[str, bool]
    ) -> A2AConfig:
        """将用户偏好应用到 A2A 配置并返回副本。"""
        return A2AConfig(
            a2a_servers=[
                server.model_copy(
                    deep=True,
                    update={"enabled": bool(server.enabled and preference_map.get(server.id, True))},
                )
                for server in a2a_config.a2a_servers
            ]
        )

    @staticmethod
    def _filter_skills_by_user_preferences(
        skills: list[Skill], preference_map: dict[str, bool]
    ) -> list[Skill]:
        """基于用户偏好过滤 Skill 列表。"""
        return [skill for skill in skills if preference_map.get(skill.id, True)]

    async def _handle_tool_event(self, event: ToolEvent) -> None:
        """额外处理工具消息，使其前端交互更友好"""
        try:
            # 1.如果事件状态为已调用则执行以下代码
            if event.status == ToolEventStatus.CALLED:
                # R1: react_graph 已将 ToolEvent.tool_name 写为 canonical category
                # （见 graphs/react_graph.py 中对 resolve_tool_source(...).category 的调用），此处直接消费。
                # R4 CS3 Task 15+16: compute envelope ONCE; all 7 enrichment branches
                # (search/mcp/a2a/shell/file/skill/skill_creator) now consume envelope.*
                # instead of event.function_result / event.function_args directly.
                from app.application.services.tool_event_envelope_v1 import (
                    project_tool_event_to_envelope_v1,
                )
                envelope = project_tool_event_to_envelope_v1(event)
                category = event.tool_name
                logger.debug("处理工具事件: tool_name=%s, function=%s, category=%s", event.tool_name, event.function_name, category)
                # 2.工具为浏览器则补全工具浏览器工具内容
                if category == "browser":
                    screenshot_url = await self._get_browser_screenshot()
                    logger.debug("浏览器截图完成: url=%s", screenshot_url[:80] if screenshot_url else "(empty)")
                    event.tool_content = BrowserToolContent(
                        screenshot=screenshot_url,
                    )
                elif category == "search":
                    # 3.工具为搜索则添加搜索工具内容
                    # R4 CS3 Task 15 migration: read envelope.function_result.data (dict)
                    # instead of event.function_result.data. Pre-migration used
                    # hasattr(data, "results") which was always False for dict input —
                    # latent bug fixed here via isinstance + SearchResultItem.model_validate.
                    from app.domain.models.search import SearchResultItem

                    fr = envelope.function_result
                    data = fr.data if fr else None
                    if isinstance(data, dict) and isinstance(data.get("results"), list):
                        try:
                            results = [SearchResultItem.model_validate(r) for r in data["results"]]
                        except Exception:
                            logger.warning("search result dict shape drift, falling back to empty")
                            results = []
                        event.tool_content = SearchToolContent(results=results)
                    else:
                        event.tool_content = SearchToolContent(results=[])
                elif category == "shell":
                    # 4.工具为shell则生成shell工具内容
                    # R4 CS3 Task 16: use envelope.function_args for consistency.
                    session_id = envelope.function_args.get("session_id", "default")
                    shell_result = await self._sandbox.read_shell_output(
                        session_id, console=True,
                    )
                    console_records = (shell_result.data or {}).get("console_records", [])
                    event.tool_content = ShellToolContent(
                        console=console_records
                    )
                    # Shell 命令可能生成输出文件，主动扫描并同步
                    exec_dir = envelope.function_args.get("exec_dir", "")
                    if exec_dir:
                        await self._sync_generated_files(exec_dir)
                elif category == "file":
                    # 5.工具为file则将文件同步到对象存储
                    # R4 CS3 Task 16: use envelope.function_args / envelope.function_name.
                    filepath = envelope.function_args.get("filepath")
                    if filepath:
                        file_read_result = await self._sandbox.read_file(filepath)
                        file_content: str = (file_read_result.data or {}).get(
                            "content", ""
                        )
                        event.tool_content = FileToolContent(content=file_content)
                        # 写操作和显式文件查看都需要把沙箱文件同步到会话文件列表
                        if envelope.function_name in (
                            "file_write",
                            "file_str_replace",
                            "file_view",
                        ):
                            await self._sync_file_to_storage(filepath)
                    else:
                        event.tool_content = FileToolContent(content="(No Content)")
                elif category in ("mcp", "a2a"):
                    # 6.工具为mcp/a2a则处理调用结果
                    # R4 CS3 Task 15 migration: projector-driven enrichment.
                    # fr.status == "ok" with data → passthrough data; ok without data → message;
                    # non-ok → placeholder from fr.message.
                    is_mcp = category == "mcp"
                    fr = envelope.function_result
                    if fr is None:
                        logger.warning("MCP/A2A工具调用结果未发现")
                        event.tool_content = (
                            MCPToolContent(result="(MCP工具无可用结果)")
                            if is_mcp
                            else A2AToolContent(a2a_result="(A2A智能体无可用结果)")
                        )
                    elif fr.status != "ok":
                        # error / denied / timeout — fall back to message
                        logger.info("MCP/A2A工具失败: status=%s, message=%s", fr.status, fr.message)
                        event.tool_content = (
                            MCPToolContent(result=fr.message)
                            if is_mcp
                            else A2AToolContent(a2a_result=fr.message)
                        )
                    elif fr.data is not None:
                        logger.info("MCP/A2A工具调用结果: %s", fr.data)
                        event.tool_content = (
                            MCPToolContent(result=fr.data)
                            if is_mcp
                            else A2AToolContent(a2a_result=fr.data)
                        )
                    else:
                        # ok but no data — use message
                        logger.info("MCP/A2A工具调用成功返回，但无 data: %s", fr.message)
                        event.tool_content = (
                            MCPToolContent(result=fr.message)
                            if is_mcp
                            else A2AToolContent(a2a_result=fr.message)
                        )
                elif category == "skill":
                    # Native skill 通过 ToolResult.data.shell_session_id 传递沙箱会话
                    # R4 CS3 Task 16: use envelope.function_result instead of event.function_result.
                    fr = envelope.function_result
                    skill_data = fr.data if fr and fr.data is not None else {}
                    shell_sid = (
                        skill_data.get("shell_session_id")
                        if isinstance(skill_data, dict)
                        else None
                    )
                    if shell_sid:
                        # 读取终端输出，供 UI 终端面板展示
                        try:
                            shell_result = await self._sandbox.read_shell_output(
                                shell_sid, console=True,
                            )
                            console_records = (shell_result.data or {}).get(
                                "console_records", []
                            )
                            event.tool_content = ShellToolContent(
                                console=console_records
                            )
                        except Exception:
                            event.tool_content = SkillToolContent(
                                skill_result=skill_data
                            )
                    elif fr and fr.data is not None:
                        event.tool_content = SkillToolContent(
                            skill_result=fr.data
                        )
                    elif fr and fr.message:
                        event.tool_content = SkillToolContent(
                            skill_result=fr.message
                        )
                    else:
                        event.tool_content = SkillToolContent(
                            skill_result="(Skill工具无可用结果)"
                        )
                    # Skill 执行可能生成输出文件，主动扫描并同步
                    skill_exec_dir = (
                        skill_data.get("exec_dir")
                        if isinstance(skill_data, dict)
                        else None
                    )
                    if skill_exec_dir:
                        await self._sync_generated_files(skill_exec_dir)
                elif category == "skill creator":
                    # R4 CS3 Task 16: use envelope.function_result instead of event.function_result.
                    fr = envelope.function_result
                    if fr and fr.data is not None:
                        event.tool_content = SkillToolContent(
                            skill_result=fr.data
                        )
                    elif fr and fr.message:
                        event.tool_content = SkillToolContent(
                            skill_result=fr.message
                        )
                    else:
                        event.tool_content = SkillToolContent(
                            skill_result="(Skill Creator 工具无可用结果)"
                        )
        except Exception as e:
            logger.exception("AgentTaskRunner生成工具内容失败: %s", e)

    def _snapshot_metrics(self) -> Optional[Dict[str, Any]]:
        """D5: Snapshot execution metrics for inclusion in terminal events."""
        if not self._flow:
            return None
        _em = getattr(self._flow, "_execution_metrics", None)
        if not _em:
            return None
        try:
            return _em.to_dict()
        except Exception:
            return None

    async def _emit_flow_event(self, task: Task, event: BaseEvent) -> str | None:
        """Emit a single flow event to the task output stream and apply side effects.

        Handles:
        - Streaming chunked assistant messages
        - Persisting events (skip partial MessageEvents)
        - Side effects: TitleEvent, MessageEvent, WaitEvent, ControlEvent

        Returns:
            "wait"     — caller should return immediately (WaitEvent received)
            "takeover" — caller should return immediately (ControlEvent REQUESTED received)
            None       — continue normally
        """
        emitted_events: List[Event] = []
        if isinstance(event, MessageEvent) and event.role == "assistant":
            async for chunked_event in self._stream_assistant_message_event(event):
                emitted_events.append(chunked_event)
        else:
            emitted_events.append(event)

        for emitted_event in emitted_events:
            should_persist = not (
                isinstance(emitted_event, MessageEvent) and emitted_event.partial
            )
            await self._put_and_add_event(task, emitted_event, persist=should_persist)

            if isinstance(emitted_event, TitleEvent):
                async with self._uow:
                    await self._uow.session.update_title(
                        self._session_id, emitted_event.title
                    )
            elif isinstance(emitted_event, MessageEvent) and not emitted_event.partial:
                async with self._uow:
                    await self._uow.session.update_latest_message(
                        self._session_id,
                        emitted_event.message,
                        emitted_event.created_at,
                    )
                    await self._uow.session.increment_unread_message_count(
                        self._session_id
                    )
            elif isinstance(emitted_event, (WaitEvent, ToolConfirmationEvent)):
                async with self._uow:
                    await self._uow.session.update_status(
                        self._session_id, SessionStatus.WAITING
                    )
                return "wait"
            elif (
                isinstance(emitted_event, ControlEvent)
                and emitted_event.action == ControlAction.REQUESTED
            ):
                async with self._uow:
                    await self._uow.session.update_status(
                        self._session_id, SessionStatus.TAKEOVER_PENDING
                    )
                return "takeover"
            # D5: Track watchdog termination
            elif isinstance(emitted_event, HealthEvent) and emitted_event.status in (
                HealthStatus.TERMINATING, HealthStatus.TERMINATED,
            ):
                self._was_timed_out = True

        return None

    async def _run_flow(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """根据消息对象运行PlannerReActFlow"""
        # 1.判断传递的消息是否为空
        if not message.message:
            logger.warning(f"AgentTaskRunner接收了一条空消息")
            yield ErrorEvent(error="空消息错误")
            return

        # 2.调用流并运行获取事件信息
        async for event in self._flow.invoke(message):
            # 3.判断是否为工具事件，如果是则额外处理
            if isinstance(event, ToolEvent):
                await self._handle_tool_event(event)
                if event.status == ToolEventStatus.CALLED:
                    logger.debug("enrichment结果: tool_name=%s, function=%s, tool_content=%s, is_none=%s", event.tool_name, event.function_name, type(event.tool_content).__name__, event.tool_content is None)
            elif isinstance(event, MessageEvent):
                # 4.如果是消息事件则将AI消息事件中的附件同步到存储中
                await self._sync_message_attachments_to_storage(event)

            # 5.将事件直接返回
            yield event

        # 6.流消费完毕后，读取压缩结果并发送上下文状态/压缩事件（B3）
        compaction_result = getattr(self._flow, "_last_compaction_result", None)
        if compaction_result is not None:
            overflow_config = getattr(self._flow, "_overflow_config", None)
            context_window = 0
            if overflow_config is not None:
                try:
                    from app.domain.services.context.model_context_window import resolve_context_window
                    context_window = resolve_context_window(
                        overflow_config.model_name, overflow_config
                    )
                except Exception as exc:
                    logger.warning("Failed to resolve context window for SSE event: %s", exc)
                    context_window = overflow_config.context_window or 0

            soft_threshold = overflow_config.soft_trigger_ratio if overflow_config else 0.85
            hard_threshold = overflow_config.hard_trigger_ratio if overflow_config else 0.95

            yield ContextStatusEvent(
                used_tokens=compaction_result.tokens_after,
                context_window=context_window,
                usage_ratio=compaction_result.usage_ratio_after,
                soft_threshold=soft_threshold,
                hard_threshold=hard_threshold,
            )

            if compaction_result.level_applied > 0:
                yield CompactionEvent(
                    level=compaction_result.level_applied,
                    tokens_before=compaction_result.tokens_before,
                    tokens_after=compaction_result.tokens_after,
                    messages_removed=compaction_result.messages_removed,
                    usage_ratio_after=compaction_result.usage_ratio_after,
                )
                # D5: Track compaction count for metrics
                if self._flow:
                    _em = getattr(self._flow, "_execution_metrics", None)
                    if _em:
                        _em.compaction_count += 1
                        _em.context_usage_ratio = compaction_result.usage_ratio_after

    async def _do_persist_and_flush(self) -> None:
        """Phase 1+2: persist state then submit flush.

        Runs inside asyncio.shield() so CancelledError cannot interrupt
        mid-persist (which includes an LLM summary call when enabled).
        """
        flow = self._flow

        # Phase 1: persist (Memory/ConversationSummary/flush gate/overflow check)
        await flow._persist_after_graph(
            flow._deferred_final_state, flow._deferred_summaries
        )

        # Phase 2: flush submit (synchronous fire-and-forget)
        flush_batch = getattr(flow, "_pending_flush_batch", None)
        if flush_batch and self._memory_flusher:
            self._memory_flusher.submit(flush_batch)

    async def _do_postprocess(self, task: Task) -> None:
        """Execute post-processing: persist + flush + user-visible summary.

        Phase 1+2 are shielded from cancellation to guarantee persistence
        completes even when the user sends a follow-up message.
        Phase 3 failure -> silent degradation (summary loss is acceptable).
        """
        # Shield Phase 1+2: CancelledError cannot reach _persist_after_graph
        # or flush submit.  If the outer task is cancelled while shield is
        # running, CancelledError is raised HERE after shield finishes.
        await asyncio.shield(self._do_persist_and_flush())

        # Phase 3: user-visible streaming summary
        flow = self._flow
        messages = flow._deferred_final_state.get("messages", [])
        if messages:
            summary_stream_id: str | None = None

            try:
                async def _on_summary_event(evt: BaseEvent) -> None:
                    nonlocal summary_stream_id
                    if isinstance(evt, MessageEvent) and evt.stream_id:
                        summary_stream_id = evt.stream_id
                    is_final = isinstance(evt, MessageEvent) and not evt.partial
                    await self._put_and_add_event(task, evt, persist=is_final)
                    # Final summary drives sidebar preview + unread count
                    # (matches original summarizer_node behavior in main event loop)
                    if is_final:
                        try:
                            async with self._uow:
                                await self._uow.session.update_latest_message(
                                    self._session_id,
                                    evt.message,
                                    evt.created_at,
                                )
                                await self._uow.session.increment_unread_message_count(
                                    self._session_id
                                )
                        except Exception as e:
                            logger.warning("Summary latest_message update failed: %s", e)

                _summary_lang = (flow._deferred_final_state or {}).get("language", "zh")
                # B4 M0: thread the session-scoped cost callback into the
                # graph-external summary call so its tokens reach the ledger
                # with node_name="background_summary". Use ``getattr`` so
                # tests that bypass ``__init__`` (``object.__new__``) still
                # work — the attribute is simply absent → no callbacks.
                _cost_handler = getattr(self, "_cost_callback_handler", None)
                _cost_callbacks = (
                    [_cost_handler] if _cost_handler is not None else None
                )
                await run_background_summary(
                    messages,
                    flow.summary_llm,
                    _on_summary_event,
                    lang=_summary_lang,
                    callbacks=_cost_callbacks,
                )
            except asyncio.CancelledError:
                # Send a final non-partial message to clear the ghost partial
                # in the frontend (which upserts by stream_id).
                if summary_stream_id:
                    try:
                        await self._put_and_add_event(task, MessageEvent(
                            role="assistant",
                            message="",
                            stream_id=summary_stream_id,
                            partial=False,
                        ))
                    except Exception:
                        pass
                raise
            except Exception as e:
                logger.warning("Phase 3 用户可见摘要失败（静默降级）: %s", e)

    async def _run_postprocess_or_cancel(self, task: Task) -> bool:
        """Run post-processing; cancel if new message arrives. Returns True if cancelled."""
        postprocess = asyncio.create_task(self._do_postprocess(task))

        try:
            while not postprocess.done():
                if not await task.input_stream.is_empty():
                    postprocess.cancel()
                    try:
                        await postprocess
                    except asyncio.CancelledError:
                        pass
                    return True
                await asyncio.sleep(0.2)

            await postprocess  # propagate exceptions

            # Drain check: catch messages that arrived between last poll and completion
            if not await task.input_stream.is_empty():
                return True

            return False
        except asyncio.CancelledError:
            postprocess.cancel()
            try:
                await postprocess
            except asyncio.CancelledError:
                pass
            raise

    async def _set_terminal_status(self, status: SessionStatus) -> None:
        """Set session to a terminal status and fire the on_session_complete callback.

        Consolidates the COMPLETED/TIMED_OUT write + lifecycle suspend notification
        so all completion paths (invoke, resume, CancelledError, Exception) go through
        one place.

        B4 M0: before marking the session terminal, drain any in-flight
        CostCallbackHandler persist tasks so the cost ledger is complete when
        the UI first queries ``GET /cost``. A failing flush is logged but
        does not block the terminal transition — the degraded-session marker
        (design Issue 1D) will be added alongside the shield wrap in a
        follow-up.
        """
        if self._cost_callback_handler is not None:
            try:
                # Cap the drain at 3s so a stuck DB / connection pool doesn't
                # wedge the session in FINISHING. Pending tasks keep running
                # in the background; any failures are already logged by
                # ``_persist_safely``.
                await self._cost_callback_handler.flush_pending(timeout=3.0)
            except Exception as exc:  # noqa: BLE001 — terminal path must not stall
                logger.warning(
                    "CostCallbackHandler.flush_pending failed on terminal "
                    "transition (session_id=%s, status=%s): %s",
                    self._session_id, status, exc,
                )
        async with self._uow:
            await self._uow.session.update_status(self._session_id, status)
        if self._on_session_complete is not None:
            try:
                await self._on_session_complete(self._session_id)
            except Exception:
                logger.debug(
                    "on_session_complete callback failed for session %s",
                    self._session_id,
                )

    async def _cleanup_tools(self) -> None:
        """清理MCP和A2A工具资源，确保在同一任务上下文中释放

        注意：该方法必须在初始化MCP/A2A的同一个asyncio Task中调用，
        否则anyio的cancel scope会检测到任务上下文切换并抛出RuntimeError。
        """
        try:
            if self._mcp_tool:
                await self._mcp_tool.cleanup()
        except Exception as e:
            logger.warning(f"清理MCP工具资源时出错: {e}")
        try:
            if self._a2a_tool and self._a2a_tool.manager:
                await self._a2a_tool.manager.cleanup()
        except Exception as e:
            logger.warning(f"清理A2A工具资源时出错: {e}")
        try:
            if self._skill_bundle_sync:
                await self._skill_bundle_sync.cleanup()
        except Exception as e:
            logger.warning(f"清理Skill bundle同步任务时出错: {e}")
        try:
            if self._skill_tool:
                await self._skill_tool.cleanup()
        except Exception as e:
            logger.warning(f"清理Skill工具资源时出错: {e}")
        self._session_skill_pool = []
        self._last_effective_selected_skills = []
        self._last_substantive_user_message = ""
        self._continuation_decision_cache.clear()
        self._step_skill_state = None
        self._current_message_selected_skills = []
        self._current_message_text = ""
        self._last_initialized_skill_ids = ()
        self._last_skill_risk_fp = ()
        self._last_initialized_skills = []

    async def invoke(self, task: Task) -> None:
        """根据传递的任务处理agent消息队列并运行agent流"""
        try:
            # 1.任务一启动先推进会话状态，避免前端长期显示pending
            async with self._uow:
                await self._uow.session.update_status(
                    self._session_id, SessionStatus.RUNNING
                )

            # 2.确保沙箱、mcp、a2a均初始化完成
            logger.info(f"AgentTaskRunner任务处理开始")
            await self._sandbox.ensure_sandbox()
            mcp_preference_map = await self._load_user_preferences_map(ToolType.MCP)
            a2a_preference_map = await self._load_user_preferences_map(ToolType.A2A)
            skill_preference_map = await self._load_user_preferences_map(ToolType.SKILL)
            filtered_mcp_config = self._apply_user_preferences_to_mcp_config(
                self._mcp_config,
                mcp_preference_map,
            )
            filtered_a2a_config = self._apply_user_preferences_to_a2a_config(
                self._a2a_config,
                a2a_preference_map,
            )
            await self._mcp_tool.initialize(filtered_mcp_config)
            await self._a2a_tool.initialize(filtered_a2a_config)
            enabled_skills = await self._load_enabled_skills()
            self._session_skill_pool = self._filter_skills_by_user_preferences(
                enabled_skills,
                skill_preference_map,
            )
            # Phase 1: Embedding 索引构建
            embedding_config = getattr(self._agent_config, 'skill_embedding', None)
            if embedding_config and embedding_config.enabled and embedding_config.api_base and embedding_config.api_key:
                try:
                    from app.infrastructure.external.embedding.openai_embedding_provider import OpenAIEmbeddingProvider
                    from app.infrastructure.external.embedding.skill_embedding_index import SkillEmbeddingIndex
                    provider = OpenAIEmbeddingProvider(
                        api_base=embedding_config.api_base,
                        api_key=embedding_config.api_key,
                        model=embedding_config.model,
                        dimensions=embedding_config.dimensions,
                    )
                    # 尝试使用 Redis 缓存
                    embedding_cache = None
                    try:
                        from app.infrastructure.external.embedding.redis_embedding_cache import RedisEmbeddingCache
                        from app.infrastructure.storage.redis import get_redis
                        redis_client = get_redis()
                        _ = redis_client.client  # Raises RuntimeError if not initialized
                        embedding_cache = RedisEmbeddingCache(redis_client.client)
                    except (RuntimeError, Exception):
                        embedding_cache = None
                    self._embedding_index = SkillEmbeddingIndex(provider, cache=embedding_cache)
                    await self._embedding_index.build(self._session_skill_pool)
                    self._embedding_available = True
                    logger.info("Embedding 索引构建完成: skills=%d, model=%s", len(self._session_skill_pool), embedding_config.model)
                except Exception:
                    logger.warning("Embedding 不可用，将使用 token-overlap", exc_info=True)
                    self._embedding_available = False
            initial_skills = self._select_skills_from_pool(
                self._session_skill_pool,
                "",
            )
            await self._skill_bundle_sync.prepare_startup_sync(
                skill_pool=self._session_skill_pool,
                initial_selected=initial_skills,
            )
            await self._skill_bundle_sync.await_initial_sync()
            await self._apply_preselected_skills(initial_skills)
            self._skill_bundle_sync.start_background_sync()

            # 传递 skill pool getter 和 file listings getter 给 flow，用于 get_skill_guide 按需加载
            # 使用 hasattr duck-typing guard，兼容测试中的 mock flow
            if hasattr(self._flow, '_skill_pool_getter'):
                self._flow._skill_pool_getter = lambda: self._session_skill_pool
                self._flow._file_listings_getter = lambda: self._skill_bundle_sync.get_file_listing_all()
                self._flow._sandbox_skill_root = self._skill_bundle_sync.sandbox_skill_root
            # MCP progressive loading: pass discovery dependencies to flow
            if hasattr(self._flow, '_mcp_tool_ref'):
                self._flow._mcp_tool_ref = lambda: self._mcp_tool
                self._flow._activated_mcp_tools_ref = lambda: self._activated_mcp_tools
                self._flow._mcp_always_bind_names = self._get_always_bind_tool_names()

            # 3. 主消息循环 + FINISHING 后处理
            try:
                while True:
                    # Phase A: 处理所有待处理消息
                    while not await task.input_stream.is_empty():
                        event = await self._pop_event(task)
                        if event is None:
                            continue
                        message = ""

                        # 5.判断事件类型是否为消息事件，如果是则处理消息并将附件同步到沙箱中
                        image_content_blocks: list[dict] = []
                        if isinstance(event, MessageEvent):
                            message = event.message or ""
                            await self._sync_message_attachments_to_sandbox(event)
                            # 构建图片附件的多模态内容块，使 LLM 能直接"看到"图片
                            logger.debug(
                                "before _build_image_blocks: attachments count=%d, types=%s, mimes=%s",
                                len(event.attachments),
                                [type(a).__name__ for a in event.attachments],
                                [getattr(a, "mime_type", "N/A") for a in event.attachments],
                            )
                            # A7 Task 2.7: honor profile.accepts_image_url. Filter
                            # non-File attachments upstream (legacy behavior that
                            # lived inside the old method body).
                            _image_attachments = [
                                a for a in event.attachments if isinstance(a, File)
                            ]
                            image_content_blocks = await self._build_image_blocks(
                                _image_attachments
                            )
                            logger.debug("after _build_image_blocks: blocks=%d", len(image_content_blocks))
                            logger.info(
                                "AgentTaskRunner接收到新消息(len=%s, digest=%s, images=%d)",
                                len(message),
                                hashlib.sha256(message.encode("utf-8")).hexdigest(),
                                len(image_content_blocks),
                            )

                        # 6.将消息事件转换称消息对象
                        # 附件路径附带外部可访问 URL（MCP 工具无法访问沙箱文件系统）
                        attachment_paths: list[str] = []
                        if isinstance(event, MessageEvent):
                            for att in event.attachments:
                                path = att.filepath
                                url = self._image_url_map.get(path)
                                if url:
                                    attachment_paths.append(
                                        f"{path} (external_url: {url})"
                                    )
                                else:
                                    attachment_paths.append(path)
                        message_obj = Message(
                            message=message,
                            attachments=attachment_paths,
                            image_content_blocks=image_content_blocks,
                            skill_confirmation_action=(
                                event.skill_confirmation_action
                                if isinstance(event, MessageEvent)
                                else None
                            ),
                            language=self._current_language,  # B5 #29: bootstrap hint
                        )

                        selected_skills, _ = await self._select_skills_for_message(
                            self._session_skill_pool,
                            message_obj.message,
                        )
                        self._current_message_text = message_obj.message
                        self._current_message_selected_skills = list(selected_skills)
                        self._step_skill_state = None
                        self._last_virtual_step_id = ""
                        self._activated_mcp_tools.clear()  # Reset MCP activation per message
                        # B5 C5a: clear lc_tools cache at the message boundary.
                        # Cache entries are keyed by (mode, skill_ids, mcp_activation)
                        # but also implicitly depend on instance state that can't
                        # be practically added to the key (e.g. _skill_tool internal
                        # state, _memory_session_factory wiring). Clearing here is
                        # cheap and eliminates cross-message staleness risk.
                        self._lc_tools_cache.clear()
                        await self._apply_preselected_skills(selected_skills, scores=self._current_embedding_scores)

                        # Phase 2+3: 设置 LangGraph configurable 回调
                        if hasattr(self._flow, '_skill_context_refresher'):
                            self._flow._skill_context_refresher = self._refresh_skill_context_for_step
                            self._flow._react_graph_provider = self._build_step_react_graph
                            self._flow._skill_guide_injector = SkillGuideInjector(
                                selected_skills, self._tier2_preloaded_skill_ids
                            )
                        # B5 post-audit LOW #1: inject language callback so
                        # main_graph.planner_node can notify us of the real
                        # session language after parsing the plan. The flow
                        # stores it and ``_build_config`` forwards it via
                        # ``configurable["language_callback"]``.
                        if hasattr(self._flow, "_language_callback"):
                            self._flow._language_callback = self.set_language
                        # TODO #30: clock 2 replacement — provider callback
                        # reading runner's _last_skill_context. Wired here
                        # (not in __init__) because this spec intentionally
                        # matches the existing invoke-main-loop wiring pattern;
                        # moving it to __init__ is part of the deferred #25 work.
                        # See spec §3.4 / §1.4.
                        if hasattr(self._flow, "_skill_context_provider"):
                            self._flow._skill_context_provider = lambda: self._last_skill_context

                        # 7.传递消息对象并运行PlannerReActFlow
                        async for event in self._run_flow(message_obj):
                            await self._handle_step_skill_lock(event, message_obj.message)
                            # 8-12. 发送事件到输出流并处理各类侧效应
                            result = await self._emit_flow_event(task, event)
                            if result in ("wait", "takeover"):
                                return

                        # 单条消息执行结束后重置step锁定状态
                        self._step_skill_state = None

                    # Phase B: 无消息且有延迟后处理 -> FINISHING
                    if self._flow and getattr(self._flow, '_deferred_final_state', None):
                        async with self._uow:
                            await self._uow.session.update_status(
                                self._session_id, SessionStatus.FINISHING
                            )
                        await self._put_and_add_event(task, FinishingEvent())

                        try:
                            cancelled = await self._run_postprocess_or_cancel(task)
                        except Exception as e:
                            logger.error("后处理失败 (postprocess_incomplete): %s", e)
                            await self._put_and_add_event(task, DoneEvent(metrics=self._snapshot_metrics()))
                            break

                        if cancelled:
                            # Reset deferred state to prevent stale re-entry
                            self._flow._deferred_final_state = None
                            self._flow._deferred_summaries = None
                            async with self._uow:
                                await self._uow.session.update_status(
                                    self._session_id, SessionStatus.RUNNING
                                )
                            continue
                        else:
                            await self._put_and_add_event(task, DoneEvent(metrics=self._snapshot_metrics()))
                            break
                    else:
                        break

                # Normal completion (or watchdog termination)
                if self._was_timed_out:
                    # D5: Emit HealthEvent(TERMINATED) with execution metrics
                    await self._put_and_add_event(task, HealthEvent(
                        status=HealthStatus.TERMINATED,
                        reason="执行已超时终止，请查看已完成的进展",
                        action="terminated",
                        metrics=self._snapshot_metrics(),
                    ))
                    await self._set_terminal_status(SessionStatus.TIMED_OUT)
                else:
                    await self._set_terminal_status(SessionStatus.COMPLETED)

            except asyncio.CancelledError:
                cancel_reason = getattr(task, "cancel_reason", "stop")
                logger.info("AgentTaskRunner任务运行取消，reason=%s", cancel_reason)

                if cancel_reason in {"takeover_start", "takeover_timeout"}:
                    raise

                if cancel_reason == "session_delete":
                    raise

                await self._put_and_add_event(task, DoneEvent())
                await self._set_terminal_status(SessionStatus.COMPLETED)
                raise

            except Exception as e:
                logger.exception(f"AgentTaskRunner运行出错: {str(e)}")
                await self._put_and_add_event(
                    task, ErrorEvent(error=f"AgentTaskRunner出错: {str(e)}")
                )
                await self._set_terminal_status(SessionStatus.COMPLETED)
        finally:
            # 17.在同一个asyncio Task上下文中清理MCP/A2A工具资源
            # 这是关键：streamablehttp_client内部使用anyio.create_task_group()，
            # 要求在同一个Task中进入和退出cancel scope，
            # 所以必须在invoke()的finally块（即初始化MCP的同一个Task）中清理
            await self._cleanup_tools()

    async def resume(self, task: Task, command: Any) -> None:
        """Resume a paused task with a LangGraph Command.

        Streams events from flow.resume(command) and bridges them to the task's
        output stream using the same event-handling logic as invoke().
        Includes FINISHING state and DoneEvent emission to match invoke() behaviour.
        """
        try:
            async for event in self._flow.resume(command):
                # Enrich ToolEvents and sync MessageEvent attachments (mirrors _run_flow)
                if isinstance(event, ToolEvent):
                    await self._handle_tool_event(event)
                    if event.status == ToolEventStatus.CALLED:
                        logger.debug(
                            "enrichment结果: tool_name=%s, function=%s, tool_content=%s, is_none=%s",
                            event.tool_name, event.function_name,
                            type(event.tool_content).__name__, event.tool_content is None,
                        )
                elif isinstance(event, MessageEvent):
                    await self._sync_message_attachments_to_storage(event)

                result = await self._emit_flow_event(task, event)
                if result in ("wait", "takeover"):
                    return

            # FINISHING: run deferred post-processing then emit DoneEvent (mirrors invoke())
            if self._flow and getattr(self._flow, "_deferred_final_state", None):
                async with self._uow:
                    await self._uow.session.update_status(
                        self._session_id, SessionStatus.FINISHING
                    )
                await self._put_and_add_event(task, FinishingEvent())

                try:
                    cancelled = await self._run_postprocess_or_cancel(task)
                except Exception as e:
                    logger.error("resume 后处理失败 (postprocess_incomplete): %s", e)
                    await self._put_and_add_event(task, DoneEvent(metrics=self._snapshot_metrics()))
                    await self._set_terminal_status(SessionStatus.COMPLETED)
                    return

                if not cancelled:
                    await self._put_and_add_event(task, DoneEvent(metrics=self._snapshot_metrics()))
                    _final_status = SessionStatus.TIMED_OUT if self._was_timed_out else SessionStatus.COMPLETED
                    await self._set_terminal_status(_final_status)
            else:
                await self._put_and_add_event(task, DoneEvent(metrics=self._snapshot_metrics()))
                _final_status = SessionStatus.TIMED_OUT if self._was_timed_out else SessionStatus.COMPLETED
                await self._set_terminal_status(_final_status)

        except Exception as e:
            logger.exception(f"AgentTaskRunner.resume 运行出错: {str(e)}")
            await self._put_and_add_event(
                task, ErrorEvent(error=f"AgentTaskRunner.resume 出错: {str(e)}")
            )

    async def destroy(self) -> None:
        """销毁任务运行器并释放资源（best-effort：每步独立 try/except，确保后续清理不被跳过）

        Note: sandbox lifecycle is managed by SandboxLifecycleService (I3).
        This method only releases the handle and cleans up tools/flow.
        """
        logger.info("开始清除销毁AgentTaskRunner资源")
        try:
            # 1. Release sandbox handle (lifecycle service owns actual destruction)
            if self._sandbox and hasattr(self._sandbox, "release"):
                logger.info("释放 AgentTaskRunner 的沙箱 handle")
                self._sandbox.release()
        except Exception as exc:
            logger.warning("sandbox.release() 失败（继续清理）: %s", exc)

        try:
            # 2.清除mcp和a2a工具（幂等操作，如果invoke()中已清理则不会重复执行）
            await self._cleanup_tools()
        except Exception as exc:
            logger.warning("_cleanup_tools() 失败（继续清理）: %s", exc)

        try:
            # 3.清理 checkpointer 引用
            if hasattr(self._flow, "close"):
                await self._flow.close()
        except Exception as exc:
            logger.warning("flow.close() 失败: %s", exc)

    async def on_done(self, task: Task) -> None:
        """任务结束时执行的回调函数"""
        logger.info(f"AgentTaskRunner任务执行结束")
