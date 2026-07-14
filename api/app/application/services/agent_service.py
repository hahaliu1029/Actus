import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Callable, Dict, FrozenSet, List, Optional, Type

from app.application.errors.exceptions import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ServiceUnavailableError,
)
from langchain_core.language_models import BaseChatModel

from app.domain.external.file_storage import FileStorage
from app.domain.external.memory_flusher import MemoryFlusher
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.external.mailbox_publisher import MailboxPublisher
from app.domain.external.supervisor_registry import SupervisorRegistryPort
from app.domain.external.task import Task
from app.domain.errors.supervisor import SupervisorContractError
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    LifecycleRuntimeConfig,
    MCPConfig,
    SkillRiskPolicy,
    ToolRuntimeConfig,
)
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.event import (
    BaseEvent,
    ControlAction,
    ControlEvent,
    ControlScope,
    ControlSource,
    DoneEvent,
    ErrorEvent,
    Event,
    MessageEvent,
    SessionModeChangedEvent,
    WaitEvent,
)
from app.domain.models.file import File
from app.domain.models.lifecycle import RETRY_BUDGET_INITIAL, RetryLifecycleContext
from app.domain.models.message import SkillConfirmationAction
from app.domain.models.session import SandboxBindingState, Session, SessionStatus

# from app.domain.repositories.file_repository import FileRepository
# from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.mailbox_skip_helper import _should_skip_mailbox_lifecycle
from app.domain.services.session.mode_event import ModeChangedEventSink
from app.domain.services.permission.confirmation_queue import ConfirmationQueue as ConfirmationManager
from app.domain.services.permission.errors import PermissionConfigurationError
from app.infrastructure.external.message_queue import STREAM_TTL_SECONDS
from app.interfaces.schemas.session import SupervisorSnapshot
from core.config import get_settings
from langgraph.types import Command
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)


async def _commit_uow_if_real(uow) -> None:
    """C3 PR-3c (codex r11 [HIGH CONTRACT] fix).

    Explicitly commit the underlying DB session when ``uow`` is a real
    :class:`~app.infrastructure.repositories.db_uow.DBUnitOfWork` (which
    exposes ``db_session.commit()``) so a terminal-write site can detect
    commit failure synchronously and skip downstream side-effects
    (mailbox supervisor stop, Redis revoke, notification emit).

    Why explicit: ``DBUnitOfWork.__aexit__`` swallows
    ``asyncio.CancelledError`` on commit for SSE-disconnect ergonomics,
    so "left the with-block normally" does NOT imply "commit durably
    succeeded". This helper lets callers raise commit failure as an
    exception that breaks out before the supervisor stop runs.

    Test stubs (e.g. unit tests' ``_Uow`` lacking ``db_session``) get a
    no-op so this fix doesn't break their mocked paths. Mirrors the
    runner's ``await uow.db_session.commit()`` at
    ``agent_task_runner.py:3092``.
    """
    db_session = getattr(uow, "db_session", None)
    if db_session is None:
        return
    commit = getattr(db_session, "commit", None)
    if commit is None:
        return
    await commit()


# C3 PR-3c (codex r9 [HIGH CONTRACT] fix) — GC anchor + observability for
# shielded mailbox-stop tasks. Without a hard reference, asyncio can GC the
# task spawned by ``_maybe_stop_supervisor_for_session`` before its done
# callback fires, swallowing exceptions. Same pattern as
# ``_PENDING_TERMINAL_TASKS`` in ``agent_task_runner.py:119``.
_PENDING_MAILBOX_STOP_TASKS: set[asyncio.Task] = set()


def _on_mailbox_stop_task_done(task: asyncio.Task) -> None:
    _PENDING_MAILBOX_STOP_TASKS.discard(task)
    if task.cancelled():
        # Shouldn't happen: ``asyncio.shield`` protects the inner task
        # from the outer cancel and we don't cancel it directly.
        logger.warning(
            "mailbox stop task %s was cancelled unexpectedly", task.get_name()
        )
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "mailbox stop task %s raised: %s",
            task.get_name(),
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
OUTPUT_STREAM_POLL_BLOCK_MS = 1000
TAKEOVER_CANCEL_TIMEOUT_SECONDS = 15
TAKEOVER_LEASE_TTL_SECONDS = 15 * 60
_REDIS_STREAM_ID_RE = re.compile(r"^\d+-\d+$")

# C2 coordinator-cancel — hard upper bound (seconds) on the user-stop child-cancel
# fanout. The fanout never raises (INV-C2), but a slow DB enumeration / Redis
# publish must never BLOCK the parent's own terminalization. On timeout the
# dispatched children fall back to the <=300s watchdog (NG1). Module-level so
# tests can monkeypatch it small.
_PARENT_CANCEL_FANOUT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class _ConfigSnapshot:
    """Immutable config-dependent dependency bundle. Atomically swapped on refresh."""
    llm: BaseChatModel
    agent_config: "AgentConfig"
    mcp_config: "MCPConfig"
    a2a_config: "A2AConfig"
    skill_risk_policy: "SkillRiskPolicy"
    overflow_config: "ContextOverflowConfig"
    summary_llm: BaseChatModel | None
    vision_fallback_model: BaseChatModel | None
    skill_creator_service: "SkillCreatorService"
    supports_vision: bool
    supports_pdf_input: bool
    file_understanding_config: "FileUnderstandingConfig | None"
    tool_runtime: ToolRuntimeConfig = field(default_factory=ToolRuntimeConfig)
    # M1 PR-4+8 memory gate — snapshot holds the pre-resolved BaseChatModel
    # (from the settings.memory_gate_llm string key) so config refresh can
    # atomically swap it; threshold/batch_cap are immutable numbers that
    # also move via the snapshot.
    memory_gate_llm: BaseChatModel | None = None
    memory_gate_threshold: float = 0.7
    memory_gate_batch_cap: int = 20
    # C7: lifecycle flag 快照（default-OFF；config refresh 原子换新）
    lifecycle_runtime: LifecycleRuntimeConfig = field(default_factory=LifecycleRuntimeConfig)


@dataclass
class _ResumeToolConfirmationState:
    """R5b-3 (Codex round-2 HIGH fix) preflight → drive 之间的上下文。

    ``preflight_resume_tool_confirmation`` 阶段同步完成 claim / audit / 取建 task
    （所有可能抛 HTTP 异常的工作都在这里做完）；随后 ``drive_...`` 只负责
    ``task.resume`` + yield events。两段拆开使得 ConflictError/NotFoundError/
    BadRequestError 能在 ``EventSourceResponse`` 创建**之前**抛到 FastAPI
    exception handler → 映射为明确 HTTP 状态码（I4 对外合同）。
    """

    session: "Session"
    detail: Any  # ConfirmationDetail（domain 层，此处 Any 避开循环 import）
    task: Any
    decision_id: Optional[str]
    persistent_scope: bool
    action: str
    scope: str
    tool_call_id: str
    owner_user_id: str
    session_id: str
    # PE-0 Phase 8.1 (C-P0-8): claim_nonce produced by pe.preflight_resume;
    # forwarded in drive_resume_tool_confirmation → Command(resume=...)
    # so graph-layer commit_resume can validate nonce before writing the grant.
    # None when using the legacy path (feature flag off or PE unavailable).
    claim_nonce: Optional[str] = None


class AgentService:
    """Manus智能体服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        config_snapshot: _ConfigSnapshot,
        sandbox_cls: Type[Sandbox],
        task_cls: Type[Task],
        search_engine: SearchEngine,
        file_storage: FileStorage,
        redis_client: object | None = None,
        checkpointer_pool: object | None = None,
        memory_flusher: MemoryFlusher | None = None,
        memory_embedding_provider=None,
        memory_session_factory=None,
        memory_repo_factory=None,
        memory_write_service=None,  # PR-3: MemoryManagementService — for memory_save tool
        memory_session_save_cap: int = 20,  # PR-3: per-session memory_save cap
        event_recovery=None,
        sandbox_lifecycle_service=None,
        memory_gate_breaker=None,  # PR-4+8: MemoryGateBreaker shared across sessions
        memory_gate_daily_cap=None,  # PR-4+8: MemoryGateDailyCap
        memory_notification_emitter=None,  # PR-4+8: MemoryNotificationEmitter
        memory_gate_rebuild_fn: Callable[
            ["_ConfigSnapshot"], "tuple[object | None, object | None]"
        ] | None = None,
        # PR-4+8: hot-refresh hook. Called from ``_refresh_config`` when the
        # snapshot's ``memory_gate_llm`` identity changes so breaker +
        # daily_cap track the new LLM. None = static wiring (tests /
        # legacy callers that don't reshape gate config at runtime).
        supervisor_registry: SupervisorRegistryPort | None = None,
        # C3 PR-3c: per-pod MailboxSupervisor registry. Forwarded into every
        # ``AgentTaskRunner`` constructed in ``_create_task`` so the runner
        # can spawn/stop a supervisor when its root session enters
        # RUNNING / terminal. None when the mailbox plane is disabled
        # (deployment-time ``settings.mailbox_supervisor_enabled=False``).
        mailbox_publisher: "MailboxPublisher | None" = None,
        # C3 PR-4.5: child-side envelope publisher. Forwarded into every
        # ``AgentTaskRunner`` so mailbox-plane children can publish
        # SPAWN_ACK / RESULT_READY / CANCEL_ACK / PROGRESS_UPDATE
        # envelopes back to the supervisor.
        coord_deps: object | None = None,
        # PR-9b-A Task A8 — lifespan-scoped ``_CoordinatorRuntimeDeps``
        # aggregator. Forwarded into every ``AgentTaskRunner`` constructed
        # in ``_create_task`` and from there into
        # ``PlannerReActFlow.__init__(_coord_deps=...)``. Default ``None``
        # preserves the legacy/test path: the runner falls back to
        # ``_NullCoordinatorRuntimeDeps`` so ``_build_config()`` SKIPS the
        # 18 coordinator cfg keys when wiring is absent.
        coordinator_parent_cancel_fanout: object | None = None,
        # C2 coordinator-cancel — optional fanout consumed ONLY by
        # ``stop_session`` to cancel dispatched coordinator children on
        # user-stop. None -> legacy/test path (stop_session skips). Built in
        # ``_build_agent_service`` from ``coord_deps``; NOT threaded into
        # ``AgentTaskRunner`` and NOT added to ``_CoordinatorRuntimeDeps``.
        policy_snapshot_sink: object | None = None,  # C5a Seam B sink (DI-provided)
        # B9 Task 20 (R3#2 必经转发点): extension stats recorder. Stored on self,
        # forwarded to every AgentTaskRunner built by _create_task, which threads
        # it into PlannerReActFlow → react_graph configurable. None = flag off.
        extension_stats_recorder: object | None = None,
        # D1a §4.1 (R2#F12): governance AdmissionPort. Stored on self, forwarded
        # to every AgentTaskRunner built by _create_task → SkillTool /
        # SkillBundleSyncManager. None = mode off (lifespan constructs nothing).
        extension_admission_port: object | None = None,
    ) -> None:
        """构造函数，完成Agent服务初始化"""
        self._config_snapshot = config_snapshot
        self._uow_factory = uow_factory
        self._sandbox_cls = sandbox_cls
        self._task_cls = task_cls
        self._sandbox_lifecycle_service = sandbox_lifecycle_service
        self._search_engine = search_engine
        self._file_storage = file_storage
        self._redis_client = redis_client
        self._checkpointer_pool = checkpointer_pool
        # A4-1 §6: unconditional SSM (status-write authority on every write path)
        self._build_unconditional_ssm()
        self._memory_flusher = memory_flusher
        self._memory_embedding_provider = memory_embedding_provider
        self._memory_session_factory = memory_session_factory
        self._memory_repo_factory = memory_repo_factory
        self._memory_write_service = memory_write_service
        self._memory_session_save_cap = memory_session_save_cap
        self._memory_gate_breaker = memory_gate_breaker
        self._memory_gate_daily_cap = memory_gate_daily_cap
        self._memory_notification_emitter = memory_notification_emitter
        self._memory_gate_rebuild_fn = memory_gate_rebuild_fn
        self._supervisor_registry = supervisor_registry
        self._mailbox_publisher = mailbox_publisher
        # PR-9b-A Task A8 — lifespan-scoped coordinator runtime deps.
        # Forwarded to every AgentTaskRunner constructed by _create_task,
        # which threads it into PlannerReActFlow.__init__(_coord_deps=...).
        self._coord_deps = coord_deps
        # C2 coordinator-cancel — consumed only by stop_session (INV-C4 null-safe).
        self._coordinator_parent_cancel_fanout = coordinator_parent_cancel_fanout
        self._policy_snapshot_sink = policy_snapshot_sink
        self._extension_stats_recorder = extension_stats_recorder  # B9 Task 20
        self._extension_admission_port = extension_admission_port  # D1a §4.1

        # codex r5 [HIGH CONTRACT] — partial-bind protection.
        # ``AgentTaskRunner._set_terminal_status._terminal_op`` calls
        # ``_maybe_stop_mailbox_supervisor`` (shielded), so runner-driven
        # terminal paths reliably stop the supervisor. But ``AgentService``
        # has SEVEN other direct ``update_to_terminal`` call sites (admin
        # cancel, takeover-pending timeout, takeover-lease timeout, force
        # terminate via approval, retry budget exhaustion, etc.) that do
        # NOT flow through the runner. Codex r5 caught those leaking a
        # spawned root supervisor task after these non-runner terminals.
        # ``_maybe_stop_supervisor_for_session`` is the shared helper —
        # idempotent, swallowing failures, safe to call on subagent IDs
        # (the registry's ``stop`` is a no-op on unknown roots), so every
        # terminal-write site below just appends one ``await`` and stays
        # symmetric with the runner path.
        self._event_recovery = event_recovery
        self._background_tasks: set[asyncio.Task] = set()
        self._pending_timeout_tasks: dict[str, asyncio.Task] = {}
        self._takeover_timeout_tasks: dict[str, asyncio.Task] = {}
        self._confirmation_sweep_task: asyncio.Task | None = None
        self._settings = get_settings()
        # Eagerly init ConfirmationManager so sweep and chat() guard work from startup
        self._confirmation_manager: ConfirmationManager | None = None
        if redis_client and hasattr(redis_client, "client"):
            try:
                self._confirmation_manager = ConfirmationManager(
                    redis=redis_client.client,
                    timeout_seconds=self._settings.tool_confirmation_timeout_seconds,
                )
            except Exception:
                logger.warning("Failed to init ConfirmationManager at startup")
        logger.info("AgentService初始化成功")

    def _build_unconditional_ssm(self) -> None:
        # A4-1 §6: SessionStateMachine must be present on EVERY production write
        # path, independent of the tool-confirmation master switch. It is cheap
        # and stateless (uow_factory + optional redis), so we build it once here.
        # Lazy import mirrors the existing _create_task pattern and avoids any
        # import cycle with application.composition.
        from app.application.composition.graph_assembly import (
            build_session_state_machine,
        )

        self._ssm = build_session_state_machine(
            uow_factory=self._uow_factory,
            redis=(
                self._redis_client.client
                if self._redis_client and hasattr(self._redis_client, "client")
                else None
            ),
        )

    def _refresh_config(self, snapshot: _ConfigSnapshot) -> None:
        """Atomically replace config snapshot. CPython GIL guarantees single-attr assignment is atomic.

        **PR-4+8**: memory gate breaker + daily_cap are derived from
        ``snapshot.memory_gate_llm``; if the deployer hot-refreshes
        ``summary_model`` (or any config that changes the gate LLM),
        the refresh must also rebuild breaker + daily_cap or else we'd
        leave them at the init-time None while the gate is now enabled
        —— auto-promote would bypass both protections.

        Identity check is conservative: ``is`` instead of equality
        because LLMConfig clones by value would compare equal even when
        we want to reset state. Only ``is`` reliably signals "same
        instance, keep breaker counter".
        """
        old_gate_llm = self._config_snapshot.memory_gate_llm
        new_gate_llm = snapshot.memory_gate_llm
        self._config_snapshot = snapshot

        if old_gate_llm is not new_gate_llm and self._memory_gate_rebuild_fn is not None:
            new_breaker, new_daily_cap = self._memory_gate_rebuild_fn(snapshot)
            # Atomic-enough for CPython: each attr swap is single-bytecode.
            # A task in flight might briefly see the old breaker + new
            # daily_cap (or vice versa); both combinations are valid —
            # worst case is one extra flush going through the stale
            # breaker while the new cap is already live. Not worth a lock.
            self._memory_gate_breaker = new_breaker
            self._memory_gate_daily_cap = new_daily_cap

    async def _get_task(self, session: Session) -> Optional[Task]:
        """根据传递的任务会话获取任务实例"""
        # 1.从会话中取出任务id
        task_id = session.task_id
        if not task_id:
            return None

        # 2.调用人物类的get方法获取对应的任务实例
        return self._task_cls.get(task_id)

    @staticmethod
    def _resolve_effective_tool_filter(
        session: Session,
        tool_filter: Optional[FrozenSet[str]],
    ) -> Optional[FrozenSet[str]]:
        """T12 / Phase 1 PR-X — F8 pod-restart resilience.

        Precedence:

        1. Explicit caller ``tool_filter`` (including ``frozenset()`` "deny-all")
           **always wins** — fresh-chat paths already know the right policy.
        2. If caller passed ``None`` *and* ``session.tool_filter_preset`` is
           non-``None``, resolve the preset to its allowlist via
           ``resolve_preset(...)``. ``resolve_preset`` raises ``ValueError``
           on unknown presets (including ``""``) so a code/data drift
           surfaces loudly instead of silently dropping the restriction.
        3. Otherwise return ``None`` (no restriction).

        Centralised here so every ``_create_task`` reconstruction path
        (chat, resume, FINISHING, orphan sweep, preflight rebuild) gets the
        same restore semantics without each caller needing to remember it.

        Contract note (codex R1 P2): the predicate is ``is None``, NOT
        ``not preset_name``. The empty string ``""`` MUST flow through to
        ``resolve_preset`` (which fails closed with ``ValueError``) rather
        than being coerced to "no restriction" by truthiness — that
        coercion would silently widen permissions on a malformed row.
        """
        if tool_filter is not None:
            return tool_filter
        preset_name = getattr(session, "tool_filter_preset", None)
        if preset_name is None:
            return None
        from app.domain.services.tool_filter_presets import resolve_preset

        restored = resolve_preset(preset_name)
        if restored is not None:
            logger.info(
                "[T12] 会话[%s] tool_filter 从 preset 还原: preset=%s allow=%d",
                session.id,
                preset_name,
                len(restored),
            )
        return restored

    async def _create_task(
        self,
        session: Session,
        *,
        tool_filter: Optional[FrozenSet[str]] = None,
        force_initial_compaction: bool = False,
        retry_lifecycle_context: Optional[RetryLifecycleContext] = None,
    ) -> Task:
        """根据传递的会话创建一个新任务

        Args:
            session: 会话实例
            tool_filter: Phase 1 minimal subagent — 可选的工具名白名单
                （``None`` = 不过滤，沿用历史行为；空集合 = 显式拒绝所有工具）。
                值会原封不动透传给 ``AgentTaskRunner``，仅在 fresh chat
                创建任务的路径上由 ``chat()`` 注入。

                T12 / Phase 1 PR-X：resume / FINISHING / orphan / preflight
                等重建路径调用方仍然传 ``None``——本函数会在入口处检查
                ``session.tool_filter_preset``，若有值则通过
                ``resolve_preset(...)`` 还原同一份白名单，从而填上 F8
                pod-restart 安全缺口。显式传入的 ``tool_filter`` 优先级
                高于 preset：caller 已经知道自己要的是什么。
        """
        # T12: pod-restart resilience — restore tool_filter from preset when
        # caller didn't pass one (resume / FINISHING / orphan paths). See
        # ``_resolve_effective_tool_filter`` for the precedence contract.
        tool_filter = self._resolve_effective_tool_filter(session, tool_filter)

        snap = self._config_snapshot  # local capture — immune to concurrent refresh

        # 1. 通过 lifecycle service 获取或创建沙箱 handle（I5: 禁止隐式复活）
        from app.domain.errors.sandbox_lifecycle import (
            SessionFinalizedError,
            SessionSuspendedError,
            SessionUnboundError,
        )

        if self._sandbox_lifecycle_service:
            try:
                sandbox = await self._sandbox_lifecycle_service.acquire(session.id)
            except SessionUnboundError:
                sandbox = await self._sandbox_lifecycle_service.bind_new(
                    session.id, user_id=session.user_id
                )
            except SessionSuspendedError:
                # I2 + §6: SUSPENDED → ACTIVE 必须显式 resume()，_create_task 不隐式 unsuspend。
                # Caller（chat 的 reopen 分支、resume_tool_confirmation 等）负责判断是否 resume。
                raise
            except SessionFinalizedError:
                raise RuntimeError(f"会话[{session.id}]的沙箱已终止，无法创建任务")
        else:
            # Fallback for tests without lifecycle service
            _sandbox = None
            binding_id = session.sandbox_binding.id
            if binding_id:
                _sandbox = await self._sandbox_cls.get(binding_id)
            if not _sandbox:
                _sandbox = await self._sandbox_cls.create(user_id=session.user_id)
                session.sandbox_binding = session.sandbox_binding.model_copy(
                    update={"id": _sandbox.id}
                )
                async with self._uow_factory() as uow:
                    await uow.session.save(session)
            sandbox = _sandbox

        # 4.从沙箱中获取浏览器实例
        browser = await sandbox.get_browser()
        if not browser:
            logger.error(f"获取沙箱[{sandbox.id}]中的浏览器实例失败")
            raise RuntimeError(f"获取沙箱[{sandbox.id}]中的浏览器实例失败")

        # 5.构造 file_view 处理器（延迟到此处，因为需要运行时 sandbox + file_storage）
        file_processor_lookup = None
        if snap.file_understanding_config:
            from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

            async def _upload_bytes(file_bytes: bytes, filename: str) -> str | None:
                from io import BytesIO
                from fastapi import UploadFile
                try:
                    upload = UploadFile(file=BytesIO(file_bytes), filename=filename, size=len(file_bytes))
                    file_obj = await self._file_storage.upload_file(upload)
                    return await self._file_storage.get_presigned_url(file_obj)
                except Exception:
                    logger.warning("file_uploader failed for %s", filename, exc_info=True)
                    return None

            file_processor_lookup = FileProcessorRegistry(
                sandbox=sandbox,
                file_uploader=_upload_bytes,
                vision_model=snap.vision_fallback_model,
                audio_config=snap.file_understanding_config.audio,
                video_config=snap.file_understanding_config.video,
                pdf_page_parallel_enabled=snap.tool_runtime.pdf_page_parallel_enabled,
            )

        # R5b-4 (cleanup): ApprovalCache 已被 ApprovalStateReader (R5b-2) +
        # ApprovalStateWriter (R5b-3) 完全替换；旧 Redis-based cache 构造移除。
        #
        # R5b-2: Build ApprovalStateReader (DB-only; I7: Redis 不可用时仍可 allow/deny)
        # PE-4d1: legacy tool_approval_rules fallback retired — reader reads
        # grants only (Priority 1-4). No legacy query injection.
        approval_state_reader = None
        try:
            from app.application.services.approval_state_adapters import (
                UowApprovalGrantQuery,
            )
            from app.domain.services.approval_state_reader import ApprovalStateReader

            grant_query = UowApprovalGrantQuery(uow_factory=self._uow_factory)
            approval_state_reader = ApprovalStateReader(query=grant_query)
        except Exception:
            logger.warning(
                "Failed to build ApprovalStateReader; tool pre-check will "
                "fall through to user confirmation (fail-open by design)"
            )

        # R5b-3: Build ApprovalStateWriter (CS4 单一 Writer；SmartApprove / 未来 callsite 统一出口)
        approval_state_writer = None
        try:
            from app.application.services.approval_state_writer import (
                ApprovalStateWriter,
            )
            approval_state_writer = ApprovalStateWriter(
                uow_factory=self._uow_factory,
            )
        except Exception:
            logger.warning(
                "Failed to build ApprovalStateWriter; SmartApprove persistence "
                "will be skipped (fail-open) until next session init"
            )

        # Reuse the service-level ConfirmationManager (initialized in __init__)
        confirmation_manager_inst = self._confirmation_manager

        # PE-0 Phase 7 (C-R2-P0-1): build PermissionEngine + SessionStateMachine
        # per-task. Both are None when the pre-conditions aren't met (no writer,
        # no reader, no confirmation queue) so the legacy path in react_graph
        # and PlannerReActFlow stays active until Phase 9 fully cuts over.
        #
        # P1#1 (round-11 fix): move the feature-flag gate INTO _create_task so that
        # task._flow._permission_engine is the single authoritative signal for
        # "PE is active for this task".  Previously the gate lived only in
        # PlannerReActFlow._build_config, so a PE object could exist on the flow
        # (built here) while _build_config skipped PE injection into the graph
        # configurable — a split-brain that the HTTP preflight guard then failed to
        # detect because it re-read the *current* config snapshot rather than the
        # snapshot that was in effect at task-creation time.
        #
        # With this gate here, when the tc.enabled master switch is off, we never
        # build PE/SSM and both remain None.  The HTTP preflight split-brain check
        # reduces to the simple and reliable:
        # ``task._flow._permission_engine is not None``.
        _tc_at_create = getattr(snap.agent_config, "tool_confirmation", None)
        # PE-4c: per-source flags retired. Build PE/SSM iff the confirmation
        # master switch is on, OR no tool_confirmation config is present at
        # all (default-on). Invariant (a): no-config session still builds PE;
        # invariant (b): master enabled=False → never build PE.
        _flag_pe_active_at_create = (
            _tc_at_create is None or getattr(_tc_at_create, "enabled", True)
        )

        ssm = None
        permission_engine = None
        if _flag_pe_active_at_create:
            # PE-4c fail-closed: PE is the SOLE tool-confirmation path (the legacy
            # native risk gate is deleted in PE-4c). When the master switch is on
            # (or no tool_confirmation config is present → default-on), confirmation
            # is REQUIRED, so a missing PE dependency must fail CLOSED. Silently
            # leaving permission_engine=None would let risk-bearing native tools
            # (shell_execute=high, file_write/file_str_replace=medium,
            # browser_console_exec=high) run UNCONFIRMED, because tool_node skips
            # _pe_dispatch when PE is absent and the PE-4b native fail-closed guard
            # only fires when PE is present. Operators who genuinely want to run
            # without confirmation must set tool_confirmation.enabled=False
            # explicitly (→ _flag_pe_active_at_create False → no PE, no raise).
            if (
                approval_state_writer is None
                or approval_state_reader is None
                or confirmation_manager_inst is None
            ):
                _missing_pe_deps = ", ".join(
                    name
                    for name, val in (
                        ("approval_state_writer", approval_state_writer),
                        ("approval_state_reader", approval_state_reader),
                        ("confirmation_queue", confirmation_manager_inst),
                    )
                    if val is None
                )
                raise PermissionConfigurationError(
                    "tool_confirmation is enabled but the PermissionEngine cannot "
                    f"be built (missing: {_missing_pe_deps}); refusing to run tools "
                    "without the confirmation boundary (PE-4c fail-closed). Set "
                    "tool_confirmation.enabled=False to run without confirmation."
                )
            try:
                from app.application.composition.graph_assembly import (
                    build_decision_recorder,
                    build_permission_engine,
                    build_session_state_machine,
                )

                ssm = build_session_state_machine(
                    uow_factory=self._uow_factory,
                    redis=(
                        self._redis_client.client
                        if self._redis_client and hasattr(self._redis_client, "client")
                        else None
                    ),
                )
                # P1#5: read SmartApprove gate flags from tool_confirmation config.
                # Default to enabled=True so existing sessions with no explicit
                # config keep the same behavior (backward-compat).
                _tc_cfg = getattr(snap.agent_config, "tool_confirmation", None)
                _sa_enabled = bool(
                    getattr(_tc_cfg, "smart_approve_enabled", True)
                    if _tc_cfg is not None else True
                )
                _sa_medium_only = bool(
                    getattr(_tc_cfg, "smart_approve_medium_only", False)
                    if _tc_cfg is not None else False
                )
                # P2#5: forward smart_approve_timeout_seconds from ToolRuntimeConfig
                # so operator-configured timeouts (1-300s range) take effect for PE
                # as well as the legacy path.  Default to 30.0 (build_permission_engine
                # default) when the field is absent for backward-compat.
                _sa_timeout = float(
                    getattr(snap.tool_runtime, "smart_approve_timeout_seconds", 30.0)
                )
                # P2#3: forward confirmation_timeout_seconds from ToolConfirmationConfig
                # so the ConfirmationQueue deadline matches the ToolConfirmationEvent
                # timeout shown to the frontend.
                _confirm_timeout = int(
                    getattr(_tc_cfg, "timeout_seconds", 300)
                    if _tc_cfg is not None else 300
                )
                # PE-1 §2.6: construct the initial source registry. NativeSource is
                # always registered here; SkillSource is registered post-task_runner
                # construction (skill_tool lives on AgentTaskRunner, not on
                # AgentService at this point). validate_pe_source_registry is
                # called after the late-registration block below — the broad
                # except contract re-raises PermissionConfigurationError as a
                # deploy-time bug (PE-1 §3.2 Round 4 P1#2).
                from app.domain.services.permission.skill_refresher import (
                    SkillRiskRefresher,
                )
                from app.domain.services.permission.sources import (
                    A2aSource,
                    McpSource,
                    NativeSource,
                    SkillSource,
                )

                _redis_for_skill_source = (
                    self._redis_client.client
                    if self._redis_client and hasattr(self._redis_client, "client")
                    else None
                )
                _skill_tool_for_source = getattr(self, "_skill_tool", None)
                _skill_source = (
                    SkillSource(
                        refresher=SkillRiskRefresher(_skill_tool_for_source),
                        redis=_redis_for_skill_source,
                    )
                    if _redis_for_skill_source is not None
                    and _skill_tool_for_source is not None
                    else None
                )
                _pe_sources: dict[str, Any] = {
                    "native": NativeSource(),
                    "mcp": McpSource(),
                    "a2a": A2aSource(),
                }
                if _skill_source is not None:
                    _pe_sources["skill"] = _skill_source

                permission_engine = build_permission_engine(
                    uow_factory=self._uow_factory,
                    writer=approval_state_writer,
                    reader=approval_state_reader,
                    queue=confirmation_manager_inst,
                    session_machine=ssm,
                    summary_llm=snap.summary_llm,
                    smart_approve_enabled=_sa_enabled,
                    smart_approve_medium_only=_sa_medium_only,
                    smart_approve_timeout_seconds=_sa_timeout,
                    confirmation_timeout_seconds=_confirm_timeout,
                    decision_recorder=build_decision_recorder(),  # P3#1: wire OTel recorder
                    sources=_pe_sources,
                )
            except PermissionConfigurationError:
                # PE-1 §3.2 Round 4 P1#2: PE registry mismatch is a deploy-time
                # bug — never silently fall back to legacy and pretend nothing
                # is wrong. Surface to the caller; HTTP layer will 500 with
                # correlation_id from the catch-all handler.
                raise
            except Exception as _pe_build_exc:
                # PE-4c fail-closed: the legacy confirmation path this USED to fall
                # back to is DELETED. Running risk-bearing native tools unconfirmed
                # while the master switch is on is unacceptable, so surface the
                # build failure as PermissionConfigurationError (HTTP catch-all →
                # 5xx + correlation_id) instead of silently leaving
                # permission_engine=None.
                logger.error(
                    "Failed to build PermissionEngine/SessionStateMachine while "
                    "tool_confirmation is enabled; failing closed (PE-4c).",
                    exc_info=True,
                )
                raise PermissionConfigurationError(
                    "Failed to build the PermissionEngine while tool_confirmation "
                    "is enabled; refusing to run tools without the confirmation "
                    "boundary (PE-4c fail-closed)."
                ) from _pe_build_exc

        # B5 #29: compute bootstrap language from the already-loaded session.
        # ``_get_accessible_session`` upstream already hydrated events via
        # ``get_by_id().to_domain()``, so ``session.get_latest_plan()`` is a
        # zero-cost in-memory Python lookup. Falls back to "zh" for brand-new
        # sessions (no plan history) or plan.language == "" (empty string).
        latest_plan = session.get_latest_plan()
        initial_language = (
            latest_plan.language
            if latest_plan is not None and latest_plan.language
            else "zh"
        )

        # 6.创建AgentTaskRunner
        # B4 M0 / B3-core PR-3b: build the session-scoped cost callback here
        # so every LLM call emits a CostRecord and mirrors LLM inflight state
        # through ExecutionSupervisor.
        from app.application.services.cost_callback_factory import (
            build_supervisor_aware_callback_handler,
        )

        cost_callback_handler = build_supervisor_aware_callback_handler(
            supervisor=self._supervisor,
            session_id=session.id,
            user_id=session.user_id,
            uow_factory=self._uow_factory,
        )

        task_runner = AgentTaskRunner(
            uow_factory=self._uow_factory,
            llm=snap.llm,
            agent_config=snap.agent_config,
            mcp_config=snap.mcp_config,
            a2a_config=snap.a2a_config,
            skill_risk_policy=snap.skill_risk_policy,
            overflow_config=snap.overflow_config,
            session_id=session.id,
            user_id=session.user_id,
            file_storage=self._file_storage,
            browser=browser,
            search_engine=self._search_engine,
            sandbox=sandbox,
            skill_creator_service=snap.skill_creator_service,
            summary_llm=snap.summary_llm,
            checkpointer_pool=self._checkpointer_pool,
            supports_vision=snap.supports_vision,
            supports_pdf_input=snap.supports_pdf_input,
            file_processor_lookup=file_processor_lookup,
            memory_flusher=self._memory_flusher,
            memory_embedding_provider=self._memory_embedding_provider,
            memory_session_factory=self._memory_session_factory,
            memory_repo_factory=self._memory_repo_factory,
            memory_write_service=self._memory_write_service,
            memory_session_redis=(
                self._redis_client.client
                if self._redis_client and hasattr(self._redis_client, "client")
                else None
            ),
            memory_session_save_cap=self._memory_session_save_cap,
            memory_gate_llm=snap.memory_gate_llm,
            memory_gate_breaker=self._memory_gate_breaker,
            memory_gate_daily_cap=self._memory_gate_daily_cap,
            memory_gate_threshold=snap.memory_gate_threshold,
            memory_gate_batch_cap=snap.memory_gate_batch_cap,
            memory_notification_emitter=self._memory_notification_emitter,
            approval_state_reader=approval_state_reader,
            confirmation_manager=confirmation_manager_inst,
            permission_engine=permission_engine,
            session_state_machine=self._ssm,
            initial_language=initial_language,
            tool_runtime=snap.tool_runtime,
            # C7: lifecycle flags 前半段布线终点——runner 持有，graph/flow 不接（spec §8）
            lifecycle_runtime=snap.lifecycle_runtime,
            retry_lifecycle_context=retry_lifecycle_context,
            # C7 §5 R10#A3 — epoch 从持久列派生（每次建 task 都算，retry 后的
            # follow-up task 亦携带当前 epoch）；禁止内存计数。
            lifecycle_task_epoch=max(
                0, RETRY_BUDGET_INITIAL - session.retry_budget_remaining
            ),
            on_session_complete=self._compose_completion_callbacks(
                original=self._on_task_runner_complete,
                session_id=session.id,
                user_id=str(session.user_id),
            ),
            # A7 Task 2.7: forward the LLM's ProviderProfile so
            # ``_build_image_blocks`` can honor ``accepts_image_url``.
            # ActusChatModel / ActusResponsesModel / ActusFallbackChatModel all
            # expose ``.profile`` (see _build_llm in service_dependencies.py).
            profile=getattr(snap.llm, "profile", None),
            cost_callback_handler=cost_callback_handler,
            event_seq_client=(
                self._redis_client.client
                if self._redis_client and hasattr(self._redis_client, "client")
                else None
            ),
            execution_supervisor=self._supervisor,
            idle_watchdog=getattr(self, "_idle_watchdog", None),
            was_background=session.was_background,
            tool_filter=tool_filter,
            force_initial_compaction=force_initial_compaction,  # B11 §8
            # C3 PR-3c: ``getattr`` mirrors the ``_idle_watchdog`` line above —
            # several tests build ``AgentService`` via ``__new__`` to bypass the
            # heavy ctor wiring, then drive ``_create_task``; defensiveness keeps
            # those paths green while production wiring (main.py:305) always
            # passes a real value (or ``None`` when the mailbox flag is off).
            supervisor_registry=getattr(self, "_supervisor_registry", None),
            mailbox_supervisor_enabled=getattr(
                getattr(self, "_settings", None), "mailbox_supervisor_enabled", False
            ),
            # C3 PR-4.5: child-side publisher. Same ``getattr`` defense for
            # ``__new__``-bypass tests; production wiring threads the
            # ``RedisMailboxPublisher`` through from ``_build_agent_service``.
            mailbox_publisher=getattr(self, "_mailbox_publisher", None),
            # PR-9b-A Task A8: lifespan-scoped _CoordinatorRuntimeDeps
            # aggregator. AgentTaskRunner forwards this as
            # ``_coord_deps`` into PlannerReActFlow.__init__. When the
            # aggregator is None (legacy/test path), AgentTaskRunner falls
            # back to ``_NullCoordinatorRuntimeDeps`` so _build_config()
            # SKIPS the 18 coordinator cfg keys.
            coord_deps=getattr(self, "_coord_deps", None),
            policy_snapshot_sink=getattr(self, "_policy_snapshot_sink", None),
            # B9 Task 20: forward the lifespan-scoped stats recorder. getattr
            # defense mirrors the lines above for __new__-bypass tests.
            extension_stats_recorder=getattr(self, "_extension_stats_recorder", None),
            # D1a §4.1: forward the governance AdmissionPort (None when mode off).
            # getattr defense mirrors the line above for __new__-bypass tests.
            extension_admission_port=getattr(self, "_extension_admission_port", None),
        )

        # PE-1 §2.6: skill_tool lives on the live task_runner (constructed above);
        # register SkillSource into the PE built earlier and validate the final
        # registry. If validate fails, surface as PermissionConfigurationError
        # (broad-except contract re-raises).
        if permission_engine is not None:
            from app.application.composition.graph_assembly import (
                validate_pe_source_registry,
            )
            from app.domain.services.permission.skill_refresher import (
                SkillRiskRefresher,
            )
            from app.domain.services.permission.sources import SkillSource

            _redis_for_skill_source_late = (
                self._redis_client.client
                if self._redis_client and hasattr(self._redis_client, "client")
                else None
            )
            _task_skill_tool = getattr(task_runner, "_skill_tool", None)
            if (
                _redis_for_skill_source_late is not None
                and _task_skill_tool is not None
                and "skill" not in (getattr(permission_engine, "_sources", None) or {})
            ):
                permission_engine.register_source(
                    "skill",
                    SkillSource(
                        refresher=SkillRiskRefresher(_task_skill_tool),
                        redis=_redis_for_skill_source_late,
                    ),
                )
            # ALWAYS validate when PE exists — deploys missing the skill plumbing
            # fail fast at this point (post-task_runner construction).
            validate_pe_source_registry(
                getattr(permission_engine, "_sources", None) or {}
            )

        # 6.创建任务Task并更新会话中的信息
        task = self._task_cls.create(task_runner=task_runner)
        session.task_id = task.id
        async with self._uow_factory() as uow:
            if self._sandbox_lifecycle_service:
                persisted_session = await uow.session.get_by_id(session.id)
                if persisted_session is None:
                    raise RuntimeError(f"会话[{session.id}]不存在，无法创建任务")
                persisted_session.task_id = task.id
                await uow.session.save(persisted_session)
            else:
                await uow.session.save(session)

        supervisor = getattr(self, "_supervisor", None)
        if supervisor is not None:
            supervisor._register_runner(
                session_id=session.id,
                runner=task,
            )

        # PR2 §10: register a live event sink so lifecycle events reach the SSE
        # stream in real-time (not just PG recovery poll).
        # Returns the Redis stream message ID so the caller can unify IDs.
        if self._sandbox_lifecycle_service:
            async def _push_lifecycle_event(evt: Any) -> Optional[str]:
                return await task.output_stream.put(evt.model_dump_json())

            self._sandbox_lifecycle_service.registry.register_live_event_sink(
                session.id, _push_lifecycle_event
            )

        return task

    async def _on_task_runner_complete(self, session_id: str) -> None:
        """Callback from AgentTaskRunner._set_terminal_status.

        Called after task_runner writes COMPLETED/TIMED_OUT. Transitions sandbox
        binding to SUSPENDED (I2) — covers ALL completion paths including
        _resume_tool_confirmation, confirmation_sweep, and _resume_task_with_handoff.
        """
        if self._sandbox_lifecycle_service:
            # C3 PR-6 — legacy retired (spec §11.7). Subagent suspend owned
            # exclusively by MailboxSupervisor via the destroy hook (M1
            # single-writer). Root sessions retain AgentService suspend.
            # ``_should_skip_mailbox_lifecycle`` is referenced literally so the
            # AST CI gate (§13.3) still marks this site as guarded; the
            # ``worker_type=="root"`` check replaces the legacy ``else``
            # suspend fallback PR-4.5 left in place for rollback safety.
            #
            # codex r12 [R12-3] / r13 [R13-3] / r25 [R25-2, HIGH
            # ARCH] — ``get_session()`` lookup must be try-guarded.
            # On lookup exception we ATTEMPT ONE RETRY (transient
            # DB blip is the most common cause). If retry also
            # fails, we conservatively suspend (best-effort): a
            # lost session row is almost always a root whose
            # sandbox would orphan without cleanup, and the M1
            # race for an unknown mailbox child is the lesser
            # risk. The live event sink release still runs
            # unconditionally.
            session = None
            try:
                session = await self.get_session(session_id)
            except Exception:
                logger.debug(
                    "on_task_runner_complete: get_session(%s) raised on first "
                    "try — retrying once before falling back",
                    session_id,
                    exc_info=True,
                )
                try:
                    session = await self.get_session(session_id)
                except Exception:
                    logger.warning(
                        "on_task_runner_complete: get_session(%s) raised on retry "
                        "— falling back to defensive suspend (best-effort cleanup)",
                        session_id,
                        exc_info=True,
                    )
            if session is not None and _should_skip_mailbox_lifecycle(session):
                logger.debug(
                    "on_task_runner_complete: skip suspend session=%s "
                    "— mailbox supervisor owns lifecycle (M1)",
                    session_id,
                )
            elif session is None or session.worker_type == "root":
                try:
                    await self._sandbox_lifecycle_service.suspend(session_id)
                except Exception:
                    logger.debug(
                        "on_task_runner_complete: suspend for session %s skipped",
                        session_id,
                    )
            else:
                logger.debug(
                    "on_task_runner_complete: skip suspend session=%s — "
                    "worker_type=%s is not root and not mailbox-plane "
                    "(unreachable post-PR-6 legacy retirement)",
                    session_id,
                    session.worker_type,
                )
            # Release the live event sink AFTER suspend so the SUSPENDED event
            # reaches the SSE stream. The subsequent DoneEvent from task_runner
            # goes through task.output_stream directly — it doesn't need the sink.
            self._sandbox_lifecycle_service.registry.release_live_event_sink(
                session_id
            )

    def _compose_completion_callbacks(
        self,
        *,
        original: Callable[[str], Any],
        session_id: str,
        user_id: str,
    ) -> Callable[[str], Any]:
        async def composed(passed_session_id: str) -> None:
            if passed_session_id != session_id:
                raise AssertionError(
                    "composed callback session_id mismatch: "
                    f"expected={session_id} got={passed_session_id}"
                )

            cancel_reason: str | None = None
            redis_client = getattr(self, "_redis_client", None)
            redis = (
                redis_client.client
                if redis_client is not None and hasattr(redis_client, "client")
                else redis_client
            )
            if redis is not None:
                try:
                    hot = await redis.hgetall(f"supervisor:hot:{session_id}")
                    raw = hot.get("pending_terminal_reason") or hot.get(
                        b"pending_terminal_reason"
                    )
                    if raw is not None:
                        cancel_reason = (
                            raw.decode() if isinstance(raw, bytes) else raw
                        )
                except Exception:
                    logger.warning(
                        "compose: failed to read pending_terminal_reason for %s",
                        session_id,
                        exc_info=True,
                    )

            try:
                await original(passed_session_id)
            finally:
                supervisor = getattr(self, "_supervisor", None)
                if supervisor is not None:
                    try:
                        await supervisor._on_runner_session_complete(
                            session_id=session_id,
                            user_id=user_id,
                            cancel_reason=cancel_reason,
                        )
                    except Exception:
                        logger.exception(
                            "supervisor cleanup failed for session=%s",
                            session_id,
                        )

        return composed

    async def _safe_update_unread_count(self, session_id: str) -> None:
        """在独立的后台任务中安全地更新未读消息计数

        该方法通过asyncio.create_task()调用，运行在一个全新的asyncio Task中，
        因此不受sse_starlette的anyio cancel scope影响，数据库操作可以正常完成。
        使用uow_factory创建全新的UoW实例，避免与被取消的上下文共享数据库连接。
        """
        try:
            uow = self._uow_factory()
            async with uow:
                await uow.session.update_unread_message_count(session_id, 0)
        except Exception as e:
            logger.warning(f"会话[{session_id}]后台更新未读消息计数失败: {e}")

    async def _batch_has_non_pe_eligible_pending(
        self,
        task_flow: object,
        session_id: str,
        pending_tool_name: str,
        tc: object,
    ) -> bool:
        """PE-1 §3.2 Correction E — source-aware batch eligibility check.

        Returns True if the checkpointed tool_calls batch contains any tool
        that is NOT PE-eligible under the current
        ``ToolConfirmationConfig``. A tool is PE-eligible iff
        ``is_pe_eligible_tool_source(tool_source, tc)`` returns True — which
        requires the source to be in ``PE_SUPPORTED_SOURCES``,
        the master ``enabled`` switch on (PE-4c retired the per-source flags),
        AND (for ``source="skill"``) the category to be ``"skill"`` so
        creator/guide tools fall back to legacy (Round 2 P1#2).

        Mirrors the routing logic ``_pe_dispatch`` uses post-PE-1: when
        any tool in the batch is not PE-eligible, ``_pe_dispatch`` falls
        back to the legacy tool_node path for the whole batch, so
        preflight must also route to legacy or we end up with a stuck
        claim_nonce / split-brain.
        """
        from app.domain.services.permission.sources import (
            is_pe_eligible_tool_source,
        )
        from app.domain.services.tools.tool_source_resolver import (
            ToolSourceUnknownError as _TSUErr,
            resolve_tool_source as _resolve,
        )
        from langchain_core.messages import AIMessage as _AIMsg
        from langchain_core.messages import ToolMessage as _TMMsg

        try:
            _main_graph = getattr(task_flow, "_main_graph", None)
            if _main_graph is None:
                _ensure = getattr(task_flow, "_ensure_graphs", None)
                if _ensure is not None:
                    await _ensure()
                _main_graph = getattr(task_flow, "_main_graph", None)
                if _main_graph is None:
                    return False
            _graph_config = task_flow._build_config()  # type: ignore[union-attr]
            _graph_snap = await _main_graph.aget_state(_graph_config, subgraphs=True)

            _gs_messages: list = []
            _snap_tasks = getattr(_graph_snap, "tasks", None) or []
            for _task in _snap_tasks:
                _sub_state = getattr(_task, "state", None)
                if _sub_state is None:
                    continue
                _sub_values = getattr(_sub_state, "values", None) or {}
                _sub_messages = _sub_values.get("messages", []) if isinstance(_sub_values, dict) else []
                if _sub_messages:
                    _gs_messages = _sub_messages
                    break

            if not _gs_messages and _graph_snap:
                _gs_messages = ((_graph_snap.values or {}).get("messages", [])
                                if _graph_snap else [])

            _batch_tool_calls: list[dict] = []
            _batch_ai_idx: int = -1
            for _idx, _msg in enumerate(reversed(_gs_messages)):
                if isinstance(_msg, _AIMsg) and _msg.tool_calls:
                    _batch_tool_calls = list(_msg.tool_calls)
                    _batch_ai_idx = len(_gs_messages) - 1 - _idx
                    break
            if not _batch_tool_calls:
                return False
            _already_done: set[str] = set()
            for _tm in _gs_messages[_batch_ai_idx + 1:]:
                if isinstance(_tm, _TMMsg) and _tm.tool_call_id:
                    _already_done.add(_tm.tool_call_id)
            for _tc in _batch_tool_calls:
                _tc_id = _tc.get("id", "")
                if _tc_id in _already_done:
                    continue
                _tc_name = _tc.get("name", "")
                if _tc_name == "message_ask_user":
                    continue
                try:
                    _tc_source = _resolve(_tc_name)
                except _TSUErr:
                    # Round 2 P1#1: unknown source aligns with
                    # ``react_graph._pe_dispatch`` — treat as non-PE-eligible
                    # (the whole batch falls back to legacy). Passing None
                    # to ``is_pe_eligible_tool_source`` returns False, which
                    # surfaces as "non-PE-eligible found" → True here.
                    _tc_source = None
                if not is_pe_eligible_tool_source(_tc_source, tc):
                    logger.info(
                        "PE preflight mixed-batch guard (PE-1 §3.2 + Round 2 "
                        "P1#1/P1#2): session=%s batch contains non-PE-eligible "
                        "tool '%s' (source=%s, category=%s) alongside pending "
                        "tool '%s'. _pe_dispatch will fall back to legacy for "
                        "the whole batch — routing preflight to legacy path "
                        "to avoid split-brain.",
                        session_id,
                        _tc_name,
                        getattr(_tc_source, "source", None),
                        getattr(_tc_source, "category", None),
                        pending_tool_name,
                    )
                    return True
            return False
        except Exception as _err:
            logger.debug(
                "PE preflight _batch_has_non_pe_eligible_pending: graph state read failed for "
                "session=%s (best-effort, returning False): %s",
                session_id, _err,
            )
            return False

    def _build_pe_ssm_for_resume(self, snap: "_ConfigSnapshot") -> "tuple[object | None, object | None]":
        """Per-call helper that mirrors ``_create_task`` PE/SSM construction.

        Used by HTTP-resume entry points (``preflight_resume_tool_confirmation``).
        Returns ``(pe, ssm)`` or ``(None, None)`` when the feature flag is off
        or required dependencies are unavailable — ensuring HTTP preflight and
        graph resume consistently use the legacy path (no split-brain).
        """
        from app.application.composition.graph_assembly import (
            build_decision_recorder,
            build_permission_engine,
            build_session_state_machine,
        )

        # Feature flag gate (mirrors PlannerReActFlow._build_config gate):
        # PE-4c: per-source flags retired. Master switch is the only gate.
        # Invariant (b): master enabled=False → return (None, None) so the
        # graph also takes the legacy path (no split-brain). Invariant (a):
        # tc is None → fall through and build PE (default-on).
        tc = getattr(snap.agent_config, "tool_confirmation", None)
        if tc is not None and not getattr(tc, "enabled", True):
            return None, None  # confirmation master switch off

        confirmation_queue = self._confirmation_manager
        if confirmation_queue is None:
            return None, None

        # Build reader (fail-open)
        # PE-4d1: legacy tool_approval_rules fallback retired — grants-only reader.
        approval_state_reader = None
        try:
            from app.application.services.approval_state_adapters import (
                UowApprovalGrantQuery,
            )
            from app.domain.services.approval_state_reader import ApprovalStateReader

            grant_query = UowApprovalGrantQuery(uow_factory=self._uow_factory)
            approval_state_reader = ApprovalStateReader(query=grant_query)
        except Exception:
            pass

        # Build writer (fail-open)
        approval_state_writer = None
        try:
            from app.application.services.approval_state_writer import ApprovalStateWriter

            approval_state_writer = ApprovalStateWriter(uow_factory=self._uow_factory)
        except Exception:
            pass

        if approval_state_writer is None or approval_state_reader is None:
            return None, None

        try:
            ssm = build_session_state_machine(
                uow_factory=self._uow_factory,
                redis=(
                    self._redis_client.client
                    if self._redis_client and hasattr(self._redis_client, "client")
                    else None
                ),
            )
            # P1#5 (resume path): mirror the _create_task gate so SmartApprove
            # is consistently absent when disabled in config.
            _tc_cfg_r = getattr(snap.agent_config, "tool_confirmation", None)
            _sa_enabled_r = bool(
                getattr(_tc_cfg_r, "smart_approve_enabled", True)
                if _tc_cfg_r is not None else True
            )
            _sa_medium_only_r = bool(
                getattr(_tc_cfg_r, "smart_approve_medium_only", False)
                if _tc_cfg_r is not None else False
            )
            # P2#5 (resume path): mirror _create_task — forward operator-configured
            # smart_approve_timeout_seconds from ToolRuntimeConfig so the PE used
            # for HTTP preflight respects the same timeout as the graph-time PE.
            _sa_timeout_r = float(
                getattr(snap.tool_runtime, "smart_approve_timeout_seconds", 30.0)
            )
            # P2#3 (resume path): mirror _create_task — ConfirmationQueue deadline
            # must match the ToolConfirmationEvent timeout sent to the frontend.
            _confirm_timeout_r = int(
                getattr(_tc_cfg_r, "timeout_seconds", 300)
                if _tc_cfg_r is not None else 300
            )
            # PE-1 §2.6: construct the source registry for the resume path.
            # SkillSource registration here is best-effort — preflight_resume_
            # tool_confirmation re-attaches the SkillSource from the live
            # task_runner once the pending tool's source is known to be 'skill'
            # and calls validate_pe_source_registry there. build_permission_engine
            # no longer validates internally (PE-1 §2.6 fix: caller-driven).
            from app.domain.services.permission.skill_refresher import (
                SkillRiskRefresher,
            )
            from app.domain.services.permission.sources import (
                A2aSource,
                McpSource,
                NativeSource,
                SkillSource,
            )

            _redis_for_skill_source_r = (
                self._redis_client.client
                if self._redis_client and hasattr(self._redis_client, "client")
                else None
            )
            _skill_tool_for_source_r = getattr(self, "_skill_tool", None)
            _skill_source_r = (
                SkillSource(
                    refresher=SkillRiskRefresher(_skill_tool_for_source_r),
                    redis=_redis_for_skill_source_r,
                )
                if _redis_for_skill_source_r is not None
                and _skill_tool_for_source_r is not None
                else None
            )
            _pe_sources_r: dict[str, Any] = {
                "native": NativeSource(),
                "mcp": McpSource(),
                "a2a": A2aSource(),
            }
            if _skill_source_r is not None:
                _pe_sources_r["skill"] = _skill_source_r

            pe = build_permission_engine(
                uow_factory=self._uow_factory,
                writer=approval_state_writer,
                reader=approval_state_reader,
                queue=confirmation_queue,
                session_machine=ssm,
                summary_llm=snap.summary_llm,
                smart_approve_enabled=_sa_enabled_r,
                smart_approve_medium_only=_sa_medium_only_r,
                smart_approve_timeout_seconds=_sa_timeout_r,
                confirmation_timeout_seconds=_confirm_timeout_r,
                decision_recorder=build_decision_recorder(),  # P3#1: wire OTel recorder
                sources=_pe_sources_r,
            )
            return pe, ssm
        except PermissionConfigurationError:
            # PE-1 §3.2 Round 4 P1#2: PE registry mismatch is a deploy-time
            # bug — never silently fall back to legacy and pretend nothing
            # is wrong. Surface to the caller; HTTP layer will 500 with
            # correlation_id from the catch-all handler.
            raise
        except Exception:
            logger.warning(
                "_build_pe_ssm_for_resume: failed to build PE/SSM, "
                "falling back to legacy preflight path.",
                exc_info=True,
            )
            return None, None

    async def preflight_resume_tool_confirmation(
        self,
        session_id: str,
        user_id: str,
        is_admin: bool,
        tool_confirmation: object,
    ) -> "_ResumeToolConfirmationState":
        """PE-0 Phase 8.1: Delegating preflight entry point.

        When PE is available (feature flag on + deps resolved), routes through
        ``pe.preflight_resume`` so the claim_nonce is produced by PE and carried
        into ``drive_resume_tool_confirmation`` → graph ``commit_resume``.

        When PE is unavailable (flag off or build failure), falls back to the
        original legacy implementation (``_preflight_resume_tool_confirmation_legacy``),
        which preserves R5b-3 behavior exactly.
        """
        from app.domain.services.permission.context import (
            EvaluationContext,
            ResumeSignal,
        )
        from app.domain.services.permission.errors import SessionModeViolation
        from app.domain.services.permission.tool_call_spec import ToolCallSpec

        snap = self._config_snapshot
        pe, ssm = self._build_pe_ssm_for_resume(snap)

        if pe is None or ssm is None:
            return await self._preflight_resume_tool_confirmation_legacy(
                session_id=session_id,
                user_id=user_id,
                is_admin=is_admin,
                tool_confirmation=tool_confirmation,
            )

        # PE path
        action: str = getattr(tool_confirmation, "action", "deny")
        scope: str = getattr(tool_confirmation, "scope", "once")
        tool_call_id: str = getattr(tool_confirmation, "tool_call_id", "")

        # Validate session access
        session = await self._get_accessible_session(session_id, user_id, is_admin)

        # Read pending confirmation detail
        if not self._confirmation_manager:
            raise BadRequestError("ConfirmationManager 不可用，无法处理工具确认")

        pending_detail = await self._confirmation_manager.read(session_id, tool_call_id)
        if pending_detail is None:
            # P2#7: Mirror the legacy path's late-duplicate handling.
            # detail=None means the confirmation was already processed and cleaned up
            # by a previous /resume (winner path), or it truly never existed / expired.
            # Raising SessionModeViolation (410) would break the reconnect/retry contract.
            # Instead: check for a persistent grant (same as legacy); if found → 409
            # so the frontend can replay via /events?since=...; if truly missing → 404.
            #
            # P2#4: PE commit_resume writes the grant with
            # confirmation_id=f"{session_id}:{tool_call_id}" (via _cid(call)),
            # so we must query with the same composite key, not the bare
            # tool_call_id.  Using only tool_call_id causes find_by_confirmation_id
            # to miss the row → 404 instead of 409 on late-duplicate retry.
            _pe_confirmation_id = f"{session_id}:{tool_call_id}"
            try:
                async with self._uow_factory() as _lookup_uow:
                    existing_grant = await _lookup_uow.approval_grants.find_by_confirmation_id(
                        _pe_confirmation_id
                    )
                    # P2#2 (Codex round-10): split-brain compatibility.
                    # PE writes grants with composite confirmation_id
                    # (f"{session_id}:{tool_call_id}") but the legacy path writes
                    # with the bare tool_call_id.  On a session that started with a
                    # legacy preflight and was then retried after PE became available,
                    # pending_detail is already cleaned up, so we land here.  The PE
                    # composite lookup misses the legacy-written row → 404 instead of
                    # 409.  Fall back to the bare id before declaring "not found".
                    if existing_grant is None:
                        existing_grant = await _lookup_uow.approval_grants.find_by_confirmation_id(
                            tool_call_id
                        )
            except Exception as _lookup_err:
                logger.warning(
                    "PE path late-duplicate grant lookup failed tool_call=%s: %s",
                    tool_call_id, _lookup_err,
                )
                raise ServiceUnavailableError(
                    f"工具确认[{tool_call_id}]状态查询暂时失败，请稍后重试"
                ) from _lookup_err
            if existing_grant is not None:
                raise ConflictError(
                    f"工具确认[{tool_call_id}]已被处理完成（grant 已持久）；"
                    "请通过 /events?since=<last_event_id> 重连 SSE 复播结果"
                )
            raise NotFoundError(
                f"工具确认请求[{tool_call_id}]不存在或已过期"
            )

        # PE-1 §3.2 + Round 2 P1#1/P1#2: source+category-aware per-call gate.
        # PE-0 was native-only; PE-1 supports native + skill
        # (PE_SUPPORTED_SOURCES). For any source that is NOT in
        # PE_SUPPORTED_SOURCES — or when the master switch is off (PE-4c
        # retired the per-source flags) — OR for skill creator /
        # skill guide tools (``source="skill"`` but ``category != "skill"``,
        # which ``SkillSource.build_skill_call_metadata`` cannot resolve) —
        # OR for hallucinated / unknown tool names — we route to the legacy
        # confirmation path so the graph layer's ``_pe_dispatch`` and the
        # HTTP preflight stay in lock-step.
        from app.domain.services.permission.sources import (
            is_pe_eligible_tool_source,
        )
        from app.domain.services.tools.tool_source_resolver import (
            ToolSourceUnknownError,
            resolve_tool_source,
        )
        try:
            _pending_tool_source_obj = resolve_tool_source(pending_detail.tool_name)
        except ToolSourceUnknownError:
            # Round 2 P1#1: align with ``react_graph._pe_dispatch`` which
            # treats unknown source as non-PE-eligible (whole batch legacy).
            # The previous fallback to ``"native"`` wrote PE claim /
            # pe_resume_outcomes that the graph then ignored, causing
            # split-brain. ``None`` makes ``is_pe_eligible_tool_source``
            # return False so we route to legacy.
            _pending_tool_source_obj = None

        _tc_for_per_call_gate = getattr(snap.agent_config, "tool_confirmation", None)
        if pending_detail is not None and not is_pe_eligible_tool_source(
            _pending_tool_source_obj, _tc_for_per_call_gate,
        ):
            logger.info(
                "PE preflight: tool_name=%s has source=%s category=%s — "
                "per-source gate disabled, source unsupported, OR skill "
                "creator/guide. Routing to legacy confirmation path.",
                pending_detail.tool_name,
                getattr(_pending_tool_source_obj, "source", None),
                getattr(_pending_tool_source_obj, "category", None),
            )
            return await self._preflight_resume_tool_confirmation_legacy(
                session_id=session_id,
                user_id=user_id,
                is_admin=is_admin,
                tool_confirmation=tool_confirmation,
            )

        # PE-1 §3.2 step 5.d + Round 2 P1#2: when the pending tool is a
        # dynamic Skill (source="skill" AND category="skill" — only the
        # SkillTool wrappers, NOT skill creator/guide), register a live
        # SkillSource into the existing PE. _build_pe_ssm_for_resume cannot
        # do this on its own because the SkillTool only exists on the live
        # task_runner; here we have a chance to walk
        # ``_existing_task._task_runner._flow._skill_tool`` and wire the
        # adapter before pe.preflight_resume is invoked. Guarded so we only
        # register once and only when a usable SkillTool + redis is
        # available.
        #
        # Note: at this point ``is_pe_eligible_tool_source`` already returned
        # True, so for ``source="skill"`` we know ``category == "skill"`` —
        # the explicit check below is defensive only.
        if (
            _pending_tool_source_obj is not None
            and _pending_tool_source_obj.source == "skill"
            and _pending_tool_source_obj.category == "skill"
        ):
            try:
                _existing_task_for_skill = await self._get_task(session)
            except Exception:
                _existing_task_for_skill = None
            if _existing_task_for_skill is not None:
                _runner_for_skill = getattr(
                    _existing_task_for_skill, "_task_runner", None,
                )
                _flow_for_skill = getattr(_runner_for_skill, "_flow", None)
                _skill_tool_live = getattr(_flow_for_skill, "_skill_tool", None)
                _pe_sources_attr = getattr(pe, "_sources", None) or {}
                _redis_for_late_skill = (
                    self._redis_client.client
                    if self._redis_client and hasattr(self._redis_client, "client")
                    else None
                )
                if (
                    _skill_tool_live is not None
                    and _redis_for_late_skill is not None
                    and "skill" not in _pe_sources_attr
                ):
                    from app.domain.services.permission.skill_refresher import (
                        SkillRiskRefresher,
                    )
                    from app.domain.services.permission.sources import SkillSource

                    try:
                        pe.register_source(
                            "skill",
                            SkillSource(
                                refresher=SkillRiskRefresher(_skill_tool_live),
                                redis=_redis_for_late_skill,
                            ),
                        )
                    except ValueError:
                        # register_source raises on duplicate — concurrent
                        # preflight may already have registered the source;
                        # safe to ignore.
                        logger.debug(
                            "PE preflight: skill source already registered for "
                            "session=%s; skipping duplicate.",
                            session_id,
                        )
                    # PE-1 §2.6: validate now that the full registry is wired.
                    # build_permission_engine no longer validates internally; the
                    # caller (here) is responsible for the post late-register check.
                    from app.application.composition.graph_assembly import (
                        validate_pe_source_registry,
                    )
                    validate_pe_source_registry(
                        getattr(pe, "_sources", None) or {}
                    )

        # P1#1 (split-brain guard): Check whether the existing task for this session
        # was created with PE enabled.  A task built before PE was enabled (feature
        # flag off, build failure, or first session before Phase 7 rollout) has
        # _flow._permission_engine = None, meaning graph commit_resume uses the
        # legacy path which does NOT consume pe_resume_outcomes / claim_nonce.
        #
        # Round-11 simplification: the flag gate is now enforced inside _create_task
        # itself (P1#1 fix), so task._flow._permission_engine is None iff:
        #   (a) PE build failed, OR
        #   (b) tc.enabled=False at creation time (PE-4c: per-source flags retired)
        # Both cases mean the graph will walk the legacy commit path — PE preflight
        # must NOT proceed.  Checking _task_pe is not None is now sufficient and
        # immune to config hot-reload because the flag check already ran when the task
        # was first built.
        #
        # Guard: peek at the existing task (without creating a new one).  If the
        # task exists but is a legacy task (no PE), route to legacy path.
        # If the task doesn't exist yet (will be created by drive_resume), we
        # tentatively proceed with PE preflight and re-check after _create_task (P1#2).
        _existing_task = await self._get_task(session)
        if _existing_task is not None:
            _task_runner = getattr(_existing_task, "_task_runner", None)
            _task_flow = getattr(_task_runner, "_flow", None)
            _task_pe = getattr(_task_flow, "_permission_engine", None)
            # Simplified check: _task_pe is None means the task is legacy.
            # Both flag=off and build-failure produce _task_pe=None (see _create_task).
            if _task_pe is None:
                logger.info(
                    "PE preflight split-brain guard: session=%s task is not PE-active "
                    "(pe_instance=None — flag was off or PE build failed at task creation). "
                    "Routing to legacy confirmation path.",
                    session_id,
                )
                return await self._preflight_resume_tool_confirmation_legacy(
                    session_id=session_id,
                    user_id=user_id,
                    is_admin=is_admin,
                    tool_confirmation=tool_confirmation,
                )

            # P2 (round-16) / PE-1 §3.2 Correction E: Mixed-batch routing
            # consistency guard. _pe_dispatch in react_graph routes the ENTIRE
            # batch to legacy when ANY tool_call in the batch is NOT
            # PE-eligible (per is_pe_enabled_for_source). If we ran PE preflight
            # on an eligible tool that happens to share a batch with an
            # ineligible tool, the graph would walk the legacy commit path and
            # never consume pe_resume_outcomes / claim_nonce → split-brain.
            #
            # Fix: mirror the _pe_dispatch batch-level check here. Read the
            # active tool_calls batch from the graph checkpoint and if ANY
            # ineligible tool is present → fall back to legacy preflight for
            # the whole batch.
            if await self._batch_has_non_pe_eligible_pending(
                _task_flow, session_id, pending_detail.tool_name,
                _tc_for_per_call_gate,
            ):
                return await self._preflight_resume_tool_confirmation_legacy(
                    session_id=session_id,
                    user_id=user_id,
                    is_admin=is_admin,
                    tool_confirmation=tool_confirmation,
                )

        # Read session mode
        mode, mode_rev = await ssm.get_mode_with_revision(session_id)
        if mode not in (SessionStatus.RUNNING, SessionStatus.WAITING):
            raise SessionModeViolation(
                f"resume not allowed in mode={mode.value if hasattr(mode, 'value') else mode}"
            )

        # Build PE input DTOs. At this point the gate above has already
        # ensured ``_pending_tool_source_obj is not None`` (else we would
        # have routed to legacy), so ``_pending_tool_source_obj.source``
        # is safe to read.
        call_spec = ToolCallSpec(
            tool_name=pending_detail.tool_name,
            tool_args=dict(pending_detail.tool_args),
            tool_source=_pending_tool_source_obj.source,  # PE-1 §3.2 — accept skill/mcp/a2a
            user_id=pending_detail.user_id,
            session_id=session_id,
            arg_digest=pending_detail.arg_digest,
            primary_arg=getattr(pending_detail, "primary_arg", None),
            dir_arg=getattr(pending_detail, "dir_arg", None),
            tool_call_id=tool_call_id,
        )
        ctx = EvaluationContext(
            session_mode=mode,
            session_mode_revision=mode_rev,
        )
        signal = ResumeSignal(
            confirmation_id=f"{session_id}:{tool_call_id}",
            action=action,  # type: ignore[arg-type]
            grant_scope=scope,  # type: ignore[arg-type]
            actor="user_click",
        )

        # Codex round-28 P2#1: pe.preflight_resume can raise mid-CAS.  We
        # distinguish the exception types because the safe response differs
        # depending on whether we own the Redis claim:
        #
        # - PolicyConflict: we lost the CAS race or pre-CAS validation failed
        #   (approval_already_claimed / no_pending_confirmation /
        #   arg_digest_mismatch).  We do NOT own the claim — never rollback.
        #
        # - CancelledError (Codex round-30 P2#1 update): cancellation can arrive
        #   either pre-CAS (no claim written) OR post-CAS (claim_nonce written
        #   but preflight never returned).  Because we cannot prove which side
        #   of the CAS we are on, and the claim_nonce is unknown in either case,
        #   a background rollback with claim_nonce=None would skip the nonce
        #   guard and unconditionally mark_pending — which can DELETE another
        #   concurrent caller's legitimate claim (single-flight race).  Safer:
        #   do NOT rollback on cancel.  Any genuinely orphaned 'processing'
        #   entry will be reclaimed by the orphan sweeper (bounded by the
        #   sweep threshold — see _confirmation_sweep_loop).
        #
        # - Any other exception: PE raises from pre-CAS validation paths and
        #   does not write a claim_nonce.  No rollback needed — propagate.
        from app.domain.services.permission.errors import PolicyConflict as _PolicyConflict

        preflight = None
        try:
            preflight = await pe.preflight_resume(call_spec, ctx, signal)
        except asyncio.CancelledError:
            # Codex round-30 P2#1: we cannot prove ownership of the claim when
            # cancellation arrives during preflight_resume.  Rolling back with
            # claim_nonce=None would race against a concurrent winner and delete
            # their nonce.  Let the orphan sweeper reclaim any stuck 'processing'
            # entry instead (max delay = orphan threshold).
            raise
        except _PolicyConflict:
            # We lost the CAS race (approval_already_claimed) or the validation
            # failed before CAS (no_pending_confirmation / arg_digest_mismatch).
            # In all PolicyConflict sub-cases we do NOT own the claim — do NOT
            # rollback, as that would delete another concurrent caller's state.
            raise
        # Any other exception: PE raises from pre-CAS validation paths and does not
        # write a claim_nonce.  No rollback needed — just propagate.

        # P2#7: If _get_task / _create_task fails after preflight succeeded, the
        # Redis entry is stuck in 'processing' with no graph to commit_resume.
        # Roll back to 'pending' so a subsequent /resume can retry.
        try:
            # Get/create the task (same as legacy path)
            task = await self._get_task(session)
            _task_was_created_now = False
            if task is None:
                task = await self._create_task(session)
                _task_was_created_now = True
                if not task:
                    raise ServiceUnavailableError(
                        f"会话[{session_id}]创建任务失败，请稍后重试"
                    )
        except BaseException:
            # P2#7 / P1#2: Use BaseException (not Exception) so asyncio.CancelledError
            # (which is NOT an Exception subclass in Python 3.8+) is also caught
            # and the claim is rolled back before propagating.  Without this,
            # client disconnect leaves the entry stuck in 'processing' + claim_nonce
            # forever (sweeper skips 'processing' state), causing resume conflicts.
            #
            # Codex round-20 P2#2: use _spawn_background_rollback_if_present (same
            # pattern as drive_resume_tool_confirmation / legacy path) so that if
            # the caller is cancelled (CancelledError), the mark_pending call itself
            # is not also cancelled — the independent asyncio.Task survives the parent
            # cancel scope and completes the rollback.  The "_if_present" variant
            # additionally guards against resurrecting a partial Redis hash when
            # commit_resume has already cleaned up the confirmation entry.
            self._spawn_background_rollback_if_present(
                persistent_scope=False,
                decision_id=None,
                session_id=session_id,
                tool_call_id=tool_call_id,
                claim_nonce=preflight.claim_nonce,
            )
            raise

        # P1#2 (round-11 fix): When a new task was just created, verify that PE was
        # actually wired into it.  Between pe.preflight_resume (which writes claim_nonce
        # into Redis) and _create_task there is a window where:
        #   - The operator toggled tc.enabled=false (PE-4c: per-source flags
        #     retired), OR
        #   - build_permission_engine raised an exception (build failure)
        # In either case _create_task returns a task with _flow._permission_engine=None,
        # meaning the graph will walk the legacy commit path and never call
        # pe.commit_resume.  Without this check the Redis entry stays stuck in
        # 'processing' (claim_nonce set, nobody consumes it → processing orphan).
        #
        # Fix: detect the mismatch, roll back the PE claim to 'pending', and
        # re-run preflight on the legacy path so the user gets a working confirmation.
        if _task_was_created_now:
            _new_runner = getattr(task, "_task_runner", None)
            _new_flow = getattr(_new_runner, "_flow", None)
            _new_pe = getattr(_new_flow, "_permission_engine", None)
            if _new_pe is None:
                logger.warning(
                    "PE preflight split-brain guard (P1#2): session=%s "
                    "newly created task has no PE wired "
                    "(flag toggled or build failed between preflight and _create_task). "
                    "Rolling back PE claim and falling back to legacy confirmation path.",
                    session_id,
                )
                if self._confirmation_manager is not None:
                    try:
                        await self._confirmation_manager.mark_pending(session_id, tool_call_id)
                    except BaseException as _rb_err:
                        logger.warning(
                            "PE preflight P1#2 rollback (mark_pending) failed for %s:%s: %s",
                            session_id, tool_call_id, _rb_err,
                        )
                return await self._preflight_resume_tool_confirmation_legacy(
                    session_id=session_id,
                    user_id=user_id,
                    is_admin=is_admin,
                    tool_confirmation=tool_confirmation,
                )

            # P2#1 (round-17) / PE-1 §3.2 Correction E: Mixed-batch guard for
            # the post-create path. When a worker restarts, in-memory task is
            # lost and _create_task builds a new task. The existing-task
            # mixed-batch guard above was skipped (_existing_task was None).
            # PE preflight has already claimed the Redis entry (claim_nonce
            # written). But if the checkpointed batch contains tools whose
            # source is NOT PE-eligible under the current
            # ToolConfirmationConfig, _pe_dispatch will fall back to legacy
            # for the whole batch on resume and never consume
            # pe_resume_outcomes / claim_nonce → split-brain / stuck
            # 'processing' entry.
            #
            # Fix: apply the same mixed-batch check here. On mismatch, roll
            # back the claim to 'pending' and redirect to legacy preflight
            # (same rollback pattern as P1#2 above).
            if _new_flow is not None and await self._batch_has_non_pe_eligible_pending(
                _new_flow, session_id, pending_detail.tool_name,
                _tc_for_per_call_gate,
            ):
                logger.warning(
                    "PE preflight mixed-batch guard (P2#1 post-create): session=%s "
                    "newly created task has a mixed batch in the checkpoint. "
                    "Rolling back PE claim and falling back to legacy confirmation path.",
                    session_id,
                )
                if self._confirmation_manager is not None:
                    try:
                        await self._confirmation_manager.mark_pending(session_id, tool_call_id)
                    except BaseException as _rb_err:
                        logger.warning(
                            "PE preflight P2#1 rollback (mark_pending) failed for %s:%s: %s",
                            session_id, tool_call_id, _rb_err,
                        )
                return await self._preflight_resume_tool_confirmation_legacy(
                    session_id=session_id,
                    user_id=user_id,
                    is_admin=is_admin,
                    tool_confirmation=tool_confirmation,
                )

        owner_user_id = pending_detail.user_id
        persistent_scope = scope in ("session", "always")

        # NOTE: mark_processing is intentionally NOT called here.
        # pe.preflight_resume already performed an atomic CAS via
        # mark_processing_if_pending (which writes the claim_nonce).  A second
        # mark_processing call here would be a no-op at best; at worst, if it
        # raises transiently, HTTP preflight returns an error while the entry
        # remains stuck in 'processing' with no graph to commit_resume — causing
        # a resume conflict on the next attempt. (P2#2 fix)

        logger.info(
            "会话[%s] PE preflight OK: tool_call_id=%s action=%s scope=%s claim_nonce=%s",
            session_id, tool_call_id, action, scope,
            preflight.claim_nonce[:8] + "..." if preflight.claim_nonce else "none",
        )
        return _ResumeToolConfirmationState(
            session=session,
            detail=pending_detail,
            task=task,
            decision_id=None,  # PE commit_resume owns the write; no pre-claim decision_id
            persistent_scope=persistent_scope,
            action=action,
            scope=scope,
            tool_call_id=tool_call_id,
            owner_user_id=owner_user_id,
            session_id=session_id,
            claim_nonce=preflight.claim_nonce,
        )

    async def _preflight_resume_tool_confirmation_legacy(
        self,
        session_id: str,
        user_id: str,
        is_admin: bool,
        tool_confirmation: object,
    ) -> "_ResumeToolConfirmationState":
        """R5b-3 (Codex round-2 HIGH fix) preflight 阶段——同步完成所有可能抛
        HTTP 异常的工作，返 drive 阶段所需的上下文。

        **调用点约定**：HTTP 层（``session_routes.chat``）在 ``EventSourceResponse``
        创建**之前** await 本方法。抛出的 ``ConflictError`` / ``NotFoundError`` /
        ``BadRequestError`` / ``ForbiddenError`` 会被 FastAPI exception handler 映射到
        明确 HTTP 状态码（409/404/400/403），满足 I4 对外合同。

        生效步骤：
        1. 校验会话访问权限
        2. 读 ``ConfirmationDetail``（不存在/已清理 → 404；状态非 pending → 409
           归一，不再因时序分叉成 400）
        3. scope ∈ {session, always}: ``Writer.write()`` 赢 UNIQUE(confirmation_id)
           claim；``newly_created=False`` → 409；``ValueError`` (scope/effect 冲突) → 400
        4. scope="once": ``ConfirmationManager.mark_processing_if_pending`` CAS；
           失败 → 409
        5. 取/建 task；失败回滚 claim 并抛 ``ServiceUnavailableError`` (503)
        6. once 路径立即写 audit 证据（赢 claim 后）

        PE-0 Phase 8.1: This is the legacy path retained as a fallback when
        the master ``enabled`` switch is False or PE build fails (PE-4c:
        per-source flags retired).
        Direct writer calls (write/write_audit_only/delete_grant) here are
        intentional and whitelisted under INV-1b for the legacy code path.
        """
        action: str = getattr(tool_confirmation, "action", "deny")
        scope: str = getattr(tool_confirmation, "scope", "once")
        tool_call_id: str = getattr(tool_confirmation, "tool_call_id", "")

        # 1. 校验会话访问权限
        session = await self._get_accessible_session(session_id, user_id, is_admin)

        # 2. 读 confirmation detail
        if not self._confirmation_manager:
            raise BadRequestError("ConfirmationManager 不可用，无法处理工具确认")
        confirmation_mgr = self._confirmation_manager
        detail = await confirmation_mgr.read(session_id, tool_call_id)
        if not detail:
            # Codex round-6 HIGH: I4 late-duplicate 合同——winner 已完成并
            # cleanup 了 confirmation_detail，但 grant 行（persistent scope）
            # 持久保留；再次 /resume 同 confirmation_id 应返 409（前端凭此走
            # /events?since=... 重连复播已完成的 tool event），**不是**当
            # "已过期" 报 404。只有 detail 和 grant 都不存在才真正 404。
            #
            # Round-19 P2 (hot-switch): PE writes grants with composite
            # confirmation_id (f"{session_id}:{tool_call_id}"); legacy path
            # writes with bare tool_call_id.  When PE was active for a prior
            # /resume (composite grant written) and then PE becomes unavailable
            # (config degradation / build failure) so the retry lands here,
            # the bare-only lookup misses the PE-written row → 404 instead of
            # 409.  Mirror the PE path's dual-lookup: try bare first (legacy
            # writes), then composite (PE writes), before declaring "not found".
            try:
                async with self._uow_factory() as _lookup_uow:
                    existing_grant = await _lookup_uow.approval_grants.find_by_confirmation_id(
                        tool_call_id
                    )
                    if existing_grant is None:
                        # Fall back to PE composite key to cover hot-switch case.
                        _pe_composite_id = f"{session_id}:{tool_call_id}"
                        existing_grant = await _lookup_uow.approval_grants.find_by_confirmation_id(
                            _pe_composite_id
                        )
            except Exception as _lookup_err:
                # Codex round-7 MEDIUM: 不再伪装成 404。真实 late-duplicate 但
                # grant lookup 遭 DB/UoW 瞬时故障时，客户端必须知道这是基础设施
                # 暂态错误（可重试），而不是 confirmation 永久丢失。
                logger.warning(
                    "R5b-3 late-duplicate grant lookup 失败 tool_call=%s: %s",
                    tool_call_id, _lookup_err,
                )
                raise ServiceUnavailableError(
                    f"工具确认[{tool_call_id}]状态查询暂时失败，请稍后重试"
                ) from _lookup_err
            if existing_grant is not None:
                raise ConflictError(
                    f"工具确认[{tool_call_id}]已被处理完成（grant 已持久）；"
                    "请通过 /events?since=<last_event_id> 重连 SSE 复播结果"
                )
            raise NotFoundError(
                f"工具确认请求[{tool_call_id}]不存在或已过期"
            )
        # Codex round-2 MEDIUM fix: 非 pending 归一成 ConflictError，不再根据时序
        # 分叉成 400——"already claimed / reconnect" 是同一语义。
        if detail.status != "pending":
            raise ConflictError(
                f"工具确认[{tool_call_id}]已被处理（status={detail.status}）"
            )

        owner_user_id = detail.user_id  # grant / audit owner = session owner，与请求者解耦
        persistent_scope = scope in ("session", "always")
        decision_id: Optional[str] = None

        # 3/4. Claim（persistent → Writer UNIQUE；once → ConfirmationManager CAS）
        if persistent_scope:
            from app.application.services.approval_state_writer import (
                ApprovalStateWriter,
            )
            from app.domain.models.approval_grant import ApprovalDecision
            from app.domain.services.approval_grant_policy import (
                session_grant_expires_at,
            )
            from app.domain.services.tools.tool_source_resolver import (
                ToolSourceUnknownError,
                resolve_tool_source,
            )

            try:
                _tool_source = resolve_tool_source(detail.tool_name).source
            except ToolSourceUnknownError:
                _tool_source = "native"

            _effect = "approve" if action == "approve" else "deny"
            _expires_at = (
                session_grant_expires_at() if scope == "session" else None
            )
            decision = ApprovalDecision(
                user_id=owner_user_id,
                session_id=session_id if scope == "session" else None,
                tool_name=detail.tool_name,
                tool_source=_tool_source,
                arg_digest=detail.arg_digest,
                primary_arg=detail.primary_arg,
                dir_arg=detail.dir_arg or "",
                scope=scope,
                effect=_effect,
                source_type="user_click",
                confirmation_id=tool_call_id,
                expires_at=_expires_at,
                risk_level=detail.risk_level,
            )
            writer = ApprovalStateWriter(uow_factory=self._uow_factory)

            # Codex round-4 CRITICAL: shield claim + post-cancel rollback callback
            async def _do_write() -> tuple[Optional[str], bool]:
                return await writer.write(decision)

            try:
                decision_id, newly_created = await self._claim_with_post_cancel_rollback(
                    _do_write,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    persistent_scope=True,
                )
            except ValueError as _claim_err:
                raise BadRequestError(
                    f"确认参数冲突: {_claim_err}"
                ) from _claim_err
            if not newly_created:
                raise ConflictError(
                    f"工具确认[{tool_call_id}]已被处理"
                )
        else:
            # Codex round-4 CRITICAL: once 路径的 CAS 同样走 envelope
            async def _do_cas() -> tuple[Optional[str], bool]:
                # PE-0: claim_nonce / processing_started_at are intentionally omitted here;
                # this path will be rewritten in Phase 8 to go through pe.preflight_resume
                # (which produces the nonce). The sweeper added in Phase 11 only fires for
                # entries that have processing_started_at set, so this preflight path is
                # inert under the sweeper until Phase 8 lands. See plan §Phase 8.1.
                claimed_local = await confirmation_mgr.mark_processing_if_pending(
                    session_id, tool_call_id
                )
                # 统一 shape 成 (decision_id, newly_created)；once 无 decision_id
                return None, claimed_local

            _, claimed = await self._claim_with_post_cancel_rollback(
                _do_cas,
                session_id=session_id,
                tool_call_id=tool_call_id,
                persistent_scope=False,
            )
            if not claimed:
                raise ConflictError(
                    f"工具确认[{tool_call_id}]已被处理"
                )

        # 5. 取/建 task（可能失败 → 回滚 claim 后抛 503）
        try:
            task = await self._get_task(session)
            if task is None:
                task = await self._create_task(session)
                if not task:
                    # Codex round-3 MEDIUM fix: 可预期的基础设施失败必须是 503
                    # （裸 RuntimeError 会被 exception handler 映射成 500）
                    raise ServiceUnavailableError(
                        f"会话[{session_id}]创建任务失败，请稍后重试"
                    )
            if persistent_scope:
                # persistent 赢 claim 后主动把状态推到 processing（once 已由 CAS 推过）
                await confirmation_mgr.mark_processing(session_id, tool_call_id)
        except BaseException:
            # Codex round-4 CRITICAL: 独立 task 做 rollback，不在被 cancel 的 task
            # 里直接 await（否则 delete_grant/mark_pending 会立刻被取消 → orphan
            # claim 永久卡死；sweeper 又跳过 processing，前端只会持续 409）
            self._spawn_background_rollback(
                persistent_scope=persistent_scope,
                decision_id=decision_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
            )
            raise

        # 6. once 路径立即写 audit（赢 claim 后持久化证据，不等 drive 完成）
        # CS4 single-writer contract: once audit 必须经 ApprovalStateWriter 入口，
        # 不得在此处直写 tool_approval_log.create（详见 writer.write_audit_only docstring）。
        if not persistent_scope:
            from app.application.services.approval_state_writer import (
                ApprovalStateWriter,
            )
            once_audit_writer = ApprovalStateWriter(uow_factory=self._uow_factory)
            try:
                await once_audit_writer.write_audit_only(
                    user_id=owner_user_id,
                    session_id=session_id,
                    tool_name=detail.tool_name,
                    tool_args=detail.tool_args,
                    risk_level=detail.risk_level,
                    action=action,
                    scope=scope,
                    approved_by="user",
                )
            except Exception as _log_err:
                # 普通错误吞成 warning（audit 丢失可容忍，但不能阻塞 resume）
                logger.warning(
                    "R5b-3 once scope audit 写入失败: %s", _log_err
                )
            except BaseException:
                # Codex round-5 CRITICAL: CancelledError 到 once audit 时 CAS 已把
                # confirmation 推到 processing、task 已取建；此时 preflight 主任务
                # 被取消，drive 根本不会开始，但 confirmation 卡在 processing →
                # sweeper 跳过 processing → orphan 永久 409。必须在此窗口 spawn
                # rollback 把 ConfirmationManager 状态 mark_pending 回来。
                self._spawn_background_rollback(
                    persistent_scope=False,
                    decision_id=None,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                )
                raise

        logger.info(
            "会话[%s] 工具确认 preflight OK: tool_call_id=%s action=%s scope=%s decision_id=%s owner=%s",
            session_id, tool_call_id, action, scope,
            decision_id or "none", owner_user_id,
        )
        return _ResumeToolConfirmationState(
            session=session,
            detail=detail,
            task=task,
            decision_id=decision_id,
            persistent_scope=persistent_scope,
            action=action,
            scope=scope,
            tool_call_id=tool_call_id,
            owner_user_id=owner_user_id,
            session_id=session_id,
        )

    async def _rollback_resume_claim(
        self,
        *,
        persistent_scope: bool,
        decision_id: Optional[str],
        session_id: str,
        tool_call_id: str,
    ) -> None:
        """Resume claim 回滚：persistent 路径删 grant + audit；all 路径 mark_pending。"""
        if persistent_scope and decision_id is not None:
            try:
                from app.application.services.approval_state_writer import (
                    ApprovalStateWriter,
                )
                writer = ApprovalStateWriter(uow_factory=self._uow_factory)
                await writer.delete_grant(decision_id)
            except Exception as _del_err:
                logger.warning(
                    "R5b-3 claim 回滚 delete_grant 失败 decision_id=%s: %s",
                    decision_id, _del_err,
                )
        if self._confirmation_manager is not None:
            try:
                await self._confirmation_manager.mark_pending(session_id, tool_call_id)
            except Exception:
                logger.warning(
                    "回退确认状态为 pending 失败: %s:%s",
                    session_id, tool_call_id,
                )

    def _spawn_background_rollback(
        self,
        *,
        persistent_scope: bool,
        decision_id: Optional[str],
        session_id: str,
        tool_call_id: str,
    ) -> Optional[asyncio.Task]:
        """Codex round-4 CRITICAL fix: 把 rollback 扔进独立 asyncio.Task 跑，
        规避 SSE cancel scope 传播（同一 task 里 await rollback 会被立刻取消，
        delete_grant/mark_pending 都跑不完 → orphan claim 永久卡死 processing）。

        Pattern 对齐 ``_safe_update_unread_count``（line ~454）：
        - 独立 task 不继承父 cancel scope
        - ``uow_factory()`` 每次建新 UoW，不共享已被 close 的 session

        返 ``asyncio.Task`` 方便测试 await 等待完成；生产路径 fire-and-forget。
        """
        try:
            return asyncio.create_task(
                self._rollback_resume_claim(
                    persistent_scope=persistent_scope,
                    decision_id=decision_id,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                )
            )
        except RuntimeError:
            logger.warning(
                "R5b-3 round-4: 无法创建后台 rollback task session=%s tool_call=%s",
                session_id, tool_call_id,
            )
            return None

    async def _rollback_resume_claim_if_present(
        self,
        *,
        persistent_scope: bool,
        decision_id: Optional[str],
        session_id: str,
        tool_call_id: str,
        claim_nonce: Optional[str],
    ) -> None:
        """P2#3 (Codex round-10): Conditional rollback — only mark_pending when
        the confirmation still exists in the queue AND our nonce matches.

        commit_resume cleans up the Redis hash on the success path.  If cleanup
        already ran, calling mark_pending would resurrect a partial hash that
        lacks required fields (session_id, tool_name, …) → subsequent read()
        raises KeyError / returns incomplete data.

        Guard order:
        1. confirmation no longer in queue → commit_resume succeeded → skip
        2. nonce mismatch → another claim owner → skip
        3. otherwise → delegate to the unconditional _rollback_resume_claim
        """
        if self._confirmation_manager is not None and claim_nonce is not None:
            try:
                detail = await self._confirmation_manager.read(session_id, tool_call_id)
            except Exception as _read_err:
                logger.warning(
                    "P2#3 条件回滚: read() 失败 %s:%s — 跳过 mark_pending 避免脏写: %s",
                    session_id, tool_call_id, _read_err,
                )
                return
            if detail is None:
                # commit_resume already cleaned up — nothing to rollback
                logger.debug(
                    "P2#3 条件回滚: confirmation 已清理 %s:%s — 跳过",
                    session_id, tool_call_id,
                )
                return
            if detail.claim_nonce != claim_nonce:
                # Another owner claimed this slot — leave it alone
                logger.debug(
                    "P2#3 条件回滚: nonce 不匹配 %s:%s (expected=%s, found=%s) — 跳过",
                    session_id, tool_call_id, claim_nonce, detail.claim_nonce,
                )
                return
        # Safe to rollback: confirmation still exists and we own the claim
        await self._rollback_resume_claim(
            persistent_scope=persistent_scope,
            decision_id=decision_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
        )

    def _spawn_background_rollback_if_present(
        self,
        *,
        persistent_scope: bool,
        decision_id: Optional[str],
        session_id: str,
        tool_call_id: str,
        claim_nonce: Optional[str],
    ) -> Optional[asyncio.Task]:
        """Background-task wrapper for _rollback_resume_claim_if_present.

        Used by the PE path in drive_resume_tool_confirmation to avoid
        resurrecting a partial Redis hash when commit_resume has already
        cleaned up the confirmation entry.

        Returns asyncio.Task for testability; production code ignores return value.
        """
        try:
            return asyncio.create_task(
                self._rollback_resume_claim_if_present(
                    persistent_scope=persistent_scope,
                    decision_id=decision_id,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    claim_nonce=claim_nonce,
                )
            )
        except RuntimeError:
            logger.warning(
                "P2#3: 无法创建条件回滚后台 task session=%s tool_call=%s",
                session_id, tool_call_id,
            )
            return None

    async def _claim_with_post_cancel_rollback(
        self,
        claim_factory,  # () -> Awaitable[tuple[Optional[str], bool]]
        *,
        session_id: str,
        tool_call_id: str,
        persistent_scope: bool,
    ) -> tuple[Optional[str], bool]:
        """Codex round-4 CRITICAL envelope for claim operations.

        问题：``await writer.write(decision)`` / ``await mark_processing_if_pending(...)``
        执行到 commit 一半时父 task 被 cancel → UoW __aexit__ 处理 CancelledError
        吞掉，grant/Redis 状态可能 **partial 残留**，但调用方不知道 decision_id
        → 无法 rollback → orphan。

        方案：
        1. claim 丢到独立 ``asyncio.Task``（不继承父 cancel scope）
        2. 主路径 ``await asyncio.shield(task)`` —— 外部 cancel 时 shield 抛
           CancelledError 给父，但内部 task 继续跑完 commit
        3. ``add_done_callback`` 在 claim task 完成后：若 ``newly_created=True``
           表示 claim 真写成功，spawn 独立 rollback task 清理

        返 ``(decision_id, newly_created)``；decision_id 可能为 None（CAS 路径）。
        """
        task = asyncio.create_task(claim_factory())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            def _on_done(t: asyncio.Task) -> None:
                if t.cancelled():
                    return
                if t.exception() is not None:
                    return  # claim 失败，没写任何 claim 状态
                try:
                    result = t.result()
                    did, newly = result
                except Exception as _extract_err:
                    logger.warning(
                        "R5b-3 round-4: 提取 claim 结果失败 tool_call=%s: %s",
                        tool_call_id, _extract_err,
                    )
                    return
                if not newly:
                    return  # loser 分支，claim 没成功写
                self._spawn_background_rollback(
                    persistent_scope=persistent_scope,
                    decision_id=did,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                )

            task.add_done_callback(_on_done)
            raise

    async def drive_resume_tool_confirmation(
        self, state: _ResumeToolConfirmationState,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Drive 阶段：task.resume + cleanup + yield events。所有可能抛 HTTP
        的错已在 ``preflight_...`` 阶段抛完；本函数进 SSE 后只产事件。

        I2 kickoff 失败：``task.resume`` 抛异常 → 回滚 claim 并把异常转为
        ``ErrorEvent`` yield（SSE 已建连，无法再发 HTTP 状态码）。
        """
        confirmation_mgr = self._confirmation_manager
        try:
            try:
                # PE-0 Phase 8.2: add claim_nonce so graph-layer commit_resume
                # can validate the nonce before writing the grant.
                # claim_nonce is None on the legacy path (feature flag off or
                # PE unavailable) — graph interrupt_helper handles None gracefully.
                resume_value = {
                    "action": state.action,
                    "scope": state.scope,
                    "claim_nonce": state.claim_nonce,
                }
                await state.task.resume(Command(resume=resume_value))
            except BaseException:
                # Codex round-3 CRITICAL + round-4 reinforcement：
                # 客户端 SSE 断连会以 CancelledError 抵达这里；rollback 必须在
                # 独立 asyncio.Task 里跑，否则和父 task 一起被取消，delete_grant /
                # mark_pending 根本跑不完 → orphan claim 永远 processing。
                # 独立 task 不继承父 cancel scope（_safe_update_unread_count pattern）。
                #
                # P2#3 (Codex round-10): when claim_nonce is set (PE path), use the
                # conditional rollback that checks the confirmation still exists in
                # the queue before calling mark_pending.  commit_resume cleans up the
                # hash on success; an unconditional mark_pending would resurrect a
                # partial hash that read() then fails to parse.
                if state.claim_nonce is not None:
                    self._spawn_background_rollback_if_present(
                        persistent_scope=state.persistent_scope,
                        decision_id=state.decision_id,
                        session_id=state.session_id,
                        tool_call_id=state.tool_call_id,
                        claim_nonce=state.claim_nonce,
                    )
                else:
                    self._spawn_background_rollback(
                        persistent_scope=state.persistent_scope,
                        decision_id=state.decision_id,
                        session_id=state.session_id,
                        tool_call_id=state.tool_call_id,
                    )
                raise

            # P1#1: PE path — commit_resume already called queue.cleanup() internally
            # (AllowSuccess/Denied paths in DefaultPermissionEngine.commit_resume each
            # call self._queue.cleanup before returning).  Calling cleanup again here
            # would delete the hash that was already removed, which is a no-op, but on
            # a concurrent replay or slow network the race would delete the entry before
            # commit_resume reads it → PolicyConflict("no_pending_confirmation").
            # Only run cleanup on the legacy path (claim_nonce is None on legacy).
            if confirmation_mgr is not None and state.claim_nonce is None:
                await confirmation_mgr.cleanup(state.session_id, state.tool_call_id)

            latest_event_id = None
            while True:
                event_id, event_str = await state.task.output_stream.get(
                    start_id=latest_event_id, block_ms=OUTPUT_STREAM_POLL_BLOCK_MS
                )
                if event_str is None:
                    if state.task.done:
                        break
                    continue
                latest_event_id = event_id

                event = TypeAdapter(Event).validate_json(event_str)
                event.id = event_id

                async with self._uow_factory() as uow:
                    await uow.session.update_unread_message_count(state.session_id, 0)

                yield event
                if isinstance(event, (DoneEvent, ErrorEvent, WaitEvent, ControlEvent)):
                    break

            logger.info(f"会话[{state.session_id}]工具确认恢复完成")
        except Exception as e:
            logger.error(f"会话[{state.session_id}]工具确认驱动出错: {str(e)}")
            event = ErrorEvent(error=str(e))
            try:
                async with self._uow_factory() as uow:
                    await uow.session.add_event(state.session_id, event)
            except (asyncio.CancelledError, Exception) as add_err:
                logger.warning(
                    f"会话[{state.session_id}]添加错误事件失败: {add_err}"
                )
            yield event
        finally:
            try:
                asyncio.create_task(
                    self._safe_update_unread_count(state.session_id)
                )
            except RuntimeError:
                logger.warning(
                    f"会话[{state.session_id}]无法创建后台任务更新未读消息计数"
                )

    async def _resume_tool_confirmation(
        self,
        session_id: str,
        user_id: str,
        is_admin: bool,
        tool_confirmation: object,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Backward-compat 组合入口：preflight → drive。

        生产 SSE 路径在 ``session_routes.chat`` 里已改为**分开调用** preflight +
        ``EventSourceResponse(drive(...))``，使得 HTTP 409/404/400 能在
        ``EventSourceResponse`` 之前抛出。本方法保留给既有测试 / 其他非 SSE
        caller 使用，preflight 异常会在 generator 第一次 ``__anext__`` 时抛出。
        """
        state = await self.preflight_resume_tool_confirmation(
            session_id=session_id,
            user_id=user_id,
            is_admin=is_admin,
            tool_confirmation=tool_confirmation,
        )
        async for event in self.drive_resume_tool_confirmation(state):
            yield event
        return

    async def _get_accessible_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> Session:
        """根据用户权限获取可访问会话"""
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
        if not session:
            logger.error(f"尝试访问不存在的会话[{session_id}]")
            raise NotFoundError("任务会话不存在, 请核实后重试")
        if not is_admin and session.user_id != user_id:
            logger.error(f"用户[{user_id}]无权访问会话[{session_id}]")
            raise ForbiddenError("无权访问此会话")
        return session

    # ----- B3-core PR-1: producer-side seq stamping + helpers -----

    async def get_session(self, session_id: str) -> Optional["Session"]:
        """B3-core PR-1 §6.5: thin wrapper around ``uow.session.get_by_id``.

        For callers that don't need authorization (e.g. supervisor lifespan
        reconciler, auto-degrade detached task in PR-3c/PR-4). Authorization
        paths must keep using ``_get_accessible_session``.

        Returns ``None`` if not found (NOT raises) — supervisor callers handle
        absent sessions gracefully (race with delete).
        """
        async with self._uow_factory() as uow:
            return await uow.session.get_by_id(session_id)

    async def _emit_event(self, session_id: str, event: BaseEvent) -> Optional[str]:
        """B3-core PR-1 §3.2 / §6.6: producer-side event emit.

        Stamps ``event.seq`` via INCR ``session:seq:{sid}`` (24h TTL on first set),
        then XADDs to ``task:output:{task_id}`` of the session's running task.

        Returns the Redis Stream message_id (so callers can correlate) or ``None``
        if no live task exists OR the redis client is not wired (test fallback).

        **Single-writer-per-session invariant:**
        The INCR ↔ XADD pair is NOT atomic — there are intervening awaits for
        the uow.session lookup. Two concurrent ``_emit_event`` calls on the
        same session_id may interleave such that Stream order diverges from
        ``event.seq`` order (consumer always sorts by message_id). Callers
        MUST ensure single-writer discipline per session.

        Intended call sites (PR-3c/PR-4 will land these): the per-session
        ``agent_task_runner`` event bridge plus detached tasks for
        ``_do_auto_degrade`` and explicit cancel emit, all of which run in a
        single asyncio task per session. PR-1 ships ``_emit_event`` itself;
        production hot-path callers wire in PR-3c/PR-4. If multi-writer becomes
        possible later, add a per-session ``asyncio.Lock`` registry here.
        """
        if not self._redis_client:
            logger.debug("_emit_event: no redis_client — skip emit for %s", session_id)
            return None
        client = (
            self._redis_client.client
            if hasattr(self._redis_client, "client")
            else self._redis_client
        )

        # 1. Stamp seq (atomic INCR; 24h TTL on first set — first INCR returns 1).
        seq_key = f"session:seq:{session_id}"
        try:
            new_seq = await client.incr(seq_key)
            if new_seq == 1:
                # First emit for this session — arm 24h TTL.
                try:
                    await client.expire(seq_key, STREAM_TTL_SECONDS)
                except Exception:
                    logger.warning(
                        "first-write EXPIRE for %s failed", seq_key, exc_info=True
                    )
            event.seq = int(new_seq)
        except Exception:
            logger.warning(
                "seq stamp failed for session=%s", session_id, exc_info=True
            )
            # Fall through — emit without seq (degrades to legacy resume path).

        # 2. XADD to task:output stream via the existing task's output_stream.
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
        if session is None or not session.task_id:
            logger.debug("_emit_event: no task for %s — backlog skip", session_id)
            return None

        task = (
            self._task_cls.get(session.task_id)
            if hasattr(self._task_cls, "get")
            else None
        )
        if task is None:
            # Task object isn't in the in-process registry (e.g., another worker
            # owns it, or it already finished). Backlog emit by writing directly
            # to the Stream key — auto-degrade still wants the event in the
            # durable backlog so the reconnecting client sees it.
            #
            # Note: ``RedisStreamMessageQueue.put`` runs the EXISTS-before-XADD
            # branch (§3.2) so for backlog emits on already-existing streams
            # the EXPIRE re-arm is skipped — abandoned streams age out as
            # designed.
            stream_name = f"task:output:{session.task_id}"
            try:
                from app.infrastructure.external.message_queue.redis_stream_message_queue import (
                    RedisStreamMessageQueue,
                )
                fallback_queue = RedisStreamMessageQueue(stream_name)
                return await fallback_queue.put(event.model_dump_json())
            except Exception:
                logger.warning(
                    "_emit_event fallback put failed for %s",
                    session_id,
                    exc_info=True,
                )
                return None

        try:
            return await task.output_stream.put(event.model_dump_json())
        except Exception:
            logger.warning(
                "_emit_event task.output_stream.put failed for %s",
                session_id,
                exc_info=True,
            )
            return None

    async def get_events_since(
        self,
        session_id: str,
        since_event_id: str | None,
        user_id: str,
        is_admin: bool = False,
        since_seq: int | None = None,  # B3-core PR-1 §3.3
    ) -> dict:
        """获取 session 在 since_event_id (or since_seq) 之后的增量事件。

        PG 为主（跨 invoke 权威来源），Redis 补充当前 task 的 in-flight 事件。

        B3-core PR-1 §3.3 / PR-4:
        - When both ``since_seq`` and ``since_event_id`` are provided,
          ``since_seq`` filters sequenced events, while ``since_event_id`` is
          retained as the legacy ``seq is None`` floor (logged at WARNING).
        - Result dict gains ``last_seq`` (max event.seq seen, or since_seq fallback)
          and ``supervisor_snapshot``.
        """
        if since_seq is not None and since_event_id is not None:
            logger.warning(
                "get_events_since: both since_seq=%s and since_event_id=%s given — using since_seq with event-id fallback (B3 §3.3)",
                since_seq,
                since_event_id,
            )

        session = await self._get_accessible_session(session_id, user_id, is_admin)

        # 1. PG 主路径：按 since_seq 或 since_event_id 切片
        pg_events = session.events or []
        if since_seq is not None:
            # B3-core PR-1 §3.3 — seq cursor filters sequenced events.
            # If the client also provides an event-id cursor, use it as the
            # legacy floor so seq=None events written by old/direct producers
            # after the cursor are still recovered.
            legacy_floor_available = since_event_id is not None
            if since_event_id:
                found_idx = None
                for i, evt in enumerate(pg_events):
                    if getattr(evt, "id", None) == since_event_id:
                        found_idx = i
                        break
                if found_idx is not None:
                    pg_events = pg_events[found_idx + 1:]

            filtered_pg_events = []
            for evt in pg_events:
                event_seq = getattr(evt, "seq", None)
                if event_seq is None:
                    if legacy_floor_available:
                        filtered_pg_events.append(evt)
                    continue
                if int(event_seq) > since_seq:
                    filtered_pg_events.append(evt)
            pg_events = filtered_pg_events
        elif since_event_id:
            found_idx = None
            for i, evt in enumerate(pg_events):
                if getattr(evt, "id", None) == since_event_id:
                    found_idx = i
                    break
            if found_idx is not None:
                pg_events = pg_events[found_idx + 1:]
            # else: 找不到 → 返回全量（宁可多发不漏发）

        # 2. Redis 补充路径
        redis_only_events = []
        redis_has_more = False
        if session.task_id and self._event_recovery:
            redis_start_id = None
            if since_seq is not None:
                # since_seq already removes duplicates for sequenced events.
                # Keep the client's event-id cursor as the legacy floor instead
                # of advancing to the PG tail; otherwise Redis-only gaps before
                # a later PG-persisted event are skipped.
                redis_start_id = (
                    since_event_id
                    if self._is_valid_redis_stream_id(since_event_id)
                    else None
                )
            else:
                # Legacy event-id path: use the last PG stream id as Redis
                # start to avoid replaying already persisted PG events.
                for evt in reversed(pg_events):
                    candidate_id = getattr(evt, "id", None)
                    if self._is_valid_redis_stream_id(candidate_id):
                        redis_start_id = candidate_id
                        break
                if not redis_start_id:
                    redis_start_id = (
                        since_event_id
                        if self._is_valid_redis_stream_id(since_event_id)
                        else None
                    )

            try:
                # B3-core PR-1: pass through after_seq when provided.
                recovery_result = await self._event_recovery.get_recent_events(
                    task_id=session.task_id,
                    after_event_id=redis_start_id,
                    after_seq=since_seq,
                )
                # 过滤掉 PG 中已有的 event_id
                pg_event_ids = {
                    getattr(e, "id", None)
                    for e in pg_events
                    if getattr(e, "id", None)
                }
                redis_only_events = [
                    e
                    for e in recovery_result.events
                    if getattr(e, "id", None) not in pg_event_ids
                ]
                redis_has_more = recovery_result.has_more
            except Exception:
                logger.warning(
                    "event_recovery: Redis 补充失败 session=%s task=%s",
                    session_id,
                    session.task_id,
                )

        merged = list(pg_events) + redis_only_events
        if since_seq is not None and redis_only_events:
            merged = self._sort_recovered_events(merged)

        # B3-core PR-1 §3.3: derive last_seq.
        seqs_seen = [
            int(getattr(e, "seq", None))
            for e in merged
            if getattr(e, "seq", None) is not None
        ]
        last_seq = max(seqs_seen) if seqs_seen else (since_seq or 0)
        supervisor_snapshot = await self.build_supervisor_snapshot(session)

        return {
            "events": merged,
            "session_status": session.status,
            "has_more": redis_has_more,
            # B3-core PR-1 additions:
            "last_seq": last_seq,
            "supervisor_snapshot": supervisor_snapshot,
        }

    async def build_supervisor_snapshot(self, session: Session) -> SupervisorSnapshot:
        return await self._build_supervisor_snapshot(session)

    async def _build_supervisor_snapshot(self, session: Session) -> SupervisorSnapshot:
        hot_hash = await self._read_supervisor_hot_hash(session.id)
        last_progress_at = self._parse_hot_unix_timestamp(
            self._redis_hash_get(hot_hash, "last_activity_at")
        )
        age_seconds = (
            (datetime.now(timezone.utc) - last_progress_at).total_seconds()
            if last_progress_at is not None
            else -1
        )
        is_alive = last_progress_at is not None and 0 <= age_seconds < 60

        if session.terminal_reason == "user_cancel":
            cancellation_state = "cancelled"
        elif self._redis_hash_value_is_one(
            self._redis_hash_get(hot_hash, "cancellation_pending")
        ):
            cancellation_state = "cancelling"
        else:
            cancellation_state = "none"

        return SupervisorSnapshot(
            execution_mode=session.execution_mode,
            execution_phase=session.execution_phase,
            background_reason=session.background_reason,
            expires_at=session.expires_at,
            retry_budget_remaining=session.retry_budget_remaining,
            suspended_reason=session.suspended_reason,
            terminal_reason=session.terminal_reason,
            last_progress_at=last_progress_at,
            is_alive=is_alive,
            cancellation_state=cancellation_state,
            execution_revision=session.execution_revision,
        )

    async def _read_supervisor_hot_hash(self, session_id: str) -> dict[Any, Any]:
        redis_client = getattr(self, "_redis_client", None)
        if redis_client is None:
            return {}

        try:
            redis = getattr(redis_client, "client", redis_client)
            if redis is None:
                return {}
            hot_hash = await redis.hgetall(f"supervisor:hot:{session_id}")
        except Exception:
            logger.warning(
                "get_events_since: failed to read supervisor hot hash for %s",
                session_id,
                exc_info=True,
            )
            return {}

        return hot_hash if isinstance(hot_hash, dict) else {}

    @staticmethod
    def _redis_hash_get(hot_hash: dict[Any, Any], field: str) -> Any:
        if field in hot_hash:
            return hot_hash[field]
        encoded_field = field.encode("utf-8")
        if encoded_field in hot_hash:
            return hot_hash[encoded_field]
        return None

    @staticmethod
    def _decode_redis_value(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return value

    @classmethod
    def _parse_hot_unix_timestamp(cls, value: Any) -> datetime | None:
        value = cls._decode_redis_value(value)
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return None
        if timestamp <= 0:
            return None
        try:
            return datetime.fromtimestamp(timestamp, timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    @classmethod
    def _redis_hash_value_is_one(cls, value: Any) -> bool:
        value = cls._decode_redis_value(value)
        return str(value).strip() == "1"

    @staticmethod
    def _is_valid_redis_stream_id(event_id: object) -> bool:
        if not isinstance(event_id, str):
            return False
        return bool(_REDIS_STREAM_ID_RE.match(event_id.strip()))

    @staticmethod
    def _redis_stream_id_tuple(event_id: object) -> tuple[int, int] | None:
        if not AgentService._is_valid_redis_stream_id(event_id):
            return None
        left, right = str(event_id).split("-", 1)
        return int(left), int(right)

    @staticmethod
    def _sort_recovered_events(events: list[BaseEvent]) -> list[BaseEvent]:
        def sort_key(indexed_event: tuple[int, BaseEvent]) -> tuple[int, int, int, int]:
            index, event = indexed_event
            event_seq = getattr(event, "seq", None)
            if event_seq is not None:
                return (0, int(event_seq), 0, index)

            stream_id = AgentService._redis_stream_id_tuple(getattr(event, "id", None))
            if stream_id is not None:
                return (1, stream_id[0], stream_id[1], index)

            return (2, index, 0, 0)

        return [event for _, event in sorted(enumerate(events), key=sort_key)]

    async def _check_attachments_access(
        self, attachments: Optional[List[str]], user_id: str, is_admin: bool = False
    ) -> None:
        """检查用户是否有权限使用传递的附件"""
        if not attachments:
            return

        async with self._uow_factory() as uow:
            for attachment_id in attachments:
                file = await uow.file.get_by_id(attachment_id)
                if not file:
                    raise NotFoundError(f"附件[{attachment_id}]不存在")
                if not is_admin and (not file.user_id or file.user_id != user_id):
                    raise ForbiddenError(f"无权使用附件[{attachment_id}]")

    async def chat(
        self,
        session_id: str,
        user_id: str,
        is_admin: bool = False,
        message: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        skill_confirmation_action: SkillConfirmationAction | None = None,
        tool_confirmation: object | None = None,
        latest_event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        tool_filter: Optional[FrozenSet[str]] = None,
        team_slug: Optional[str] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """根据传递的信息调用Agent服务发起对话请求

        Args:
            tool_filter: Phase 1 minimal subagent — optional allowlist of
                tool names. ``None`` (default) preserves existing behavior
                (no filtering). Empty ``frozenset()`` means "deny all".
                Propagated only into the fresh-chat ``_create_task`` path;
                resume / FINISHING / tool_confirmation paths keep ``None``
                because the caller has no fresh ``tool_filter`` context.
                F8 known gap: ``tool_filter`` is process-local; if the pod
                restarts mid-session, the rebuilt task will lose the
                allowlist. Callers must replay if persistence is required.
        """
        latest_event_id = (
            latest_event_id
            if self._is_valid_redis_stream_id(latest_event_id)
            else None
        )

        # 危险工具确认恢复路径：直接走 resume 流程，不走正常 chat 分支
        if tool_confirmation is not None:
            async for event in self._resume_tool_confirmation(
                session_id, user_id, is_admin, tool_confirmation
            ):
                yield event
            return

        try:
            # 1.检查会话是否存在
            session = await self._get_accessible_session(session_id, user_id, is_admin)
            await self._check_attachments_access(attachments, user_id, is_admin)

            # 2.获取对应会话任务
            task = await self._get_task(session)

            logger.info(
                "会话[%s] chat请求: message_present=%s task_exists=%s session_status=%s",
                session_id,
                bool(message),
                task is not None,
                session.status.value,
            )
            is_suspended_background = (
                session.status == SessionStatus.RUNNING
                and session.execution_mode == "background"
                and session.execution_phase == "suspended"
                and task is None
            )
            if is_suspended_background:
                if message or attachments:
                    raise ConflictError("后台任务已挂起，请先重试后台任务")
                logger.info(
                    "会话[%s]后台任务已挂起，空 chat 续流保持 suspended 状态",
                    session_id,
                )
                return

            # 3.判断是否传递了message
            if message:
                if session.status in {
                    SessionStatus.TAKEOVER_PENDING,
                    SessionStatus.TAKEOVER,
                }:
                    raise BadRequestError("当前会话处于接管状态，暂不支持聊天输入")

                # 4.判断会话的状态是什么,如果不是运行中则表示已完成或者空闲中
                if session.status == SessionStatus.FINISHING:
                    # FINISHING: task 仍在运行（invoke 在后处理阶段）
                    # 复用现有 task，push 到 input_stream 触发 cancel 后处理
                    task = await self._get_task(session)
                    if task is None:
                        task = await self._create_task(session)
                        if not task:
                            logger.error(f"会话[{session_id}]创建任务失败")
                            raise RuntimeError(f"会话[{session_id}]创建任务失败")
                elif session.status != SessionStatus.RUNNING or task is None:
                    if session.status == SessionStatus.WAITING:
                        # Check if waiting due to tool_confirmation — if so, reject plain text.
                        # Tool confirmations must go through the dedicated _resume_tool_confirmation() path.
                        if self._confirmation_manager:
                            try:
                                has_pending = await self._confirmation_manager.has_pending_for_session(session_id)
                                if has_pending:
                                    raise BadRequestError(
                                        "当前会话正在等待工具确认，请通过确认卡片操作，不支持文本输入"
                                    )
                            except BadRequestError:
                                raise
                            except Exception:
                                pass  # Redis failure should not block normal chat
                        logger.info(
                            "会话[%s] WAITING状态恢复: 将创建新任务并从数据库加载中断状态",
                            session_id,
                        )
                    # 5.不在运行中需要创建一个新的task并启动
                    # I2: COMPLETED/TIMED_OUT 会话的 sandbox binding 为 SUSPENDED，
                    # chat() 是用户显式发消息，视为明确的 resume 意图。
                    if (
                        self._sandbox_lifecycle_service
                        and session.status in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT)
                    ):
                        try:
                            await self._sandbox_lifecycle_service.resume(session.id)
                        except Exception:
                            pass  # acquire inside _create_task will handle the actual state
                    # B11 §8: consume a pending manual /compact request. Only at
                    # COMPLETED/TIMED_OUT (same gate as sandbox-resume) — WAITING/
                    # FINISHING resume paths by construction never consume it.
                    _force_compact = False
                    if session.status in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT):
                        try:
                            from app.application.services.manual_compaction_flag import (
                                consume_manual_compact_pending,
                            )
                            _force_compact = await consume_manual_compact_pending(
                                self._redis_client.client, session_id
                            )
                        except Exception:
                            _force_compact = False  # Redis failure must not block chat
                    # Phase 1 minimal subagent: only the fresh-chat creation path
                    # propagates tool_filter — resume / FINISHING / sweeper paths
                    # have no fresh allowlist context and stay on the default None.
                    task = await self._create_task(
                        session,
                        tool_filter=tool_filter,
                        force_initial_compaction=_force_compact,
                    )
                    if not task:
                        logger.error(f"会话[{session_id}]创建任务失败")
                        raise RuntimeError(f"会话[{session_id}]创建任务失败")

                # 6.传递了消息则更新会话中的最后一条消息
                async with self._uow_factory() as uow:
                    await uow.session.update_latest_message(
                        session_id=session_id,
                        message=message,
                        timestamp=timestamp or datetime.now(),
                    )

                # 7.创建一个人类消息事件
                message_event = MessageEvent(
                    role="user",
                    message=message,
                    attachments=(
                        [File(id=attachment) for attachment in attachments]
                        if attachments
                        else []
                    ),
                    skill_confirmation_action=skill_confirmation_action,
                    team_slug=team_slug,  # [S4 §7] carrier — survives enqueue/dequeue
                )

                # 8.将事件添加到任务的输入流中，好让Agent获取到数据
                event_id = await task.input_stream.put(message_event.model_dump_json())
                message_event.id = event_id
                async with self._uow_factory() as uow:
                    await uow.session.add_event(session_id, message_event)

                # 9.立刻把用户消息返回给前端，避免依赖后续拉取导致消息缺失
                yield message_event

                # 10.执行任务
                await task.invoke()
                logger.info(
                    f"往会话[{session_id}]输入消息队列写入消息: {message[:50]}..."
                )
            elif session.status == SessionStatus.RUNNING and task is None:
                logger.warning(
                    "会话[%s]状态自愈: status_reconciled=true from=running to=completed message_present=false task_exists=false",
                    session_id,
                )
                async with self._uow_factory() as uow:
                    transitioned = await self._ssm.terminate(
                        session_id,
                        SessionStatus.COMPLETED,
                        "resume_state_lost",
                        session_repo=uow.session,
                    )
                    # codex r11 [HIGH CONTRACT] — explicit commit so a
                    # swallowed CancelledError on UoW close cannot leave
                    # the supervisor stop firing without a durable
                    # terminal write. Mirrors runner pattern at
                    # agent_task_runner.py:3092.
                    await _commit_uow_if_real(uow)
                if transitioned is not False:
                    await self._emit_bg_terminal_notification_if_background(
                        session_id,
                        SessionStatus.COMPLETED,
                        "resume_state_lost",
                    )
                # C3 PR-3c (codex r5 + r11) — non-runner terminal: stop supervisor.
                await self._maybe_stop_supervisor_for_session(session_id)
                # Sync sandbox binding: ACTIVE → SUSPENDED (same as normal completion)
                # C3 PR-6 — legacy retired (spec §11.7). Subagent suspend owned
                # exclusively by MailboxSupervisor via destroy hook (M1).
                # _should_skip_mailbox_lifecycle reference satisfies AST CI gate
                # (§13.3); worker_type=="root" gate replaces the legacy else
                # fallback. ``session`` here is the row fetched above.
                if self._sandbox_lifecycle_service:
                    if _should_skip_mailbox_lifecycle(session):
                        logger.debug(
                            "status-reconcile: skip suspend %s — mailbox plane",
                            session_id,
                        )
                    elif session.worker_type == "root":
                        try:
                            await self._sandbox_lifecycle_service.suspend(session_id)
                        except Exception:
                            logger.debug("status-reconcile suspend for %s skipped", session_id)
                    else:
                        logger.debug(
                            "status-reconcile: skip suspend %s — worker_type=%s "
                            "is not root and not mailbox-plane (unreachable "
                            "post-PR-6 legacy retirement)",
                            session_id,
                            session.worker_type,
                        )
                session = session.model_copy(update={"status": SessionStatus.COMPLETED})

            # 11.记录日志展示会话已启动
            logger.info(f"会话[{session_id}]已启动")
            logger.info(f"会话[{session_id}]任务实例: {task}")

            # 12.从任务的输出流中读取数据
            while task:
                # 13.从输出消息队列中获取数据
                event_id, event_str = await task.output_stream.get(
                    start_id=latest_event_id, block_ms=OUTPUT_STREAM_POLL_BLOCK_MS
                )
                if event_str is None:
                    logger.debug(f"在会话[{session_id}]输出队列中未发现事件内容")
                    if task.done:
                        break
                    continue
                latest_event_id = event_id

                # 14.使用Pydantic提供的类型适配器将event_str转换为指定类实例
                event = TypeAdapter(Event).validate_json(event_str)
                event.id = event_id
                logger.debug(f"从会话[{session_id}]中获取事件: {type(event).__name__}")

                if isinstance(event, ControlEvent):
                    if event.action == ControlAction.REQUESTED:
                        self._schedule_pending_timeout(session_id)
                    elif event.action in {
                        ControlAction.STARTED,
                        ControlAction.REJECTED,
                        ControlAction.ENDED,
                        ControlAction.EXPIRED,
                    }:
                        self._cancel_pending_timeout(session_id)

                # 15.将未读消息数重置为0
                async with self._uow_factory() as uow:
                    await uow.session.update_unread_message_count(session_id, 0)

                # 16.将事件返回并判断事件类型是否为结束类型
                yield event
                if isinstance(event, (DoneEvent, ErrorEvent, WaitEvent, ControlEvent)):
                    break

            # 17.循环外面表示这次任务AI端的已结束
            # (suspend 由 task_runner._set_terminal_status → _on_task_runner_complete 统一处理)
            logger.info(f"会话[{session_id}]本轮运行结束")
        except (BadRequestError, ConflictError):
            raise
        except Exception as e:
            # 18.记录日志并返回错误事件
            logger.error(f"任务会话[{session_id}]对话出错: {str(e)}")
            event = ErrorEvent(error=str(e))
            try:
                async with self._uow_factory() as uow:
                    await uow.session.add_event(session_id, event)
            except (asyncio.CancelledError, Exception) as add_err:
                logger.warning(
                    f"会话[{session_id}]添加错误事件失败(可能是客户端断开连接): {add_err}"
                )
            yield event
        finally:
            # 19.会话完整传递给前端后，表示至少用户肯定收到了这些消息，所以不应该有未读消息数
            # 注意：当SSE客户端断开连接时，sse_starlette使用anyio cancel scope取消当前Task中
            # 所有的await操作（asyncio.shield也无法对抗anyio的cancel scope）。
            # 如果在finally块中直接执行数据库操作，该操作会被立即取消，并且SQLAlchemy在尝试
            # 终止被中断的连接时也会被取消，从而产生ERROR日志并可能污染连接池。
            # 解决方案：将数据库更新操作放到独立的asyncio Task中执行，新Task不受当前
            # cancel scope的影响，可以正常完成数据库操作。
            try:
                asyncio.create_task(self._safe_update_unread_count(session_id))
            except RuntimeError:
                # 事件循环已关闭（如应用正在关闭），无法创建后台任务
                logger.warning(f"会话[{session_id}]无法创建后台任务更新未读消息计数")

    async def stop_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> None:
        """根据传递的会话id停止指定会话

        I2: stop ≠ destroy. Transitions sandbox binding to SUSPENDED.
        Container stays alive for potential resume.
        """
        # 1.查找会话是否存在
        session = await self._get_accessible_session(session_id, user_id, is_admin)

        # C2 coordinator-cancel — fan out CANCEL_REQUEST to dispatched children
        # BEFORE cancelling the parent's own task, so running coordinator
        # children terminalize immediately (request_stop(PARENT_CANCEL) ->
        # CANCEL_ACK) instead of waiting for the <=300s watchdog. The fanout
        # service never raises (INV-C2); we additionally bound it with a short
        # timeout (asyncio.wait_for) so a slow DB enumeration / Redis publish can
        # never BLOCK the parent's own terminalization. The try/except is
        # defense-in-depth: an exception OR a timeout is swallowed and the parent
        # still terminalizes (the dispatched children fall back to the watchdog —
        # NG1).
        if (
            session.worker_type == "root"
            and self._coordinator_parent_cancel_fanout is not None
        ):
            try:
                await asyncio.wait_for(
                    self._coordinator_parent_cancel_fanout.cancel_children(
                        parent_session_id=session_id, reason="parent_cancel"
                    ),
                    timeout=_PARENT_CANCEL_FANOUT_TIMEOUT_SECONDS,
                )
            except Exception:
                logger.warning(
                    "coordinator child cancel fanout failed or timed out for parent=%s",
                    session_id,
                    exc_info=True,
                )

        # 2.根据会话获取任务信息
        task = await self._get_task(session)
        if task:
            task.cancel(reason="stop")

        # 3.更新会话任务状态
        async with self._uow_factory() as uow:
            transitioned = await self._ssm.terminate(
                session_id,
                SessionStatus.COMPLETED,
                "user_cancel",
                session_repo=uow.session,
            )
            # codex r11 — explicit commit (see resume_state_lost path).
            await _commit_uow_if_real(uow)
        if transitioned is not False:
            await self._emit_bg_terminal_notification_if_background(
                session_id,
                SessionStatus.COMPLETED,
                "user_cancel",
            )
        # C3 PR-3c (codex r5 + r11) — non-runner terminal: stop supervisor.
        await self._maybe_stop_supervisor_for_session(session_id)
        await self._cleanup_background_slot_if_needed(session, reason="user_cancel")

        # 4. Suspend sandbox binding (I2: ACTIVE → SUSPENDED, container stays alive)
        # C3 PR-6 — legacy retired (spec §11.7). Subagent suspend owned
        # exclusively by MailboxSupervisor via destroy hook (M1).
        # _should_skip_mailbox_lifecycle reference satisfies AST CI gate
        # (§13.3); worker_type=="root" gate replaces the legacy else fallback.
        # ``session`` here is the row fetched at line 3107.
        if self._sandbox_lifecycle_service:
            if _should_skip_mailbox_lifecycle(session):
                logger.debug(
                    "stop_session: skip suspend %s — mailbox plane",
                    session_id,
                )
            elif session.worker_type == "root":
                try:
                    await self._sandbox_lifecycle_service.suspend(session_id)
                except Exception:
                    logger.warning(
                        "Failed to suspend sandbox for session %s",
                        session_id,
                        exc_info=True,
                    )
            else:
                logger.debug(
                    "stop_session: skip suspend %s — worker_type=%s is not "
                    "root and not mailbox-plane (unreachable post-PR-6 "
                    "legacy retirement)",
                    session_id,
                    session.worker_type,
                )

    @staticmethod
    def _get_latest_control_event(session: Session) -> Optional[ControlEvent]:
        for event in reversed(session.events):
            if isinstance(event, ControlEvent):
                return event
        return None

    @staticmethod
    def _lease_key(session_id: str) -> str:
        return f"takeover:lease:{session_id}"

    @staticmethod
    def _lease_value(takeover_id: str, operator_user_id: str) -> str:
        return f"{takeover_id}:{operator_user_id}"

    @staticmethod
    def _to_unix_seconds(value: Optional[datetime]) -> Optional[int]:
        if value is None:
            return None
        return int(value.timestamp())

    @staticmethod
    def _build_lease_expiry(ttl_seconds: int) -> datetime:
        return datetime.now(timezone.utc) + timedelta(seconds=max(ttl_seconds, 1))

    def _get_redis_connection(self):
        if not self._redis_client:
            return None
        try:
            return self._redis_client.client
        except Exception as exc:
            raise BadRequestError("接管租约服务暂不可用，请稍后重试") from exc

    @staticmethod
    def _parse_csv(raw: str) -> set[str]:
        return {item.strip() for item in (raw or "").split(",") if item.strip()}

    def _resolve_operator_role(self, *, is_admin: bool, user_role: Optional[str]) -> str:
        if user_role:
            return user_role
        return "super_admin" if is_admin else "user"

    @staticmethod
    def _resolve_worker_count() -> int:
        for key in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
            raw_value = (os.getenv(key) or "").strip()
            if not raw_value:
                continue
            try:
                worker_count = int(raw_value)
            except ValueError:
                continue
            if worker_count > 0:
                return worker_count
        return 1

    def _assert_takeover_capability(
        self,
        *,
        user_id: str,
        is_admin: bool,
        user_role: Optional[str],
        scope: Optional[ControlScope] = None,
    ) -> None:
        if not self._settings.feature_takeover_enabled:
            raise ForbiddenError("接管功能未启用")

        if scope == ControlScope.BROWSER and not self._settings.feature_takeover_browser_enabled:
            raise ForbiddenError("浏览器接管功能未启用")

        role = self._resolve_operator_role(is_admin=is_admin, user_role=user_role)
        allowed_roles = self._parse_csv(self._settings.feature_takeover_allowed_roles)
        if allowed_roles and role not in allowed_roles:
            raise ForbiddenError("当前角色无接管权限")

        whitelist = self._parse_csv(self._settings.feature_takeover_user_whitelist)
        if whitelist and user_id not in whitelist:
            raise ForbiddenError("当前用户不在接管白名单中")

        if (
            self._settings.feature_takeover_single_worker_only
            and self._resolve_worker_count() > 1
        ):
            raise ForbiddenError("当前部署为多Worker模式，接管功能仅支持单Worker")

    async def _acquire_takeover_lease(
        self,
        session_id: str,
        *,
        takeover_id: str,
        operator_user_id: str,
        ttl_seconds: int = TAKEOVER_LEASE_TTL_SECONDS,
    ) -> bool:
        redis = self._get_redis_connection()
        if redis is None:
            return True

        lease_key = self._lease_key(session_id)
        lease_value = self._lease_value(takeover_id, operator_user_id)
        acquired = await redis.set(
            lease_key,
            lease_value,
            ex=max(ttl_seconds, 1),
            nx=True,
        )
        return bool(acquired)

    async def _renew_takeover_lease(
        self,
        session_id: str,
        *,
        takeover_id: str,
        operator_user_id: str,
        ttl_seconds: int = TAKEOVER_LEASE_TTL_SECONDS,
    ) -> bool:
        redis = self._get_redis_connection()
        if redis is None:
            return True

        lease_key = self._lease_key(session_id)
        lease_value = self._lease_value(takeover_id, operator_user_id)
        script = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
    return 1
else
    return 0
end
"""
        renewed = await redis.eval(
            script,
            1,
            lease_key,
            lease_value,
            max(ttl_seconds, 1),
        )
        return int(renewed) == 1

    async def _release_takeover_lease(
        self,
        session_id: str,
        *,
        takeover_id: str,
        operator_user_id: str,
    ) -> None:
        redis = self._get_redis_connection()
        if redis is None:
            return

        lease_key = self._lease_key(session_id)
        lease_value = self._lease_value(takeover_id, operator_user_id)
        script = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""
        await redis.eval(
            script,
            1,
            lease_key,
            lease_value,
        )

    async def _force_release_takeover_lease(self, session_id: str) -> None:
        redis = self._get_redis_connection()
        if redis is None:
            return
        await redis.delete(self._lease_key(session_id))

    async def _verify_takeover_lease_owner(
        self,
        *,
        session_id: str,
        takeover_id: str,
        operator_user_id: str,
    ) -> bool:
        redis = self._get_redis_connection()
        if redis is None:
            return True

        lease_value = await redis.get(self._lease_key(session_id))
        if lease_value is None:
            return False
        if isinstance(lease_value, bytes):
            lease_value = lease_value.decode("utf-8", errors="ignore")
        return lease_value == self._lease_value(takeover_id, operator_user_id)

    def _track_background_task(self, task: asyncio.Task) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _cancel_pending_timeout(self, session_id: str) -> None:
        timeout_task = self._pending_timeout_tasks.pop(session_id, None)
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()

    def _cancel_takeover_timeout(self, session_id: str) -> None:
        timeout_task = self._takeover_timeout_tasks.pop(session_id, None)
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()

    def _schedule_takeover_timeout(
        self,
        *,
        session_id: str,
        takeover_id: str,
        operator_user_id: str,
        ttl_seconds: int,
    ) -> None:
        self._cancel_takeover_timeout(session_id)
        timeout_task = asyncio.create_task(
            self._handle_takeover_lease_timeout(
                session_id=session_id,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
                ttl_seconds=max(ttl_seconds, 1),
            )
        )
        self._takeover_timeout_tasks[session_id] = timeout_task

        def _cleanup(done_task: asyncio.Task) -> None:
            current_task = self._takeover_timeout_tasks.get(session_id)
            if current_task is done_task:
                self._takeover_timeout_tasks.pop(session_id, None)

        timeout_task.add_done_callback(_cleanup)
        self._track_background_task(timeout_task)

    def _schedule_pending_timeout(self, session_id: str) -> None:
        ttl = max(self._settings.feature_takeover_pending_ttl_seconds, 1)
        self._cancel_pending_timeout(session_id)
        timeout_task = asyncio.create_task(
            self._handle_takeover_pending_timeout(session_id=session_id, ttl_seconds=ttl)
        )
        self._pending_timeout_tasks[session_id] = timeout_task

        def _cleanup(done_task: asyncio.Task) -> None:
            current_task = self._pending_timeout_tasks.get(session_id)
            if current_task is done_task:
                self._pending_timeout_tasks.pop(session_id, None)

        timeout_task.add_done_callback(_cleanup)
        self._track_background_task(timeout_task)

    async def _handle_takeover_pending_timeout(
        self,
        *,
        session_id: str,
        ttl_seconds: int,
    ) -> None:
        try:
            await asyncio.sleep(max(ttl_seconds, 1))
            transitioned = False
            uow = self._uow_factory()
            async with uow:
                # 读取时加行锁，避免与 reject_takeover 等并发状态迁移发生 TOCTOU 竞态。
                session = await uow.session.get_by_id_for_update(session_id)
                if not session or session.status != SessionStatus.TAKEOVER_PENDING:
                    return

                latest_control = self._get_latest_control_event(session)
                takeover_id = latest_control.takeover_id if latest_control else None
                await uow.session.add_event(
                    session_id,
                    ControlEvent(
                        action=ControlAction.EXPIRED,
                        source=ControlSource.SYSTEM,
                        reason="pending_timeout",
                        request_status="expired",
                        takeover_id=takeover_id,
                    ),
                )
                transitioned = await self._ssm.terminate(
                    session_id,
                    SessionStatus.COMPLETED,
                    "watchdog_timeout",
                    session_repo=uow.session,
                )
                # codex r11 — explicit commit (see resume_state_lost path).
                await _commit_uow_if_real(uow)
            if transitioned is not False:
                await self._emit_bg_terminal_notification_if_background(
                    session_id,
                    SessionStatus.COMPLETED,
                    "watchdog_timeout",
                )
            # C3 PR-3c (codex r5 explicit cite) — non-runner terminal
            # (takeover_pending TTL expired): stop supervisor.
            await self._maybe_stop_supervisor_for_session(session_id)
            await self._force_release_takeover_lease(session_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("会话[%s]处理接管待决超时失败: %s", session_id, exc)

    async def _handle_takeover_lease_timeout(
        self,
        *,
        session_id: str,
        takeover_id: str,
        operator_user_id: str,
        ttl_seconds: int,
    ) -> None:
        try:
            await asyncio.sleep(max(ttl_seconds, 1))
            uow = self._uow_factory()
            async with uow:
                session = await uow.session.get_by_id_for_update(session_id)
                if not session or session.status != SessionStatus.TAKEOVER:
                    return

                latest_control = self._get_latest_control_event(session)
                if not latest_control or latest_control.takeover_id != takeover_id:
                    return

                lease_owned = await self._verify_takeover_lease_owner(
                    session_id=session_id,
                    takeover_id=takeover_id,
                    operator_user_id=operator_user_id,
                )
                if lease_owned:
                    return

                await uow.session.add_event(
                    session_id,
                    ControlEvent(
                        action=ControlAction.EXPIRED,
                        source=ControlSource.SYSTEM,
                        reason="takeover_timeout",
                        request_status="expired",
                        takeover_id=takeover_id,
                    ),
                )
                await self._ssm.set_mode(
                    session_id,
                    SessionStatus.TAKEOVER_PENDING,
                    reason="takeover_lease_timeout",
                    session_repo=uow.session,
                )
                try:
                    _, rev = await uow.session.read_status_with_revision(session_id)
                except Exception:
                    rev = None
                await self._ssm.emit_session_mode_changed(
                    session_id,
                    to=SessionStatus.TAKEOVER_PENDING,
                    from_mode="takeover",  # guard at 3448 asserts source == TAKEOVER
                    reason="takeover_lease_timeout",
                    mode_revision=rev,
                    sink=lambda sid, ev: uow.session.add_event(sid, ev),
                )
            # 先释放 lease 再调度 pending timeout，防止释放失败时已有 timeout 在跑
            await self._force_release_takeover_lease(session_id)
            self._schedule_pending_timeout(session_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("会话[%s]处理接管租约超时失败: %s", session_id, exc)

    def _schedule_takeover_completion(
        self,
        *,
        session_id: str,
        task: Task,
        scope: ControlScope,
        takeover_id: str,
        operator_user_id: str,
        cancel_timeout_seconds: int,
        lease_ttl_seconds: int,
        expires_at: datetime,
    ) -> None:
        background_task = asyncio.create_task(
            self._complete_takeover_after_cancel(
                session_id=session_id,
                task=task,
                scope=scope,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
                cancel_timeout_seconds=cancel_timeout_seconds,
                lease_ttl_seconds=lease_ttl_seconds,
                expires_at=expires_at,
            )
        )
        self._track_background_task(background_task)

    async def _append_control_event(
        self,
        session_id: str,
        *,
        action: ControlAction,
        source: ControlSource,
        scope: Optional[ControlScope] = None,
        reason: Optional[str] = None,
        handoff_mode: Optional[str] = None,
        request_status: Optional[str] = None,
        takeover_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        task: Optional[Task] = None,
    ) -> ControlEvent:
        control_event = ControlEvent(
            action=action,
            source=source,
            scope=scope,
            reason=reason,
            handoff_mode=handoff_mode,
            request_status=request_status,
            takeover_id=takeover_id,
            expires_at=expires_at,
        )
        if task:
            try:
                event_id = await task.output_stream.put(control_event.model_dump_json())
                control_event.id = event_id
            except Exception as exc:
                logger.warning(
                    "会话[%s]写入ControlEvent到输出流失败，降级为仅落库: %s",
                    session_id,
                    exc,
                )
        uow = self._uow_factory()
        async with uow:
            await uow.session.add_event(session_id, control_event)
        return control_event

    def _sse_or_db_sink(self, task: Optional[Task]) -> ModeChangedEventSink:
        """A4-2: the caller-owned sink for SSM.emit_session_mode_changed on the
        HTTP-takeover paths. Mirrors _append_control_event's
        live-sink-or-DB-fallback: if a live task is present, put to its output
        stream (degrade-on-failure to DB-only), then ALWAYS persist via a fresh
        _uow_factory() txn. Demoted from the old A4-0 session-mode helper (which
        constructed the event); the SSM now owns construction, so this sink
        dispatches a pre-built event. The caller captured mode_revision INSIDE
        the status-write txn (INV-2) before building."""

        async def _sink(session_id: str, event: SessionModeChangedEvent) -> None:
            if task is not None:
                try:
                    event.id = await task.output_stream.put(event.model_dump_json())
                except Exception as exc:
                    logger.warning(
                        "会话[%s]写入SessionModeChangedEvent到输出流失败，降级为仅落库: %s",
                        session_id,
                        exc,
                    )
            async with self._uow_factory() as uow:
                await uow.session.add_event(session_id, event)

        return _sink

    async def _append_error_event(
        self, session_id: str, *, error: str, task: Optional[Task] = None
    ) -> ErrorEvent:
        error_event = ErrorEvent(error=error)
        if task:
            try:
                event_id = await task.output_stream.put(error_event.model_dump_json())
                error_event.id = event_id
            except Exception as exc:
                logger.warning(
                    "会话[%s]写入ErrorEvent到输出流失败，降级为仅落库: %s",
                    session_id,
                    exc,
                )
        uow = self._uow_factory()
        async with uow:
            await uow.session.add_event(session_id, error_event)
        return error_event

    async def _emit_bg_notification_if_background(
        self,
        session_id: str,
        event_type: str,
    ) -> None:
        emitter = getattr(self, "_memory_notification_emitter", None)
        if emitter is None:
            return

        try:
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)
        except Exception:
            logger.debug(
                "background notification session lookup failed for %s",
                session_id,
                exc_info=True,
            )
            return

        if (
            session is None
            or not getattr(session, "was_background", False)
            or not getattr(session, "user_id", None)
        ):
            return

        try:
            await emitter.emit(
                user_id=str(session.user_id),
                event_type=event_type,
                payload={"session_id": session_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s emit failed: session=%s err=%s",
                event_type,
                session_id,
                exc,
            )

    async def _cleanup_background_slot_if_needed(
        self,
        session: Session,
        *,
        reason: str,
    ) -> None:
        if (
            session.execution_mode != "background"
            and not getattr(session, "was_background", False)
        ):
            return
        if not getattr(session, "user_id", None):
            return
        supervisor = getattr(self, "_supervisor", None)
        if supervisor is None:
            return
        try:
            await supervisor.cleanup_background_slot(
                session_id=session.id,
                user_id=str(session.user_id),
                reason=reason,
            )
        except Exception:
            logger.warning(
                "background slot cleanup failed for session %s",
                session.id,
                exc_info=True,
            )

    @staticmethod
    def _bg_event_type_for_terminal(
        status: SessionStatus,
        terminal_reason: str,
    ) -> str | None:
        if terminal_reason == "user_cancel":
            return "bg_cancelled"
        if terminal_reason == "resume_state_lost":
            return "bg_failed_resume"
        if terminal_reason == "watchdog_timeout":
            return "bg_failed_watchdog"
        if status == SessionStatus.COMPLETED and terminal_reason == "natural":
            return "bg_completed"
        return None

    async def _emit_bg_terminal_notification_if_background(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
    ) -> None:
        event_type = self._bg_event_type_for_terminal(status, terminal_reason)
        if event_type is None:
            return
        await self._emit_bg_notification_if_background(session_id, event_type)

    async def _maybe_stop_supervisor_for_session(self, session_id: str) -> None:
        """C3 PR-3c (codex r5 [HIGH CONTRACT] fix) — stop the per-pod
        MailboxSupervisor when ``AgentService`` writes a terminal status
        outside the runner's ``_set_terminal_status`` path.

        Safe + idempotent:
          * No-op when ``_supervisor_registry`` is ``None`` (mailbox flag
            disabled or PR-4 not yet shipped — registry isn't built).
          * No-op when ``session_id`` is a subagent or otherwise has no
            slot — ``SupervisorRegistry.stop`` simply ``pop``s a missing
            entry, returns immediately. We don't pre-filter for root
            because (a) the round-trip would cost a UoW lookup per
            terminal, and (b) the registry is the single source of
            truth for "is this id a tracked root?" — keep that authority
            in one place.
          * Exceptions logged + swallowed so a transient registry hiccup
            cannot fail a terminal write that already committed.

        Called from every non-runner ``update_to_terminal`` site in
        AgentService (admin cancel, takeover-pending timeout,
        takeover-lease timeout, FORCE_TERMINATE via approval, retry
        budget exhaustion, ...). The runner's own terminal path goes
        through ``AgentTaskRunner._set_terminal_status._terminal_op``
        which calls ``self._maybe_stop_mailbox_supervisor()`` — same
        intent, different code path, both fire under the same registry
        idempotency contract so duplicate calls are harmless.
        """
        registry = self._supervisor_registry
        if registry is None:
            return
        # C3 PR-3c (codex r7→r12) — fire-and-forget the stop so this hook
        # introduces ZERO cancellation seam at the caller. Earlier rounds
        # used ``asyncio.shield(await ...)`` then re-raised CancelledError;
        # r12 caught that the re-raise still skips downstream cleanup in
        # every caller (background slot cleanup, control events, lua
        # revoke, ...). Fire-and-forget removes the await entirely:
        #
        # * The spawned task is anchored in ``_PENDING_MAILBOX_STOP_TASKS``
        #   so asyncio cannot GC it before completion.
        # * ``_on_mailbox_stop_task_done`` observes raises via the done
        #   callback (logs at ERROR with traceback).
        # * If lifespan shutdown happens before the stop completes,
        #   ``SupervisorRegistry.stop_all()`` (main.py shutdown step)
        #   sweeps any in-flight slots; the in-flight stop task's
        #   completion is a race we accept because stop is idempotent.
        #
        # This is the same trade-off the C3 spec §6.5 makes: the stop
        # hook is best-effort plumbing, NOT a synchronization barrier
        # the caller depends on.
        stop_task = asyncio.create_task(
            registry.stop(session_id),
            name=f"mailbox-stop-{session_id}",
        )
        _PENDING_MAILBOX_STOP_TASKS.add(stop_task)
        stop_task.add_done_callback(_on_mailbox_stop_task_done)

    async def _inject_handoff_message(self, task: Task, text: str) -> str:
        """向任务输入流注入一条handoff消息，用于恢复执行上下文。"""
        handoff_event = MessageEvent(
            role="system",
            message=text,
        )
        return await task.input_stream.put(handoff_event.model_dump_json())

    async def _resume_task_with_handoff(
        self, session: Session, text: str, *, retry_lifecycle_context=None,
    ) -> Task:
        """重建任务并注入handoff消息后启动任务。

        retry_lifecycle_context（C7 §5，R3#7）：独立 optional 参数、仅
        retry_from_suspend 传入——不与普通 takeover handoff 文本混流；
        透传不受 lifecycle flag 门控（R7#P3c）。
        """
        task = await self._create_task(session, retry_lifecycle_context=retry_lifecycle_context)
        await self._inject_handoff_message(task, text)
        await task.invoke()
        return task

    async def retry_from_suspend(
        self,
        session_id: str,
        user_id: str,
        *,
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ) -> Dict[str, object]:
        """Retry a suspended background task without changing ownership."""
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        if session.status != SessionStatus.RUNNING:
            raise BadRequestError("当前会话状态不支持后台重试")
        if session.execution_mode != "background":
            raise BadRequestError("当前会话不是后台任务")
        if session.execution_phase != "suspended":
            raise ConflictError("后台任务状态已变化，请刷新后重试")
        if session.retry_budget_remaining <= 0:
            raise ConflictError("后台任务重试次数已用尽")

        if session.sandbox_binding.state in (
            SandboxBindingState.DESTROYING,
            SandboxBindingState.DESTROYED,
        ):
            raise ConflictError("沙箱已终止，无法重试")
        if session.sandbox_binding.state not in (
            SandboxBindingState.ACTIVE,
            SandboxBindingState.SUSPENDED,
        ):
            raise ConflictError("沙箱状态不支持重试")

        if self._sandbox_lifecycle_service is None:
            raise ServiceUnavailableError("沙箱生命周期服务暂不可用，请稍后重试")
        supervisor = getattr(self, "_supervisor", None)
        if supervisor is None:
            raise ServiceUnavailableError("后台执行服务暂不可用，请稍后重试")

        if session.background_reason == "auto_degrade":
            expires_at = supervisor.new_auto_degrade_cleanup_expiry()
        else:
            # Explicit background expiry is a caller-owned deadline, not a
            # rolling cleanup lease. Preserve it across retry. Legacy rows
            # without a persisted value keep the previous two-hour fallback.
            expires_at = session.expires_at or (
                datetime.now(timezone.utc) + timedelta(hours=2)
            )
        original_retry_budget = session.retry_budget_remaining
        original_expires_at = session.expires_at
        original_suspended_reason = session.suspended_reason

        from app.domain.errors.sandbox_lifecycle import (
            SandboxLifecycleError,
            SessionDestroyingError,
            SessionFinalizedError,
        )

        resume_user_id = str(session.user_id or user_id)
        claimed_retry: tuple[int, int] | None = None
        admission_rc: int | None = None
        claim_rollback_attempted = False
        try:
            async with supervisor.mode_transition_fence(session_id=session.id):
                try:
                    async with self._uow_factory() as uow:
                        claimed_retry = (
                            await uow.session.claim_background_retry_from_suspend(
                                session.id,
                                expires_at=expires_at,
                            )
                        )
                        if claimed_retry is not None:
                            await _commit_uow_if_real(uow)
                except BaseException:
                    if claimed_retry is not None:
                        claim_rollback_attempted = True
                        await self._rollback_background_retry_claim(
                            session,
                            expected_execution_revision=claimed_retry[1],
                            retry_budget_remaining=original_retry_budget,
                            expires_at=original_expires_at,
                            suspended_reason=original_suspended_reason,
                        )
                    raise
                if claimed_retry is None:
                    raise ConflictError("后台任务状态已变化，请刷新后重试")
                claimed_retry_budget, retry_execution_revision = claimed_retry

                try:
                    admission_rc = await supervisor.resume(
                        session_id=session.id,
                        user_id=resume_user_id,
                        execution_mode="background",
                        expires_at=expires_at,
                        previous_expires_at=original_expires_at,
                        retry_budget_remaining=claimed_retry_budget,
                        expected_execution_revision=retry_execution_revision,
                    )
                except SupervisorContractError as exc:
                    claim_rollback_attempted = True
                    await self._rollback_background_retry_claim(
                        session,
                        expected_execution_revision=retry_execution_revision,
                        retry_budget_remaining=original_retry_budget,
                        expires_at=original_expires_at,
                        suspended_reason=original_suspended_reason,
                    )
                    message = (
                        "后台执行名额已满，请稍后重试"
                        if exc.rejection_code in ("R1", "R2")
                        else "后台任务状态已变化，请刷新后重试"
                    )
                    raise ConflictError(message) from exc
                except BaseException:
                    claim_rollback_attempted = True
                    await self._rollback_background_retry_claim(
                        session,
                        expected_execution_revision=retry_execution_revision,
                        retry_budget_remaining=original_retry_budget,
                        expires_at=original_expires_at,
                        suspended_reason=original_suspended_reason,
                    )
                    raise
        except BaseException:
            if claimed_retry is not None and not claim_rollback_attempted:
                claim_rollback_attempted = True
                rolled_back = await self._rollback_background_retry_claim(
                    session,
                    expected_execution_revision=claimed_retry[1],
                    retry_budget_remaining=original_retry_budget,
                    expires_at=original_expires_at,
                    suspended_reason=original_suspended_reason,
                )
                if rolled_back:
                    await self._rollback_background_resume_admission(
                        session,
                        user_id=resume_user_id,
                        supervisor=supervisor,
                        admission_rc=admission_rc,
                        previous_expires_at=original_expires_at,
                        expected_execution_revision=claimed_retry[1],
                    )
            raise

        try:
            await self._sandbox_lifecycle_service.resume(session.id)
            session.execution_phase = "running"
            session.suspended_reason = None
            session.expires_at = expires_at
            session.retry_budget_remaining = claimed_retry_budget
            await self._resume_task_with_handoff(
                session,
                "用户请求重试挂起的后台任务，请从上次中断处继续执行。",
                retry_lifecycle_context=RetryLifecycleContext(
                    retry_budget_remaining=claimed_retry_budget,
                ),
            )
        except (SessionDestroyingError, SessionFinalizedError) as exc:
            await self._finalize_lost_background_retry(
                session,
                user_id=resume_user_id,
                supervisor=supervisor,
                admission_rc=admission_rc,
                expected_execution_revision=retry_execution_revision,
            )
            raise ConflictError("沙箱已终止，无法重试") from exc
        except SandboxLifecycleError as exc:
            rolled_back = await self._rollback_background_retry_claim(
                session,
                expected_execution_revision=retry_execution_revision,
                retry_budget_remaining=original_retry_budget,
                expires_at=original_expires_at,
                suspended_reason=original_suspended_reason,
            )
            if rolled_back:
                await self._rollback_background_resume_admission(
                    session,
                    user_id=resume_user_id,
                    supervisor=supervisor,
                    admission_rc=admission_rc,
                    previous_expires_at=original_expires_at,
                    expected_execution_revision=retry_execution_revision,
                )
            raise ConflictError("沙箱状态不支持重试") from exc
        except BaseException:
            rolled_back = await self._rollback_background_retry_claim(
                session,
                expected_execution_revision=retry_execution_revision,
                retry_budget_remaining=original_retry_budget,
                expires_at=original_expires_at,
                suspended_reason=original_suspended_reason,
            )
            if rolled_back:
                await self._rollback_background_resume_admission(
                    session,
                    user_id=resume_user_id,
                    supervisor=supervisor,
                    admission_rc=admission_rc,
                    previous_expires_at=original_expires_at,
                    expected_execution_revision=retry_execution_revision,
                )
            # C3 PR-6 — legacy retired (spec §11.7). Subagent suspend owned
            # exclusively by MailboxSupervisor via destroy hook (M1).
            # _should_skip_mailbox_lifecycle reference satisfies AST CI gate
            # (§13.3); worker_type=="root" gate replaces the legacy else
            # fallback. retry_from_suspend rollback only fires on root
            # sessions in practice (mailbox children cannot retry_from_suspend),
            # so the explicit guard is defense-in-depth.
            if (
                rolled_back
                and session.sandbox_binding.state == SandboxBindingState.SUSPENDED
            ):
                if _should_skip_mailbox_lifecycle(session):
                    logger.debug(
                        "retry_from_suspend rollback: skip suspend %s — mailbox plane",
                        session.id,
                    )
                elif session.worker_type == "root":
                    try:
                        await self._sandbox_lifecycle_service.suspend(session.id)
                    except Exception:
                        logger.warning(
                            "retry_from_suspend rollback failed to suspend sandbox %s",
                            session.id,
                            exc_info=True,
                        )
                else:
                    logger.debug(
                        "retry_from_suspend rollback: skip suspend %s — "
                        "worker_type=%s is not root and not mailbox-plane "
                        "(unreachable post-PR-6 legacy retirement)",
                        session.id,
                        session.worker_type,
                    )
            raise

        return {
            "status": SessionStatus.RUNNING,
            "request_status": "resumed",
            "retry_budget_remaining": claimed_retry_budget,
            "expires_at": self._to_unix_seconds(expires_at),
        }

    async def _finalize_lost_background_retry(
        self,
        session: Session,
        *,
        user_id: str,
        supervisor,
        admission_rc: int | None,
        expected_execution_revision: int,
    ) -> None:
        try:
            await supervisor.terminate(
                session_id=session.id,
                user_id=str(session.user_id or user_id),
                terminal_reason="resume_state_lost",
                status=SessionStatus.TIMED_OUT,
            )
            return
        except Exception:
            logger.warning(
                "retry_from_suspend failed to terminate lost sandbox session %s",
                session.id,
                exc_info=True,
            )

        try:
            async with self._uow_factory() as uow:
                await self._ssm.terminate(
                    session.id,
                    SessionStatus.TIMED_OUT,
                    "resume_state_lost",
                    session_repo=uow.session,
                )
                # codex r11 — explicit commit (see resume_state_lost path).
                await _commit_uow_if_real(uow)
        except Exception:
            logger.warning(
                "retry_from_suspend fallback terminal update failed for session %s",
                session.id,
                exc_info=True,
            )
            return
        # C3 PR-3c (codex r5 + r11) — non-runner terminal: stop supervisor.
        await self._maybe_stop_supervisor_for_session(session.id)

        try:
            async with self._uow_factory() as uow:
                await uow.session.update_supervisor_fields(
                    session.id,
                    retry_budget_remaining=0,
                    expires_at=None,
                    suspended_reason=None,
                )
        except Exception:
            logger.warning(
                "retry_from_suspend fallback terminal cleanup failed for session %s",
                session.id,
                exc_info=True,
            )

        await self._revoke_background_resume_admission(
            session,
            user_id=user_id,
            supervisor=supervisor,
            admission_rc=admission_rc,
            expected_execution_revision=expected_execution_revision,
        )

    async def _revoke_background_resume_admission(
        self,
        session: Session,
        *,
        user_id: str,
        supervisor,
        admission_rc: int | None,
        expected_execution_revision: int,
    ) -> None:
        if admission_rc is None:
            return
        try:
            await supervisor.revoke_background_resume_admission(
                session_id=session.id,
                user_id=user_id,
                admission_rc=admission_rc,
                expected_execution_revision=expected_execution_revision,
            )
        except Exception:
            logger.warning(
                "retry_from_suspend Redis admission revoke failed for session %s",
                session.id,
                exc_info=True,
            )

    async def _rollback_background_resume_admission(
        self,
        session: Session,
        *,
        user_id: str,
        supervisor,
        admission_rc: int | None,
        previous_expires_at: Optional[datetime],
        expected_execution_revision: int,
    ) -> None:
        if admission_rc is None:
            return
        try:
            await supervisor.rollback_background_resume_admission(
                session_id=session.id,
                user_id=user_id,
                admission_rc=admission_rc,
                previous_expires_at=previous_expires_at,
                expected_execution_revision=expected_execution_revision,
            )
        except Exception:
            logger.warning(
                "retry_from_suspend Redis admission rollback failed for session %s",
                session.id,
                exc_info=True,
            )

    async def _rollback_background_retry_claim(
        self,
        session: Session,
        *,
        expected_execution_revision: int,
        retry_budget_remaining: int,
        expires_at: Optional[datetime],
        suspended_reason: Optional[str],
    ) -> bool:
        try:
            async with self._uow_factory() as uow:
                rolled_back = (
                    await uow.session.rollback_background_retry_claim_if_active(
                        session.id,
                        expected_execution_revision=expected_execution_revision,
                        retry_budget_remaining=retry_budget_remaining,
                        expires_at=expires_at,
                        suspended_reason=suspended_reason,
                    )
                )
                if rolled_back:
                    await _commit_uow_if_real(uow)
                return rolled_back
        except Exception:
            logger.warning(
                "retry_from_suspend rollback failed for session %s",
                session.id,
                exc_info=True,
            )
            return False

    async def _rollback_resume_failed(
        self,
        session_id: str,
        *,
        error: Exception,
        task: Optional[Task],
        takeover_id: Optional[str] = None,
        operator_user_id: Optional[str] = None,
    ) -> None:
        logger.exception("会话[%s]恢复执行失败: %s", session_id, error)
        self._cancel_takeover_timeout(session_id)
        if task and not task.done:
            task.cancel(reason="takeover_timeout")
        await self._append_error_event(
            session_id,
            error=f"恢复执行失败: {str(error)}",
            task=task,
        )
        uow = self._uow_factory()
        async with uow:
            transitioned = await self._ssm.terminate(
                session_id,
                SessionStatus.COMPLETED,
                "resume_state_lost",
                session_repo=uow.session,
            )
            # codex r11 — explicit commit (see other resume_state_lost path).
            await _commit_uow_if_real(uow)
        if transitioned is not False:
            await self._emit_bg_terminal_notification_if_background(
                session_id,
                SessionStatus.COMPLETED,
                "resume_state_lost",
            )
        # C3 PR-3c (codex r5) — non-runner terminal: stop supervisor.
        await self._maybe_stop_supervisor_for_session(session_id)
        await self._append_control_event(
            session_id,
            action=ControlAction.ENDED,
            source=ControlSource.SYSTEM,
            handoff_mode="complete",
            reason="resume_failed",
            request_status="failed",
            takeover_id=takeover_id,
            task=task,
        )
        if takeover_id and operator_user_id:
            await self._release_takeover_lease(
                session_id,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
            )

    async def _complete_takeover_after_cancel(
        self,
        *,
        session_id: str,
        task: Task,
        scope: ControlScope,
        takeover_id: str,
        operator_user_id: str,
        cancel_timeout_seconds: int,
        lease_ttl_seconds: Optional[int] = None,
        expires_at: Optional[datetime] = None,
    ) -> None:
        """等待running任务取消完成，异步推进接管状态。"""
        try:
            effective_lease_ttl_seconds = max(
                int(lease_ttl_seconds or self._settings.feature_takeover_lease_ttl_seconds),
                1,
            )
            effective_expires_at = expires_at or self._build_lease_expiry(
                effective_lease_ttl_seconds
            )
            deadline = time.monotonic() + max(cancel_timeout_seconds, 1)
            while not task.done and time.monotonic() < deadline:
                await asyncio.sleep(0.05)

            if task.done:
                uow = self._uow_factory()
                async with uow:
                    await self._ssm.set_mode(
                        session_id,
                        SessionStatus.TAKEOVER,
                        reason="takeover_started",
                        session_repo=uow.session,
                    )
                    try:
                        _, rev = await uow.session.read_status_with_revision(session_id)
                    except Exception:
                        rev = None
                await self._ssm.emit_session_mode_changed(
                    session_id,
                    to=SessionStatus.TAKEOVER,
                    from_mode="running",
                    reason="takeover_started",
                    mode_revision=rev,
                    sink=self._sse_or_db_sink(task),
                )
                await self._append_control_event(
                    session_id,
                    action=ControlAction.STARTED,
                    source=ControlSource.SYSTEM,
                    scope=scope,
                    request_status="started",
                    takeover_id=takeover_id,
                    expires_at=effective_expires_at,
                    task=task,
                )
                self._schedule_takeover_timeout(
                    session_id=session_id,
                    takeover_id=takeover_id,
                    operator_user_id=operator_user_id,
                    ttl_seconds=effective_lease_ttl_seconds,
                )
                return

            await self._append_control_event(
                session_id,
                action=ControlAction.REJECTED,
                source=ControlSource.SYSTEM,
                scope=scope,
                reason="cancel_timeout",
                request_status="rejected",
                takeover_id=takeover_id,
                task=task,
            )
            await self._release_takeover_lease(
                session_id,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
            )
        except Exception as exc:
            logger.exception("会话[%s]异步确认接管失败: %s", session_id, exc)
            await self._append_error_event(
                session_id,
                error=f"接管确认失败: {str(exc)}",
                task=task,
            )
            await self._release_takeover_lease(
                session_id,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
            )

    async def get_takeover(
        self,
        session_id: str,
        user_id: str,
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ) -> Dict[str, object]:
        """获取会话接管状态"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        latest_control = self._get_latest_control_event(session)
        if not latest_control:
            return {"status": session.status}

        return {
            "status": session.status,
            "takeover_id": latest_control.takeover_id,
            "request_status": latest_control.request_status,
            "reason": latest_control.reason,
            "scope": latest_control.scope.value if latest_control.scope else None,
            "handoff_mode": latest_control.handoff_mode,
            "expires_at": self._to_unix_seconds(latest_control.expires_at),
        }

    async def assert_takeover_shell_access(
        self,
        *,
        session_id: str,
        user_id: str,
        takeover_id: str,
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ) -> None:
        """校验终端接管 WebSocket 访问权限与租约有效性。"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
            scope=ControlScope.SHELL,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        if session.status != SessionStatus.TAKEOVER:
            raise BadRequestError("当前会话不处于接管状态")

        latest_control = self._get_latest_control_event(session)
        if not latest_control or not latest_control.takeover_id:
            raise ConflictError("接管租约已失效或不匹配")
        if latest_control.takeover_id != takeover_id:
            raise ConflictError("接管租约已失效或不匹配")
        if latest_control.scope and latest_control.scope != ControlScope.SHELL:
            raise BadRequestError("当前接管范围不是终端")

        lease_owned = await self._verify_takeover_lease_owner(
            session_id=session_id,
            takeover_id=takeover_id,
            operator_user_id=user_id,
        )
        if not lease_owned:
            raise ConflictError("接管租约已失效或不匹配")

    async def start_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        scope: str = "shell",
        is_admin: bool = False,
        user_role: Optional[str] = None,
        cancel_timeout_seconds: int = TAKEOVER_CANCEL_TIMEOUT_SECONDS,
    ) -> Dict[str, object]:
        """启动会话接管"""
        try:
            control_scope = ControlScope(scope)
        except ValueError as exc:
            raise BadRequestError("scope仅支持 shell 或 browser") from exc
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
            scope=control_scope,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)

        if session.status == SessionStatus.TAKEOVER:
            latest_control = self._get_latest_control_event(session)
            return {
                "request_status": "started",
                "status": SessionStatus.TAKEOVER,
                "scope": (
                    latest_control.scope.value
                    if latest_control and latest_control.scope
                    else control_scope.value
                ),
                "takeover_id": latest_control.takeover_id if latest_control else None,
                "expires_at": (
                    self._to_unix_seconds(latest_control.expires_at)
                    if latest_control
                    else None
                ),
            }

        if session.status not in {
            SessionStatus.RUNNING,
            SessionStatus.WAITING,
            SessionStatus.TAKEOVER_PENDING,
        }:
            raise BadRequestError(
                f"当前状态[{session.status.value}]不支持启动接管"
            )

        takeover_id = f"tk_{uuid.uuid4().hex[:12]}"
        lease_ttl_seconds = max(self._settings.feature_takeover_lease_ttl_seconds, 1)
        lease_expires_at = self._build_lease_expiry(lease_ttl_seconds)
        acquired = await self._acquire_takeover_lease(
            session_id,
            takeover_id=takeover_id,
            operator_user_id=user_id,
            ttl_seconds=lease_ttl_seconds,
        )
        if not acquired:
            raise ConflictError("接管租约冲突，请稍后重试")

        if session.status == SessionStatus.RUNNING:
            task = await self._get_task(session)
            if task:
                task.cancel(reason="takeover_start")
                self._schedule_takeover_completion(
                    session_id=session_id,
                    task=task,
                    scope=control_scope,
                    takeover_id=takeover_id,
                    operator_user_id=user_id,
                    cancel_timeout_seconds=cancel_timeout_seconds,
                    lease_ttl_seconds=lease_ttl_seconds,
                    expires_at=lease_expires_at,
                )
                return {
                    "takeover_id": takeover_id,
                    "request_status": "starting",
                    "status": SessionStatus.RUNNING,
                    "scope": control_scope.value,
                    "expires_at": self._to_unix_seconds(lease_expires_at),
                }
            logger.warning(
                "会话[%s]处于RUNNING但无活跃task，直接按无任务路径进入接管",
                session_id,
            )

        if session.status == SessionStatus.TAKEOVER_PENDING:
            self._cancel_pending_timeout(session_id)

        from_mode = session.status.value  # source ∈ {running, waiting, takeover_pending}
        async with self._uow_factory() as uow:
            await self._ssm.set_mode(
                session_id,
                SessionStatus.TAKEOVER,
                reason="takeover_started",
                session_repo=uow.session,
            )
            try:
                _, rev = await uow.session.read_status_with_revision(session_id)
            except Exception:
                rev = None
        await self._ssm.emit_session_mode_changed(
            session_id,
            to=SessionStatus.TAKEOVER,
            from_mode=from_mode,
            reason="takeover_started",
            mode_revision=rev,
            sink=self._sse_or_db_sink(None),
        )
        await self._append_control_event(
            session_id,
            action=ControlAction.STARTED,
            source=ControlSource.USER,
            scope=control_scope,
            request_status="started",
            takeover_id=takeover_id,
            expires_at=lease_expires_at,
        )
        self._schedule_takeover_timeout(
            session_id=session_id,
            takeover_id=takeover_id,
            operator_user_id=user_id,
            ttl_seconds=lease_ttl_seconds,
        )
        return {
            "takeover_id": takeover_id,
            "request_status": "started",
            "status": SessionStatus.TAKEOVER,
            "scope": control_scope.value,
            "expires_at": self._to_unix_seconds(lease_expires_at),
        }

    async def renew_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        takeover_id: str,
        is_admin: bool = False,
        user_role: Optional[str] = None,
        lease_ttl_seconds: Optional[int] = None,
    ) -> Dict[str, object]:
        """续期会话接管租约"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        if session.status != SessionStatus.TAKEOVER:
            raise BadRequestError("当前会话不处于接管状态")

        effective_ttl_seconds = (
            lease_ttl_seconds
            if lease_ttl_seconds is not None and lease_ttl_seconds > 0
            else self._settings.feature_takeover_lease_ttl_seconds
        )
        effective_ttl_seconds = max(int(effective_ttl_seconds), 1)
        lease_expires_at = self._build_lease_expiry(effective_ttl_seconds)
        renewed = await self._renew_takeover_lease(
            session_id,
            takeover_id=takeover_id,
            operator_user_id=user_id,
            ttl_seconds=effective_ttl_seconds,
        )
        if not renewed:
            raise ConflictError("接管租约已失效或不匹配")

        await self._append_control_event(
            session_id,
            action=ControlAction.RENEWED,
            source=ControlSource.USER,
            request_status="renewed",
            takeover_id=takeover_id,
            expires_at=lease_expires_at,
        )
        self._schedule_takeover_timeout(
            session_id=session_id,
            takeover_id=takeover_id,
            operator_user_id=user_id,
            ttl_seconds=effective_ttl_seconds,
        )
        return {
            "status": SessionStatus.TAKEOVER,
            "request_status": "renewed",
            "takeover_id": takeover_id,
            "expires_at": self._to_unix_seconds(lease_expires_at),
        }

    async def reject_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        decision: str,
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ) -> Dict[str, object]:
        """处理接管请求拒绝"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        if session.status != SessionStatus.TAKEOVER_PENDING:
            raise BadRequestError("当前会话不处于待接管状态")
        self._cancel_takeover_timeout(session_id)
        self._cancel_pending_timeout(session_id)
        latest_control = self._get_latest_control_event(session)
        takeover_id = latest_control.takeover_id if latest_control else None

        decision_normalized = (decision or "").strip().lower()
        if decision_normalized == "continue":
            resumed_task: Optional[Task] = None
            try:
                resumed_task = await self._resume_task_with_handoff(
                    session,
                    "用户拒绝接管请求，请继续执行上次任务。",
                )
            except Exception as exc:
                await self._rollback_resume_failed(
                    session_id,
                    error=exc,
                    task=resumed_task,
                )
                return {"status": SessionStatus.COMPLETED, "reason": "resume_failed"}

            async with self._uow_factory() as uow:
                await self._ssm.set_mode(
                    session_id,
                    SessionStatus.RUNNING,
                    reason="takeover_rejected",
                    session_repo=uow.session,
                )
                try:
                    _, rev = await uow.session.read_status_with_revision(session_id)
                except Exception:
                    rev = None
            await self._ssm.emit_session_mode_changed(
                session_id,
                to=SessionStatus.RUNNING,
                from_mode="takeover_pending",  # reject_takeover requires TAKEOVER_PENDING (guard ~4447)
                reason="takeover_rejected",
                mode_revision=rev,
                sink=self._sse_or_db_sink(resumed_task),
            )
            await self._append_control_event(
                session_id,
                action=ControlAction.REJECTED,
                source=ControlSource.USER,
                reason="continue",
                takeover_id=takeover_id,
                task=resumed_task,
            )
            await self._force_release_takeover_lease(session_id)
            return {"status": SessionStatus.RUNNING, "reason": "continue"}

        if decision_normalized == "terminate":
            async with self._uow_factory() as uow:
                transitioned = await self._ssm.terminate(
                    session_id,
                    SessionStatus.COMPLETED,
                    "user_cancel",
                    session_repo=uow.session,
                )
                # codex r11 — explicit commit (see resume_state_lost path).
                await _commit_uow_if_real(uow)
            if transitioned is not False:
                await self._emit_bg_terminal_notification_if_background(
                    session_id,
                    SessionStatus.COMPLETED,
                    "user_cancel",
                )
            # C3 PR-3c (codex r5 + r11) — non-runner terminal: stop supervisor.
            await self._maybe_stop_supervisor_for_session(session_id)
            await self._append_control_event(
                session_id,
                action=ControlAction.REJECTED,
                source=ControlSource.USER,
                reason="terminate",
                takeover_id=takeover_id,
            )
            await self._force_release_takeover_lease(session_id)
            return {"status": SessionStatus.COMPLETED, "reason": "terminate"}

        raise BadRequestError("decision仅支持 continue 或 terminate")

    async def end_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        handoff_mode: str = "continue",
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ) -> Dict[str, object]:
        """结束接管并交还控制"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        if session.status != SessionStatus.TAKEOVER:
            raise BadRequestError("当前会话不处于接管状态")
        self._cancel_takeover_timeout(session_id)
        latest_control = self._get_latest_control_event(session)
        takeover_id = latest_control.takeover_id if latest_control else None
        takeover_scope = latest_control.scope if latest_control else None

        mode = (handoff_mode or "").strip().lower()
        if mode == "continue":
            if takeover_scope == ControlScope.BROWSER:
                handoff_text = (
                    "用户已结束浏览器接管，浏览器页面已被用户操作，"
                    "请先获取最新页面状态再继续执行任务。"
                )
            else:
                handoff_text = "用户已结束接管并交还控制，请继续执行未完成任务。"
            resumed_task: Optional[Task] = None
            try:
                resumed_task = await self._resume_task_with_handoff(
                    session,
                    handoff_text,
                )
            except Exception as exc:
                await self._rollback_resume_failed(
                    session_id,
                    error=exc,
                    task=resumed_task,
                    takeover_id=takeover_id,
                    operator_user_id=user_id,
                )
                return {
                    "status": SessionStatus.COMPLETED,
                    "handoff_mode": "complete",
                }

            async with self._uow_factory() as uow:
                await self._ssm.set_mode(
                    session_id,
                    SessionStatus.RUNNING,
                    reason="takeover_ended",
                    session_repo=uow.session,
                )
                try:
                    _, rev = await uow.session.read_status_with_revision(session_id)
                except Exception:
                    rev = None
            await self._ssm.emit_session_mode_changed(
                session_id,
                to=SessionStatus.RUNNING,
                from_mode="takeover",
                reason="takeover_ended",
                mode_revision=rev,
                sink=self._sse_or_db_sink(resumed_task),
            )
            await self._append_control_event(
                session_id,
                action=ControlAction.ENDED,
                source=ControlSource.USER,
                handoff_mode="continue",
                takeover_id=takeover_id,
                task=resumed_task,
            )
            if takeover_id:
                await self._release_takeover_lease(
                    session_id,
                    takeover_id=takeover_id,
                    operator_user_id=user_id,
                )
            return {"status": SessionStatus.RUNNING, "handoff_mode": "continue"}

        if mode == "complete":
            async with self._uow_factory() as uow:
                transitioned = await self._ssm.terminate(
                    session_id,
                    SessionStatus.COMPLETED,
                    "natural",
                    session_repo=uow.session,
                )
                # codex r11 — explicit commit (see resume_state_lost path).
                await _commit_uow_if_real(uow)
            if transitioned is not False:
                await self._emit_bg_terminal_notification_if_background(
                    session_id,
                    SessionStatus.COMPLETED,
                    "natural",
                )
            # C3 PR-3c (codex r5 + r11) — non-runner terminal: stop supervisor.
            await self._maybe_stop_supervisor_for_session(session_id)
            await self._append_control_event(
                session_id,
                action=ControlAction.ENDED,
                source=ControlSource.USER,
                handoff_mode="complete",
                takeover_id=takeover_id,
            )
            if takeover_id:
                await self._release_takeover_lease(
                    session_id,
                    takeover_id=takeover_id,
                    operator_user_id=user_id,
                )
            return {"status": SessionStatus.COMPLETED, "handoff_mode": "complete"}

        raise BadRequestError("handoff_mode仅支持 continue 或 complete")

    async def reopen_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ):
        """已完成会话的补救接管：completed -> takeover_pending，并调度 pending timeout。"""
        # scope 在 reopen 阶段故意不传：reopen 仅恢复到 takeover_pending，
        # 具体 scope 在后续 start_takeover 时由用户选择确定
        self._assert_takeover_capability(
            user_id=user_id, is_admin=is_admin, user_role=user_role
        )
        await self._get_accessible_session(session_id, user_id, is_admin)

        window_seconds = self._settings.feature_takeover_reopen_window_seconds
        if window_seconds <= 0:
            raise BadRequestError("REOPEN_DISABLED")

        # 使用 _uow_factory() 创建独立 UoW，避免单例共享 DB session
        uow = self._uow_factory()
        async with uow:
            # 事务内加锁重读，防并发
            session = await uow.session.get_by_id_for_update(session_id)
            if not session or session.status not in (
                SessionStatus.COMPLETED,
                SessionStatus.TIMED_OUT,
            ):
                raise BadRequestError("当前状态不支持恢复接管")

            # I2: reopen_takeover 要求 binding.state ∈ {ACTIVE, SUSPENDED}
            # DESTROYING/DESTROYED 的会话不可恢复
            from app.domain.models.session import SandboxBindingState
            if session.sandbox_binding.state in (
                SandboxBindingState.DESTROYING,
                SandboxBindingState.DESTROYED,
            ):
                raise BadRequestError("沙箱已终止，无法恢复接管")
            if not session.completed_at:
                raise BadRequestError("REOPEN_WINDOW_EXPIRED")
            # 与领域模型 / ORM 保持一致，使用 datetime.now()（naive local time）
            elapsed = (datetime.now() - session.completed_at).total_seconds()
            if elapsed > window_seconds:
                raise BadRequestError("REOPEN_WINDOW_EXPIRED")
            # 使用 add_event 而非 _append_control_event，
            # 因为 completed 阶段无活跃 output stream
            await uow.session.add_event(
                session_id,
                ControlEvent(
                    action=ControlAction.REOPENED,
                    source=ControlSource.USER,
                ),
            )
            await self._ssm.set_mode(
                session_id,
                SessionStatus.TAKEOVER_PENDING,
                reason="takeover_reopened",
                session_repo=uow.session,
            )
            try:
                _, rev = await uow.session.read_status_with_revision(session_id)
            except Exception:
                rev = None
            await self._ssm.emit_session_mode_changed(
                session_id,
                to=SessionStatus.TAKEOVER_PENDING,
                # reopen source is a terminal/non-control status (the guard above
                # allows COMPLETED or TIMED_OUT) — neither is in ModeLiteral, so
                # from_mode is None (valid for Optional[ModeLiteral]).
                from_mode=None,
                reason="takeover_reopened",
                mode_revision=rev,
                sink=lambda sid, ev: uow.session.add_event(sid, ev),
            )

        self._schedule_pending_timeout(session_id)
        remaining_seconds = window_seconds - elapsed
        return {
            "status": SessionStatus.TAKEOVER_PENDING,
            "request_status": "reopened",
            "reason": None,
            "remaining_seconds": remaining_seconds,
        }

    def start_sweep_task(self) -> None:
        """Start the background confirmation-sweep asyncio.Task.

        Should be called once from the FastAPI lifespan after all services are
        initialized.  Safe to call multiple times — no-ops if already running.
        """
        if self._confirmation_sweep_task is not None and not self._confirmation_sweep_task.done():
            return
        self._confirmation_sweep_task = asyncio.create_task(
            self._confirmation_sweep_loop(),
            name="confirmation_sweep",
        )

    async def _confirmation_sweep_loop(self) -> None:
        """Background task: sweep expired and orphaned-processing confirmations every 30s."""
        import uuid
        worker_id = str(uuid.uuid4())[:8]
        while True:
            try:
                await asyncio.sleep(30)
                if not self._confirmation_manager:
                    continue
                if not await self._confirmation_manager.acquire_sweep_lock(worker_id):
                    continue

                # Phase 1: Sweep expired confirmations (timeout_fallback resume).
                expired = await self._confirmation_manager.find_expired()
                for detail in expired:
                    try:
                        await self._confirmation_manager.mark_processing(detail.session_id, detail.tool_call_id)
                        logger.info(
                            "Confirmation timeout: session=%s tool_call=%s, resuming with timeout_fallback",
                            detail.session_id,
                            detail.tool_call_id,
                        )
                        # Resume the interrupted graph with timeout_fallback action
                        resume_ok = False
                        try:
                            async with self._uow_factory() as _sweep_uow:
                                session = await _sweep_uow.session.get_by_id(detail.session_id)
                            if session:
                                task = await self._get_task(session)
                                if task is None:
                                    task = await self._create_task(session)
                                if task:
                                    from langgraph.types import Command
                                    await task.resume(Command(resume={"action": "timeout_fallback", "scope": "once"}))
                                    resume_ok = True
                        except Exception:
                            logger.exception(
                                "Timeout resume failed for session=%s, rolling back to pending",
                                detail.session_id,
                            )
                        if resume_ok:
                            await self._confirmation_manager.cleanup(detail.session_id, detail.tool_call_id)
                        else:
                            # Roll back to pending so next sweep cycle can retry
                            await self._confirmation_manager.mark_pending(detail.session_id, detail.tool_call_id)
                    except Exception:
                        logger.exception(
                            "Sweep failed for %s:%s",
                            detail.session_id,
                            detail.tool_call_id,
                        )

                # Phase 2: P1#3 — Rescue orphaned 'processing' entries.
                # These are claims where preflight_resume succeeded (status=processing +
                # claim_nonce written) but commit_resume never ran (e.g. worker crash
                # between HTTP response and graph resume).
                #
                # P2#1 fix: check deadline_ts before reopening.
                # If the orphan's user-facing deadline has already passed, the
                # confirmation can no longer be meaningfully retried — reopen
                # would let a user approve an already-timed-out tool call.
                # Expired orphans are cleaned up (not reopened); still-live
                # orphans are mark_pending so /resume retry can re-claim.
                try:
                    import time as _time  # local to avoid hoisting at module top

                    orphans = await self._confirmation_manager.find_orphaned_processing(
                        processing_age_threshold_seconds=300,
                    )
                    _now_ts = _time.time()
                    for orphan in orphans:
                        try:
                            _deadline = getattr(orphan, "deadline_ts", None)
                            if _deadline is not None and _deadline <= _now_ts:
                                # Already past user-facing timeout → advance graph
                                # past interrupt first, then cleanup queue entry.
                                # Without resuming the graph, the LangGraph checkpoint
                                # stays in the interrupt state indefinitely while the
                                # frontend can no longer retry (queue entry gone).
                                logger.warning(
                                    "Orphaned processing confirmation expired (deadline=%.0f now=%.0f)"
                                    " — resuming graph with timeout_fallback then cleaning up:"
                                    " session=%s tool_call=%s",
                                    _deadline,
                                    _now_ts,
                                    orphan.session_id,
                                    orphan.tool_call_id,
                                )
                                # P2#2 (round-17): Only cleanup the queue entry
                                # *after* the graph resume succeeds.  If resume fails
                                # (exception or no in-memory task), leave the entry so
                                # the next sweep cycle can retry — otherwise the
                                # LangGraph checkpoint stays stuck in the interrupt
                                # state while the queue entry is gone, making the
                                # frontend unable to retry.
                                #
                                # P2#1 (round-23): Task.resume() is async-submit — it
                                # enqueues the resume command but does NOT wait for the
                                # LangGraph checkpoint to actually advance.  If the
                                # background graph execution fails (checkpoint error,
                                # network issue) after resume returns, and we already
                                # cleaned up the Redis confirmation entry, the interrupt
                                # can never be retried.  Conservative fix: submit the
                                # resume but never cleanup immediately — the commit_resume
                                # path (called from within the graph when the interrupt
                                # is actually consumed) is responsible for the final
                                # cleanup.  Subsequent sweep iterations are idempotent:
                                # re-submitting timeout_fallback to an already-advanced
                                # graph is a no-op from the graph's perspective.
                                try:
                                    async with self._uow_factory() as _orphan_uow:
                                        _orphan_session = await _orphan_uow.session.get_by_id(
                                            orphan.session_id
                                        )
                                    if _orphan_session is not None:
                                        _orphan_task = await self._get_task(_orphan_session)
                                        if _orphan_task is None:
                                            _orphan_task = await self._create_task(_orphan_session)
                                        if _orphan_task is not None:
                                            # P2#2 (round-25): check whether the task is
                                            # PE-active before submitting a PE-style resume.
                                            # If the task config was hot-switched or
                                            # build_permission_engine failed after the
                                            # orphan was written, _flow._permission_engine
                                            # may be None.  In that case, forwarding
                                            # claim_nonce causes interrupt_helper to call
                                            # commit_resume on a None PE → AttributeError,
                                            # and commit_resume cleanup never runs, leaving
                                            # the Redis 'processing' entry permanently.
                                            # Fix: detect legacy task + PE orphan and
                                            # cleanup the queue entry directly here.
                                            # P2#1 (round-26): RedisStreamTask stores
                                            # the runner as self._task_runner; _flow
                                            # lives on AgentTaskRunner, NOT on the task
                                            # itself.  getattr(task, "_flow", None)
                                            # always returns None for production tasks
                                            # → _is_legacy_task never fires → PE orphan
                                            # silently falls through to task.resume()
                                            # → commit_resume hits None PE → AttributeError
                                            # → entry leaks in 'processing' forever.
                                            #
                                            # Fix: resolve _flow via _task_runner first
                                            # (production path), then fall back to direct
                                            # _flow (test-stub compatibility).
                                            _orphan_task_runner = getattr(
                                                _orphan_task, "_task_runner", None
                                            )
                                            _orphan_flow = getattr(
                                                _orphan_task_runner, "_flow", None
                                            ) or getattr(_orphan_task, "_flow", None)
                                            # Only check _permission_engine when _flow
                                            # is explicitly present.  If _flow is absent
                                            # on both paths, we cannot tell whether the
                                            # task is legacy — fall through to the normal
                                            # resume path (existing simple task stubs
                                            # without _flow are not affected).
                                            _orphan_task_pe = (
                                                getattr(
                                                    _orphan_flow, "_permission_engine", None
                                                )
                                                if _orphan_flow is not None
                                                else None  # _flow absent → treat as unknown
                                            )
                                            _is_pe_orphan = orphan.claim_nonce is not None
                                            # PE-orphan on legacy task: _flow is present
                                            # but holds no _permission_engine.  If _flow
                                            # is absent we cannot tell — fall through to
                                            # the normal resume path.
                                            _is_legacy_task = (
                                                _orphan_flow is not None
                                                and _orphan_task_pe is None
                                            )

                                            if _is_pe_orphan and _is_legacy_task:
                                                # PE orphan on a legacy task: commit_resume
                                                # won't run (no PE injected in task), so the
                                                # queue entry must be freed manually.
                                                # Codex round-29 P2#1: The LangGraph checkpoint
                                                # is still at interrupt_helper — cleanup alone
                                                # leaves the graph permanently stuck there
                                                # (queue gone → frontend can't retry, and
                                                # next sweep no longer sees an orphan).
                                                # Fix: submit a legacy-style resume (no
                                                # claim_nonce) so interrupt_helper walks the
                                                # legacy deny path and advances the graph.
                                                #
                                                # Codex round-36 P2#1 (current fix):
                                                # Legacy interrupt_helper NEVER calls
                                                # commit_resume (no PE in flow), so the
                                                # queue-cleanup path that the PE branch
                                                # relies on does not exist for legacy tasks.
                                                # Deferring cleanup to the next sweep means
                                                # we repeatedly resubmit the same legacy
                                                # deny every 30s forever — the entry stays
                                                # in 'processing' permanently.  Round 35 was
                                                # already wrong on this point.
                                                #
                                                # Correct behavior: after submitting the
                                                # legacy resume successfully, cleanup the
                                                # queue entry synchronously.  This DOES
                                                # race with a background resume failure,
                                                # but the alternative is a guaranteed
                                                # permanent leak.  The trade-off here is
                                                # explicit: legacy tasks have no
                                                # commit_resume to defer to, so the sweeper
                                                # is the only cleanup site.  If the
                                                # background graph step fails after this
                                                # point, the checkpoint stays in interrupt
                                                # state but the user-facing confirmation is
                                                # already timed out (deadline passed) and
                                                # would have produced a deny outcome
                                                # anyway.
                                                logger.warning(
                                                    "Expired orphan: PE orphan on legacy task"
                                                    " (no _permission_engine) — submitting"
                                                    " legacy deny resume and cleaning up queue"
                                                    " entry on success (legacy path has no"
                                                    " commit_resume to defer cleanup to);"
                                                    " session=%s tool_call=%s claim_nonce=%s",
                                                    orphan.session_id,
                                                    orphan.tool_call_id,
                                                    orphan.claim_nonce,
                                                )
                                                from langgraph.types import Command as _Command
                                                _legacy_resume_ok = False
                                                try:
                                                    await _orphan_task.resume(
                                                        _Command(
                                                            resume={
                                                                "tool_call_id": orphan.tool_call_id,
                                                                "action": "deny",
                                                                "scope": "once",
                                                                "reason": "confirmation_timeout",
                                                                # NO claim_nonce — forces legacy
                                                                # interrupt_helper path which
                                                                # will NOT call commit_resume.
                                                            }
                                                        )
                                                    )
                                                    _legacy_resume_ok = True
                                                    logger.info(
                                                        "Legacy resume submitted for PE-orphan-on-legacy-task;"
                                                        " cleaning up queue entry synchronously"
                                                        " (legacy path has no commit_resume) —"
                                                        " session=%s tool_call=%s",
                                                        orphan.session_id,
                                                        orphan.tool_call_id,
                                                    )
                                                except Exception:
                                                    logger.exception(
                                                        "Expired orphan: legacy deny resume"
                                                        " failed for PE orphan on legacy task"
                                                        " — leaving entry for next sweep;"
                                                        " session=%s tool_call=%s",
                                                        orphan.session_id,
                                                        orphan.tool_call_id,
                                                    )
                                                # Round 36 P2#1: cleanup ONLY on resume
                                                # success.  Failure path leaves the entry
                                                # so the next sweep can retry the resume
                                                # submission itself.
                                                if _legacy_resume_ok:
                                                    try:
                                                        await self._confirmation_manager.cleanup(
                                                            orphan.session_id,
                                                            orphan.tool_call_id,
                                                        )
                                                    except Exception:
                                                        logger.exception(
                                                            "Expired orphan: post-legacy-resume"
                                                            " cleanup failed for"
                                                            " session=%s tool_call=%s"
                                                            " (entry may leak until manual"
                                                            " intervention)",
                                                            orphan.session_id,
                                                            orphan.tool_call_id,
                                                        )
                                            else:
                                                from langgraph.types import Command as _Command
                                                await _orphan_task.resume(
                                                    _Command(
                                                        resume={
                                                            "tool_call_id": orphan.tool_call_id,
                                                            # P2#1 (round-24): use "deny" so
                                                            # ResumeSignal.action Literal check
                                                            # passes ("timeout_fallback" is not
                                                            # a valid action literal and would
                                                            # cause a ValueError in PE path).
                                                            # timeout is treated as implicit deny.
                                                            "action": "deny",
                                                            "scope": "once",
                                                            "reason": "confirmation_timeout",
                                                            # Forward the PE claim nonce so
                                                            # interrupt_helper routes to the PE
                                                            # commit_resume path (which cleans
                                                            # up the queue entry).  Without
                                                            # this, claim_nonce is None → legacy
                                                            # fallback → commit_resume cleanup
                                                            # never runs → Redis entry leaks.
                                                            "claim_nonce": orphan.claim_nonce,
                                                        }
                                                    )
                                                )
                                                logger.info(
                                                    "Expired orphan: deny resume submitted"
                                                    " (cleanup deferred to commit_resume) —"
                                                    " session=%s tool_call=%s claim_nonce=%s",
                                                    orphan.session_id,
                                                    orphan.tool_call_id,
                                                    orphan.claim_nonce,
                                                )
                                                # NOTE: Do NOT cleanup here. Task.resume is
                                                # async-fire-and-forget; actual graph
                                                # advancement happens in the background.
                                                # Cleanup is handled by commit_resume when
                                                # the graph interrupt is consumed, or by
                                                # the next sweep if the background task
                                                # fails.
                                        else:
                                            logger.warning(
                                                "Expired orphan: no in-memory task found for"
                                                " session=%s tool_call=%s — skipping cleanup,"
                                                " next sweep will retry",
                                                orphan.session_id,
                                                orphan.tool_call_id,
                                            )
                                    else:
                                        # P2#2 (round-24): session row deleted →
                                        # orphan is unrecoverable (_get_task /
                                        # _create_task will never succeed). Clean up
                                        # the Redis processing entry immediately to
                                        # prevent permanent resource leak and
                                        # repeated-sweep noise every 30 s.
                                        logger.warning(
                                            "Expired orphan: session deleted for"
                                            " session=%s tool_call=%s —"
                                            " cleaning up unrecoverable confirmation",
                                            orphan.session_id,
                                            orphan.tool_call_id,
                                        )
                                        try:
                                            await self._confirmation_manager.cleanup(
                                                orphan.session_id,
                                                orphan.tool_call_id,
                                            )
                                        except Exception:
                                            logger.exception(
                                                "Expired orphan: cleanup after session"
                                                " deletion failed for session=%s"
                                                " tool_call=%s",
                                                orphan.session_id,
                                                orphan.tool_call_id,
                                            )
                                except Exception:
                                    logger.exception(
                                        "Expired orphan: timeout resume submission failed for"
                                        " session=%s tool_call=%s — leaving entry for next"
                                        " sweep to retry",
                                        orphan.session_id,
                                        orphan.tool_call_id,
                                    )
                                # Never cleanup immediately — defer to commit_resume or next sweep
                            else:
                                # Still within deadline → reopen so user can retry
                                logger.warning(
                                    "Orphaned processing confirmation rescued: session=%s tool_call=%s",
                                    orphan.session_id,
                                    orphan.tool_call_id,
                                )
                                await self._confirmation_manager.mark_pending(
                                    orphan.session_id,
                                    orphan.tool_call_id,
                                )
                        except Exception:
                            logger.exception(
                                "Orphan rescue failed for %s:%s",
                                orphan.session_id,
                                orphan.tool_call_id,
                            )
                except Exception:
                    logger.exception("Orphaned processing sweep error")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Confirmation sweep error")

    async def shutdown(self) -> None:
        """关闭Agent服务"""
        if self._confirmation_sweep_task is not None and not self._confirmation_sweep_task.done():
            self._confirmation_sweep_task.cancel()
            try:
                await self._confirmation_sweep_task
            except asyncio.CancelledError:
                pass
        self._confirmation_sweep_task = None
        for task in list(self._pending_timeout_tasks.values()):
            if not task.done():
                task.cancel()
        self._pending_timeout_tasks.clear()
        for task in list(self._takeover_timeout_tasks.values()):
            if not task.done():
                task.cancel()
        self._takeover_timeout_tasks.clear()
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        self._background_tasks.clear()
        logger.info("正在清除所有会话任务资源并释放")
        await self._task_cls.destroy()
        logger.info("所有会话任务资源清除成功")
