import asyncio
import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from app.application.services.agent_service import AgentService
from app.application.services.app_config_service import AppConfigService
from app.application.services.file_service import FileService
from app.application.services.memory_management_service import MemoryManagementService
from app.application.services.session_service import SessionService
from app.application.services.skill_creator_service import SkillCreatorService
from app.application.services.skill_export_service import SkillExportService
from app.application.services.skill_service import SkillService
from app.application.services.status_service import StatusService
from app.application.services.user_tool_approval_policy_service import (
    UserToolApprovalPolicyService,
)

# from app.domain.repositories.session_repository import SessionRepository
from app.infrastructure.external.file_storage.minio_file_storage import MinioFileStorage
from app.infrastructure.external.health_checker.minio_health_checker import (
    MinioHealthChecker,
)
from app.infrastructure.external.health_checker.postgres_health_checker import (
    PostgresHealthChecker,
)
from app.infrastructure.external.health_checker.redis_health_checker import (
    RedisHealthChecker,
)
from app.domain.external.mailbox_publisher import MailboxPublisher
from app.domain.models.app_config import LLMConfig, SkillRiskPolicy
from app.application.services.agent_service import _ConfigSnapshot
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel
from langchain_core.language_models import BaseChatModel
from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox
from app.infrastructure.external.github_search_client import GitHubSearchClient
from app.infrastructure.external.event_recovery.redis_event_recovery import RedisEventRecovery
from app.infrastructure.external.search.bing_search import BingSearchEngine
from app.infrastructure.external.task.redis_stream_task import RedisStreamTask

# from app.infrastructure.repositories.db_file_repository import DBFileRepository
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.infrastructure.repositories.file_app_config_repository import (
    FileAppConfigRepository,
)
from app.infrastructure.repositories.db_memory_chunk_repository import DBMemoryChunkRepository
from app.infrastructure.repositories.db_memory_system_notification_repository import (
    DBMemorySystemNotificationRepository,
)
from app.infrastructure.repositories.db_user_tool_approval_policy_repository import (
    DBUserToolApprovalPolicyRepository,
)
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.storage.minio import MinioStore, get_minio
from app.infrastructure.storage.postgres import get_db_session, get_postgres, get_uow
from app.infrastructure.storage.redis import RedisClient, get_redis

# from app.interfaces.repository_dependencies import get_db_session_repository
from core.config import get_settings
from fastapi import Depends, Request
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import HTTPConnection

# from functools import lru_cache


logger = logging.getLogger(__name__)
settings = get_settings()

# D5.1: combined (primary + fallback) threshold above which fallback-path
# worst case may saturate ExecutionConfig.total_timeout_seconds (default 600s).
# Derived as: total_timeout_seconds_default / graph_retry_count = 600 / 3 = 200.
_FALLBACK_BUDGET_WARNING_THRESHOLD_SECONDS: float = 200.0

# --- Config cache (D2) ---
_config_cache: "AppConfig | None" = None
_config_mtime: float = 0.0
_config_size: int = 0
_config_expiry: float = 0.0
_config_generation: int = 0
_config_lock = threading.Lock()


# @lru_cache()
def get_app_config_service() -> AppConfigService:
    """获取应用配置服务"""
    # 1.获取数据仓库并打印日志
    logger.info("加载获取AppConfigService")
    file_app_config_repository = FileAppConfigRepository(settings.app_config_filepath)

    # 2.实例化AppConfigService
    return AppConfigService(app_config_repository=file_app_config_repository)


def get_subagent_limits() -> "SubagentLimitsConfig":
    """C1a: spawn cap config loaded once from env. Used by SessionService.create_session_with_parent."""
    from core.config import SubagentLimitsConfig  # local import avoids circular
    return settings.subagent_limits


# @lru_cache()
def get_status_service(
    db_session: AsyncSession = Depends(get_db_session),
    redis_client: RedisClient = Depends(get_redis),
    minio_store: MinioStore = Depends(get_minio),
) -> StatusService:
    """获取状态服务"""
    # 1.初始化postgres和redis健康检查器
    postgres_checker = PostgresHealthChecker(db_session)
    redis_checker = RedisHealthChecker(redis_client)
    minio_checker = MinioHealthChecker(minio_store)

    # 2.创建服务并返回
    logger.info("加载获取StatusService")
    return StatusService(checkers=[postgres_checker, redis_checker, minio_checker])


# @lru_cache()
def get_file_service(
    minio_store: MinioStore = Depends(get_minio),
) -> FileService:
    # 1.初始化文件仓库和文件存储桶
    # file_repository = DBFileRepository(db_session=db_session)
    file_storage = MinioFileStorage(
        bucket=settings.minio_bucket_name,
        minio_store=minio_store,
        uow_factory=get_uow,
    )

    # 2.构建服务并返回
    return FileService(
        uow_factory=get_uow,
        file_storage=file_storage,
    )


# @lru_cache()
def get_session_service(request: HTTPConnection) -> SessionService:
    lifecycle_service = getattr(request.app.state, "sandbox_lifecycle_service", None)
    fs_reconciler = getattr(request.app.state, "fs_reconciler", None)
    supervisor = getattr(request.app.state, "supervisor", None)
    return SessionService(
        uow_factory=get_uow,
        task_cls=RedisStreamTask,
        sandbox_lifecycle_service=lifecycle_service,
        fs_reconciler=fs_reconciler,
        execution_supervisor=supervisor,
        subagent_limits=get_subagent_limits(),
        # C3 PR-6 (spec §11.7) — legacy retired; SessionService no longer
        # consults a runtime flag. The ``mailbox_flag_reader`` kwarg used
        # to wire the rollback-aware live env reader is no longer passed.
        # SessionService unconditionally writes
        # ``subagent_control_plane='mailbox'``; the §11.6 rollback runbook
        # is decommissioned together with the alembic migration that
        # upgrades any historic ``legacy`` rows.
    )


def _load_app_config() -> "AppConfig":
    """TTL + mtime/size cached config loader. Generation counter for downstream refresh."""
    global _config_cache, _config_mtime, _config_size, _config_expiry, _config_generation
    now = time.monotonic()
    path = Path(settings.app_config_filepath).resolve()

    with _config_lock:
        if _config_cache is not None:
            try:
                st = path.stat()
            except OSError:
                return _config_cache
            if (now < _config_expiry
                    and st.st_mtime == _config_mtime
                    and st.st_size == _config_size):
                return _config_cache

        repo = FileAppConfigRepository(settings.app_config_filepath)
        _config_cache = repo.load()
        try:
            st = path.stat()
            _config_mtime = st.st_mtime
            _config_size = st.st_size
        except OSError:
            _config_mtime = 0.0
            _config_size = 0
        _config_expiry = now + settings.config_cache_ttl
        _config_generation += 1
        return _config_cache


# --- LLM cache (D2) ---
_MAX_LLM_CACHE_SIZE = 4  # main + summary + vision_fallback + 1 余量
_llm_cache: dict[str, BaseChatModel] = {}
_llm_lock = threading.Lock()


def _llm_fingerprint(llm_config: LLMConfig, supports_pdf_input: bool = False) -> str:
    """Content-addressable key for LLM instances.

    D5.1 note: timeout_seconds is included so configs differing only in
    timeout produce distinct cache entries. This is a selective hash — if
    LLMConfig gains new fields in the future, they must also be added.
    TODO(post-D5.1): consider switching to a model_dump-based fingerprint
    to avoid per-field drift between LLMConfig and this function.
    """
    parts = (
        str(llm_config.base_url),
        llm_config.api_key,
        llm_config.model_name,
        str(llm_config.temperature),
        str(llm_config.max_tokens),
        llm_config.api_type,
        str(getattr(llm_config, "supports_response_format", True)),
        str(getattr(llm_config, "supports_vision", True)),
        str(supports_pdf_input),
        # D5.1: different timeouts must produce different cached instances.
        str(llm_config.timeout_seconds),
        # D5.2: connect_timeout_seconds likewise — configs differing only in
        # the connect budget must not share a cached adapter instance.
        str(llm_config.connect_timeout_seconds),
        # A7 C2: 配置不同 provider 必须产生不同 adapter 实例。
        # 归一化与 _build_llm() 保持一致 (strip whitespace) — 否则语义等价的
        # " openai_official " / "openai_official" 会命中两个不同 cache entry。
        (llm_config.provider or "").strip(),
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _build_llm(llm_config: LLMConfig, *, supports_pdf_input: bool = False) -> BaseChatModel:
    """根据 api_type 配置构建 LLM 实例。

    - chat_completions: 仅用 Chat Completions API
    - responses: 仅用 Responses API
    - auto: 先走 Chat Completions，遇到兼容性错误自动回退到 Responses API

    NOTE: 不使用 with_fallbacks()，因为 RunnableWithFallbacks.__getattr__ 会调用
    typing.get_type_hints() 解析 BaseChatModel.with_structured_output 的类型注解，
    而 langchain-core 在 TYPE_CHECKING 块中 import builtins，运行时不可用，
    导致 NameError: name 'builtins' is not defined。
    改用 ActusFallbackChatModel 在 BaseChatModel 层面内部处理回退逻辑。
    """
    fp = _llm_fingerprint(llm_config, supports_pdf_input)
    with _llm_lock:
        if fp in _llm_cache:
            return _llm_cache[fp]

        from app.infrastructure.external.llm.actus_fallback_chat_model import ActusFallbackChatModel

        # A7 P0.1: resolve ProviderProfile from LLMConfig.provider, with
        # heuristic base_url inference when provider is empty. _build_llm is
        # the single production entry that injects the profile; direct-
        # instantiation paths fall back to generic_openai via default_factory.
        #
        # P0.1 registry only contains generic_openai + openai_official.
        # infer_provider_from_base_url may return ids (kimi_k2, deepseek_*,
        # glm, gemini_compat, ...) that aren't registered yet — those land
        # in later phases. For inferred unknown ids, fall back to
        # generic_openai + WARN so the P0.1 "no regression" contract holds
        # (spec §8 rollout).
        #
        # Explicit vs inferred distinction (Task 1.8 P2 fix):
        # - Explicit ``llm_config.provider="<typo>"`` MUST fail-fast as
        #   ConfigError — silently falling back would paper over a config
        #   typo and route traffic against the wrong profile without
        #   surfacing the bug.
        # - Inferred ``base_url`` -> unregistered id silently falls back
        #   because that id will land in a later phase (transitional).
        from app.application.errors.exceptions import ConfigError
        from app.domain.services.provider_profiles import (
            get_profile,
            infer_provider_from_base_url,
        )

        explicit_provider = (llm_config.provider or "").strip()
        if explicit_provider:
            # Fail-fast: a typo'd explicit provider id must surface, not be
            # swallowed. Bubbles to caller as ConfigError.
            profile = get_profile(explicit_provider)
        else:
            provider_id = infer_provider_from_base_url(
                str(llm_config.base_url),
                model_name=llm_config.model_name,
            )
            try:
                profile = get_profile(provider_id)
            except ConfigError:
                logger.warning(
                    "[A7] inferred provider_id=%s not registered in P0.1; "
                    "falling back to generic_openai profile",
                    provider_id,
                )
                profile = get_profile("generic_openai")

        timeout_seconds = llm_config.timeout_seconds
        connect_timeout_seconds = llm_config.connect_timeout_seconds

        # A7 P1: profile.supports_vision acts as a hard ceiling. Even if the
        # user's LLMConfig.supports_vision=True, a profile that declares no
        # vision (e.g. DeepSeek Reasoner) must force the wrapped adapter's
        # supports_vision=False so downstream image-block embedding is
        # disabled end-to-end.
        effective_supports_vision = bool(
            getattr(llm_config, "supports_vision", True)
            and profile.supports_vision
        )

        chat = ActusChatModel(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key,
            model_name=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            supports_response_format=getattr(llm_config, 'supports_response_format', True),
            supports_vision=effective_supports_vision,
            # Profile ceiling also applies to Chat adapter: profile that
            # declares no native-PDF support forces supports_pdf_input=False.
            # Computed once below the Chat adapter but reachable here via the
            # later Responses block's ``effective_supports_pdf_input``; we
            # recompute inline for locality.
            supports_pdf_input=bool(
                supports_pdf_input
                and getattr(profile, "supports_pdf_input", True)
            ),
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            profile=profile,  # A7 P0.1: profile injection
        )
        # A7 profile ceiling on supports_pdf_input: a profile that declares no
        # native-PDF support (e.g. openai_official) must force
        # supports_pdf_input=False on the adapter regardless of user config.
        # Message sanitizer + adapter wire both read this field so the ceiling
        # flows end-to-end — otherwise PDF blocks enter the wire payload even
        # when the profile says the provider does not accept native PDFs.
        effective_supports_pdf_input = bool(
            supports_pdf_input
            and getattr(profile, "supports_pdf_input", True)
        )

        responses = ActusResponsesModel(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key,
            model_name=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            supports_response_format=getattr(
                llm_config, "supports_response_format", True,
            ),
            supports_vision=effective_supports_vision,
            supports_pdf_input=effective_supports_pdf_input,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            profile=profile,  # A7 P0.1: profile injection
        )
        if llm_config.api_type == "responses":
            llm = responses
        elif llm_config.api_type == "auto":
            # D5.1 budget warning heuristic: if primary + fallback combined
            # budget is large, fallback path worst case may approach or
            # exceed D5 ExecutionWatchdog total_timeout_seconds. Log warning
            # (non-hard) so operators see it during config load.
            combined = timeout_seconds * 2
            if combined > _FALLBACK_BUDGET_WARNING_THRESHOLD_SECONDS:
                logger.warning(
                    "[D5.1 budget warning] ActusFallbackChatModel with "
                    "timeout_seconds=%.0fs may approach D5 ExecutionWatchdog "
                    "total_timeout_seconds budget. Fallback worst case = "
                    "%.0fs x 3 graph retries = %.0fs. Consider lowering "
                    "timeout_seconds or raising "
                    "ExecutionConfig.total_timeout_seconds.",
                    timeout_seconds,
                    combined,
                    combined * 3,
                )
            llm = ActusFallbackChatModel(
                primary=chat, fallback=responses, profile=profile,
            )
        else:
            llm = chat

        if len(_llm_cache) >= _MAX_LLM_CACHE_SIZE:
            oldest = next(iter(_llm_cache))
            del _llm_cache[oldest]
        _llm_cache[fp] = llm
        return llm


def _build_skill_service() -> SkillService:
    return SkillService(FileSkillRepository(settings.skills_root_dir))


def _build_config_snapshot(app_config: "AppConfig") -> _ConfigSnapshot:
    """Build an immutable config snapshot from app_config. Sub-deps use caches."""
    # Initial (user-config only) pdf_input for _build_llm; adapter-internal use.
    naive_pdf_input = bool(
        getattr(app_config.llm_config, 'supports_pdf_input', False)
        and app_config.llm_config.supports_vision
    )
    llm = _build_llm(app_config.llm_config, supports_pdf_input=naive_pdf_input)

    # A7 P1: profile.supports_vision is a hard ceiling for end-to-end vision.
    # Apply the same AND to the AgentTaskRunner / AgentService supports_vision
    # flag so image embedding is disabled when the profile declares no vision.
    _profile = getattr(llm, "profile", None)
    effective_supports_vision = bool(
        app_config.llm_config.supports_vision
        and getattr(_profile, "supports_vision", True)
    )
    # A7 capability ceiling on PDF: profile.supports_pdf_input=False forces PDF
    # off end-to-end, even when user config + supports_vision would otherwise
    # allow it. Sanitizer + both adapters read this downstream; the ceiling
    # must land here too or AgentTaskRunner will still admit PDF blocks into
    # HumanMessage content that then survives wire serialization.
    effective_pdf_input = bool(
        naive_pdf_input
        and effective_supports_vision
        and getattr(_profile, "supports_pdf_input", True)
    )

    summary_llm = None
    if app_config.agent_config.memory.summary_model:
        # D5.1: summary_timeout_seconds overrides the main LLM timeout for the
        # summarizer path. None means "inherit main llm_config.timeout_seconds".
        # The summarizer runs with no tools and short prompts, so it usually
        # wants a tighter bound than the main agent.
        summary_timeout_override = (
            app_config.agent_config.memory.summary_timeout_seconds
        )
        summary_update: dict[str, Any] = {
            "model_name": app_config.agent_config.memory.summary_model,
        }
        if summary_timeout_override is not None:
            summary_update["timeout_seconds"] = summary_timeout_override
        summary_llm_config = app_config.llm_config.model_copy(
            update=summary_update
        )
        summary_llm = _build_llm(summary_llm_config)

    vision_fallback_model = None
    vf = app_config.file_understanding.vision_fallback
    if vf.enabled and vf.model_name:
        from app.domain.models.app_config import LLMConfig as LLMConfigModel
        vision_llm_config = LLMConfigModel(
            base_url=vf.base_url or str(app_config.llm_config.base_url),
            api_key=vf.api_key or app_config.llm_config.api_key,
            model_name=vf.model_name,
            api_type=vf.api_type,
            supports_vision=True,
            timeout_seconds=app_config.llm_config.timeout_seconds,  # D5.1: inherit from main; VisionFallbackConfig has no independent timeout field
            connect_timeout_seconds=app_config.llm_config.connect_timeout_seconds,  # D5.2: inherit from main (same rationale as timeout_seconds)
        )
        vision_fallback_model = _build_llm(vision_llm_config)

    skill_creator_service = SkillCreatorService(
        llm=llm,
        github_client=GitHubSearchClient(token=settings.github_token or None),
        skill_service=_build_skill_service(),
    )

    overflow_config = ContextOverflowConfig.from_llm_config(app_config.llm_config)

    # M1 PR-4+8 memory gate LLM resolution. ``settings.memory_gate_llm`` is
    # a string **key** chosen by the deployer in config—``"summary_llm"``
    # (reuse summary_llm instance), ``"chat_llm"`` (reuse main llm), or
    # ``None`` (disabled). Other values currently fall through to None
    # rather than silently picking an arbitrary model—if ops wants a third
    # LLM just for gate, a new branch here + new LLMConfig is the
    # explicit extension path.
    memory_gate_llm_key = getattr(settings, "memory_gate_llm", None)
    memory_gate_llm: BaseChatModel | None = None
    if memory_gate_llm_key == "summary_llm":
        memory_gate_llm = summary_llm
    elif memory_gate_llm_key == "chat_llm":
        memory_gate_llm = llm
    elif memory_gate_llm_key:
        logger.warning(
            "settings.memory_gate_llm=%r not recognized; gate disabled.",
            memory_gate_llm_key,
        )

    return _ConfigSnapshot(
        llm=llm,
        agent_config=app_config.agent_config,
        mcp_config=app_config.mcp_config,
        a2a_config=app_config.a2a_config,
        skill_risk_policy=app_config.skill_risk_policy or SkillRiskPolicy(),
        overflow_config=overflow_config,
        summary_llm=summary_llm,
        vision_fallback_model=vision_fallback_model,
        skill_creator_service=skill_creator_service,
        supports_vision=effective_supports_vision,
        supports_pdf_input=effective_pdf_input,
        file_understanding_config=app_config.file_understanding,
        tool_runtime=app_config.tool_runtime,
        memory_gate_llm=memory_gate_llm,
        memory_gate_threshold=settings.memory_gate_threshold,
        memory_gate_batch_cap=settings.memory_gate_batch_cap,
    )


# --- AgentService refresh (D2) ---
_last_refresh_generation: int = 0
_refresh_lock = threading.Lock()


# codex r4 [HIGH CONTRACT] / r1 [R1-3] — PR-4.5 readiness gate (see
# ``build_supervisor_registry``).
#
# POST-PR-4.5: the flag is ``True`` (set below) and the registry
# factory wires ``_pr4_5_agent_service_callback`` — the real
# ``AgentService.stop_session`` bridge that satisfies spec §7.6
# stop-before-destroy ordering. The legacy ``_pr3c_noop_callback`` is
# preserved as a named symbol for backwards-compat tests but is no
# longer wired into the factory.
#
# The constant remains gateable: if a future refactor accidentally
# flips it back to ``False`` (or swaps the callback back to the noop)
# the lifespan still fails closed via the ``RuntimeError`` raised in
# ``build_supervisor_registry``, surfacing the misconfiguration at
# pod startup.
#
# Locked tests:
# ``tests/app/interfaces/test_build_supervisor_registry_pr4_gate.py``
# pin both the True-default and the callback identity; the legacy
# False-forced path still raises a load-bearing error with §7.6
# context for any regression that re-disables the gate.
_PR4_TERMINAL_HANDLERS_READY: bool = True
# codex r1 [R1-3, HIGH CONTRACT] (PR-4.5) — flipped True alongside the
# replacement of ``_pr3c_noop_callback`` with
# ``_pr4_5_agent_service_callback`` in this same commit. Spec §7.6 +
# the readiness-gate docstring above mandate that both halves move
# together: the real ``AgentService.stop_session`` bridge is now wired
# into ``build_supervisor_registry``'s factory below. The locked test
# at ``tests/app/interfaces/test_build_supervisor_registry_pr4_gate.py``
# has been updated in lockstep.


def build_supervisor_registry(
    *,
    redis_client: RedisClient,
    publisher: MailboxPublisher,
    sandbox_lifecycle_service: object,
) -> "SupervisorRegistry":
    """C3 PR-3c — construct the per-pod :class:`SupervisorRegistry` singleton.

    Called exclusively from ``main.py`` lifespan startup; the DI provider
    :func:`get_supervisor_registry` is a thin read of
    ``app.state.supervisor_registry`` and does NOT re-invoke this factory.
    The lifespan path owns construction so the singleton is built once
    before ``reconcile_orphans`` runs and is reused by every request.
    The registry's ``supervisor_factory`` closure builds a
    :class:`MailboxSupervisor` per root using:

    * the raw ``redis.asyncio.Redis`` client (unwrap from ``RedisClient``);
    * a fresh :class:`DbMailboxEnvelopeAuditRepository` per supervisor —
      the audit repo owns its own short-lived sessions per call (see the
      class docstring; it is NOT registered in DBUnitOfWork);
    * the shared mailbox ``publisher`` (XADD-only, no pubsub state shared
      with other callers);
    * the sandbox lifecycle service (typed as ``object`` here because
      ``SandboxLifecycleService`` lives in ``application/`` and we don't
      want a circular import — supervisor only calls ``destroy(session_id,
      reason)`` per the ``_SandboxLifecycleProtocol`` shape in
      mailbox_supervisor.py).

    ``agent_service_callback`` is ``_pr4_5_agent_service_callback`` —
    the post-PR-4.5 production bridge that dispatches
    ``CANCEL_REQUEST(TERMINATE)`` to ``AgentService.stop_session``
    (spec §7.6 stop-before-destroy) and is a no-op for every other
    envelope type. The legacy ``_pr3c_noop_callback`` is preserved as
    a named symbol for backwards-compat tests but is no longer wired
    into the registry factory.

    POST-PR-4.5 (codex r1 [R1-3] / r10 [R10-3]): the gate is OPEN by
    default. The factory wires :func:`_pr4_5_agent_service_callback`
    as ``ctx.agent_service_callback`` — the real
    ``AgentService.stop_session`` bridge that satisfies spec §7.6
    stop-before-destroy ordering on the
    ``CANCEL_REQUEST(policy=TERMINATE)`` path and is a no-op for
    every other envelope type. The legacy
    :func:`_pr3c_noop_callback` is preserved as a named symbol for
    backwards-compat tests but is NOT wired into the factory.

    If a refactor forces ``_PR4_TERMINAL_HANDLERS_READY`` back to
    ``False`` (regression), this function still raises
    ``RuntimeError`` so lifespan fails closed on misconfiguration.
    Tests can bypass via ``monkeypatch.setattr`` on the flag. The
    locked tests at
    ``tests/app/interfaces/test_build_supervisor_registry_pr4_gate.py``
    pin both the True-default and the
    :func:`_pr4_5_agent_service_callback` identity.
    """
    if not _PR4_TERMINAL_HANDLERS_READY:
        raise RuntimeError(
            "build_supervisor_registry: _PR4_TERMINAL_HANDLERS_READY is "
            "False. This is a REGRESSION post-PR-4.5: the gate should "
            "default to True so the factory wires the real "
            "_pr4_5_agent_service_callback (AgentService.stop_session "
            "bridge satisfying spec §7.6 stop-before-destroy). If "
            "you see this error, something flipped the flag back to "
            "False without also reverting the callback wiring — fix "
            "that regression rather than bypassing the gate. The "
            "locked tests are in "
            "tests/app/interfaces/test_build_supervisor_registry_pr4_gate.py "
            "(test_pr_4_5_readiness_gate_is_open + "
            "test_factory_wires_pr_4_5_callback_bridge). Post-PR-6 the "
            "MAILBOX_SUPERVISOR_ENABLED env-var rollback is decommissioned; "
            "the mailbox plane is the only supported control plane."
        )
    from app.application.services.mailbox_supervisor import (
        MailboxSupervisor,
        SupervisorContext,
    )
    from app.application.services.supervisor_registry import SupervisorRegistry
    from app.infrastructure.repositories.db_mailbox_envelope_audit_repository import (
        DbMailboxEnvelopeAuditRepository,
    )
    from core.config import resolve_mailbox_pod_id
    import uuid as _uuid

    settings_local = get_settings()
    pod_id = resolve_mailbox_pod_id(settings_local.mailbox_pod_id)
    audit_repo = DbMailboxEnvelopeAuditRepository(
        session_factory=get_postgres().session_factory,
    )

    class _TelemetryAdapter:
        """Best-effort logger-backed adapter exposing the
        ``emit(name, data) -> None`` shape the supervisor's
        ``_TelemetryProtocol`` expects.

        codex r1 [MEDIUM CONTRACT] — the previous incarnation tried to
        share ``JsonlPromptTelemetry`` for mailbox events, but that
        class has no generic ``write()`` method (only
        ``record_assembly`` / ``record_llm_invocation`` /
        ``emit_recovery_event`` — schema-typed callsites unrelated to
        mailbox supervisor telemetry). The fallback chain
        (``getattr(sink, "write", None) → logger.info``) was therefore
        ALWAYS taking the logger branch — the JsonlPromptTelemetry
        construction was dead code that promised a JSONL stream it
        never wrote to. Drop the dead path and log directly with a
        stable ``mailbox.telemetry`` prefix so ops can grep without
        reading two layers of indirection. PR-4 can wire a real
        ``mailbox_telemetry.jsonl`` writer if/when the volume warrants
        it — for the PR-3a/3b stub-handler workload, log is enough.
        """

        async def emit(self, name: str, data: dict) -> None:
            try:
                logger.info("mailbox.telemetry %s %s", name, data)
            except Exception:
                logger.debug(
                    "mailbox telemetry emit failed event=%s",
                    name,
                    exc_info=True,
                )

    telemetry_adapter = _TelemetryAdapter()

    raw_redis = redis_client.client

    # C3 PR-5 (spec §11.6 rollback runbook + R1 P2.2) — per-call session
    # adapter so ``MailboxSupervisor._check_should_stop_for_rollback`` can
    # run its read-only ``get_by_id`` + ``find_descendants`` queries
    # without holding a long-lived ``AsyncSession`` across the supervisor's
    # run lifetime. Mirrors the ``DbMailboxEnvelopeAuditRepository``
    # session-per-call pattern (see its class docstring at
    # ``infrastructure/repositories/db_mailbox_envelope_audit_repository.py:30``).
    #
    # Only the two methods used by the rollback-stop check are
    # implemented; the supervisor never calls anything else. Cast to the
    # full ``SessionRepository`` Protocol at the ``ctx.session_repo``
    # assignment site so the type checker matches the field declaration
    # (the Protocol is duck-typed at runtime so partial impl is fine,
    # but the cast keeps mypy happy).
    pg_session_factory = get_postgres().session_factory

    class _SupervisorSessionRepoAdapter:
        __slots__ = ("_session_factory",)

        def __init__(self, session_factory) -> None:  # type: ignore[no-untyped-def]
            self._session_factory = session_factory

        async def get_by_id(self, session_id: str):  # type: ignore[no-untyped-def]
            from app.infrastructure.repositories.db_session_repository import (
                DBSessionRepository,
            )
            async with self._session_factory() as db_session:
                repo = DBSessionRepository(db_session=db_session)
                return await repo.get_by_id(session_id)

        async def find_descendants(  # type: ignore[no-untyped-def]
            self,
            ancestor_id: str,
            *,
            user_id: str,
            max_depth: int,
            limit: int,
        ):
            from app.infrastructure.repositories.db_session_repository import (
                DBSessionRepository,
            )
            async with self._session_factory() as db_session:
                repo = DBSessionRepository(db_session=db_session)
                return await repo.find_descendants(
                    ancestor_id,
                    user_id=user_id,
                    max_depth=max_depth,
                    limit=limit,
                )

    supervisor_session_repo = _SupervisorSessionRepoAdapter(
        session_factory=pg_session_factory,
    )

    def _factory(root_session_id: str) -> MailboxSupervisor:
        # [C2 deferred wiring -- PR-9 composition root]
        # The following SupervisorContext fields are intentionally LEFT
        # UNSET at the live composition root; PR-9 wires them all together
        # when ``ACTUS_C2_COORDINATOR_ENABLED`` flips:
        #   - ``cost_rollup_service`` (PR-6 §14.4) -- Protocol-only stub
        #     today; concrete impl + wiring deferred to PR-9.
        #   - ``coordinator_envelope_store`` (PR-7 §12.4) -- concrete impl
        #     exists at ``DbCoordinatorResultEnvelopeStoreRepository`` but
        #     wiring deferred for atomic PR-9 flip.
        #   - ``PatchApplier._emit_event`` (PR-8 §13.5) -- wired to the per-run
        #     ``event_queue.put`` callback when the applier is constructed
        #     inside ``_run_parallel_backend``. Today the live call site passes
        #     None / a noop; the CoordinatorApplyEvent emit therefore silently
        #     no-ops in production. PR-9 wires this via the composition root
        #     when the feature flag flips.
        #   - ``CoordinatorRunOrchestrator._emit_event`` (PR-8 §13.6) -- wired
        #     via ``orchestrator_factory.build(..., emit_event=event_queue.put)``
        #     inside ``_first_time_dispatch``. Today the factory call passes
        #     no emit_event; the CoordinatorSiblingCancelEvent emit therefore
        #     silently no-ops. PR-9 wires this when the flag flips.
        # While unset (None default), the PR-6 cost-rollup PROLOGUE and
        # PR-7 persist-terminal PROLOGUE both silently no-op. The
        # supervisor otherwise functions identically to pre-coordinator
        # behavior. See ``mailbox_supervisor.py`` ResultReadyHandler /
        # CancelAckHandler ``_side_effect`` gates
        # ``if ctx.X is not None and ctx.session_repo is not None`` for
        # the runtime check.
        ctx = SupervisorContext(
            root_session_id=root_session_id,
            pod_id=pod_id,
            instance_id=_uuid.uuid4().hex[:8],
            redis=raw_redis,
            audit_repo=audit_repo,
            publisher=publisher,
            sandbox_lifecycle=sandbox_lifecycle_service,  # type: ignore[arg-type]
            # C3 PR-4.5 — real callback bridge replacing
            # ``_pr3c_noop_callback``. Supervisor handlers invoke this
            # BEFORE destroy on the TERMINATE path (spec §7.6 stop →
            # destroy ordering) so the child's agent task is stopped
            # cooperatively. ``_pr4_5_agent_service_callback`` is a
            # closure over the lifespan-scoped ``AgentService`` set via
            # ``_bind_agent_service_for_callback`` after
            # ``_build_agent_service`` returns; until binding completes
            # it is a safe no-op (lifespan ordering guarantees binding
            # happens before the supervisor consumer loop starts).
            agent_service_callback=_pr4_5_agent_service_callback,
            telemetry=telemetry_adapter,
            # C3 PR-5 (spec §11.6) — session reader for the rollback
            # stop check. Adapter holds session_factory and opens a
            # short-lived AsyncSession per query.
            session_repo=supervisor_session_repo,  # type: ignore[arg-type]
        )
        return MailboxSupervisor(
            ctx,
            block_ms=settings_local.mailbox_xreadgroup_block_ms,
            count=settings_local.mailbox_xreadgroup_count,
        )

    return SupervisorRegistry(supervisor_factory=_factory)


async def _pr3c_noop_callback(envelope) -> None:  # noqa: ANN001
    """C3 PR-3c placeholder for ``ctx.agent_service_callback`` — kept
    around for backwards compatibility with code paths and tests that
    still reference it by name (the locked gate test asserts on its
    existence and docstring).

    PR-3c placeholder behavior: no-op. PR-4.5 introduced
    ``_pr4_5_agent_service_callback`` as the production wiring; new
    callers should reference that name instead.
    """
    return None


# C3 PR-4.5 — mutable holder for the lifespan-scoped AgentService that
# ``_pr4_5_agent_service_callback`` dispatches into. Set ONCE by
# ``_bind_agent_service_for_callback`` immediately after
# ``_build_agent_service`` returns; before that, the callback waits on
# ``_PR4_5_AGENT_SERVICE_BIND_EVENT`` (codex r3 [R3-1, HIGH ARCH] —
# main.py lifespan runs ``reconcile_orphans`` BEFORE
# ``_build_agent_service`` so any supervisor that consumes a TERMINATE
# during the reconcile window must block until bind, not silently
# no-op).
#
# codex r5 [R5-1, HIGH ARCH] / codex r5 [R5-1 fix] — the event is
# created LAZILY on first ``_ensure_bind_event()`` call (which happens
# from inside an async context with a live loop). Module-load
# instantiation tied the event to whatever loop existed at import
# time (often the wrong one in pytest-anyio harnesses where each test
# spins its own loop), producing
# ``RuntimeError: <Event ...> is bound to a different event loop``.
# The lazy pattern defers loop binding to the first await.
_PR4_5_AGENT_SERVICE_FOR_CALLBACK: "AgentService | None" = None
_PR4_5_AGENT_SERVICE_BIND_EVENT: "asyncio.Event | None" = None
# How long to wait inside the callback for the AgentService to bind
# before giving up. Short enough that a permanently misconfigured
# lifespan surfaces quickly; long enough to absorb normal startup
# jitter on the reconcile → bind path.
_PR4_5_BIND_WAIT_TIMEOUT_SECONDS: float = 10.0


def _ensure_bind_event() -> "asyncio.Event":
    """Lazy ``asyncio.Event`` factory tied to the current running loop.

    Must be called from inside an async context. If the AgentService
    was already bound via ``_bind_agent_service_for_callback`` before
    this loop was entered (the common production case: bind happens in
    lifespan startup with a single asyncio loop), the freshly-created
    event is immediately set so the callback does not block on a
    permanently-cleared event.
    """
    global _PR4_5_AGENT_SERVICE_BIND_EVENT
    if _PR4_5_AGENT_SERVICE_BIND_EVENT is None:
        _PR4_5_AGENT_SERVICE_BIND_EVENT = asyncio.Event()
        if _PR4_5_AGENT_SERVICE_FOR_CALLBACK is not None:
            _PR4_5_AGENT_SERVICE_BIND_EVENT.set()
    return _PR4_5_AGENT_SERVICE_BIND_EVENT


def _bind_agent_service_for_callback(agent_service: "AgentService") -> None:
    """Wire the lifespan-scoped AgentService into the supervisor callback.

    Called from ``_build_agent_service`` after ``AgentService`` is fully
    constructed. Idempotent re-binds are safe (e.g. test harnesses that
    rebuild AgentService) but only the most recent reference wins. If a
    bind event has already been created (callback ran before bind), it
    is set so any pending awaits wake up; otherwise the event stays
    unset and ``_ensure_bind_event`` will set it lazily on first
    callback invocation.
    """
    global _PR4_5_AGENT_SERVICE_FOR_CALLBACK
    _PR4_5_AGENT_SERVICE_FOR_CALLBACK = agent_service
    if _PR4_5_AGENT_SERVICE_BIND_EVENT is not None:
        _PR4_5_AGENT_SERVICE_BIND_EVENT.set()


# codex r16 [R16-2, HIGH ARCH] — supervisor-terminate marker moved
# to ``app.domain.services.supervisor_terminate_marker`` so the
# domain-layer agent_task_runner doesn't reverse-import from
# interfaces (CLAUDE.md Clean Architecture rule). This module
# re-exports the helper for backwards-compat with any caller that
# imported the symbol from here.
from app.domain.services.supervisor_terminate_marker import (  # noqa: E402
    add_supervisor_terminate_marker as _add_supervisor_terminate_marker,
    consume_supervisor_terminate_marker as consume_supervisor_terminate_marker,  # noqa: F401
)


def _reset_agent_service_callback_state_for_tests() -> None:
    """Test helper — clear the holder + event so the unbound branch is
    exercisable. Not used in production; do NOT call from app code.
    """
    global _PR4_5_AGENT_SERVICE_FOR_CALLBACK, _PR4_5_AGENT_SERVICE_BIND_EVENT
    _PR4_5_AGENT_SERVICE_FOR_CALLBACK = None
    _PR4_5_AGENT_SERVICE_BIND_EVENT = None


async def _pr4_5_agent_service_callback(envelope) -> None:  # noqa: ANN001
    """C3 PR-4.5 supervisor → AgentService bridge (spec §7.6).

    Used by MailboxSupervisor handlers as ``ctx.agent_service_callback``.

    codex r2 [R2-1, CRITICAL ARCH] — handlers invoke this callback on
    MULTIPLE envelope types (SPAWN_REQUEST stub-dispatch, PROGRESS_UPDATE
    dispatch, ResultReadyHandler post-destroy wake, CancelAckHandler
    post-destroy wake, CancelRequestHandler TERMINATE step 1, direct-kill
    fallback). Only the TERMINATE step 1 path requires actually stopping
    the child's agent task; the others are pure wake/dispatch signals.
    A blanket ``stop_session`` on any envelope with ``child_session_id``
    would treat every heartbeat as a user-cancel.

    Behavior:

    * ``CANCEL_REQUEST`` with ``policy=TERMINATE`` ⇒ call
      ``AgentService.stop_session(child_id, is_admin=True)`` so the
      child's asyncio task is cancelled cooperatively BEFORE the
      supervisor's destroy side-effect runs (spec §7.6 stop-before-destroy).
      ``stop_session`` is fire-and-signal: ``task.cancel()`` propagates
      but the runner's terminal cleanup is async. The destroy that
      follows races a still-draining task — this is documented best-
      effort behavior; a stricter wait-for-terminal would require a new
      drain primitive on AgentService (deferred to PR-5).
    * If AgentService is not bound yet (TERMINATE arrives during the
      reconcile_orphans → bind window in lifespan startup), the call
      WAITS on the module-level bind event with a
      ``_PR4_5_BIND_WAIT_TIMEOUT_SECONDS`` cap. Timeout logs an error
      so the violation is auditable; destroy still proceeds because
      handlers swallow callback exceptions.
    * ``CANCEL_REQUEST`` with ``policy=REQUEST_CANCEL`` ⇒ no
      ``stop_session`` call (per spec §8.2 the child decides whether to
      cooperate). A future PR-5 cooperative-cancel dispatcher may add
      a non-destructive child-side signal here.
    * Any other envelope type (SPAWN_REQUEST / SPAWN_ACK /
      PROGRESS_UPDATE / RESULT_READY / CANCEL_ACK / APPROVAL_* /
      HANDOFF_REQUEST) ⇒ no-op. Heartbeat / progress envelopes do NOT
      carry a stop intent; ResultReady / CancelAck callbacks fire
      AFTER destroy and the spec uses them only as wake signals which
      the legacy notification path already handles.
    * Exceptions are logged and swallowed: the caller (handler) still
      executes the destroy side-effect; a failed callback must not
      block destroy.
    """
    # codex r2 [R2-1, CRITICAL ARCH] — type/policy gate (before the
    # bind-wait so non-TERMINATE envelopes are cheap no-ops even during
    # startup).
    from app.domain.models.mailbox_envelope import (
        CancelPolicy,
        MailboxEnvelopeType,
    )
    if getattr(envelope, "type", None) != MailboxEnvelopeType.CANCEL_REQUEST:
        return
    payload = getattr(envelope, "payload", None) or {}
    policy = payload.get("policy") if isinstance(payload, dict) else None
    if policy not in (CancelPolicy.TERMINATE.value, CancelPolicy.TERMINATE):
        return

    svc = _PR4_5_AGENT_SERVICE_FOR_CALLBACK
    if svc is None:
        # codex r3 [R3-1, HIGH ARCH] / codex r5 [R5-1] — block briefly
        # on a per-loop bind event so a TERMINATE consumed during
        # ``reconcile_orphans`` (which runs BEFORE
        # ``_build_agent_service`` in main.py lifespan) is still
        # processed correctly. ``_ensure_bind_event`` creates the
        # event inside the current loop on first call so the multi-
        # loop pytest harness doesn't see "Event bound to different
        # loop" errors.
        ev = _ensure_bind_event()
        try:
            await asyncio.wait_for(
                ev.wait(),
                timeout=_PR4_5_BIND_WAIT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "agent_service_callback: timed out waiting for "
                "AgentService bind on TERMINATE envelope=%s; destroy "
                "will proceed without cooperative stop (spec §7.6 "
                "violated — investigate lifespan misordering)",
                getattr(envelope, "envelope_id", "?"),
            )
            return
        svc = _PR4_5_AGENT_SERVICE_FOR_CALLBACK
        if svc is None:
            return

    sid = getattr(envelope, "child_session_id", None)
    if not sid:
        return
    try:
        session = await svc.get_session(sid)
        if session is None:
            # Already deleted — destroy will be a no-op anyway.
            return
        # codex r8 [R8-2] / r14 [R14-1, HIGH SEC] — defense in depth:
        # only stop sessions whose row is actually a mailbox-plane
        # subagent AND whose parent matches the envelope's claimed
        # parent. ``is_admin=True`` bypasses ownership checks (this
        # is a system-driven cancel), so without these gates a
        # malformed or cross-root envelope could end an unrelated
        # session. The supervisor's child→root mapping is trusted
        # publisher-side (per mailbox_supervisor.py docstring), so
        # the callback enforces the parent-match here as the last
        # line of defense.
        if (
            getattr(session, "worker_type", None) != "subagent"
            or getattr(session, "subagent_control_plane", None) != "mailbox"
        ):
            logger.warning(
                "agent_service_callback: refusing stop_session for sid=%s "
                "envelope_id=%s — row is not a mailbox-plane subagent "
                "(worker_type=%s, subagent_control_plane=%s); destroy will "
                "still proceed but cooperative stop is skipped",
                sid,
                getattr(envelope, "envelope_id", "?"),
                getattr(session, "worker_type", None),
                getattr(session, "subagent_control_plane", None),
            )
            return
        env_parent = getattr(envelope, "parent_session_id", None)
        row_parent = getattr(session, "parent_session_id", None)
        if env_parent and row_parent and env_parent != row_parent:
            logger.warning(
                "agent_service_callback: refusing stop_session for sid=%s "
                "envelope_id=%s — envelope.parent_session_id=%s does not "
                "match session.parent_session_id=%s (cross-root envelope "
                "or routing bug); destroy will still proceed but "
                "cooperative stop is skipped",
                sid,
                getattr(envelope, "envelope_id", "?"),
                env_parent,
                row_parent,
            )
            return
        user_id = str(session.user_id) if session.user_id else "system"
        # codex r15 [R15-1] / r16 [R16-2] — register the session in
        # the supervisor-initiated marker set BEFORE calling
        # stop_session so the runner's terminal path skips its own
        # CANCEL_ACK emission. Otherwise the child publishes
        # ``CANCEL_ACK(cancelled)`` AND the supervisor synthesizes
        # ``CANCEL_ACK(force_terminated)`` for the same correlation,
        # producing duplicate destroy invocations and confused audit.
        # Marker primitive lives in the domain layer
        # (see ``app.domain.services.supervisor_terminate_marker``)
        # so the runner-side consumer doesn't reverse-import
        # interfaces.
        _add_supervisor_terminate_marker(sid)
        await svc.stop_session(sid, user_id=user_id, is_admin=True)
    except Exception:
        logger.warning(
            "agent_service_callback stop_session failed sid=%s envelope_id=%s",
            sid,
            getattr(envelope, "envelope_id", "?"),
            exc_info=True,
        )


_supervisor_registry_singleton: "SupervisorRegistry | None" = None
_supervisor_registry_lock = threading.Lock()


def get_supervisor_registry(request: HTTPConnection) -> "SupervisorRegistry | None":
    """Return the lifespan-scoped :class:`SupervisorRegistry`.

    Post-PR-6 the registry is built unconditionally at lifespan startup
    (the ``mailbox_supervisor_enabled`` rollback is decommissioned per
    spec §11.7). The return type stays ``Optional`` for defensiveness:
    tests using stub ``app.state`` and lifespan-bypass code paths still
    return ``None`` and DI callers handle that gracefully.
    """
    return getattr(request.app.state, "supervisor_registry", None)


def _build_agent_service(
    minio_store: MinioStore,
    redis_client: RedisClient,
    checkpointer_pool: AsyncConnectionPool,
    flush_service: object | None,
    memory_embedding_provider: object | None,
    file_memory_store: object | None = None,
    sandbox_lifecycle_service: object | None = None,
    supervisor_registry: "SupervisorRegistry | None" = None,
) -> AgentService:
    """Called once in lifespan. Creates AgentService singleton and seeds generation."""
    global _last_refresh_generation
    app_config = _load_app_config()
    snapshot = _build_config_snapshot(app_config)

    # Build a lifespan-scoped MemoryManagementService for the memory_save tool.
    # The per-request get_memory_management_service() factory constructs a
    # separate instance from app.state — both walk the same config/wiring so
    # behavior is identical, this one just lives for the app's lifetime so
    # AgentTaskRunner closures can keep a stable reference across session
    # ticks without re-reading app.state in the hot path.
    memory_redis_client = (
        redis_client.client if redis_client and hasattr(redis_client, "client") else None
    )
    # Emitter 先建，MemoryManagementService + AgentService 共享同一 instance。
    # DB-only 构造（无 Redis 依赖），gate-off 部署也能让 fs_permanent_failure
    # 通知走 notification_routes 对外契约。
    from app.application.services.memory_notification_emitter import (
        DBMemoryNotificationEmitter,
    )
    memory_notification_emitter = DBMemoryNotificationEmitter(
        session_factory=get_postgres().session_factory,
    )
    memory_write_service = MemoryManagementService(
        repo_factory=DBMemoryChunkRepository,
        embedding_provider=memory_embedding_provider,
        session_factory=get_postgres().session_factory,
        file_store=file_memory_store,
        redis=memory_redis_client,
        user_daily_quota=(
            settings.memory_user_daily_quota if memory_redis_client is not None else None
        ),
        # fs 写失败 → 发 fs_permanent_failure 通知（design §183 M1 contract）。
        # 没有 emitter 时 _try_emit_fs_failure_notification 会 no-op，legacy
        # 路径不受影响；但生产部署必须注入，否则 schema 宣称的 event_type
        # 永不写入，等同于空承诺。
        notification_emitter=memory_notification_emitter,
        # codex fix P1：legacy 清理时间边界——与上方 get_memory_management_service
        # 保持同源，两条 DI 路径（lifespan write service + per-request CRUD service）
        # 必须用同一语义。
        memory_gate_rollout_at=settings.memory_gate_rollout_at,
    )

    # M1 PR-4+8 gate: single in-process breaker shared across sessions (so
    # N concurrent flushes see the same "LLM is flaky" signal); daily cap
    # is a thin Redis wrapper (stateless). Both are None when Redis or
    # gate LLM aren't available, and PlannerReActFlow's gate path treats
    # None as legacy passthrough.
    from app.domain.services.memory_gate import (
        MemoryGateBreaker,
        MemoryGateDailyCap,
    )

    def _build_memory_gate_deps(snap: "_ConfigSnapshot"):
        """Derive breaker + daily_cap from a snapshot's memory_gate_llm.

        Factored out so AgentService._refresh_config can re-run it on
        config reload when the gate LLM identity changes (PR-4+8 bug:
        first build pinned breaker/cap at init time, hot-refresh of
        ``summary_model`` would leave them stale at None while the new
        snapshot's gate was enabled, bypassing both protections).
        """
        if snap.memory_gate_llm is None:
            return (None, None)
        breaker = MemoryGateBreaker()
        daily_cap = None
        if memory_redis_client is not None:
            daily_cap = MemoryGateDailyCap(
                redis=memory_redis_client,
                cap=settings.memory_gate_daily_cap,
            )
        return (breaker, daily_cap)

    memory_gate_breaker, memory_gate_daily_cap = _build_memory_gate_deps(snapshot)
    from app.domain.services.execution_supervisor import ExecutionSupervisor
    from app.infrastructure.observability import OtelMeter

    supervisor = ExecutionSupervisor(
        redis_client=redis_client,
        uow_factory=get_uow,
        meter=OtelMeter(),
        # C3 PR-3c (codex r6 [HIGH CONTRACT]) — wire the per-pod MailboxSupervisor
        # registry through so ExecutionSupervisor's non-runner terminal-write
        # paths (idle_watchdog cancel, FINISHING reconcile at boot) can call
        # ``registry.stop`` to prevent supervisor task leaks.
        supervisor_registry=supervisor_registry,
    )
    # Notification emitter is always constructible (DB-only, no Redis
    # dep); gate-off deployments just never call it.
    # C3 PR-4.5 — single lifespan-scoped publisher so every AgentTaskRunner
    # constructed by ``AgentService._create_task`` shares one Redis-bound
    # publisher rather than re-wrapping the raw client per task.
    from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
        RedisMailboxPublisher,
    )
    redis_inner = (
        redis_client.client if redis_client and hasattr(redis_client, "client") else None
    )
    mailbox_publisher = (
        RedisMailboxPublisher(redis_inner) if redis_inner is not None else None
    )
    agent_svc = AgentService(
        uow_factory=get_uow,
        config_snapshot=snapshot,
        sandbox_cls=DockerSandbox,
        task_cls=RedisStreamTask,
        search_engine=BingSearchEngine(),
        file_storage=MinioFileStorage(
            bucket=settings.minio_bucket_name,
            minio_store=minio_store,
            uow_factory=get_uow,
        ),
        redis_client=redis_client,
        checkpointer_pool=checkpointer_pool,
        memory_flusher=flush_service,
        memory_embedding_provider=memory_embedding_provider,
        memory_session_factory=get_postgres().session_factory,
        memory_repo_factory=DBMemoryChunkRepository,
        memory_write_service=memory_write_service,
        memory_session_save_cap=settings.memory_session_save_cap,
        memory_gate_breaker=memory_gate_breaker,
        memory_gate_daily_cap=memory_gate_daily_cap,
        memory_notification_emitter=memory_notification_emitter,
        memory_gate_rebuild_fn=_build_memory_gate_deps,
        event_recovery=RedisEventRecovery(),
        sandbox_lifecycle_service=sandbox_lifecycle_service,
        supervisor_registry=supervisor_registry,
        mailbox_publisher=mailbox_publisher,
    )
    agent_svc._supervisor = supervisor
    # C3 PR-4.5 — bind AgentService into the supervisor callback bridge
    # so ``_pr4_5_agent_service_callback`` can dispatch ``stop_session``
    # on the TERMINATE path. Lifespan ordering: this returns to main.py
    # before the supervisor consumer loop is started, so the bind always
    # precedes the first envelope dispatch.
    _bind_agent_service_for_callback(agent_svc)
    _last_refresh_generation = _config_generation
    return agent_svc


def get_skill_creator_service() -> SkillCreatorService:
    app_config = _load_app_config()
    llm = _build_llm(app_config.llm_config)
    github_client = GitHubSearchClient(token=settings.github_token or None)
    skill_service = _build_skill_service()
    return SkillCreatorService(
        llm=llm,
        github_client=github_client,
        skill_service=skill_service,
    )


def get_checkpointer_pool(request: Request) -> AsyncConnectionPool:
    """Extract the checkpointer connection pool from app state."""
    return request.app.state.checkpointer_pool.pool


def get_flush_service(request: Request):
    """Extract the MemoryFlushService from app state (may be None)."""
    return getattr(request.app.state, "flush_service", None)


def get_memory_embedding_provider(request: Request):
    """Extract memory embedding provider from app state (C4 构建)."""
    return getattr(request.app.state, "memory_embedding_provider", None)


def get_sandbox_lifecycle_service(request: Request):
    """Extract SandboxLifecycleService from app state (may be None)."""
    return getattr(request.app.state, "sandbox_lifecycle_service", None)


def get_agent_service(request: HTTPConnection) -> AgentService:
    """Return app.state singleton. Atomic refresh on config generation change."""
    global _last_refresh_generation
    agent_svc = request.app.state.agent_service

    _load_app_config()  # trigger possible generation increment

    # Fast path (no lock): generation unchanged
    # NOTE: safe under CPython GIL — int read is atomic. Double-check inside lock handles races.
    if _config_generation == _last_refresh_generation:
        return agent_svc

    # Slow path: lock, pin generation, build, commit
    with _refresh_lock:
        gen = _config_generation
        if gen <= _last_refresh_generation:
            return agent_svc
        snapshot = _build_config_snapshot(_config_cache)
        agent_svc._refresh_config(snapshot)
        _last_refresh_generation = gen
        logger.info("AgentService config refreshed (generation=%d)", gen)

    return agent_svc


def get_supervisor(request: HTTPConnection):
    """Return the lifespan-scoped ExecutionSupervisor singleton."""
    return request.app.state.supervisor


def get_idle_watchdog(request: HTTPConnection):
    """Return the lifespan-scoped IdleWatchdog singleton."""
    return request.app.state.idle_watchdog


def get_skill_export_service() -> SkillExportService:
    return SkillExportService(skills_root_dir=settings.skills_root_dir)


def get_memory_management_service(
    request: Request,
) -> MemoryManagementService:
    """DI factory for :class:`MemoryManagementService`.

    - ``memory_embedding_provider`` 在 main.py 启动 lifespan 中初始化并挂在
      ``app.state``（可能是 CircuitBreaker 包装器，也可能是 DisabledEmbeddingProvider）。
    - ``session_factory`` 来自 ``postgres_client.session_factory``，不是 ``app.state``。
    - ``repo_factory`` 传类本身（``DBMemoryChunkRepository``），Service 内部会用
      AsyncSession 实例化。
    - ``redis`` / ``user_daily_quota`` 是 PR-2 的 memory 写入配额；Redis 客户端
      在 lifespan 里通过 ``get_redis()`` 单例初始化，此处只是拿引用。测试场景里
      fake app.state 不会有 redis_client，用 getattr 兜底。
    """
    from app.infrastructure.storage.redis import get_redis

    postgres_client = get_postgres()
    # file_store 在 PR-0 恒为 None（DB-only），PR-5A 由 lifespan 注入真实的
    # FsMemoryWriter。用 getattr 而非属性访问以兼容测试环境（fake app 可能
    # 未设置该属性）。
    file_store = getattr(request.app.state, "file_memory_store", None)

    # Redis 未初始化（单测绕过 lifespan）→ 同时把 quota 置 None，
    # 符合 MemoryManagementService "redis 和 quota 同传或同省略" 的契约，
    # 否则会在 __init__ 里直接 ValueError 把请求打成 500。
    redis_client = None
    user_daily_quota: int | None = None
    try:
        redis_client = get_redis().client
        user_daily_quota = settings.memory_user_daily_quota
    except RuntimeError:
        redis_client = None
        user_daily_quota = None

    # Emitter 每次 request 新建——构造是廉价的（只抓 session_factory 引用），
    # 避免 app.state 共享引入的 lifespan 初始化耦合（测试若绕过 lifespan 构造
    # app 时，app.state 可能没有 memory_notification_emitter 属性）。
    from app.application.services.memory_notification_emitter import (
        DBMemoryNotificationEmitter,
    )
    notification_emitter = DBMemoryNotificationEmitter(
        session_factory=postgres_client.session_factory,
    )
    return MemoryManagementService(
        repo_factory=DBMemoryChunkRepository,
        embedding_provider=request.app.state.memory_embedding_provider,
        session_factory=postgres_client.session_factory,
        file_store=file_store,
        redis=redis_client,
        user_daily_quota=user_daily_quota,
        notification_emitter=notification_emitter,
        # codex fix P1：legacy 清理时间边界。settings 未设 → None，
        # service 沿用旧谓词 + 前端 dialog 警告；设了 → SQL 加
        # AND created_at < rollout_at。
        memory_gate_rollout_at=settings.memory_gate_rollout_at,
    )


async def get_memory_system_notification_repository(
    db_session: AsyncSession = Depends(get_db_session),
):
    """Notification repo DI — session-scoped with explicit commit.

    Two endpoints call this: GET /unread (pure read) and POST /mark-read
    (single-row UPDATE). No multi-repo coordination, no embedding, no
    cross-layer invariants ⇒ skip a dedicated service. ``get_db_session``
    already owns the rollback-on-exception + close-on-finally lifecycle;
    we only add an explicit commit so mark-read actually sticks. If the
    endpoint raises before the ``yield repo`` completes, get_db_session's
    ``except`` branch rolls back and the ``await db_session.commit()``
    line is unreachable, so we don't double-commit.
    """

    repo = DBMemorySystemNotificationRepository(db_session)
    yield repo
    await db_session.commit()


# ----------------------------------------------------------------------
# R5b-1: ApprovalState DI wiring
# ----------------------------------------------------------------------
#
# Writer 和 Reader 的构造极轻：writer 只吃 ``uow_factory``；reader 吃两个
# Protocol adapter（``UowApprovalGrantQuery`` 始终注入；``SessionLegacyRuleQuery``
# 按 ``AppConfig.agent_config.tool_confirmation.legacy_rule_fallback`` 开关）。
#
# 这两个 factory **在 R5b-1 阶段不被任何 endpoint 直接消费**——R5b-1 只暴露
# DI 入口；R5b-2/3/4 会把 AgentService / AgentTaskRunner / planner_react 切到
# 这两条路径上。先落 factory 可以让后续 PR 只动 callsite 不动 DI 层。
def get_approval_state_writer():
    """R5 CS4 单一 Writer 工厂。无状态，per-request 构造。"""
    from app.application.services.approval_state_writer import ApprovalStateWriter

    return ApprovalStateWriter(uow_factory=get_uow)


def get_approval_state_reader():
    """R5 CS4 Reader 工厂。

    ``legacy_rule_fallback=True`` 时注入 ``SessionLegacyRuleQuery``，Reader 在
    grants miss 后查旧 ``tool_approval_rules`` 表。``False`` 时不注入，miss
    直接返 ``no_match``。切断 fallback 的时机由运维通过 config 热刷控制，不走
    环境变量旁路。
    """
    from app.application.services.approval_state_adapters import (
        SessionLegacyRuleQuery,
        UowApprovalGrantQuery,
    )
    from app.domain.services.approval_state_reader import ApprovalStateReader

    app_config = _load_app_config()
    grant_query = UowApprovalGrantQuery(uow_factory=get_uow)
    legacy_query = None
    if app_config.agent_config.tool_confirmation.legacy_rule_fallback:
        legacy_query = SessionLegacyRuleQuery(
            session_factory=get_postgres().session_factory,
        )
    return ApprovalStateReader(query=grant_query, legacy_rule_query=legacy_query)


def get_user_tool_approval_policy_service(
    db_session: AsyncSession = Depends(get_db_session),
) -> UserToolApprovalPolicyService:
    """R6 §8.2 — DI factory for ApprovalPolicy service. Overridable via
    `app.dependency_overrides` in tests (see Task 8 pattern)."""
    return UserToolApprovalPolicyService(
        DBUserToolApprovalPolicyRepository(db_session),
    )


# ----------------------------------------------------------------------
# B4 M0 Phase I: cost aggregation DI wiring
# ----------------------------------------------------------------------


def get_cost_record_repository(
    db_session: AsyncSession = Depends(get_db_session),
):
    """Per-request cost record repository backed by the current DB session."""
    from app.infrastructure.repositories.db_cost_record_repository import (
        DbCostRecordRepository,
    )

    return DbCostRecordRepository(db_session)


def get_cost_aggregation_service(
    db_session: AsyncSession = Depends(get_db_session),
):
    """B4 M0: CostAggregationService for GET /sessions/{id}/cost."""
    from app.application.services.cost_aggregation_service import (
        CostAggregationService,
    )
    from app.infrastructure.repositories.db_cost_record_repository import (
        DbCostRecordRepository,
    )

    return CostAggregationService(DbCostRecordRepository(db_session))


# ----------------------------------------------------------------------
# C1a: SessionRepository + SessionCostTreeService DI wiring
# ----------------------------------------------------------------------


def get_session_repository(
    db_session: AsyncSession = Depends(get_db_session),
):
    """C1a: per-request SessionRepository for endpoints that don't need a full UoW."""
    from app.infrastructure.repositories.db_session_repository import (
        DBSessionRepository,
    )

    return DBSessionRepository(db_session=db_session)


def get_session_cost_tree_service(
    session_repo=Depends(get_session_repository),
    cost_repo=Depends(get_cost_record_repository),
    cost_aggregator=Depends(get_cost_aggregation_service),
):
    """C1a: SessionCostTreeService composed of 3 deps. FastAPI Depends cache
    guarantees ``db_session`` is shared between session_repo and cost_repo within
    the same request.
    """
    from app.application.services.session_cost_tree_service import (
        SessionCostTreeService,
    )

    return SessionCostTreeService(
        session_repo=session_repo,
        cost_repo=cost_repo,
        cost_aggregator=cost_aggregator,
    )


# ----------------------------------------------------------------------
# PE-0 Phase 7: ConfirmationQueue DI factory
# ----------------------------------------------------------------------


def get_confirmation_queue(
    redis_client: RedisClient = Depends(get_redis),
) -> "ConfirmationQueue":
    """Return a ConfirmationQueue backed by the raw redis.asyncio.Redis client.

    C-R2-P0-3 correction: ConfirmationQueue.__init__ takes the raw redis
    client (redis.asyncio.Redis), NOT the RedisClient wrapper. Unwrap via
    ``.client`` before passing in.
    """
    from app.domain.services.permission.confirmation_queue import ConfirmationQueue

    return ConfirmationQueue(redis_client.client)


# ----------------------------------------------------------------------
# C3 PR-2: MailboxPublisher DI factory
# ----------------------------------------------------------------------


def get_mailbox_publisher(
    redis_client: RedisClient = Depends(get_redis),
) -> MailboxPublisher:
    """Return a :class:`RedisMailboxPublisher` bound to the raw
    ``redis.asyncio.Redis`` client.

    Same unwrap pattern as :func:`get_confirmation_queue` —
    ``RedisMailboxPublisher.__init__`` accepts ``redis.asyncio.Redis``, not
    the ``RedisClient`` wrapper. The publisher only needs SET / XADD; no
    pubsub or pipeline state is shared with other Redis users, so the same
    singleton ``RedisClient.client`` is safe to reuse.

    Consumer-side (:class:`RedisMailboxConsumer`) is **not** wired here —
    it's constructed per-supervisor in PR-3a (one consumer per root
    session) and never injected through FastAPI Depends.
    """
    from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
        RedisMailboxPublisher,
    )

    return RedisMailboxPublisher(redis_client.client)


def build_cost_callback_handler(session_id: str, user_id: str):
    """B4 M0: thin wrapper around the application-layer factory.

    Uses the global ``get_uow`` UoW factory; the real implementation (which
    agent_service uses with ``self._uow_factory``) lives in
    ``application/services/cost_callback_factory.py`` so the layering stays
    clean (no interfaces → application reverse imports).
    """
    from app.application.services.cost_callback_factory import (
        build_cost_callback_handler as _build,
    )

    return _build(session_id=session_id, user_id=user_id, uow_factory=get_uow)


# ----------------------------------------------------------------------
# Phase 1 minimal subagent research probe DI wiring
# ----------------------------------------------------------------------
#
# Five factories wire ``SubagentResearchService`` through FastAPI's
# ``Depends()`` graph. ``summary_llm`` is read from the same
# ``_ConfigSnapshot`` the main chat pipeline uses (canonical path; spec
# v3 fix per Codex round 2 P0#2 — there is no ``settings.summary_llm``
# attribute on Settings). ``TokenEstimator`` is constructed with a fixed
# ``"hybrid"`` strategy here because a configurable estimator field is
# not currently exposed on ``Settings``; if a future PR needs another
# strategy, add the field and read it here.


def get_token_estimator() -> "TokenEstimator":
    """Phase 1 minimal: TokenEstimator instance using hybrid strategy.

    No mutable state after __init__ — safe to construct per-request.
    """
    from app.domain.services.graphs.token_estimator import TokenEstimator

    return TokenEstimator(strategy="hybrid")


def get_summary_llm() -> "BaseChatModel":
    """Phase 1 minimal: summary join LLM from the live config snapshot.

    Resolves through ``_load_app_config`` + ``_build_config_snapshot``
    so a hot config reload (which updates ``_config_generation``) is
    honored by the next request. Returns ``None`` only if the config
    has no ``agent_config.memory.summary_model`` — caller should treat
    that as a misconfiguration (the probe service requires summary_llm
    for the join step and will surface a clear error downstream).
    """
    snapshot = _build_config_snapshot(_load_app_config())
    return snapshot.summary_llm


def get_subagent_research_classifier(
    summary_llm: "BaseChatModel" = Depends(get_summary_llm),
) -> "SubagentResearchClassifier":
    """Classifier reuses the summary LLM (single LLM, no separate config)."""
    from app.domain.services.subagent_research_classifier import (
        SubagentResearchClassifier,
    )

    return SubagentResearchClassifier(llm=summary_llm)


def get_probe_quota_service(
    redis_client: RedisClient = Depends(get_redis),
) -> "ProbeQuotaService":
    """Phase 1 minimal: per-user active probe quota (atomic Lua acquire/release)."""
    from app.infrastructure.cache.probe_quota import ProbeQuotaService

    return ProbeQuotaService(redis_client=redis_client)


def get_subagent_research_service(
    session_service: SessionService = Depends(get_session_service),
    agent_service: AgentService = Depends(get_agent_service),
    supervisor=Depends(get_supervisor),
    token_estimator: "TokenEstimator" = Depends(get_token_estimator),
    summary_llm: "BaseChatModel" = Depends(get_summary_llm),
    classifier: "SubagentResearchClassifier" = Depends(
        get_subagent_research_classifier
    ),
    sandbox_lifecycle_service=Depends(get_sandbox_lifecycle_service),
    quota_service: "ProbeQuotaService" = Depends(get_probe_quota_service),
    supervisor_registry: "SupervisorRegistry | None" = Depends(get_supervisor_registry),
    mailbox_publisher: MailboxPublisher = Depends(get_mailbox_publisher),
) -> "SubagentResearchService":
    """Phase 1 minimal: research probe orchestrator.

    Per-request construction is cheap (the service is stateless across
    requests — each ``run_research`` call owns its own probe_run_id +
    children list + tasks). Sharing across requests would require
    splitting hot dependencies (e.g. supervisor) from per-request scope.
    """
    from app.application.services.subagent_research_service import (
        SubagentResearchService,
    )

    return SubagentResearchService(
        session_service=session_service,
        agent_service=agent_service,
        execution_supervisor=supervisor,
        token_estimator=token_estimator,
        summary_llm=summary_llm,
        classifier=classifier,
        sandbox_lifecycle_service=sandbox_lifecycle_service,
        quota_service=quota_service,
        # C3 PR-4.5 — mailbox-plane wiring (registry may be None when the
        # supervisor flag is off; publisher always available since it's a
        # thin Redis wrapper).
        supervisor_registry=supervisor_registry,
        mailbox_publisher=mailbox_publisher,
    )
