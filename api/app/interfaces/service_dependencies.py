import hashlib
import logging
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
    return SessionService(
        uow_factory=get_uow,
        task_cls=RedisStreamTask,
        sandbox_lifecycle_service=lifecycle_service,
        fs_reconciler=fs_reconciler,
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


def _build_agent_service(
    minio_store: MinioStore,
    redis_client: RedisClient,
    checkpointer_pool: AsyncConnectionPool,
    flush_service: object | None,
    memory_embedding_provider: object | None,
    file_memory_store: object | None = None,
    sandbox_lifecycle_service: object | None = None,
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
    # Notification emitter is always constructible (DB-only, no Redis
    # dep); gate-off deployments just never call it.
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
    )
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
