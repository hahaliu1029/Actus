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
    return SessionService(
        uow_factory=get_uow,
        task_cls=RedisStreamTask,
        sandbox_lifecycle_service=lifecycle_service,
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

        timeout_seconds = llm_config.timeout_seconds
        connect_timeout_seconds = llm_config.connect_timeout_seconds

        chat = ActusChatModel(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key,
            model_name=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            supports_response_format=getattr(llm_config, 'supports_response_format', True),
            supports_vision=getattr(llm_config, 'supports_vision', True),
            supports_pdf_input=supports_pdf_input,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        responses = ActusResponsesModel(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key,
            model_name=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            supports_vision=getattr(llm_config, 'supports_vision', True),
            supports_pdf_input=supports_pdf_input,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
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
            llm = ActusFallbackChatModel(primary=chat, fallback=responses)
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
    effective_pdf_input = (
        getattr(app_config.llm_config, 'supports_pdf_input', False)
        and app_config.llm_config.supports_vision
    )
    llm = _build_llm(app_config.llm_config, supports_pdf_input=effective_pdf_input)

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
        supports_vision=app_config.llm_config.supports_vision,
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
    memory_write_service = MemoryManagementService(
        repo_factory=DBMemoryChunkRepository,
        embedding_provider=memory_embedding_provider,
        session_factory=get_postgres().session_factory,
        file_store=file_memory_store,
        redis=memory_redis_client,
        user_daily_quota=(
            settings.memory_user_daily_quota if memory_redis_client is not None else None
        ),
    )

    # M1 PR-4+8 gate: single in-process breaker shared across sessions (so
    # N concurrent flushes see the same "LLM is flaky" signal); daily cap
    # is a thin Redis wrapper (stateless). Both are None when Redis or
    # gate LLM aren't available, and PlannerReActFlow's gate path treats
    # None as legacy passthrough.
    from app.application.services.memory_notification_emitter import (
        DBMemoryNotificationEmitter,
    )
    from app.domain.services.memory_gate import (
        MemoryGateBreaker,
        MemoryGateDailyCap,
    )
    memory_gate_breaker = MemoryGateBreaker() if snapshot.memory_gate_llm else None
    memory_gate_daily_cap: MemoryGateDailyCap | None = None
    if snapshot.memory_gate_llm and memory_redis_client is not None:
        memory_gate_daily_cap = MemoryGateDailyCap(
            redis=memory_redis_client,
            cap=settings.memory_gate_daily_cap,
        )
    # Notification emitter is always constructible (DB-only, no Redis
    # dep); gate-off deployments just never call it.
    memory_notification_emitter = DBMemoryNotificationEmitter(
        session_factory=get_postgres().session_factory,
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

    return MemoryManagementService(
        repo_factory=DBMemoryChunkRepository,
        embedding_provider=request.app.state.memory_embedding_provider,
        session_factory=postgres_client.session_factory,
        file_store=file_store,
        redis=redis_client,
        user_daily_quota=user_daily_quota,
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
