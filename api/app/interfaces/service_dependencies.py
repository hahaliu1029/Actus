import asyncio
import hashlib
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from app.application.services.agent_service import AgentService
from app.application.services.app_config_service import AppConfigService
from app.application.services.cost_rollup_service import CostRollupService
from app.application.services.file_service import FileService
from app.application.services.memory_management_service import MemoryManagementService
from app.application.services.plugin_install_service import PLUGIN_STORE_ROOT
from app.application.services.runtime_extension_service import (
    RuntimeExtensionService,
)
from app.infrastructure.external.governance.plugin_display_name import (
    read_plugin_display_name,
)
from app.application.services.session_service import SessionService
from app.application.services.user_tool_enablement_service import (
    UserToolEnablementService,
)
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
from app.domain.external.policy_snapshot_sink import PolicySnapshotSink
from app.domain.models.app_config import LLMConfig, SkillRiskPolicy
from app.domain.repositories.coordinator_result_envelope_store_repository import (
    CoordinatorResultEnvelopeStoreRepository,
)
from app.domain.repositories.subagent_run_repository import SubagentRunRepository
from app.application.services.agent_service import _ConfigSnapshot
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel
from langchain_core.language_models import BaseChatModel
from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox
from app.infrastructure.external.sandbox.parent_sandbox_adapter import ParentSandboxAdapter
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
from app.infrastructure.repositories.db_user_tool_enablement_repository import (
    DBUserToolEnablementRepository,
)
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.repositories.file_team_repository import FileTeamRepository
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

_ORPHAN_OWNER_COMPARE_DELETE_SCRIPT = """
-- coordinator-orphan-owner-compare-delete-v1
local key = KEYS[1]
if redis.call('GET', key) ~= ARGV[1] then
  return 0
end
return redis.call('DEL', key)
"""

# --- Config cache (D2) ---
_config_cache: "AppConfig | None" = None
_config_mtime: float = 0.0
_config_size: int = 0
_config_expiry: float = 0.0
_config_generation: int = 0
_config_lock = threading.Lock()


def get_app_config_service(request: Request) -> AppConfigService:
    """获取应用配置服务（R3#15：FastAPI 依赖形态——注入 D1a 治理 reconciler/read_port）。

    off（默认 ``EXTENSION_GOVERNANCE_MODE=off``）→ app.state 上两句柄均 None → 双 None
    注入 → 六个写方法行为与治理接线前 byte 级一致。非 request 上下文（lifespan/T23
    closure 局部构造）继续直构 ``AppConfigService(repo, reconciler=..., registry_read_port=...)``。
    ``@lru_cache`` 不再适用（Request 不可哈希）——所有消费处均经 ``Depends`` 透明适配。
    """
    # 1.获取数据仓库并打印日志
    logger.info("加载获取AppConfigService")
    file_app_config_repository = FileAppConfigRepository(settings.app_config_filepath)

    # 2.实例化AppConfigService（治理挂钩 off=None 零行为变化）
    return AppConfigService(
        app_config_repository=file_app_config_repository,
        reconciler=getattr(request.app.state, "extension_reconciler", None),
        registry_read_port=getattr(request.app.state, "extension_registry_read_port", None),
    )


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
        # SPM PR-1c Task 17: provision metrics for vnc/takeover trigger surfaces.
        sandbox_provision_metrics=getattr(
            request.app.state, "sandbox_provision_metrics", None
        ),
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


def get_runtime_extension_service(
    request: Request,
    db_session: AsyncSession = Depends(get_db_session),
) -> RuntimeExtensionService:
    """B9：聚合器无状态每请求新建；probe/stats 单例句柄从 app.state 读（PR-2/3 起非 None）。"""
    return RuntimeExtensionService(
        config_provider=_load_app_config,
        # 无独立工厂——skill repo 按 get_skill_service 同款构造
        # （service_dependencies.py `SkillService(FileSkillRepository(settings.skills_root_dir))`）；
        # enablement 按 user_tools_v2_routes.py:31-33 同款
        # （`UserToolEnablementService(DBUserToolEnablementRepository(db_session))`）。
        skill_repository=FileSkillRepository(settings.skills_root_dir),
        enablement_service=UserToolEnablementService(
            DBUserToolEnablementRepository(db_session)
        ),
        probe_view=getattr(request.app.state, "extension_probe_service", None),
        liveness_view=getattr(request.app.state, "runtime_liveness_registry", None),
        stats_reader=getattr(request.app.state, "extension_stats", None),
        # D1a Task 25：治理只读 port（mode=off → None，INV-D1-0 零行为）+ plugin 展示名
        # resolver（helper 偏函数 bind PLUGIN_STORE_ROOT；registry 无 name 列 → best-effort
        # 读 plugin.json，失败 fallback ext_id）。
        registry_read_port=getattr(
            request.app.state, "extension_registry_read_port", None
        ),
        plugin_name_resolver=partial(read_plugin_display_name, PLUGIN_STORE_ROOT),
        # 有效能力 = 后台任务启动成功 AND 当前 flag on（flag 经 TTL config loader 动态读取，
        # 支持运行中翻转——R5#3/R18#1/R1#3 统一口径）
        probe_enabled_provider=lambda: (
            bool(getattr(request.app.state, "extension_probe_started", False))
            and _load_app_config().tool_runtime.extension_probe_enabled
        ),
        stats_enabled_provider=lambda: (
            bool(getattr(request.app.state, "extension_stats_started", False))
            and _load_app_config().tool_runtime.extension_stats_enabled
        ),
    )


def get_extension_probe_service(request: Request):
    """B9：手动探测端点（Task 15/16）用——直接返回 lifespan 起的单例句柄。

    句柄由 ``main._start_b9_probe`` 挂到 ``app.state.extension_probe_service``（后台
    task 无条件构造）；fail-open 启动失败时为 ``None``。有效能力（flag on 且循环启动
    成功）由端点层二次复核 / ``probe_flag_provider`` 判定——本 getter 只做句柄穿透，
    不在此合成能力（与 ``get_runtime_extension_service`` 的 probe_enabled_provider
    口径分离：GET 走聚合器 stub，手动探测走真单例）。
    """
    return getattr(request.app.state, "extension_probe_service", None)


def get_extension_registry_read_port(request: Request):
    """D1a Task 13（G6 façade §4.2）：治理只读 port 单例句柄穿透。

    句柄由 lifespan 治理段构造挂到 ``app.state.extension_registry_read_port``
    （mode≠off → ``DbExtensionRegistryReadPort``；mode=off → ``None``，INV-D1-0）。
    与 ``get_extension_probe_service`` 同款——只做句柄穿透，不在此合成能力：G6 门
    读到 ``None`` 即恒等直通（治理关闭时装配咽喉零行为）。
    """
    return getattr(request.app.state, "extension_registry_read_port", None)


def get_extension_install_service(request: Request):
    """D1a Task 19：MCP/A2A 两阶段安装管道单例句柄穿透（§7.1/§9.2）。

    句柄由 lifespan 治理段构造挂到 ``app.state.extension_install_service``
    （``mode≠off`` → ``ExtensionInstallService``；``mode=off`` → ``None``，INV-D1-0）。
    路由 ``Depends`` 的真实注入面：读到 ``None`` → 走旧直通（治理关闭零行为变化）；
    非 ``None`` → dry_run/commit 两阶段管道。挂 app.state 本身不满足 ``Depends``——
    本 provider 是承载面（R7#2）。"""
    return getattr(request.app.state, "extension_install_service", None)


def get_extension_governance_service(request: Request):
    """D1a Task 20：§9.2 治理服务单例句柄穿透（行政动作 + 观测刷新 + 审计翻页）。

    句柄由 lifespan 治理段构造挂到 ``app.state.extension_governance_service``
    （``mode≠off`` → ``ExtensionGovernanceService``；``mode=off`` → ``None``，INV-D1-0）。
    治理路由 ``Depends`` 的真实注入面（R7#2）：读到 ``None`` → 除 ``GET /governance``
    返回字面量零外一律 409 ``governance_disabled``；挂 app.state 本身不满足 ``Depends``。"""
    return getattr(request.app.state, "extension_governance_service", None)


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
            provider=vf.provider,
            supports_response_format=vf.supports_response_format,
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
        lifecycle_runtime=app_config.lifecycle_runtime,
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
    coordinator_envelope_store: CoordinatorResultEnvelopeStoreRepository,
    cost_rollup_service: CostRollupService,
    coordinator_liveness_service: object | None = None,
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
    from app.application.composition.graph_assembly import (
        build_session_state_machine,
    )
    from app.application.services.coordinator_terminal_transition import (
        CoordinatorTerminalCommand,
        terminalize_authoritative_coordinator_child,
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
    terminal_state_machine = build_session_state_machine(uow_factory=get_uow)

    async def _terminalize_child(
        command: CoordinatorTerminalCommand,
    ) -> bool:
        """Composition adapter: authoritative Supervisor terminal CAS.

        ``MailboxSupervisor`` stays application-layer and sees only this
        callable port. The locked authority read, validation, CAS and commit
        share the same composition-owned UoW.
        """
        return await terminalize_authoritative_coordinator_child(
            command,
            state_machine=terminal_state_machine,
            uow_factory=get_uow,
        )

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

        async def find_running_mailbox_children_for_parent(
            self, parent_session_id: str,
        ):
            from app.infrastructure.repositories.db_session_repository import (
                DBSessionRepository,
            )
            async with self._session_factory() as db_session:
                repo = DBSessionRepository(db_session=db_session)
                return await repo.find_running_mailbox_children_for_parent(
                    parent_session_id,
                )

    supervisor_session_repo = _SupervisorSessionRepoAdapter(
        session_factory=pg_session_factory,
    )
    if coordinator_liveness_service is None:
        from app.application.services.coordinator_liveness_lease_service import (
            CoordinatorLivenessLeaseService,
        )
        coordinator_liveness_service = CoordinatorLivenessLeaseService(
            redis=raw_redis,
            session_repository=supervisor_session_repo,  # type: ignore[arg-type]
        )

    # [C4.1a §5.1] repo-or-None：flag OFF → None（不构造 → 两 seat 门跳过 → INV-0）。
    # repo stateless → 建一次复用（match supervisor_session_repo）。
    subagent_run_repo = get_subagent_run_repository(settings_local)

    def _factory(root_session_id: str) -> MailboxSupervisor:
        # [C2 deferred wiring -- PR-9 composition root]
        # The following SupervisorContext-adjacent emit hooks remain
        # deferred to PR-9b-A6; the SupervisorContext ports for
        # ``cost_rollup_service`` (PR-6 §14.4) and
        # ``coordinator_envelope_store`` (PR-7 §12.4) are now wired
        # below (PR-9b-A4 INV-A1/A2). Wiring is always-live; the runtime
        # feature flag (``ACTUS_C2_COORDINATOR_ENABLED``) gates ENTRY
        # (whether coordinator behaviour fires), not WIRING.
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
        # The PR-6 cost-rollup PROLOGUE and PR-7 persist-terminal
        # PROLOGUE no longer silently no-op in production: their
        # SupervisorContext slots are populated unconditionally by this
        # factory. See ``mailbox_supervisor.py`` ResultReadyHandler /
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
            terminalize_child=_terminalize_child,
            # C3 PR-5 / Task 4 — authoritative session reader shared by the
            # rollback stop check and terminal-ownership row guard. Adapter
            # opens a short-lived AsyncSession per query.
            session_repo=supervisor_session_repo,  # type: ignore[arg-type]
            # PR-9b-A4 (INV-A1 / INV-A2) — coordinator ports wired at the
            # composition root. Wiring is always-live; the feature flag
            # only gates ENTRY (whether coordinator behaviour fires), not
            # WIRING (whether deps are threaded through). The
            # SupervisorContext fields remain Optional so legacy unit
            # tests that construct contexts without DI still work.
            coordinator_envelope_store=coordinator_envelope_store,
            cost_rollup_service=cost_rollup_service,
            # [C4.1a §5.1] subagent-run observation sink (repo-or-None built once
            # in the outer scope; None on flag-OFF → ResultReadyHandler's third
            # PROLOGUE gate skips → byte-identical, INV-C4.1-1).
            subagent_run_repo=subagent_run_repo,
            liveness_service=coordinator_liveness_service,  # type: ignore[arg-type]
        )
        return MailboxSupervisor(
            ctx,
            block_ms=settings_local.mailbox_xreadgroup_block_ms,
            count=settings_local.mailbox_xreadgroup_count,
        )

    return SupervisorRegistry(supervisor_factory=_factory)


@dataclass(frozen=True)
class ChildRunnerSharedDeps:
    """[C2 finish-core §5.1 — R1 fix] The process-shared deps a child runner
    needs, resolved LAZILY (post-lifespan) from the live AgentService."""
    uow_factory: object
    llm: object
    agent_config: object
    mcp_config: object
    a2a_config: object
    file_storage: object
    search_engine: object
    checkpointer_pool: object
    execution_supervisor: object
    session_state_machine: object = None  # A4-1 §6: status-write authority for the child runner
    tool_runtime: object = None  # B1: root 的 ToolRuntimeConfig（child/root flag 一致性，spec R2#1）
    # B9 Task 20 (R4#1): ExtensionStatsRecorder | None. Coordinator child /
    # subagent extension calls count into the same stats as root (no root-only
    # amputation). main.py's lazy resolver fills this from app.state.extension_stats.
    extension_stats_recorder: object = None
    # B12 follow-up: file_view multimodal deps for the coordinator-child path.
    # Sandbox-INDEPENDENT ingredients only — the child builds its OWN
    # FileProcessorRegistry bound to its own sandbox inside `_build` (registry
    # processors are sandbox-bound; the child runs in a separate sandbox). Left
    # None keeps today's behavior (no file_view binding). `tool_runtime` (above)
    # already carries the B12 flags.
    supports_vision: object = None
    supports_pdf_input: object = None
    file_understanding_config: object = None
    vision_fallback_model: object = None
    # C7: root 的 LifecycleRuntimeConfig | None（child/root flag 同源，§12-8；
    # None（legacy/test deps）→ runner ctor 内坍缩为 default-OFF 配置）。
    lifecycle_runtime: object = None
    # D1a §4.1 注入链第 2 跳：root/child 共享同一 AdmissionPort 实例（admission 必须同源，
    # 与 B12 file_view child 自建 registry 先例不同）。None=off/legacy 向后兼容。
    extension_admission_port: object = None
    # [child SPAWN_ACK fix] 每 pod 的 SupervisorRegistry 单例（main.py lifespan
    # 构建，"mailbox plane mandatory"）。child runner 的
    # ``_maybe_emit_spawn_ack_and_heartbeat`` 需要它做 best-effort ensure-parent-
    # supervisor 再发 SPAWN_ACK/heartbeat；缺席时该路径整体跳过（supervisor 只能靠
    # orphan_reconcile 兜底发现 child）。None=legacy/test 向后兼容（child 不自建）。
    supervisor_registry: object = None


def _make_shared_child_runner_builder(
    *, resolve_child_runner_deps: Callable[[], "ChildRunnerSharedDeps"]
) -> "ChildRunnerBuilder":
    """[C2 finish-core §5.1 Shape-1-variant — R1 fix] Return a ChildRunnerBuilder
    callable that resolves the shared deps at child-build time. The child
    agent_config is derived once (cached) with tool_confirmation.enabled=False
    (§5.1.7) so the lease-bound child never blocks on human confirmation. The
    child is NOT given coord_deps (must not be a nested coordinator)."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    _cache: dict = {}

    def _build(
        *, session_id, tool_filter, mailbox_publisher,
        terminal_envelope_publisher_disabled, sandbox_accessor, browser_accessor,
        user_id, cost_callback_handler, external_terminal_owner=False,
        external_heartbeat_owner=False,
    ):
        deps = resolve_child_runner_deps()  # live, post-lifespan
        child_agent_config = _cache.get("child_agent_config")
        if child_agent_config is None:
            child_agent_config = deps.agent_config.model_copy(update={
                "tool_confirmation": deps.agent_config.tool_confirmation.model_copy(
                    update={"enabled": False}
                ),
            })
            _cache["child_agent_config"] = child_agent_config
        # B12 follow-up: build the child's OWN FileProcessorRegistry bound to the
        # child's sandbox (registry processors are sandbox-bound; the child runs in
        # a separate sandbox from root). Mirrors agent_service._create_task. Producer
        # gate: only when file_understanding is configured (else None →
        # create_native_tools skips file_view binding — today's behavior).
        child_file_processor_lookup = None
        if deps.file_understanding_config:
            from app.infrastructure.external.file_processors.registry import (
                FileProcessorRegistry,
            )

            _file_storage = deps.file_storage

            async def _upload_bytes(file_bytes: bytes, filename: str) -> "str | None":
                from io import BytesIO

                from fastapi import UploadFile
                try:
                    upload = UploadFile(
                        file=BytesIO(file_bytes), filename=filename, size=len(file_bytes)
                    )
                    file_obj = await _file_storage.upload_file(upload)
                    return await _file_storage.get_presigned_url(file_obj)
                except Exception:
                    logger.warning(
                        "child file_uploader failed for %s", filename, exc_info=True
                    )
                    return None

            child_file_processor_lookup = FileProcessorRegistry(
                # [SPM Task 10] FileProcessorRegistry is sandbox-bound (raw handle);
                # unwrap via peek() (Eager → never None in PR-1b). The runner ctor
                # below receives the accessor itself.
                sandbox=sandbox_accessor.peek(),
                file_uploader=_upload_bytes,
                vision_model=deps.vision_fallback_model,
                audio_config=deps.file_understanding_config.audio,
                video_config=deps.file_understanding_config.video,
                pdf_page_parallel_enabled=getattr(
                    deps.tool_runtime, "pdf_page_parallel_enabled", False
                ),
            )
        return AgentTaskRunner(
            uow_factory=deps.uow_factory,
            llm=deps.llm,
            agent_config=child_agent_config,
            mcp_config=deps.mcp_config,
            a2a_config=deps.a2a_config,
            session_id=session_id,
            user_id=user_id,
            file_storage=deps.file_storage,
            browser_accessor=browser_accessor,
            search_engine=deps.search_engine,
            sandbox_accessor=sandbox_accessor,
            checkpointer_pool=deps.checkpointer_pool,
            cost_callback_handler=cost_callback_handler,
            tool_filter=tool_filter,
            mailbox_publisher=mailbox_publisher,
            terminal_envelope_publisher_disabled=terminal_envelope_publisher_disabled,
            external_terminal_owner=external_terminal_owner,
            external_heartbeat_owner=external_heartbeat_owner,
            coord_deps=None,  # child is NOT a nested coordinator
            session_state_machine=deps.session_state_machine,
            tool_runtime=deps.tool_runtime,
            # C7: child runner 同源持有 lifecycle flags（§12-8）——与 tool_runtime
            # 同模式，防 root 翻 flag 后 child 静默停留 default-OFF（B1 spec R2#1 教训）。
            lifecycle_runtime=deps.lifecycle_runtime,
            # B9 Task 20 (R4#1): child extension calls also feed stats.
            extension_stats_recorder=deps.extension_stats_recorder,
            # B12 follow-up: coordinator-child file_view parity. supports_vision/
            # supports_pdf_input inherit the root's resolved capability ceiling;
            # None (un-populated legacy/test deps) falls back to the runner default
            # so pre-B12 behavior is preserved. file_processor_lookup is the child's
            # own sandbox-bound registry (None when file_understanding unconfigured).
            supports_vision=(
                deps.supports_vision if deps.supports_vision is not None else True
            ),
            supports_pdf_input=(
                deps.supports_pdf_input
                if deps.supports_pdf_input is not None
                else False
            ),
            file_processor_lookup=child_file_processor_lookup,
            # D1a §4.1: child inherits the SAME AdmissionPort instance as root
            # (admission 必须同源). None when mode off. deps default None keeps
            # legacy/test child builds working unchanged.
            extension_admission_port=deps.extension_admission_port,
            # [child SPAWN_ACK fix] Same per-pod SupervisorRegistry instance the
            # root runners get — the child-side SPAWN_ACK/heartbeat path skips
            # entirely when this is None (it can't confirm a consumer exists).
            supervisor_registry=deps.supervisor_registry,
        )

    return _build


def build_coordinator_runtime_deps(
    *,
    app_state: Any,
    redis_client: RedisClient,
    resolve_child_runner_deps: Any = None,   # () -> ChildRunnerSharedDeps (lazy, post-lifespan)
    sandbox_lifecycle_service: Any = None,    # _DeferredLifecycle proxy
    child_runner_task_cls: Any = None,        # RedisStreamTask
) -> Any:
    """[PR-9b-A Task A8] Composition root for the lifespan-scoped
    ``_CoordinatorRuntimeDeps`` aggregator.

    Constructs every lifespan-scoped singleton the coordinator dispatch path
    needs, stores each on ``app_state.*`` for downstream DI lookups, and
    returns the populated ``_CoordinatorRuntimeDeps`` value object.

    Wiring is **always-live**: the runtime feature flag
    (``ACTUS_C2_COORDINATOR_ENABLED``) gates **ENTRY** (whether coordinator
    behaviour fires at ``main_graph.py:695``) — NOT **WIRING** (whether the
    deps are threaded through). Per spec §704 + plan INV-A1, the composition
    root MUST populate these singletons regardless of the C2 flag so a
    config flip is purely a runtime decision, never a redeploy.

    The helper is intentionally side-effect-free apart from constructing
    objects and writing to ``app_state``: no DB / Redis / file-system I/O
    runs here. Network-touching reconcile loops are wired separately by the
    lifespan handler (see ``main.py``).
    """
    # ── Imports (lazy to keep module load time bounded) ─────────────────────
    from app.application.services.child_agent_runner_factory import (
        ChildAgentTaskRunnerFactory,
    )
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )
    from app.application.services.coordinator_envelope_factory import (
        CoordinatorEnvelopeFactory,
    )
    from app.application.services.coordinator_rehydrate_service import (
        ApplyLeaseObservation,
        CoordinatorRehydrateService,
    )
    from app.application.services.patch_applier import (
        COORDINATOR_APPLY_LOCK_TTL_SECONDS,
        bind_coordinator_apply_lock_token,
        compare_delete_coordinator_apply_lock,
        coordinator_apply_lock_key,
        coordinator_apply_reconcile_marker_key,
    )
    from app.application.services.coordinator_parent_execution_lease import (
        CoordinatorParentExecutionLease,
    )
    from app.application.services.coordinator_runtime_deps import (
        _CoordinatorRuntimeDeps,
    )
    from app.application.services.coordinator_terminal_envelope_waiter import (
        CoordinatorTerminalEnvelopeWaiter,
    )
    from app.application.services.coordinator_wait_guard import CoordinatorWaitGuard
    from app.application.services.db_cost_rollup_service import (
        DbCostRollupService,
    )
    from app.application.services.patch_applier_deps import PatchApplierDeps
    from app.application.services.patch_reducer_service import PatchReducerService
    from app.application.services.rollback_snapshot_store import (
        LocalFSRollbackSnapshotStore,
    )
    from app.application.services.session_service import SessionService
    from app.domain.services.coordinator_limits import load_coordinator_limits_from_env
    from app.domain.services.graphs.parallel_execution_subgraph import (
        build_parallel_execution_subgraph,
    )
    from app.infrastructure.cache.probe_quota import ProbeQuotaService
    from app.infrastructure.external.file_storage.minio_file_storage import (
        MinioFileStorage,
    )
    from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
        RedisMailboxPublisher,
    )
    from app.infrastructure.external.mailbox.redis_mailbox_subscriber import (
        RedisMailboxSubscriber,
    )
    from app.infrastructure.repositories.db_coordinator_apply_audit_repository import (
        DbCoordinatorApplyAuditRepository,
    )
    from app.infrastructure.repositories.db_coordinator_result_envelope_store_repository import (
        DbCoordinatorResultEnvelopeStoreRepository,
    )
    from app.infrastructure.repositories.db_session_repository import (
        DBSessionRepository,
    )

    # ── Resolve session_factory ──────────────────────────────────────────────
    pg_session_factory = get_postgres().session_factory
    raw_redis = redis_client.client

    # ── 1. Compiled parallel-execution subgraph (singleton; ``checkpointer=False``
    #      because outer graph owns checkpoint). ─────────────────────────────
    parallel_execution_subgraph = build_parallel_execution_subgraph()

    # ── 2. Coordinator limits + envelope factory — one shared budget source. ─
    coordinator_limits = load_coordinator_limits_from_env()
    envelope_factory = CoordinatorEnvelopeFactory(limits=coordinator_limits)

    # ── 3. Mailbox publisher / subscriber — share the raw Redis client with
    #      the supervisor registry's publisher (single transport wire). ─────
    mailbox_publisher = RedisMailboxPublisher(raw_redis)
    mailbox_subscriber = RedisMailboxSubscriber(raw_redis)

    # ── 4. CoordinatorTerminalEnvelopeWaiter is assembled after the shared
    #      liveness/session ports below so its terminal-vs-stale race uses the
    #      same authority as dispatch and MailboxSupervisor. ────────────────

    # ── 5. Repositories (DB-backed, short-session pattern). ─────────────────
    coordinator_envelope_store = DbCoordinatorResultEnvelopeStoreRepository(
        session_factory=pg_session_factory,
    )
    coordinator_apply_audit_repo = DbCoordinatorApplyAuditRepository(
        session_factory=pg_session_factory,
    )

    # The ``session_repository`` slot is the lightweight session-factory-bound
    # adapter the orchestrator + child_runner_starter consume for read-only
    # queries (read_mode_revision, find_descendants, ...). We reuse the same
    # short-session pattern as the supervisor's adapter so both paths share
    # one source of truth for parent/child reads.
    class _CoordinatorSessionRepoAdapter:
        """Session-per-call SessionRepository adapter for coordinator paths.

        Mirrors ``_SupervisorSessionRepoAdapter`` (see ``build_supervisor_registry``)
        and the DbMailboxEnvelopeAuditRepository session-per-call pattern.
        """

        __slots__ = ("_sf",)

        def __init__(self, session_factory) -> None:  # type: ignore[no-untyped-def]
            self._sf = session_factory

        async def read_mode_revision(self, session_id: str) -> int:
            # Direct call: ``DBSessionRepository.read_mode_revision`` is part
            # of the SessionRepository Protocol (see
            # ``domain/repositories/session_repository.py:282``); if it is
            # ever removed the AttributeError must propagate loudly rather
            # than silently returning 0 for everyone.
            async with self._sf() as s:
                repo = DBSessionRepository(db_session=s)
                return await repo.read_mode_revision(session_id)

        async def find_children_by_coordinator_run(
            self, *, coordinator_run_id: str, parent_session_id: str,
        ):
            # Direct call: ``DBSessionRepository.find_children_by_coordinator_run``
            # is part of the SessionRepository Protocol (see
            # ``domain/repositories/session_repository.py:370``). An empty-list
            # fallback would silently mask the supervisor's fail-closed posture
            # for rehydrate — let AttributeError propagate instead.
            async with self._sf() as s:
                repo = DBSessionRepository(db_session=s)
                return await repo.find_children_by_coordinator_run(
                    coordinator_run_id=coordinator_run_id,
                    parent_session_id=parent_session_id,
                )

        async def get_by_id(self, session_id: str):
            async with self._sf() as s:
                repo = DBSessionRepository(db_session=s)
                return await repo.get_by_id(session_id)

        async def find_descendants(  # type: ignore[no-untyped-def]
            self, ancestor_id: str, *, user_id: str, max_depth: int, limit: int,
        ):
            async with self._sf() as s:
                repo = DBSessionRepository(db_session=s)
                return await repo.find_descendants(
                    ancestor_id, user_id=user_id, max_depth=max_depth, limit=limit,
                )

        async def count_descendants(
            self, ancestor_id: str, *, user_id: str, cap: int,
        ) -> int:
            # [PR-9b-A codex R2#1] parallel_execution_subgraph.py:376 calls
            # ``session_repo.count_descendants(...)`` during the descendants-cap
            # preflight; without this method the first flag-on dispatch raises
            # AttributeError. ``DBSessionRepository.count_descendants`` is the
            # source of truth at infrastructure/repositories/db_session_repository.py:937.
            async with self._sf() as s:
                repo = DBSessionRepository(db_session=s)
                return await repo.count_descendants(
                    ancestor_id, user_id=user_id, cap=cap,
                )

        async def find_running_mailbox_children_for_parent(self, parent_session_id):  # type: ignore[no-untyped-def]
            # C2 cancel fanout: session-per-call (mirrors count_descendants).
            # ``DBSessionRepository.find_running_mailbox_children_for_parent`` is
            # the source of truth (db_session_repository.py).
            async with self._sf() as s:
                repo = DBSessionRepository(db_session=s)
                return await repo.find_running_mailbox_children_for_parent(
                    parent_session_id
                )

    session_repository_adapter = _CoordinatorSessionRepoAdapter(pg_session_factory)

    # Build quota before liveness so the heartbeat callback has no latent
    # unbound-closure window during composition.
    probe_quota = ProbeQuotaService(redis_client=redis_client)

    async def _renew_sandbox_cleanup_lease(session_id: str) -> None:
        """Reset one active sandbox cleanup lease without retaining a handle."""
        if sandbox_lifecycle_service is None:
            return
        handle = await sandbox_lifecycle_service.acquire(session_id)
        try:
            await handle.renew_timeout_lease()
        finally:
            handle.release()

    async def _renew_child_sandbox_from_heartbeat(lease) -> None:  # noqa: ANN001
        # ``record_heartbeat`` runs callbacks only after the child row and
        # liveness CAS agree, so the child id here is already authoritative.
        await _renew_sandbox_cleanup_lease(lease.child_session_id)

    async def _renew_ordinary_research_sandboxes(row) -> None:  # noqa: ANN001
        # Ordinary research uses process-local mailbox liveness rather than a
        # coordinator Redis lease, but its child and waiting parent sandboxes
        # still need rolling cleanup leases during hour-scale work.
        session_ids = (row.id, row.parent_session_id)
        for session_id in session_ids:
            if not isinstance(session_id, str) or not session_id:
                continue
            try:
                await _renew_sandbox_cleanup_lease(session_id)
            except Exception:
                # Renew the other sandbox even if one binding disappeared or
                # its server is transiently unavailable.
                logger.warning(
                    "ordinary research sandbox renew skipped session=%s",
                    session_id,
                    exc_info=True,
                )

    async def _renew_quota_from_child_heartbeat(lease) -> None:  # noqa: ANN001
        # ``record_heartbeat`` invokes callbacks only after schema validation,
        # DB lineage authorization, and the Redis refresh CAS. Re-read the
        # authoritative child row for its user identity rather than trusting
        # an envelope or a non-persisted lease field.
        from app.domain.models.session import SessionStatus

        row = await session_repository_adapter.get_by_id(
            lease.child_session_id
        )
        if (
            row is None
            or row.id != lease.child_session_id
            or row.coordinator_run_id != lease.coordinator_run_id
            or row.status is not SessionStatus.RUNNING
            or not isinstance(row.user_id, str)
            or not row.user_id.strip()
        ):
            logger.warning(
                "coordinator heartbeat quota renew skipped: authoritative "
                "child identity missing or changed child=%s run=%s",
                lease.child_session_id,
                lease.coordinator_run_id,
            )
            return
        renewed = await probe_quota.renew_coordinator_concurrency(
            user_id=row.user_id,
            coordinator_run_id=lease.coordinator_run_id,
        )
        if not renewed:
            # Never call acquire here: a missing/expired member means this
            # heartbeat no longer owns quota and must not resurrect it.
            logger.warning(
                "coordinator heartbeat quota lease lost user=%s run=%s "
                "child=%s",
                row.user_id,
                lease.coordinator_run_id,
                lease.child_session_id,
            )

    # Shared Task 6 liveness authority. A DB-authorized child heartbeat also
    # renews the existing same-run quota member; child sandbox renewal remains
    # a separate owner-scoped concern.
    from app.application.services.coordinator_liveness_lease_service import (
        COORDINATOR_CHILD_LEASE_TTL_SECONDS,
        CoordinatorLivenessLeaseService,
    )
    coordinator_liveness_service = CoordinatorLivenessLeaseService(
        redis=raw_redis,
        session_repository=session_repository_adapter,  # type: ignore[arg-type]
        renew_child_sandbox=(
            _renew_child_sandbox_from_heartbeat
            if sandbox_lifecycle_service is not None
            else None
        ),
        renew_ordinary_sandboxes=(
            _renew_ordinary_research_sandboxes
            if sandbox_lifecycle_service is not None
            else None
        ),
        renew_quota=_renew_quota_from_child_heartbeat,
    )

    async def _read_persisted_terminal_for_waiter(
        *,
        child_session_id: str,
        root_session_id: str,
        coordinator_run_id: str | None,
    ):
        if coordinator_run_id is None:
            return None
        from datetime import datetime, timezone
        from app.domain.models.mailbox_envelope import (
            MailboxEnvelope,
            MailboxEnvelopeType,
            ProducerRole,
        )

        rows = await coordinator_envelope_store.find_terminal_envelopes_by_run(
            coordinator_run_id,
        )
        for row in rows:
            if row.child_session_id != child_session_id:
                continue
            envelope_type = MailboxEnvelopeType(row.envelope_type)
            raw_payload = dict(row.payload or {})
            if envelope_type == MailboxEnvelopeType.RESULT_READY:
                allowed = {
                    "outcome", "patch_manifest", "patch_manifest_ref",
                    "cost_summary", "needs_authorization_details",
                }
                payload = {
                    key: value for key, value in raw_payload.items()
                    if key in allowed
                }
                payload.setdefault("summary", "persisted terminal result")
            else:
                payload = {
                    "final_state": raw_payload.get("final_state", "cancelled")
                }
            digest = hashlib.sha256(
                f"{coordinator_run_id}|{child_session_id}|{row.envelope_type}".encode()
            ).hexdigest()[:32]
            return MailboxEnvelope(
                envelope_id=f"persisted:{digest}",
                type=envelope_type,
                parent_session_id=root_session_id,
                child_session_id=child_session_id,
                correlation_id=coordinator_run_id,
                emitted_at=getattr(row, "received_at", None)
                or datetime.now(timezone.utc),
                producer_role=ProducerRole.CHILD_AGENT,
                payload=payload,
            )
        return None

    async def _reconcile_stale_waiter_child(
        *,
        child_session_id: str,
        root_session_id: str,
        coordinator_run_id: str | None,
    ) -> None:
        if coordinator_run_id is None:
            raise RuntimeError("stale coordinator waiter requires run id")
        from app.domain.models.mailbox_envelope import CancelPolicy

        digest = hashlib.sha256(
            f"{root_session_id}|{coordinator_run_id}|{child_session_id}".encode()
        ).hexdigest()
        dedup_key = f"coordinator:orphan-reconcile:{digest}"
        owner_token = uuid.uuid4().hex
        claimed = await raw_redis.set(
            dedup_key,
            owner_token,
            nx=True,
            ex=COORDINATOR_CHILD_LEASE_TTL_SECONDS,
        )
        if not claimed:
            return
        try:
            await mailbox_publisher.publish(
                envelope_factory.make_cancel_request(
                    parent_session_id=root_session_id,
                    child_session_id=child_session_id,
                    correlation_id=coordinator_run_id,
                    reason="orphan_timeout",
                    policy=CancelPolicy.TERMINATE,
                )
            )
        except BaseException:
            # A failed publish must be retryable by the next stale observer.
            # The claim may have expired and been replaced while publish was
            # in flight; only its exact owner may remove it.
            try:
                await raw_redis.eval(
                    _ORPHAN_OWNER_COMPARE_DELETE_SCRIPT,
                    1,
                    dedup_key,
                    owner_token,
                )
            except BaseException:
                logger.exception(
                    "failed to compare-delete orphan reconcile claim key=%s",
                    dedup_key,
                )
            raise

    terminal_waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=mailbox_subscriber,
        liveness_service=coordinator_liveness_service,
        orphan_reconciler=_reconcile_stale_waiter_child,
        persisted_terminal_reader=_read_persisted_terminal_for_waiter,
    )

    # ── 6. CoordinatorRehydrateService — wraps repo + envelope_store + audit. ─
    # Apply ownership is authoritative for in-progress recovery. One Lua call
    # atomically observes the canonical lock and updates the run marker. PTTL
    # is the elapsed-time source, so cross-pod wall-clock skew cannot shorten
    # or extend the 30s grace. A live owner clears an old missing cycle in the
    # same atomic operation; Redis errors fail closed in the service.
    _APPLY_LEASE_OBSERVE_SCRIPT = r"""
-- coordinator-apply-lease-observe-v1
local token = redis.call('GET', KEYS[1])
local lock_ttl_ms = redis.call('PTTL', KEYS[1])
if token and string.len(token) > 0 and lock_ttl_ms > 0 then
  redis.call('DEL', KEYS[2])
  return {1, 0}
end

local configured_ttl_ms = tonumber(ARGV[1])
local marker_full_ttl_ms = redis.call('GET', KEYS[2])
local marker_ttl_ms = redis.call('PTTL', KEYS[2])
if (not marker_full_ttl_ms) or marker_ttl_ms <= 0 then
  redis.call('SET', KEYS[2], ARGV[1], 'PX', configured_ttl_ms, 'NX')
  marker_full_ttl_ms = redis.call('GET', KEYS[2])
  marker_ttl_ms = redis.call('PTTL', KEYS[2])
end

local full_ttl_ms = tonumber(marker_full_ttl_ms)
if (not full_ttl_ms) or marker_ttl_ms < 0 then
  return {2, 0}
end
local elapsed_ms = full_ttl_ms - marker_ttl_ms
if elapsed_ms < 0 then
  elapsed_ms = 0
end
return {0, elapsed_ms}
"""

    _APPLY_CRASH_FENCE_SCRIPT = r"""
-- coordinator-apply-crash-fence-v1
if redis.call('GET', KEYS[1]) then
  return {0, 0}
end

local marker_full_ttl_ms = redis.call('GET', KEYS[2])
local marker_ttl_ms = redis.call('PTTL', KEYS[2])
local full_ttl_ms = tonumber(marker_full_ttl_ms)
if (not marker_full_ttl_ms) or (not full_ttl_ms) or marker_ttl_ms < 0 then
  return {0, 0}
end
local elapsed_ms = full_ttl_ms - marker_ttl_ms
if elapsed_ms < 0 then
  return {2, 0}
end
if elapsed_ms < tonumber(ARGV[3]) then
  return {0, elapsed_ms}
end

local acquired = redis.call(
  'SET', KEYS[1], ARGV[1], 'NX', 'PX', tonumber(ARGV[2])
)
if not acquired then
  return {0, elapsed_ms}
end
redis.call('DEL', KEYS[2])
return {1, elapsed_ms}
"""

    class _RedisApplyLeaseObserver:
        def __init__(self, redis) -> None:  # type: ignore[no-untyped-def]
            self._redis = redis

        async def __call__(
            self,
            coordinator_run_id: str,
            *,
            marker_ttl_seconds: int,
        ) -> ApplyLeaseObservation:
            result = await self._redis.eval(
                _APPLY_LEASE_OBSERVE_SCRIPT,
                2,
                coordinator_apply_lock_key(coordinator_run_id),
                coordinator_apply_reconcile_marker_key(coordinator_run_id),
                marker_ttl_seconds * 1000,
            )
            if (
                not isinstance(result, (list, tuple))
                or len(result) != 2
            ):
                raise RuntimeError(
                    f"invalid apply lease observation result: {result!r}",
                )
            state = int(result[0])
            elapsed_ms = int(result[1])
            if state == 2:
                raise RuntimeError("corrupt apply reconcile marker")
            if state not in (0, 1) or elapsed_ms < 0:
                raise RuntimeError(
                    f"invalid apply lease observation values: {result!r}",
                )
            return ApplyLeaseObservation(
                owner_is_live=state == 1,
                missing_for_seconds=elapsed_ms / 1000.0,
            )

    class _RedisApplyCrashFence:
        """Execute the audit CAS under a rolling canonical apply lease."""

        def __init__(self, redis) -> None:  # type: ignore[no-untyped-def]
            self._redis = redis

        async def __call__(
            self,
            coordinator_run_id: str,
            operation,
            *,
            minimum_missing_seconds: int,
        ) -> bool | None:
            lock_key = coordinator_apply_lock_key(coordinator_run_id)
            marker_key = coordinator_apply_reconcile_marker_key(
                coordinator_run_id,
            )
            lock = self._redis.lock(
                lock_key,
                blocking=False,
                timeout=COORDINATOR_APPLY_LOCK_TTL_SECONDS,
            )
            token = uuid.uuid4().hex
            try:
                result = await self._redis.eval(
                    _APPLY_CRASH_FENCE_SCRIPT,
                    2,
                    lock_key,
                    marker_key,
                    token,
                    str(int(COORDINATOR_APPLY_LOCK_TTL_SECONDS * 1_000)),
                    str(minimum_missing_seconds * 1_000),
                )
            except BaseException:
                # The response may be lost after SET. Exact-token cleanup is
                # safe and leaves a bounded TTL if Redis is also unavailable.
                try:
                    await compare_delete_coordinator_apply_lock(
                        self._redis,
                        lock_key,
                        token,
                    )
                except BaseException:
                    logger.exception(
                        "ambiguous apply crash fence cleanup failed run=%s",
                        coordinator_run_id,
                    )
                raise
            if (
                not isinstance(result, (list, tuple))
                or len(result) != 2
            ):
                raise RuntimeError(
                    f"invalid apply crash fence result: {result!r}",
                )
            state = int(result[0])
            if state == 2:
                raise RuntimeError("corrupt apply reconcile marker age")
            if state == 0:
                return None
            if state != 1:
                raise RuntimeError(
                    f"invalid apply crash fence state: {result!r}",
                )
            try:
                bind_coordinator_apply_lock_token(lock, token)
            except BaseException:
                try:
                    await compare_delete_coordinator_apply_lock(
                        self._redis,
                        lock_key,
                        token,
                    )
                except BaseException:
                    logger.exception(
                        "apply crash fence token-bind cleanup failed run=%s",
                        coordinator_run_id,
                    )
                raise

            lost = asyncio.Event()

            async def renew() -> None:
                try:
                    while True:
                        await asyncio.sleep(
                            COORDINATOR_APPLY_LOCK_TTL_SECONDS / 3.0,
                        )
                        if not await lock.owned():
                            lost.set()
                            return
                        if not await lock.extend(
                            COORDINATOR_APPLY_LOCK_TTL_SECONDS,
                            replace_ttl=True,
                        ):
                            lost.set()
                            return
                except asyncio.CancelledError:
                    raise
                except Exception:
                    lost.set()
                    logger.exception(
                        "apply crash fence renew failed run=%s",
                        coordinator_run_id,
                    )

            renew_task = asyncio.create_task(
                renew(),
                name=f"apply-crash-fence-renew:{coordinator_run_id}",
            )
            try:
                if not await lock.owned():
                    return None
                result = await operation()
                if lost.is_set() or not await lock.owned():
                    return None
                return result
            finally:
                renew_task.cancel()
                try:
                    await renew_task
                except asyncio.CancelledError:
                    pass
                try:
                    if await lock.owned():
                        await lock.release()
                except Exception:
                    logger.exception(
                        "apply crash fence release failed run=%s",
                        coordinator_run_id,
                    )

    from datetime import datetime, timezone

    rehydrate_service = CoordinatorRehydrateService(
        session_repository=session_repository_adapter,
        envelope_store=coordinator_envelope_store,
        audit_repository=coordinator_apply_audit_repo,
        publisher=mailbox_publisher,
        apply_lease_observer=_RedisApplyLeaseObserver(raw_redis),
        apply_crash_fence=_RedisApplyCrashFence(raw_redis),
        clock=lambda: datetime.now(timezone.utc),
    )

    # ── 7. CoordinatorRunOrchestrator factory — per-run wrapper around the
    #      lifespan-scoped publisher + envelope_factory + subscriber.
    #      Per spec §11.3-§11.4, one orchestrator is constructed per
    #      coordinator run; the *factory* captures the lifespan deps.
    #
    #      PR-9b-A audit round-1 P1: the consumer at
    #      ``parallel_execution_subgraph._first_time_dispatch`` invokes
    #      ``orchestrator_factory.build(...)`` — a plain function has no
    #      ``.build`` attribute and would AttributeError on first flag-on
    #      dispatch. Wrap the closure-style factory in a tiny class so the
    #      ``.build(...)`` surface matches the consumer 1:1.
    #
    #      ``root_session_id`` IS accepted by ``build(...)`` (the consumer
    #      passes it for parity with ``orchestrator.run(root_session_id=...)``
    #      one line below in the dispatch path) but is NOT forwarded into
    #      ``CoordinatorRunOrchestrator.__init__`` — the orchestrator ctor
    #      (coordinator_run_orchestrator.py:185-216) does not accept it
    #      because the orchestrator reads root_session_id from the per-run
    #      ``run(...)`` kwarg (coordinator_run_orchestrator.py:230) instead.
    class _OrchestratorFactory:
        """Per-run CoordinatorRunOrchestrator factory.

        Captures the lifespan-scoped publisher / envelope_factory /
        mailbox_subscriber once at composition-root time and produces a
        fresh orchestrator from per-run parameters on every
        ``.build(...)`` call. Spec §11.3-§11.4 mandates one orchestrator
        per coordinator run.
        """

        def __init__(
            self,
            *,
            mailbox_publisher,
            envelope_factory,
            mailbox_subscriber,
            coordinator_limits,
            cost_rollup_service_provider,
        ) -> None:
            self._mailbox_publisher = mailbox_publisher
            self._envelope_factory = envelope_factory
            self._mailbox_subscriber = mailbox_subscriber
            self._coordinator_limits = coordinator_limits
            self._cost_rollup_service_provider = cost_rollup_service_provider

        def build(
            self,
            *,
            parent_session_id: str,
            coordinator_run_id: str,
            root_session_id: str | None = None,
            emit_event=None,
        ):
            from app.application.services.coordinator_run_orchestrator import (
                CoordinatorRunOrchestrator,
            )

            # ``root_session_id`` is accepted for consumer-call-shape parity
            # (dispatch_node passes it next to ``orchestrator.run(...)``) but
            # the orchestrator ctor does not store it — ``run(...)`` receives
            # it as a per-call kwarg instead.
            _ = root_session_id
            return CoordinatorRunOrchestrator(
                publisher=self._mailbox_publisher,
                parent_session_id=parent_session_id,
                coordinator_run_id=coordinator_run_id,
                envelope_factory=self._envelope_factory,
                mailbox_subscriber=self._mailbox_subscriber,
                emit_event=emit_event,
                cost_rollup_service=self._cost_rollup_service_provider(),
                coordinator_limits=self._coordinator_limits,
            )

    _orchestrator_factory = _OrchestratorFactory(
        mailbox_publisher=mailbox_publisher,
        envelope_factory=envelope_factory,
        mailbox_subscriber=mailbox_subscriber,
        coordinator_limits=coordinator_limits,
        cost_rollup_service_provider=lambda: cost_rollup_service,
    )

    # ── 8. SessionService (coordinator path). ───────────────────────────────
    coordinator_session_service = SessionService(
        uow_factory=get_uow,
        task_cls=RedisStreamTask,
        sandbox_lifecycle_service=getattr(app_state, "sandbox_lifecycle_service", None),
        fs_reconciler=getattr(app_state, "fs_reconciler", None),
        execution_supervisor=getattr(app_state, "supervisor", None),
        subagent_limits=get_subagent_limits(),
        # SPM PR-1c Task 17: provision metrics for vnc/takeover trigger surfaces.
        sandbox_provision_metrics=getattr(
            app_state, "sandbox_provision_metrics", None
        ),
    )

    # ── 9. ProbeQuotaService was constructed above for liveness callback use.

    # ── 10b. CoordinatorMetrics — C2b budget D10 instrument bundle. ─────────
    #      OtelMeter() defaults to get_meter("actus"): a no-op proxy before
    #      setup_observability runs, so constructing here (lifespan, possibly
    #      obs-disabled) is side-effect-safe. Threaded into the starter ctor
    #      below + the coord_deps aggregate; the budget finalizer emits
    #      actus_coordinator_budget_exhaustion_total through it (INV-B9).
    from app.infrastructure.observability import OtelMeter
    from app.infrastructure.observability.coordinator_telemetry import (
        CoordinatorMetrics,
    )
    coordinator_metrics = CoordinatorMetrics(OtelMeter())
    # [C2b rollout WS1b] App-layer recorder wrapping the metrics bundle + the
    # user_id_hash salt (the domain reducer must not hold the salt). Threaded
    # BOTH into the starter ctor (→ adapter, for tool_calls) AND into
    # coord_deps (→ _build_config cfg → reducer, for run-level metrics).
    from app.application.services.coordinator_metrics_recorder import (
        CoordinatorMetricsRecorder,
    )
    coordinator_metrics_recorder = CoordinatorMetricsRecorder(
        metrics=coordinator_metrics,
        user_id_hash_salt=settings.user_id_hash_salt,
    )

    # ── 11. PatchReducerService — pure, stateless. ──────────────────────────
    patch_reducer_service = PatchReducerService()

    # ── 12. PatchApplierDeps — three lifespan ports. ────────────────────────
    snapshot_store = LocalFSRollbackSnapshotStore()
    patch_applier_deps = PatchApplierDeps(
        snapshot_store=snapshot_store,
        audit_repo=coordinator_apply_audit_repo,
        redis=raw_redis,
    )

    # ── 13. ArtifactStorage — MinIO-backed content-addressed blob store.
    #      The MinioFileStorage class implements the ArtifactStoragePort
    #      Protocol (put_content_addressed_bytes + get_bytes). It's reused
    #      here as the lifespan singleton for the coordinator path.
    minio_store = get_minio()
    artifact_storage = MinioFileStorage(
        bucket=settings.minio_bucket_name,
        minio_store=minio_store,
        uow_factory=get_uow,
    )

    # ── 14. DbCostRollupService — pull-cost authority + push-only metric hook.
    #      [PR-9b-B Task B3] Supersedes PR-9b-A's MetricHookCostRollupService
    #      stub. DbCostRollupService implements BOTH:
    #        * ``aggregate(...)`` — pull authority for
    #          ``CoordinatorReduceEvent.cost_total`` (INV-B3: queries
    #          cost_records JOIN sessions filtered by coordinator_run_id +
    #          child_session_ids; mismatched-run rows excluded + surfaced
    #          via ``missing_children`` for the reducer's diagnostics).
    #        * ``rollup_to_parent(...)`` — push-only hook preserved from A3
    #          (idempotent on idempotency_key, never raises). ─────────────
    class _NoopMetricSink:
        """Inert metric sink — the metric pipeline destination is wired
        separately. Calls to ``rollup_to_parent`` traverse the same code
        path but the underlying ``record()`` is a no-op until the real
        sink lands. This keeps the lifespan composition root
        flag-on-deploy-safe.
        """

        def record(self, **kwargs) -> None:  # noqa: ANN003, D401
            return None

    metric_sink = getattr(app_state, "metric_sink", None) or _NoopMetricSink()
    app_state.metric_sink = metric_sink
    cost_rollup_service = DbCostRollupService(
        async_session_factory=pg_session_factory,
        metric_sink=metric_sink,
    )

    # ── 15. ChildAgentTaskRunnerFactory + DefaultCoordinatorChildRunnerStarter.
    #      [C2 finish-core F1.7 — R1 fix] The ``runner_class`` injection is the
    #      shared child-runner builder (a ChildRunnerBuilder callable, NOT the
    #      bare AgentTaskRunner class — that would TypeError on the missing
    #      required ctor args). The builder resolves the process-shared deps
    #      LAZILY via ``resolve_child_runner_deps`` (post-lifespan, off the live
    #      AgentService) and forces ``tool_confirmation.enabled=False`` on the
    #      child-scoped agent_config (§5.1.7).
    #
    #      When ``resolve_child_runner_deps`` is None (the 2-kwarg comp-root
    #      test callers that never dispatch a child), an unwired sentinel
    #      builder is injected so construction stays TypeError-free; it raises
    #      loudly only if a child is actually dispatched without wiring.
    def _unwired_builder(**_kw):
        raise RuntimeError(
            "coordinator child runner builder not wired (resolve_child_runner_deps "
            "is None) — production lifespan must pass it; tests that don't dispatch "
            "a child never reach here."
        )
    runner_class = (
        _make_shared_child_runner_builder(resolve_child_runner_deps=resolve_child_runner_deps)
        if resolve_child_runner_deps is not None else _unwired_builder
    )
    runner_factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class,
        mailbox_publisher=mailbox_publisher,
        task_cls=child_runner_task_cls,
    )
    child_runner_starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=runner_factory,
        mailbox_publisher=mailbox_publisher,
        mailbox_subscriber=mailbox_subscriber,
        envelope_factory=envelope_factory,
        session_repository=session_repository_adapter,  # type: ignore[arg-type]
        coordinator_envelope_store=coordinator_envelope_store,
        cost_rollup_service=cost_rollup_service,
        artifact_storage=artifact_storage,
        coordinator_limits=coordinator_limits,
        sandbox_lifecycle_service=sandbox_lifecycle_service,
        resolve_child_runner_deps=resolve_child_runner_deps,
        coordinator_metrics=coordinator_metrics,  # [C2b budget D10]
        coordinator_metrics_recorder=coordinator_metrics_recorder,  # [C2b rollout WS1b]
        # SPM PR-1c Task 17: provision metrics for the child_spawn trigger surface.
        sandbox_provision_metrics=getattr(
            app_state, "sandbox_provision_metrics", None
        ),
    )

    # ── 16. Pin everything on app_state so downstream DI / lifespan teardown
    #      can resolve them. The plan calls out these four explicitly:
    #      ``cost_rollup_service`` / ``coordinator_envelope_store`` /
    #      ``patch_applier_deps`` / ``supervisor_registry`` (the last one is
    #      wired by main.py lifespan directly via build_supervisor_registry).
    app_state.parallel_execution_subgraph = parallel_execution_subgraph
    app_state.coordinator_envelope_factory = envelope_factory
    app_state.coordinator_mailbox_publisher = mailbox_publisher
    app_state.coordinator_mailbox_subscriber = mailbox_subscriber
    app_state.coordinator_terminal_waiter = terminal_waiter
    app_state.coordinator_liveness_service = coordinator_liveness_service
    app_state.coordinator_envelope_store = coordinator_envelope_store
    app_state.coordinator_apply_audit_repo = coordinator_apply_audit_repo
    app_state.coordinator_session_repository = session_repository_adapter
    app_state.coordinator_rehydrate_service = rehydrate_service
    app_state.coordinator_orchestrator_factory = _orchestrator_factory
    app_state.coordinator_session_service = coordinator_session_service
    app_state.coordinator_probe_quota = probe_quota
    app_state.coordinator_limits = coordinator_limits
    app_state.patch_reducer_service = patch_reducer_service
    app_state.snapshot_store = snapshot_store
    app_state.patch_applier_deps = patch_applier_deps
    app_state.coordinator_artifact_storage = artifact_storage
    app_state.cost_rollup_service = cost_rollup_service
    app_state.coordinator_child_runner_starter = child_runner_starter

    # [finish-core §5.2 G2] Adapter factory: wraps a per-run raw SandboxHandle
    # into a ParentSandboxPort. Injected as a coord dep so the domain flow
    # (planner_react._build_config) never imports infrastructure to wrap it.
    def _parent_sandbox_adapter_factory(handle):  # noqa: ANN001, ANN202
        return ParentSandboxAdapter(handle)

    def _parent_execution_lease_factory(**kwargs):  # noqa: ANN003, ANN202
        # Task 7 wires durable child liveness plus the existing parent
        # supervisor-activity touch. Task 8-10 supply auto-degrade/sandbox/
        # quota renew callbacks without changing this fixed factory surface.
        async def _touch_parent_activity(context) -> None:  # noqa: ANN001
            # ``build_coordinator_runtime_deps`` runs before IdleWatchdog is
            # attached during lifespan. Resolve it lazily at tick time, then
            # reuse its existing supervisor hot-hash touch contract.
            idle_watchdog = getattr(app_state, "idle_watchdog", None)
            if idle_watchdog is not None:
                await idle_watchdog.touch_activity(
                    session_id=context.parent_session_id,
                )

        async def _renew_auto_degrade(context) -> None:  # noqa: ANN001
            # The shared ExecutionSupervisor is attached after AgentService is
            # built, so resolve it lazily on every lease tick. Its PG CAS is
            # authoritative; Redis expiry is synchronized only after commit.
            supervisor = getattr(app_state, "supervisor", None)
            if supervisor is not None:
                await supervisor.renew_auto_degrade_expiry_if_running(
                    session_id=context.parent_session_id,
                )

        async def _renew_parent_sandbox(context) -> None:  # noqa: ANN001
            await _renew_sandbox_cleanup_lease(context.parent_session_id)

        async def _renew_quota(context) -> None:  # noqa: ANN001
            if context.user_id is None:
                logger.warning(
                    "coordinator quota renew skipped: missing user_id run=%s",
                    context.coordinator_run_id,
                )
                return
            renewed = await probe_quota.renew_coordinator_concurrency(
                user_id=context.user_id,
                coordinator_run_id=context.coordinator_run_id,
            )
            if not renewed:
                # A lost/expired member is never silently re-acquired. The
                # backend can continue to its terminal cleanup while ops retain
                # a visible signal that concurrency ownership was lost.
                logger.warning(
                    "coordinator quota lease lost user=%s run=%s",
                    context.user_id,
                    context.coordinator_run_id,
                )

        async def _reconcile_all_children_stale(context) -> None:  # noqa: ANN001
            # Reuse the same idempotent, owner-token-protected orphan path as
            # the terminal waiter. One broken child must not prevent siblings
            # from being reconciled; aggregate failures keep the parent lease
            # retryable while successful siblings remain protected by NX.
            failures: list[tuple[str, BaseException]] = []
            for child_session_id in context.child_session_ids:
                try:
                    await _reconcile_stale_waiter_child(
                        child_session_id=child_session_id,
                        root_session_id=context.root_session_id,
                        coordinator_run_id=context.coordinator_run_id,
                    )
                except asyncio.CancelledError as exc:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling() > 0:
                        raise
                    failures.append((child_session_id, exc))
                    logger.warning(
                        "coordinator all-stale orphan reconciliation failed "
                        "root=%s run=%s child=%s; continuing siblings",
                        context.root_session_id,
                        context.coordinator_run_id,
                        child_session_id,
                        exc_info=True,
                    )
                except Exception as exc:
                    failures.append((child_session_id, exc))
                    logger.warning(
                        "coordinator all-stale orphan reconciliation failed "
                        "root=%s run=%s child=%s; continuing siblings",
                        context.root_session_id,
                        context.coordinator_run_id,
                        child_session_id,
                        exc_info=True,
                    )
            if failures:
                failed_children = ",".join(child_id for child_id, _ in failures)
                raise RuntimeError(
                    "coordinator all-stale orphan reconciliation failed for "
                    f"children: {failed_children}"
                ) from failures[0][1]

        return CoordinatorParentExecutionLease(
            liveness_service=coordinator_liveness_service,
            touch_parent_activity=_touch_parent_activity,
            renew_auto_degrade=_renew_auto_degrade,
            renew_parent_sandbox=(
                _renew_parent_sandbox
                if sandbox_lifecycle_service is not None
                else None
            ),
            renew_quota=_renew_quota,
            on_all_children_stale=_reconcile_all_children_stale,
            **kwargs,
        )

    def _coordinator_wait_guard_factory(*, watchdog=None):  # noqa: ANN001, ANN202
        return CoordinatorWaitGuard(
            watchdog=watchdog,
            parent_lease_factory=_parent_execution_lease_factory,
        )

    # ── 17. Assemble the immutable _CoordinatorRuntimeDeps value object. ────
    coord_deps = _CoordinatorRuntimeDeps(
        parallel_execution_subgraph=parallel_execution_subgraph,
        session_service=coordinator_session_service,
        rehydrate_service=rehydrate_service,
        child_runner_starter=child_runner_starter,
        mailbox_publisher=mailbox_publisher,
        mailbox_subscriber=mailbox_subscriber,
        envelope_factory=envelope_factory,
        orchestrator_factory=_orchestrator_factory,
        terminal_waiter=terminal_waiter,
        probe_quota=probe_quota,
        coordinator_limits=coordinator_limits,
        session_repository=session_repository_adapter,
        patch_reducer_service=patch_reducer_service,
        patch_applier_deps=patch_applier_deps,
        artifact_storage=artifact_storage,
        cost_rollup_service=cost_rollup_service,
        coordinator_envelope_store=coordinator_envelope_store,
        parent_sandbox_adapter_factory=_parent_sandbox_adapter_factory,
        coordinator_metrics=coordinator_metrics,  # [C2b budget D10]
        coordinator_metrics_recorder=coordinator_metrics_recorder,  # [C2b rollout WS1b]
        # [S4 §5] Coordinator team DI — fresh FileTeamRepository (teams/ sibling
        # of skills_root_dir; no dedicated settings field exists) + a FRESH
        # FileSkillRepository (no shared singleton — [codex-R2-F3]). Threaded via
        # _build_config into cfg and read by the team expander
        # (_run_parallel_backend) + planner/updater teaching load; active only when
        # ACTUS_C2_AGENT_TEAMS_ENABLED + a team_slug are set (flag-gated, INV-0).
        team_repository=FileTeamRepository(Path(settings.skills_root_dir).parent / "teams"),
        skill_repository=FileSkillRepository(settings.skills_root_dir),
        coordinator_wait_guard_factory=_coordinator_wait_guard_factory,
        coordinator_liveness_service=coordinator_liveness_service,
    )
    app_state.coord_deps = coord_deps
    return coord_deps


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
    coord_deps: object | None = None,
    # B9 Task 20 (R9#2): lifespan-scoped extension stats recorder. Mirrors the
    # ``flush_service: object | None`` style. main.py passes
    # ``app.state.extension_stats`` (None when flag off). Threaded into
    # AgentService → AgentTaskRunner → PlannerReActFlow → react_graph.
    extension_stats_recorder: object | None = None,
    # D1a §4.1 (R2#F12): lifespan-scoped governance AdmissionPort. main.py passes
    # ``app.state.extension_admission_port`` (None when mode off). Threaded into
    # AgentService → AgentTaskRunner → SkillTool / SkillBundleSyncManager.
    extension_admission_port: object | None = None,
    # SPM PR-1c Task 17: lifespan-scoped SandboxProvisionMetrics singleton.
    # main.py passes ``app.state.sandbox_provision_metrics``. Threaded into
    # AgentService (run_start trigger + on_demand provisioner/flusher wiring).
    sandbox_provision_metrics: object | None = None,
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
    from app.application.composition.graph_assembly import (
        build_session_state_machine,
    )
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
        session_state_machine=build_session_state_machine(uow_factory=get_uow),
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
    # C2 coordinator-cancel — build the parent-cancel fanout from the
    # lifespan-scoped coord_deps. ``coord_deps is None`` (pure legacy/test)
    # -> fanout stays None -> stop_session skips (INV-C4). The production
    # ``_CoordinatorRuntimeDeps`` carries the real session_repository /
    # envelope_factory / mailbox_publisher / child_runner_starter; the
    # ``_NullCoordinatorRuntimeDeps`` sentinel yields None for every field, so
    # the fanout's null-deps guard makes ``cancel_children`` a no-op.
    coordinator_parent_cancel_fanout = None
    if coord_deps is not None:
        from app.application.services.coordinator_parent_cancel_fanout import (
            CoordinatorParentCancelFanout,
        )

        coordinator_parent_cancel_fanout = CoordinatorParentCancelFanout(
            session_repository=getattr(coord_deps, "session_repository", None),
            envelope_factory=getattr(coord_deps, "envelope_factory", None),
            mailbox_publisher=getattr(coord_deps, "mailbox_publisher", None),
            child_runner_starter=getattr(coord_deps, "child_runner_starter", None),
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
        # PR-9b-A Task A8: lifespan-scoped coordinator runtime deps.
        # Forwarded to every AgentTaskRunner constructed by _create_task.
        coord_deps=coord_deps,
        # C2 coordinator-cancel — consumed only by stop_session.
        coordinator_parent_cancel_fanout=coordinator_parent_cancel_fanout,
        policy_snapshot_sink=get_policy_snapshot_sink(),  # C5a: Logging sink when flag ON, else Noop
        # B9 Task 20: root-path stats recorder (None when flag off).
        extension_stats_recorder=extension_stats_recorder,
        # D1a §4.1: root-path governance AdmissionPort (None when mode off).
        extension_admission_port=extension_admission_port,
        # SPM PR-1c Task 17: provision-flow metrics singleton (run_start trigger +
        # on_demand provisioner/flusher wiring).
        sandbox_provision_metrics=sandbox_provision_metrics,
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
# Writer 和 Reader 的构造极轻：writer 只吃 ``uow_factory``；reader 吃一个
# Protocol adapter（``UowApprovalGrantQuery``，始终注入）。
#
# PE-4d1：legacy ``tool_approval_rules`` fallback 已退役——reader 只读 grants，
# 不再按 config 开关注入旧表适配器。
def get_approval_state_writer():
    """R5 CS4 单一 Writer 工厂。无状态，per-request 构造。"""
    from app.application.services.approval_state_writer import ApprovalStateWriter

    return ApprovalStateWriter(uow_factory=get_uow)


def get_approval_state_reader():
    """R5 CS4 Reader 工厂（grants-only，PE-4d1 退役 legacy fallback）。"""
    from app.application.services.approval_state_adapters import (
        UowApprovalGrantQuery,
    )
    from app.domain.services.approval_state_reader import ApprovalStateReader

    grant_query = UowApprovalGrantQuery(uow_factory=get_uow)
    return ApprovalStateReader(query=grant_query)


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


def get_subagent_run_repository(settings_obj) -> Optional[SubagentRunRepository]:
    """C4.1a repo-or-None（spec §5.0）。

    flag OFF → None（不 import infra 实现、不 touch DB）；ON → DbSubagentRunRepository
    持 get_postgres().session_factory。**不是** FastAPI Depends（会在查 flag 前
    eager resolve）——由 coordinator _factory / research DI 显式调用。
    """
    if not settings_obj.subagent_run_record_enabled:
        return None
    from app.infrastructure.repositories.db_subagent_run_repository import (
        DbSubagentRunRepository,
    )
    return DbSubagentRunRepository(session_factory=get_postgres().session_factory)


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
    limits=Depends(get_subagent_limits),
):
    """C1a: SessionCostTreeService composed of 4 deps. FastAPI Depends cache
    guarantees ``db_session`` is shared between session_repo and cost_repo within
    the same request. S3: ``limits`` injects the runtime spawn-depth ceiling.
    """
    from app.application.services.session_cost_tree_service import (
        SessionCostTreeService,
    )

    return SessionCostTreeService(
        session_repo=session_repo,
        cost_repo=cost_repo,
        cost_aggregator=cost_aggregator,
        max_subagent_depth=limits.max_subagent_depth,
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


def get_policy_snapshot_sink() -> PolicySnapshotSink:
    """C5a: log-only sink when the flag is ON, else a no-op.

    Belt-and-suspenders: the two seams ALSO read the flag and skip snapshot
    construction entirely when OFF (INV-0), so the no-op is only a default.
    Reads the module-level ``settings`` singleton (service_dependencies.py:81).
    """
    if settings.sandbox_policy_compiler_enabled:
        from app.infrastructure.external.safety.logging_policy_snapshot_sink import (
            LoggingPolicySnapshotSink,
        )
        return LoggingPolicySnapshotSink()
    from app.domain.external.policy_snapshot_sink import NoopPolicySnapshotSink
    return NoopPolicySnapshotSink()


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

    # C7 §4.5 R15 — research child lifecycle sink. Injected unconditionally
    # (the master ∧ subagent AND-gate is evaluated at runtime inside the sink,
    # so config hot-reload takes effect immediately). Best-effort delivery into
    # the parent's main stream via AgentService._emit_event + PG session.events.
    from app.application.services.research_lifecycle_sink import ResearchLifecycleSink
    lifecycle_sink = ResearchLifecycleSink(
        emit_event=agent_service._emit_event,
        uow_factory=agent_service._uow_factory,
        flags_getter=lambda: agent_service._config_snapshot.lifecycle_runtime,
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
        # C4.1a PR-4 — research subagent-run observation sink. flag OFF → None.
        subagent_run_repo=get_subagent_run_repository(settings),
        # C7 §4.5 R15 — research lifecycle sink (injected unconditionally).
        lifecycle_sink=lifecycle_sink,
    )
