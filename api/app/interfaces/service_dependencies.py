import logging

from app.application.services.agent_service import AgentService
from app.application.services.app_config_service import AppConfigService
from app.application.services.file_service import FileService
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
from app.domain.models.app_config import LLMConfig
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel
from langchain_core.language_models import BaseChatModel
from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox
from app.infrastructure.external.github_search_client import GitHubSearchClient
from app.infrastructure.external.search.bing_search import BingSearchEngine
from app.infrastructure.external.task.redis_stream_task import RedisStreamTask

# from app.infrastructure.repositories.db_file_repository import DBFileRepository
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.infrastructure.repositories.file_app_config_repository import (
    FileAppConfigRepository,
)
from app.infrastructure.repositories.db_memory_chunk_repository import DBMemoryChunkRepository
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.storage.minio import MinioStore, get_minio
from app.infrastructure.storage.postgres import get_db_session, get_postgres, get_uow
from app.infrastructure.storage.redis import RedisClient, get_redis

# from app.interfaces.repository_dependencies import get_db_session_repository
from core.config import get_settings
from fastapi import Depends, Request
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncSession

# from functools import lru_cache


logger = logging.getLogger(__name__)
settings = get_settings()


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
def get_session_service() -> SessionService:
    return SessionService(
        uow_factory=get_uow,
        sandbox_cls=DockerSandbox,
        task_cls=RedisStreamTask,
    )


def _load_app_config():
    app_config_repository = FileAppConfigRepository(
        config_path=settings.app_config_filepath
    )
    return app_config_repository.load()


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
    from app.infrastructure.external.llm.actus_fallback_chat_model import ActusFallbackChatModel

    chat = ActusChatModel(
        base_url=str(llm_config.base_url),
        api_key=llm_config.api_key,
        model_name=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        supports_response_format=getattr(llm_config, 'supports_response_format', True),
        supports_vision=getattr(llm_config, 'supports_vision', True),
        supports_pdf_input=supports_pdf_input,
    )
    responses = ActusResponsesModel(
        base_url=str(llm_config.base_url),
        api_key=llm_config.api_key,
        model_name=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        supports_vision=getattr(llm_config, 'supports_vision', True),
        supports_pdf_input=supports_pdf_input,
    )
    if llm_config.api_type == "responses":
        return responses
    if llm_config.api_type == "auto":
        return ActusFallbackChatModel(primary=chat, fallback=responses)
    return chat


def _build_skill_service() -> SkillService:
    return SkillService(FileSkillRepository(settings.skills_root_dir))


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


# @lru_cache()
def get_agent_service(
    minio_store: MinioStore = Depends(get_minio),
    redis_client: RedisClient = Depends(get_redis),
    checkpointer_pool: AsyncConnectionPool = Depends(get_checkpointer_pool),
    flush_service=Depends(get_flush_service),
    memory_embedding_provider=Depends(get_memory_embedding_provider),
) -> AgentService:
    # 1.获取应用配置信息(读取配置需要实时获取,所以不配置缓存)
    app_config = _load_app_config()
    # file_repository = DBFileRepository(db_session=db_session)
    overflow_config = ContextOverflowConfig.from_llm_config(app_config.llm_config)

    # 2.构建依赖实例
    effective_pdf_input = (
        getattr(app_config.llm_config, 'supports_pdf_input', False)
        and app_config.llm_config.supports_vision
    )
    llm = _build_llm(app_config.llm_config, supports_pdf_input=effective_pdf_input)
    summary_llm = None
    if app_config.agent_config.memory.summary_model:
        summary_llm_config = app_config.llm_config.model_copy(
            update={"model_name": app_config.agent_config.memory.summary_model}
        )
        summary_llm = _build_llm(summary_llm_config)
    file_storage = MinioFileStorage(
        bucket=settings.minio_bucket_name,
        minio_store=minio_store,
        uow_factory=get_uow,
    )
    skill_creator_service = SkillCreatorService(
        llm=llm,
        github_client=GitHubSearchClient(token=settings.github_token or None),
        skill_service=_build_skill_service(),
    )

    # 3.构造 vision fallback model（非多模态主模型时用于描述图片/视频帧）
    vision_fallback_model = None
    vf = app_config.file_understanding.vision_fallback
    if vf.enabled and vf.model_name:
        from app.domain.models.app_config import LLMConfig as LLMConfigModel
        vision_llm_config = LLMConfigModel(
            base_url=vf.base_url or str(app_config.llm_config.base_url),
            api_key=vf.api_key or app_config.llm_config.api_key,
            model_name=vf.model_name,
            api_type=vf.api_type,
            supports_vision=True,  # Force: fallback model MUST support vision
        )
        vision_fallback_model = _build_llm(vision_llm_config)

    # 4.实例Agent服务并返回
    return AgentService(
        uow_factory=get_uow,
        llm=llm,
        agent_config=app_config.agent_config,
        mcp_config=app_config.mcp_config,
        a2a_config=app_config.a2a_config,
        overflow_config=overflow_config,
        skill_risk_policy=app_config.skill_risk_policy,
        sandbox_cls=DockerSandbox,
        task_cls=RedisStreamTask,
        search_engine=BingSearchEngine(),
        file_storage=file_storage,
        redis_client=redis_client,
        skill_creator_service=skill_creator_service,
        summary_llm=summary_llm,
        checkpointer_pool=checkpointer_pool,
        supports_vision=app_config.llm_config.supports_vision,
        supports_pdf_input=effective_pdf_input,
        file_understanding_config=app_config.file_understanding,
        vision_fallback_model=vision_fallback_model,
        memory_flusher=flush_service,
        memory_embedding_provider=memory_embedding_provider,
        memory_session_factory=get_postgres().session_factory,
        memory_repo_factory=DBMemoryChunkRepository,
        # file_repository=file_repository,
    )


def get_skill_export_service() -> SkillExportService:
    return SkillExportService(skills_root_dir=settings.skills_root_dir)
