import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from alembic import command
from alembic.config import Config
from app.infrastructure.logging import setup_logging
from app.infrastructure.observability import setup_observability
from app.infrastructure.storage.minio import get_minio
from app.infrastructure.storage.postgres import get_postgres
from app.infrastructure.storage.redis import get_redis
from app.interfaces.endpoints.routes import router as api_router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.middlewares.observability_middleware import (
    ObservabilityMiddleware,
)

from core.config import get_settings
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# 加载配置信息
settings = get_settings()

# 初始化日志记录
setup_logging()
# B5 PR-S2-1: 装载 OTel SDK（默认 no-op，零外发零本地输出）。setup_logging 必须先跑，
# 这样 setup_observability 内部 ``LoggingHandler`` 就接到已经装好 RedactingFormatter
# 的 root logger 上；任何 stdlib 日志在落 OTel pipeline 之前都已经走过 redaction。
setup_observability()
logger = logging.getLogger()

logger.info("应用程序启动中...")

# 定义FastApi路由tags标签
openapi_tags = [
    {
        "name": "状态模块",
        "description": "包含 **状态监测** 等API 接口，用于监测系统的运行状态。",
    }
]


def _build_alembic_database_url() -> str:
    """构建 Alembic 使用的数据库连接串（同步驱动 + 连接超时）"""
    db_url = settings.sqlalchemy_database_url
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)

    parsed = urlparse(db_url)
    query = dict(parse_qsl(parsed.query))
    query.setdefault("connect_timeout", "5")
    return urlunparse(parsed._replace(query=urlencode(query)))


def _mask_database_url(url: str) -> str:
    """脱敏数据库连接串中的密码"""
    parsed = urlparse(url)
    if parsed.password is None:
        return url
    user = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{user}:***@{host}{port}"
    return urlunparse(parsed._replace(netloc=netloc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """创建FastAPI应用生命周期上下文管理器"""
    # 1.日志打印代码已经开始执行了
    logger.info("Manus应用正在初始化")

    # 2.运行数据库迁移(将数据同步到生产环境)
    _api_root = Path(__file__).resolve().parent.parent
    alembic_cfg = Config(str(_api_root / "alembic.ini"))
    alembic_db_url = _build_alembic_database_url()
    alembic_cfg.set_main_option("sqlalchemy.url", alembic_db_url)
    logger.info(f"数据库迁移开始，连接地址: {_mask_database_url(alembic_db_url)}")
    command.upgrade(alembic_cfg, "head")
    logger.info("数据库迁移完成")

    # 3.初始化Redis/Postgres/Cos客户端

    # 2. 初始化Redis客户端
    logger.info("开始初始化 Redis 客户端")
    redis_client = get_redis()
    await redis_client.init()
    logger.info("Redis 客户端初始化完成")

    # 3. 初始化Postgres数据库客户端
    logger.info("开始初始化 Postgres 客户端")
    postgres_client = get_postgres()
    await postgres_client.init()
    logger.info("Postgres 客户端初始化完成")

    # 4. 初始化MinIO对象存储客户端
    logger.info("开始初始化 MinIO 客户端")
    minio_client = get_minio()
    await minio_client.init()
    logger.info("MinIO 客户端初始化完成")

    # 5. 初始化 Checkpointer 连接池（在 try 内，确保失败时已初始化的基础设施能被清理）
    checkpointer_pool = None
    try:
        logger.info("开始初始化 Checkpointer 连接池")
        from app.infrastructure.checkpointer_pool import CheckpointerPool
        checkpointer_pool = CheckpointerPool(
            db_url=settings.sqlalchemy_database_url,
            min_size=settings.checkpointer_pool_min_size,
            max_size=settings.checkpointer_pool_max_size,
            timeout=settings.checkpointer_pool_timeout,
        )
        await checkpointer_pool.open()
        app.state.checkpointer_pool = checkpointer_pool
        logger.info("Checkpointer 连接池初始化完成")

        # 6. 初始化 Memory Embedding Provider（C4 维度串联 + 容错）
        from app.interfaces.service_dependencies import _load_app_config
        _app_config = _load_app_config()
        _memory_cfg = _app_config.agent_config.memory

        from app.domain.external.embedding_provider import DisabledEmbeddingProvider
        from app.infrastructure.external.embedding.circuit_breaker_embedding_provider import (
            CircuitBreakerEmbeddingProvider,
        )
        from app.infrastructure.models.memory_chunk_orm import MEMORY_EMBEDDING_DIM

        if _memory_cfg.embedding_enabled:
            if not _memory_cfg.embedding_api_base or not _memory_cfg.embedding_api_key:
                raise RuntimeError(
                    "embedding_enabled=True but embedding_api_base or "
                    "embedding_api_key is empty. "
                    "请在 config.yaml 中配置 memory.embedding_api_base "
                    "和 memory.embedding_api_key。"
                )
            if _memory_cfg.embedding_dim != MEMORY_EMBEDDING_DIM:
                raise RuntimeError(
                    f"MemoryConfig.embedding_dim ({_memory_cfg.embedding_dim}) != "
                    f"MEMORY_EMBEDDING_DIM ({MEMORY_EMBEDDING_DIM}). "
                    f"修改维度需要新 migration 重建 pgvector 列。"
                )
            from app.infrastructure.external.embedding.openai_embedding_provider import (
                OpenAIEmbeddingProvider,
            )
            _inner_provider = OpenAIEmbeddingProvider(
                api_base=_memory_cfg.embedding_api_base,
                api_key=_memory_cfg.embedding_api_key,
                model=_memory_cfg.embedding_model,
                dimensions=_memory_cfg.embedding_dim,
            )
            app.state.memory_embedding_provider = CircuitBreakerEmbeddingProvider(
                inner=_inner_provider,
                threshold=_memory_cfg.embedding_circuit_breaker_threshold,
                recovery_seconds=_memory_cfg.embedding_circuit_breaker_recovery_seconds,
            )
            logger.info("Memory Embedding Provider 初始化完成（CircuitBreaker wrapper）")
        else:
            app.state.memory_embedding_provider = DisabledEmbeddingProvider()
            logger.info("Memory Embedding 未启用，使用 DisabledEmbeddingProvider")

        # 7. 初始化 MemoryFlushService（C5.0 调度骨架 + C5.1 embed/write）
        from app.application.services.memory_flush_service import MemoryFlushService
        from app.infrastructure.repositories.db_memory_chunk_repository import (
            DBMemoryChunkRepository,
        )

        flush_service = MemoryFlushService(
            embedding_provider=app.state.memory_embedding_provider,
            session_factory=postgres_client.session_factory,
            repo_factory=DBMemoryChunkRepository,
            max_retries=_memory_cfg.flush_max_retries,
            circuit_breaker_threshold=_memory_cfg.flush_circuit_breaker_threshold,
        )
        app.state.flush_service = flush_service
        logger.info("MemoryFlushService 初始化完成")

        # 7b. FileMemoryStore 初始化（M1 PR-5A）：
        # FsMemoryWriter 接 ``memory_root_container`` 下的落盘路径。容器内的
        # 真实物理目录由 docker-compose 在 PR-6A 做 host 端 bind-mount；
        # 在开发 / 单元测试环境下目录可能尚不存在（M0 spike 未跑或 config
        # 指向 ``~/.actus/memory`` 默认值），FsMemoryWriter 内部首次 write
        # 时会 mkdir -p 创建用户子目录，不依赖构造期目录存在。
        from app.infrastructure.external.memory import FsMemoryWriter, FsReconciler

        _memory_root = settings.memory_root_container
        # EnsureUserMemoryDir lifespan hook（design P2）：lifespan 起手 mkdir -p
        # 一次 memory_root_container，让 compose bind 未就位的 dev / CI 环境也
        # 能跑起来；DockerSandbox mount + FsMemoryWriter 首次写都依赖这个根目录
        # 存在。非 root api 容器只要父目录可写即可 mkdir 子路径（design L78）。
        try:
            Path(_memory_root).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "memory_root_container %s mkdir 失败（可能是只读挂载或权限问题）: %s",
                _memory_root,
                exc,
            )
        app.state.file_memory_store = FsMemoryWriter(memory_root=_memory_root)
        logger.info(
            "FsMemoryWriter 初始化完成，memory_root=%s", _memory_root
        )

        # 7c. FsReconciler（PR-5B）—— 启动后台扫 fs_synced=false 的行做补写。
        # 懒式 per-user walk 由 SessionService.create_session 触发（DI 侧注入）。
        # 后台扫描以 best-effort 方式起：失败不拖垮 lifespan，日志里能看到
        # 就行；ops 可手动 `python -m app.cli.memory_reconcile` 兜底。
        fs_reconciler = FsReconciler(
            session_factory=postgres_client.session_factory,
            repo_factory=DBMemoryChunkRepository,
            file_store=app.state.file_memory_store,
            memory_root=_memory_root,
        )
        app.state.fs_reconciler = fs_reconciler

        async def _background_fs_reconcile() -> None:
            try:
                await fs_reconciler.scan_pending_fs_sync()
            except Exception:
                logger.exception("FsReconciler 启动扫描失败（非致命，等下轮）")

        app.state._fs_reconciler_task = asyncio.create_task(
            _background_fs_reconcile()
        )
        logger.info("FsReconciler 初始化完成，后台扫描已触发")

        # 8. 初始化 SandboxLifecycleService 单例（同 checkpointer_pool 模式，eng review #9）
        from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
        from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox
        from app.infrastructure.storage.postgres import get_uow
        SandboxLifecycleService.check_single_worker_argv()
        sandbox_lifecycle_service = SandboxLifecycleService(
            sandbox_cls=DockerSandbox,
            uow_factory=get_uow,
        )
        app.state.sandbox_lifecycle_service = sandbox_lifecycle_service
        logger.info(
            "SandboxLifecycleService 单例初始化完成 "
            "(Actus sandbox lifecycle running in SINGLE-WORKER mode)"
        )

        # 9. Reconcile orphans BEFORE confirmation sweep (eng review #12)
        await sandbox_lifecycle_service.reconcile_orphans()
        logger.info("Sandbox orphan reconciliation 完成")

        # 10. 创建 AgentService 单例 (D2)
        from app.interfaces.service_dependencies import _build_agent_service
        app.state.agent_service = _build_agent_service(
            minio_store=minio_client,
            redis_client=redis_client,
            checkpointer_pool=checkpointer_pool.pool,
            flush_service=flush_service,
            memory_embedding_provider=app.state.memory_embedding_provider,
            file_memory_store=getattr(app.state, "file_memory_store", None),
            sandbox_lifecycle_service=sandbox_lifecycle_service,
        )
        logger.info("AgentService 单例初始化完成")

        # 11. 启动 Confirmation Sweep 后台任务（扫描超时的危险工具确认）
        app.state.agent_service.start_sweep_task()
        logger.info("Confirmation sweep task 已启动")

        # Clean stale FINISHING sessions (best-effort: deferred_final_state lost on restart)
        try:
            from sqlalchemy import update
            from app.infrastructure.models.session import SessionModel
            from datetime import datetime, timedelta

            stale_threshold = datetime.now() - timedelta(seconds=120)
            async with postgres_client.session_factory() as db_session:
                stmt = (
                    update(SessionModel)
                    .where(
                        SessionModel.status == "finishing",
                        SessionModel.updated_at < stale_threshold,
                    )
                    .values(status="completed", completed_at=datetime.now())
                )
                result = await db_session.execute(stmt)
                await db_session.commit()
                if result.rowcount > 0:
                    logger.warning(
                        "postprocess_skipped_on_restart: cleaned %d stale FINISHING sessions",
                        result.rowcount,
                    )
        except Exception as e:
            logger.warning("Failed to clean stale FINISHING sessions: %s", e)

        # R3: Background scan for existing skills missing scan_report
        # Must start BEFORE yield (startup phase). After yield is shutdown.
        async def _background_skill_scan():
            """Startup background scan for existing skills (30s timeout)"""
            try:
                from pathlib import Path as _Path
                from app.domain.services.trust_matrix import scan_skill_source
                from app.domain.services.skills_guard import SkillsGuard
                from app.infrastructure.repositories.file_skill_repository import FileSkillRepository

                repo = FileSkillRepository(settings.skills_root_dir)
                skills_root = _Path(settings.skills_root_dir)
                skills = await repo.list()
                scanned = 0
                for skill in skills:
                    skill_dir = skills_root / skill.id
                    if skill.scan_report and skill.scan_report.get("content_hash"):
                        current_hash = SkillsGuard.compute_content_hash(skill_dir)
                        if current_hash == skill.scan_report["content_hash"]:
                            continue
                    report = scan_skill_source(skill.runtime_type, skill_dir)
                    skill.scan_report = report.to_dict()
                    if not skill.trust_origin or skill.trust_origin == "":
                        skill.trust_origin = "user_installed"
                    await repo.upsert(skill)
                    scanned += 1
                logger.info("R3 startup scan complete: %d skills scanned", scanned)
            except asyncio.TimeoutError:
                logger.warning("R3 startup scan timed out (30s), remaining skills keep dangerous default")
            except Exception:
                logger.exception("R3 startup scan failed (non-fatal)")

        app.state._r3_scan_task = asyncio.create_task(
            asyncio.wait_for(_background_skill_scan(), timeout=30.0)
        )

        # lifespan分界点
        yield
    finally:
        try:
            logger.info("Manus应用正在关闭")
            agent_svc = getattr(app.state, "agent_service", None)
            if agent_svc:
                await asyncio.wait_for(agent_svc.shutdown(), timeout=30.0)
            logger.info("Agent服务成功关闭")
        except asyncio.TimeoutError:
            logger.warning("Agent服务关闭超时, 强制关闭, 部分任务将被释放")
        except Exception as e:
            logger.error(f"Agent服务关闭期间出现错误: {str(e)}")

        # 关闭 SandboxLifecycleService（在 AgentService 之后——agent 可能持有 handle）
        lifecycle_svc = getattr(app.state, "sandbox_lifecycle_service", None)
        if lifecycle_svc:
            try:
                await asyncio.wait_for(lifecycle_svc.shutdown(), timeout=15.0)
                logger.info("SandboxLifecycleService 关闭成功")
            except asyncio.TimeoutError:
                logger.warning("SandboxLifecycleService 关闭超时")
            except Exception as e:
                logger.warning(f"SandboxLifecycleService 关闭时出错: {e}")

        # 关闭 MemoryFlushService（等待后台 flush 任务完成）
        flush_service = getattr(app.state, "flush_service", None)
        if flush_service:
            try:
                await flush_service.shutdown()
                logger.info("MemoryFlushService 关闭成功")
            except Exception as e:
                logger.warning(f"MemoryFlushService 关闭时出错: {e}")

        # 应用关闭前的清理工作
        if checkpointer_pool is not None:
            await checkpointer_pool.close()
        await redis_client.shutdown()
        await postgres_client.shutdown()
        await minio_client.shutdown()
        logger.info("Manus应用关闭成功")


app = FastAPI(
    title="Actus通用智能体",
    description="Actus是一个通用的AI Agent系统，可以完全私有部署，使用A2A+MCP连接Agent/Tool，同时支持在沙箱中运行各种内置工具和操作",
    lifespan=lifespan,
    openapi_tags=openapi_tags,
    version="1.0.0",
)

# 配置CORS中间件，解决跨域问题
_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """请求体大小限制中间件"""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_request_body_size:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large"},
        )
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if len(body) > settings.max_request_body_size:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )
    return await call_next(request)


# B5 PR-S1-5: ObservabilityMiddleware 最后注册 → user_middleware[0]
# → outermost frame，先于 CORS / body_limit / 异常处理器执行，使
# 后续中间件、handler、lifespan 都能在 contextvar 上读到 trace_id /
# request_id。Starlette 通过 ``user_middleware.insert(0, …)`` 实现
# "最后 add → 最外层" 语义；测试 ``test_observability_middleware
# _registration_outermost`` 锁死该顺序。
app.add_middleware(ObservabilityMiddleware)

# 注册全局异常处理器
register_exception_handlers(app)

app.include_router(api_router, prefix="/api")

logger.info("FastAPI应用程序实例已创建。")
