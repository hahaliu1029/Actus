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
from app.interfaces.endpoints.extension_governance_routes import (
    router as extension_governance_router,
)
from app.interfaces.endpoints.plugin_routes import (
    router as plugin_router,
)
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


# C3 PR-3c (codex r13 [HIGH CANCELLATION] fix) — GC anchor + observability
# for the lifespan stale-FINISHING cleanup fire-and-forget supervisor stops.
# Without this hard-reference set, asyncio could collect the task before
# its done callback fires, swallowing any exception inside the stop body.
# Mirrors ``_PENDING_MAILBOX_STOP_TASKS`` in agent_service.py /
# execution_supervisor.py — same purpose, distinct namespace so the
# lifespan-only batch is observable separately from per-request stops.
_STALE_FINISHING_STOP_TASKS: set[asyncio.Task] = set()


def _on_stale_finishing_stop_done(task: asyncio.Task) -> None:
    _STALE_FINISHING_STOP_TASKS.discard(task)
    if task.cancelled():
        logger.warning(
            "stale-FINISHING stop task %s was cancelled unexpectedly",
            task.get_name(),
        )
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "stale-FINISHING cleanup: supervisor stop task %s raised: %s",
            task.get_name(),
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


# ---------------------------------------------------------------------------
# B9 运行时扩展探测（Task 14）—— 后台循环 + lifespan 接线
# ---------------------------------------------------------------------------
def _start_b9_probe(app: FastAPI) -> None:
    """B9 探测启动——任何失败 fail-open：只禁能力，不阻断 app startup（spec §3.5）。"""
    # R2#2：安全默认值先落位，一切 import（含新模块导入期异常）都在 try 内
    app.state.runtime_liveness_registry = None
    app.state.extension_probe_service = None
    app.state.extension_probe_started = False
    try:
        from app.application.services.extension_probe_service import (
            DefaultExtensionProber, ExtensionProbeService,
        )
        from app.domain.services.runtime_liveness_registry import runtime_liveness_registry
        from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
        from app.interfaces.service_dependencies import _load_app_config

        app.state.runtime_liveness_registry = runtime_liveness_registry
        probe_service = ExtensionProbeService(
            config_provider=_load_app_config,
            skill_repository=FileSkillRepository(settings.skills_root_dir),
            prober=DefaultExtensionProber(),
            probe_flag_provider=lambda: _load_app_config().tool_runtime.extension_probe_enabled,
            liveness_view=runtime_liveness_registry,
            # R2#F6：治理观测生产链第 5 跳（mode=off → None；T9 保证 port 构造先于 probe）。
            # getattr 兜底：port 未装配（off / 早期 state）→ None = 零 registry 交互。
            admission_port=getattr(app.state, "extension_admission_port", None),
        )
        app.state.extension_probe_service = probe_service
        task = asyncio.create_task(probe_service.run_loop())
        task.add_done_callback(_log_b9_probe_exit)
        app.state._extension_probe_task = task
        app.state.extension_probe_started = True
    except Exception:
        logger.warning("B9 探测服务启动失败（fail-open，能力禁用）", exc_info=True)
        app.state.extension_probe_service = None
        app.state.extension_probe_started = False


def _log_b9_probe_exit(task: "asyncio.Task") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("B9 探测循环非预期退出", exc_info=exc)


# B9 shutdown 等待上限（秒）——提取为模块常量，测试可 monkeypatch 缩小。
PROBE_SHUTDOWN_WAIT_SECONDS = 5.0


async def _stop_b9_probe(app: FastAPI) -> None:
    """B9 探测后台 task 停机（belt-and-suspenders 三重防护）。

    底层 MCP init/cleanup（``MCPClientManager.initialize()`` / ``cleanup()``，被
    prober 在 ``_probe_locked`` 内调用）含 ``except BaseException`` 块，可能吞掉
    lifespan 投递的 ``asyncio.CancelledError``。单靠 ``task.cancel()`` 无法保证退出，
    故三层叠加：
    1. **cooperative ``_stopping``**（``service.shutdown()``）——置 ``_stopping=True``，
       即便 cancel 被吞，``run_loop`` 也会在至多一次 tick+sleep 后自然退出。
    2. **``task.cancel()``**——正常路径立即中断在 sleep/await 上的循环。
    3. **bounded ``asyncio.wait``**——即便前两者都失效，也只等 ``PROBE_SHUTDOWN_WAIT_SECONDS``
       就 fail-open 放弃，绝不让 shutdown 无限挂起（不用 ``wait_for``：它 cancel 后
       还要等任务确认取消，对吞 cancel 的任务同样会挂——正是本修复要防的失效模式）。
    """
    probe_service = getattr(app.state, "extension_probe_service", None)
    if probe_service is not None:
        # 先置 _stopping（cooperative stop），保证 cancel 被吞时循环仍会退出。
        try:
            await probe_service.shutdown()
        except Exception:  # noqa: BLE001 - shutdown 仅置 flag，异常不阻断停机
            logger.warning("B9 probe service.shutdown() 出错（继续 cancel）", exc_info=True)

    probe_task = getattr(app.state, "_extension_probe_task", None)
    if probe_task:
        probe_task.cancel()
        # 用 ``asyncio.wait``（非 ``wait_for``）做 bounded 等待：``wait_for`` 在超时时
        # 会去 cancel 并 await 内层 task 的取消确认——若底层 MCP init 的
        # ``except BaseException`` 把取消也吞掉，``wait_for`` 会永久挂在等确认上，达不到
        # "5s 上限" 的承诺。``asyncio.wait`` 只返回 (done, pending)，不要求 task 确认取消，
        # 因此即便 task 完全吞掉 cancel 也保证 ≤5s 返回（真·fail-open）。cooperative
        # ``_stopping`` 已在上面置位，生产路径下 task 会自行退出、落 done。
        try:
            done, _pending = await asyncio.wait(
                {probe_task}, timeout=PROBE_SHUTDOWN_WAIT_SECONDS
            )
        except (asyncio.CancelledError, Exception):
            return
        if not done:
            logger.warning(
                "B9 probe task 未在 5s 内退出——放弃等待（fail-open shutdown）"
            )
        else:
            # task 已退出——消费其异常（若有）避免 "exception never retrieved" 警告。
            exc_task = next(iter(done))
            if not exc_task.cancelled():
                exc_task.exception()


# ---------------------------------------------------------------------------
# B9 运行时扩展统计（Task 19）—— 有界队列 recorder + Redis flusher + lifespan 接线
# ---------------------------------------------------------------------------
def _start_b9_stats(app: FastAPI, redis_client) -> None:
    """B9 统计启动——**仅 flag on 时**构造 recorder/flusher 并置 started=True。

    **与 probe 侧的有意差异（此处冻结，R2#3）**：统计记录是工具执行热路径的组装期
    注入——recorder 若无条件构造，flag-off 部署仍会记录。故 stats 用**条件注入**：
    ``extension_stats_enabled`` on 才构造 ``RedisExtensionStats`` + 起 flusher 后台
    task；off→on 翻转需重启进程才开始记录（wart 已随 plan 审查日志归档）；on→off 翻转
    时 GET 顶层 ``stats_enabled`` 位即时变 False（getter 以 ``started ∧ flag`` 合成），
    但已注入的 recorder 持续记录到重启——可接受。

    任何失败 fail-open：只禁能力（``extension_stats=None`` / ``started=False``），
    不阻断 app startup。
    """
    # R2#2 同款：安全默认值先落位，一切 import / 构造异常都在 try 内。
    app.state.extension_stats = None
    app.state.extension_stats_started = False
    try:
        from app.interfaces.service_dependencies import _load_app_config

        if not _load_app_config().tool_runtime.extension_stats_enabled:
            logger.info("B9 扩展统计未启用（flag off）——跳过 recorder/flusher 构造")
            return

        from app.infrastructure.external.runtime_stats.redis_extension_stats import (
            RedisExtensionStats,
        )

        stats = RedisExtensionStats(redis_client.client)
        app.state.extension_stats = stats
        task = asyncio.create_task(stats.run_flusher())
        task.add_done_callback(_log_b9_stats_exit)
        app.state._extension_stats_task = task
        app.state.extension_stats_started = True
    except Exception:
        logger.warning("B9 统计服务启动失败（fail-open，能力禁用）", exc_info=True)
        app.state.extension_stats = None
        app.state.extension_stats_started = False


def _log_b9_stats_exit(task: "asyncio.Task") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("B9 统计 flusher 循环非预期退出", exc_info=exc)


# B9 统计 shutdown 等待上限（秒）——drain ≤2s 由 RedisExtensionStats.shutdown() 内部
# 保证，这里再包一层 bounded wait 兜底（对齐 probe 侧防吞 cancel 的三层防护动机）。
STATS_SHUTDOWN_WAIT_SECONDS = 5.0


async def _stop_b9_stats(app: FastAPI) -> None:
    """B9 统计停机（对齐 probe 侧 shape）：recorder 停收 → flusher 限时 drain → cancel。

    顺序（spec §9）：先 ``stats.shutdown()``（置 ``_closed`` 拒收 + 限时 drain ≤2s），
    再 cancel flusher task，最后 bounded ``asyncio.wait`` 兜底——即便 flusher 卡住也只
    等 ``STATS_SHUTDOWN_WAIT_SECONDS`` 就 fail-open 放弃，绝不让 shutdown 无限挂起。
    无 stats（flag off / 构造失败 fail-open 路径）时静默 no-op。
    """
    stats = getattr(app.state, "extension_stats", None)
    if stats is not None:
        try:
            await stats.shutdown()
        except Exception:  # noqa: BLE001 - drain 异常不阻断停机
            logger.warning("B9 stats.shutdown() 出错（继续 cancel）", exc_info=True)

    stats_task = getattr(app.state, "_extension_stats_task", None)
    if stats_task:
        stats_task.cancel()
        try:
            done, _pending = await asyncio.wait(
                {stats_task}, timeout=STATS_SHUTDOWN_WAIT_SECONDS
            )
        except (asyncio.CancelledError, Exception):
            return
        if not done:
            logger.warning(
                "B9 stats flusher task 未在 %.0fs 内退出——放弃等待（fail-open shutdown）",
                STATS_SHUTDOWN_WAIT_SECONDS,
            )
        else:
            exc_task = next(iter(done))
            if not exc_task.cancelled():
                exc_task.exception()


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

        # D1a §4.1：治理 port 单例（mode=off → 不构造=None，INV-D1-0 ①层）
        # 位置=最早段（R2#F2）：DB ready 后、首次 config load 与 _start_b9_probe 之前，
        # 使 _governance_mode 与 port 类对后续 T14 sweep / T23 closure 在此前已定义。
        _governance_mode = settings.extension_governance_mode
        if _governance_mode != "off":
            from app.infrastructure.external.governance.db_extension_admission import (
                DbExtensionAdmissionPort,
            )
            from app.infrastructure.external.governance.db_extension_registry import (
                DbExtensionRegistryReadPort,
                DbExtensionRegistryWritePort,
            )
            app.state.extension_admission_port = DbExtensionAdmissionPort(
                postgres_client.session_factory, mode=_governance_mode)
            app.state.extension_registry_write_port = DbExtensionRegistryWritePort(
                postgres_client.session_factory)
            app.state.extension_registry_read_port = DbExtensionRegistryReadPort(
                postgres_client.session_factory)
        else:
            app.state.extension_admission_port = None
            app.state.extension_registry_write_port = None
            app.state.extension_registry_read_port = None
        # off = 零新增日志 I/O：仅治理开启时才记 mode 行（off 缺省下不 emit）。
        if _governance_mode != "off":
            logger.info("D1a extension governance mode=%s", _governance_mode)

        # D1a startup 四段全序①（②saga 收尾在 T23 插入本行之后、normal load 之前）
        # 清扫 config 目录孤儿 temp（SIGKILL 残留含 secrets，不过夜；§8.3-3 R51#3）。
        # mode=off zero behavior change：仅 mode≠off 才做此启动新工作。
        if _governance_mode != "off":
            from app.infrastructure.repositories.file_app_config_repository import (
                sweep_orphan_config_temps,
            )
            sweep_orphan_config_temps(str(settings.app_config_filepath))

        # D1a startup 四段全序②（saga 收尾）：孤儿 in_progress install → 逆序补偿；孤儿 uninstall →
        # 标 failed；终态 op 的 staging 目录整删。**必须先于 ③normal config load**——config 不可读
        # 场景下普通 load 抛 ServerRequestsError 中止 boot，closure 排后则恢复不可达（R52#2）；
        # migration 已在更早段（command.upgrade）跑完，DB 可用窗口成立。局部构造最小实例（不依赖后续
        # lifespan 服务）；config 不可读经 _tolerant_load_config_entries 三态兜底（不触 config 依赖）。
        # mode=off zero behavior change：off 下治理阻断与收尾全退场（§6.0/INV-D1-0）。
        if _governance_mode != "off":
            try:
                from app.application.services.app_config_service import AppConfigService
                from app.application.services.extension_reconciler import (
                    ExtensionReconciler,
                )
                from app.application.services.plugin_install_service import (
                    run_startup_saga_closure,
                )
                from app.application.services.skill_service import SkillService
                from app.infrastructure.external.governance.plugin_saga_store import (
                    PluginSagaStore,
                )
                from app.infrastructure.repositories.file_app_config_repository import (
                    FileAppConfigRepository,
                )
                from app.infrastructure.repositories.file_skill_repository import (
                    FileSkillRepository,
                )

                _closure_write_port = app.state.extension_registry_write_port
                _closure_read_port = app.state.extension_registry_read_port
                _closure_reconciler = ExtensionReconciler(
                    _closure_write_port, app.state.extension_admission_port)
                await run_startup_saga_closure(
                    saga_store=PluginSagaStore(postgres_client.session_factory),
                    app_config_service=AppConfigService(
                        FileAppConfigRepository(settings.app_config_filepath),
                        reconciler=_closure_reconciler,
                        registry_read_port=_closure_read_port),
                    skill_service=SkillService(
                        FileSkillRepository(settings.skills_root_dir),
                        registry_write_port=_closure_write_port,
                        registry_read_port=_closure_read_port),
                    write_port=_closure_write_port,
                    config_path=str(settings.app_config_filepath))
            except Exception:
                logger.error(
                    "D1a saga closure failed; boot continues (registry 阻断态保守)",
                    exc_info=True)

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

        # 7a. B9 运行时扩展探测（Task 14）——service + 后台 task 无条件构造/启动
        # （不以 flag 为条件；循环内每 tick 经 probe_flag_provider 自检，flag off 空转）。
        # 任何失败 fail-open：只禁能力，不阻断 startup。
        _start_b9_probe(app)
        logger.info(
            "B9 扩展探测启动完成 (started=%s)",
            getattr(app.state, "extension_probe_started", False),
        )

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

        # C3 PR-3c — forward reference holder for the deferred lifecycle
        # service. The supervisor factory closure needs the lifecycle
        # service, but SandboxLifecycleService itself needs the supervisor
        # registry to be injectable (see ``reconcile_orphans``). We break
        # the cycle by giving the factory a tiny adapter that resolves
        # lifecycle lazily through this dict, then filling the dict
        # immediately after construct. The supervisor isn't consuming
        # envelopes until PR-4 flips the flag-gated handlers, so the
        # brief construct-then-fill sequence is safe.
        _pending_lifecycle_ref: dict[str, object] = {}

        class _DeferredLifecycle:
            async def destroy(self, session_id: str, reason) -> None:
                svc = _pending_lifecycle_ref.get("svc")
                if svc is None:
                    logger.warning(
                        "mailbox supervisor destroy called before "
                        "lifecycle service ready session=%s",
                        session_id,
                    )
                    return
                await svc.destroy(session_id, reason)

            async def bind_new(self, session_id: str, *, user_id=None):
                # C2 finish-core F1.7 — the coordinator child_runner_starter
                # binds a fresh per-child sandbox lease via this proxy. The
                # forward reference is filled immediately after
                # SandboxLifecycleService construction (a few lines below), so
                # any real child dispatch happens well after bind. A None svc
                # here is a composition-order bug, not a normal startup race.
                svc = _pending_lifecycle_ref.get("svc")
                if svc is None:
                    raise RuntimeError(
                        "coordinator child bind_new before SandboxLifecycleService "
                        "ready — composition order bug"
                    )
                return await svc.bind_new(session_id, user_id=user_id)

        # C3 PR-6 (spec §11.7) — mailbox plane is the only supported
        # control plane. SupervisorRegistry is built unconditionally;
        # the ``mailbox_supervisor_enabled`` env-var rollback gate is
        # decommissioned. Init failure still raises (fail-closed) so
        # mis-configuration surfaces at pod-start instead of as silent
        # zero-throughput.
        from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
            RedisMailboxPublisher,
        )
        from app.interfaces.service_dependencies import (
            build_coordinator_runtime_deps,
            build_supervisor_registry,
            get_policy_snapshot_sink,
        )

        mailbox_publisher = RedisMailboxPublisher(redis_client.client)

        # PR-9b-A Task A8 — coordinator runtime deps composition root.
        # Wiring is **always-live**: the runtime feature flag
        # ``ACTUS_C2_COORDINATOR_ENABLED`` gates ENTRY at
        # ``main_graph.py:695`` (whether coordinator behaviour fires), NOT
        # WIRING (whether the deps are threaded through). The helper
        # populates ``app.state.cost_rollup_service``,
        # ``app.state.coordinator_envelope_store``,
        # ``app.state.patch_applier_deps`` (and many more) — required by the
        # two new ``build_supervisor_registry`` kwargs immediately below.
        #
        # The narrow ``except RuntimeError`` below ONLY catches the specific
        # lifespan-mock test-env condition where the canonical
        # ``app.infrastructure.storage.postgres.get_postgres`` singleton has
        # not been initialised. ``Postgres.session_factory`` raises
        # ``RuntimeError("Postgres数据库客户端未初始化，请先调用init方法进行初始化。")``
        # at ``postgres.py:82`` in that case. Any other exception
        # (ImportError, TypeError, missing attribute, ...) is a real bug
        # and MUST propagate so lifespan fails loudly rather than silently
        # downgrading to the null-deps fallback.
        # C2 finish-core F1.7 — one shared _DeferredLifecycle instance is
        # reused by BOTH the coordinator child_runner_starter (bind_new) and
        # the supervisor registry (destroy); both resolve the live
        # SandboxLifecycleService lazily through ``_pending_lifecycle_ref``
        # which is filled immediately after construction below.
        deferred_lifecycle = _DeferredLifecycle()
        from app.infrastructure.external.task.redis_stream_task import (
            RedisStreamTask,
        )

        def _resolve_child_runner_deps():
            # Pragmatic coupling: reads AgentService internals to assemble child deps (no public accessor); resolved lazily post-lifespan when app.state.agent_service is live.
            # C2 finish-core F1.7 — resolve the process-shared child-runner deps
            # LAZILY off the live AgentService (built later in this lifespan).
            # Only invoked at child-build time (a real coordinator dispatch),
            # which is always well after AgentService construction.
            from app.interfaces.service_dependencies import ChildRunnerSharedDeps
            svc = app.state.agent_service
            snap = svc._config_snapshot
            return ChildRunnerSharedDeps(
                uow_factory=get_uow,
                llm=snap.llm,
                agent_config=snap.agent_config,
                mcp_config=snap.mcp_config,
                a2a_config=snap.a2a_config,
                file_storage=svc._file_storage,
                search_engine=svc._search_engine,
                checkpointer_pool=svc._checkpointer_pool,
                execution_supervisor=svc._supervisor,
                session_state_machine=svc._ssm,
                # B1: same-source as the root runner (agent_service.py builds
                # the root with tool_runtime=snap.tool_runtime); child inherits
                # the identical ToolRuntimeConfig so B1 flags flipped on root
                # don't silently stay default-OFF for child graphs (spec R2#1).
                tool_runtime=snap.tool_runtime,
                # C7: child runner 同源持有 lifecycle flags——child 照常投影其
                # plan/step/tool lifecycle（§12-8 positive acceptance）；task 层
                # 由 _is_root_session() gate 在 runner 内拦（PR4）。
                lifecycle_runtime=snap.lifecycle_runtime,
                # B9 Task 20 (R4#1): child extension calls feed the same stats
                # recorder as root. None when flag off.
                extension_stats_recorder=getattr(
                    app.state, "extension_stats", None
                ),
                # B12 follow-up: thread the same file_view multimodal deps the root
                # runner gets (agent_service builds the root registry from these).
                # The child builds its OWN sandbox-bound registry in _build.
                supports_vision=snap.supports_vision,
                supports_pdf_input=snap.supports_pdf_input,
                file_understanding_config=snap.file_understanding_config,
                vision_fallback_model=snap.vision_fallback_model,
                # D1a §4.1 注入链第 2 跳：root/child 共享同一 AdmissionPort 实例
                # （admission 必须同源）。None when mode off. Built earlier in lifespan.
                extension_admission_port=app.state.extension_admission_port,
            )

        coord_deps = None
        try:
            coord_deps = build_coordinator_runtime_deps(
                app_state=app.state,
                redis_client=redis_client,
                resolve_child_runner_deps=_resolve_child_runner_deps,
                sandbox_lifecycle_service=deferred_lifecycle,
                child_runner_task_cls=RedisStreamTask,
            )
            logger.info(
                "_CoordinatorRuntimeDeps composition root initialised "
                "(PR-9b-A: 17 lifespan-scoped singletons on app.state)"
            )
        except RuntimeError as exc:
            msg = str(exc)
            if (
                "数据库客户端未初始化" not in msg
                and "client not initialised" not in msg.lower()
                and "client not initialized" not in msg.lower()
            ):
                # Real production failure (e.g. coordinator-limit env parse
                # error raising RuntimeError) — surface loudly. Do NOT
                # silently fall back to null-deps; that would mask a config
                # bug as a "lifespan-mock context".
                raise
            # Test-env path: ``app.main.get_postgres`` is patched but the
            # canonical ``app.infrastructure.storage.postgres.get_postgres``
            # is not, so ``Postgres.session_factory`` raises. Log and fall
            # through to the null-deps sentinel so the rest of lifespan
            # still drives existing assertions; production lifespan always
            # initialises the canonical client before reaching here.
            logger.warning(
                "build_coordinator_runtime_deps skipped — Postgres client "
                "not initialized (expected only in lifespan-mock test "
                "contexts): %s",
                exc,
            )
            from app.application.services.coordinator_runtime_deps import (
                _NullCoordinatorRuntimeDeps,
            )

            coord_deps = _NullCoordinatorRuntimeDeps()
            app.state.coord_deps = coord_deps
            app.state.coordinator_envelope_store = getattr(
                app.state, "coordinator_envelope_store", None,
            )
            app.state.cost_rollup_service = getattr(
                app.state, "cost_rollup_service", None,
            )
            app.state.patch_applier_deps = getattr(
                app.state, "patch_applier_deps", None,
            )
        supervisor_registry = build_supervisor_registry(
            redis_client=redis_client,
            publisher=mailbox_publisher,
            # C2 finish-core F1.7 — same deferred lifecycle instance as the
            # coordinator child_runner_starter (single source of truth).
            sandbox_lifecycle_service=deferred_lifecycle,
            coordinator_envelope_store=app.state.coordinator_envelope_store,
            cost_rollup_service=app.state.cost_rollup_service,
        )
        logger.info(
            "SupervisorRegistry 单例初始化完成 (C3 PR-6 — mailbox plane mandatory)"
        )
        app.state.supervisor_registry = supervisor_registry

        sandbox_lifecycle_service = SandboxLifecycleService(
            sandbox_cls=DockerSandbox,
            uow_factory=get_uow,
            supervisor_registry=supervisor_registry,
            sink=get_policy_snapshot_sink(),  # C5a: Logging sink when flag ON, else Noop
            policy_snapshot_enabled=get_settings().sandbox_policy_compiler_enabled,  # C5a Seam A gate (read once)
            runtime_hardening_enabled=get_settings().sandbox_runtime_hardening_enabled,  # C5c gate (read once)
        )
        app.state.sandbox_lifecycle_service = sandbox_lifecycle_service
        # Fill the forward reference now that lifecycle service is live.
        _pending_lifecycle_ref["svc"] = sandbox_lifecycle_service
        logger.info(
            "SandboxLifecycleService 单例初始化完成 "
            "(Actus sandbox lifecycle running in SINGLE-WORKER mode)"
        )

        # 9. Reconcile orphans BEFORE confirmation sweep (eng review #12)
        await sandbox_lifecycle_service.reconcile_orphans()
        logger.info("Sandbox orphan reconciliation 完成")

        # 9b. B9 运行时扩展统计（Task 19）——**必须在 _build_agent_service 之前**：
        # recorder 单例先就绪，才能进 AgentService → AgentTaskRunner 的热路径组装链
        # （启动顺序硬约束）。stats 用条件注入（仅 flag on 时构造，见 _start_b9_stats
        # docstring 冻结语义）；任何失败 fail-open 只禁能力。
        _start_b9_stats(app, redis_client)
        logger.info(
            "B9 扩展统计启动完成 (started=%s)",
            getattr(app.state, "extension_stats_started", False),
        )

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
            supervisor_registry=supervisor_registry,
            # PR-9b-A Task A8: forward the lifespan-scoped
            # ``_CoordinatorRuntimeDeps`` aggregator built above into the
            # AgentService → AgentTaskRunner → PlannerReActFlow chain.
            coord_deps=app.state.coord_deps,
            # B9 Task 20: lifespan-scoped extension stats recorder. None when
            # flag off (see _start_b9_stats). Threaded into the hot path so
            # react_graph埋点 records ext tool calls.
            extension_stats_recorder=getattr(app.state, "extension_stats", None),
            # D1a §4.1 (R2#F12): governance AdmissionPort singleton built above
            # (None when mode off). Threaded into AgentService → AgentTaskRunner →
            # SkillTool / SkillBundleSyncManager. Root/child same-source instance.
            extension_admission_port=app.state.extension_admission_port,
        )
        logger.info("AgentService 单例初始化完成")

        from app.domain.services.idle_watchdog import IdleWatchdog

        app.state.supervisor = app.state.agent_service._supervisor
        app.state.idle_watchdog = IdleWatchdog(
            redis_client=redis_client,
            supervisor=app.state.supervisor,
            uow_factory=get_uow,
            notification_emitter=getattr(
                app.state.agent_service,
                "_memory_notification_emitter",
                None,
            ),
        )
        app.state.agent_service._idle_watchdog = app.state.idle_watchdog
        await app.state.supervisor.script_load_all()
        await app.state.supervisor.reconcile_running_background_at_boot(
            notification_emitter=getattr(
                app.state.agent_service,
                "_memory_notification_emitter",
                None,
            )
        )
        app.state.idle_watchdog.start()
        logger.info("ExecutionSupervisor / IdleWatchdog 单例初始化完成")

        # 11. 启动 Confirmation Sweep 后台任务（扫描超时的危险工具确认）
        app.state.agent_service.start_sweep_task()
        logger.info("Confirmation sweep task 已启动")

        # Clean stale FINISHING sessions (best-effort: deferred_final_state lost on restart)
        try:
            from datetime import datetime, timedelta

            from sqlalchemy import select

            from app.domain.models.session import SessionStatus
            from app.infrastructure.models.session import SessionModel
            from app.infrastructure.repositories.db_session_repository import (
                DBSessionRepository,
            )

            stale_threshold = datetime.now() - timedelta(seconds=120)
            async with postgres_client.session_factory() as db_session:
                stmt = (
                    select(SessionModel.id).where(
                        SessionModel.status == "finishing",
                        SessionModel.updated_at < stale_threshold,
                    )
                )
                result = await db_session.execute(stmt)
                session_ids = [str(row.id) for row in result.all()]
                from app.application.composition.graph_assembly import (
                    build_session_state_machine,
                )

                repo = DBSessionRepository(db_session=db_session)
                ssm = build_session_state_machine(uow_factory=get_uow)
                for session_id in session_ids:
                    await ssm.terminate(
                        session_id,
                        SessionStatus.COMPLETED,
                        "server_restart",
                        session_repo=repo,
                    )
                await db_session.commit()
                if session_ids:
                    logger.warning(
                        "postprocess_skipped_on_restart: cleaned %d stale FINISHING sessions",
                        len(session_ids),
                    )
                # codex r7 [HIGH CONTRACT] — stale-FINISHING cleanup is a
                # third non-runner terminal-write path (besides AgentService
                # and ExecutionSupervisor). ``reconcile_orphans`` ran above
                # at lifespan step 9 (line ~303) and may have spawned a
                # MailboxSupervisor for any of these stale roots via the
                # PR-3c §11.3 dual path; stop those slots NOW so the
                # registry stays consistent with the freshly written
                # terminal status. Best-effort — failure logged + swallowed
                # because the cleanup outer try/except already catches.
                # C3 PR-3c (codex r13 [HIGH CANCELLATION]) — fire-and-forget.
                # Earlier rounds awaited each stop in this loop, but that
                # creates the same cancellation/hang seam codex removed from
                # AgentService and ExecutionSupervisor in r12: an outer
                # CancelledError or a hung registry.stop() would skip the
                # remaining stale-root stops. Spawn tasks that asyncio
                # anchors via the supervisor's done callback chain — they
                # complete in the background; ``SupervisorRegistry.stop_all``
                # at lifespan shutdown sweeps any in-flight ones.
                if session_ids and supervisor_registry is not None:
                    for session_id in session_ids:
                        stop_task = asyncio.create_task(
                            supervisor_registry.stop(session_id),
                            name=f"mailbox-stop-stale-finishing-{session_id}",
                        )
                        _STALE_FINISHING_STOP_TASKS.add(stop_task)
                        stop_task.add_done_callback(
                            _on_stale_finishing_stop_done
                        )
        except Exception as e:
            logger.warning("Failed to clean stale FINISHING sessions: %s", e)

        # C2b child-row reaper (durable zombie-RUNNING backstop).
        # Terminalize coordinator-child rows still RUNNING whose terminal
        # envelope is already persisted (the runner's best-effort row write
        # failed, or the runner was SIGKILLed after the envelope was
        # persisted). Match-only — never backfills; skips no-envelope
        # children so reconcile_orphans' supervisor re-spawn trigger is
        # preserved (spec §4.2). Mirrors the stale-FINISHING cleanup above.
        # OUTER best-effort try: a query/DI failure logs + is swallowed so it
        # never aborts lifespan startup.
        try:
            from app.application.composition.graph_assembly import (
                build_session_state_machine,
            )
            from app.application.services.child_terminal_reconciler import (
                sweep_running_mailbox_children,
            )
            from app.infrastructure.repositories.db_session_repository import (
                DBSessionRepository,
            )

            child_reaper_store = getattr(
                app.state, "coordinator_envelope_store", None
            )
            if child_reaper_store is None:
                logger.info(
                    "child_row_reaper: coordinator_envelope_store unavailable "
                    "— skipping sweep"
                )
            else:
                async with postgres_client.session_factory() as db_session:
                    repo = DBSessionRepository(db_session=db_session)
                    ssm = build_session_state_machine(uow_factory=get_uow)
                    stats = await sweep_running_mailbox_children(
                        session_repo=repo,
                        envelope_store=child_reaper_store,
                        state_machine=ssm,
                        uow_factory=get_uow,
                    )
                    if stats.terminalized or stats.errored:
                        logger.warning(
                            "child_row_reaper: scanned=%d terminalized=%d "
                            "skipped=%d errored=%d",
                            stats.scanned,
                            stats.terminalized,
                            stats.skipped,
                            stats.errored,
                        )
        except Exception as e:
            logger.warning("child_row_reaper: sweep failed (swallowed): %s", e)

        # C2 coordinator-cancel Part B — leaked-sandbox startup reaper.
        # On user-stop the root MailboxSupervisor is killed before consuming the
        # cancelled children's CANCEL_ACK, so each child's per-child sandbox
        # container is left ACTIVE with no reaper (F0.6/F0.7). This match-only
        # startup sweep destroys those leaked containers idempotently
        # (restart-bounded — NG8). Mirrors the C2b child-row reaper above.
        # OUTER best-effort try: a query/DI failure logs + is swallowed so it
        # never aborts lifespan startup.
        try:
            from app.application.services.sandbox_terminal_reaper import (
                sweep_terminal_coordinator_active_sandboxes,
            )
            from app.infrastructure.repositories.db_session_repository import (
                DBSessionRepository,
            )

            sandbox_reaper_svc = getattr(
                app.state, "sandbox_lifecycle_service", None
            )
            if sandbox_reaper_svc is None:
                logger.info(
                    "sandbox_reaper: lifecycle service unavailable — skipping sweep"
                )
            else:
                async with postgres_client.session_factory() as db_session:
                    repo = DBSessionRepository(db_session=db_session)
                    # Total budget for the sweep. NOTE: this wait_for only bounds
                    # the async-cancellable portion — DockerSandbox.get()/destroy()
                    # still make SYNCHRONOUS Docker SDK calls on the event loop
                    # (docker_sandbox.py: containers.get/reload/remove), so a fully
                    # hung Docker daemon can still block startup past this budget.
                    # That is a PRE-EXISTING systemic exposure shared with
                    # reconcile_orphans() above (which awaits the same Docker path
                    # with no bound at all); the complete fix (async-safe
                    # DockerSandbox via asyncio.to_thread + per-call client timeout)
                    # is a deferred follow-up. On a cancellable timeout the outer
                    # best-effort except logs + swallows; leaked sandboxes are
                    # re-scanned next boot (restart-bounded — NG8).
                    stats = await asyncio.wait_for(
                        sweep_terminal_coordinator_active_sandboxes(
                            session_repo=repo,
                            lifecycle_service=sandbox_reaper_svc,
                        ),
                        timeout=30.0,
                    )
                    if stats.destroyed or stats.errored:
                        logger.warning(
                            "sandbox_reaper: scanned=%d destroyed=%d "
                            "already_gone=%d errored=%d",
                            stats.scanned,
                            stats.destroyed,
                            stats.already_gone,
                            stats.errored,
                        )
        except Exception as e:
            logger.warning("sandbox_reaper: sweep failed (swallowed): %s", e)

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

        # B8: 召回 telemetry 通道 2——logger→OTel metrics 桥（幂等、失败吞掉）
        from app.infrastructure.telemetry.memory_recall_telemetry import (
            mount_memory_recall_metrics,
        )
        mount_memory_recall_metrics()

        # D1a startup 四段全序④：全量 reconcile（ports/服务构造后 inline await）。
        # advisory lock(74520011) 保证并发 pod 单跑者；reconcile 失败 fail-open 不阻断启动。
        # mode=off zero behavior change：mode≠off 才构造 reconciler + 跑对账。
        if _governance_mode != "off":
            from sqlalchemy import text

            from app.application.services.extension_reconciler import (
                D1A_STARTUP_RECONCILE_LOCK_KEY,
                ExtensionReconciler,
            )
            from app.infrastructure.repositories.file_skill_repository import (
                FileSkillRepository,
            )

            _reconciler = ExtensionReconciler(
                app.state.extension_registry_write_port,
                app.state.extension_admission_port,
            )
            app.state.extension_reconciler = _reconciler

            # D1a Task 19：MCP/A2A 两阶段安装管道单例（mode≠off；off 分支下 None）。
            # 自建 reconciler-wired AppConfigService（stateless——每次 load 重读文件）；
            # commit 经它的 install_context 触发 reconciler → WritePort 记账。
            from app.application.services.app_config_service import AppConfigService
            from app.application.services.extension_install_service import (
                ExtensionInstallService,
            )
            from app.application.services.extension_probe_service import (
                DefaultExtensionProber,
            )
            from app.infrastructure.repositories.file_app_config_repository import (
                FileAppConfigRepository,
            )

            _install_app_config_service = AppConfigService(
                FileAppConfigRepository(settings.app_config_filepath),
                reconciler=_reconciler,
                registry_read_port=app.state.extension_registry_read_port,
            )
            app.state.extension_install_service = ExtensionInstallService(
                _install_app_config_service,
                app.state.extension_registry_read_port,
                DefaultExtensionProber(),
                _governance_mode,
            )
            # D1a Task 20：§9.2 治理服务单例（mode≠off；off 分支下 None）。行政动作 +
            # 观测刷新（skill/plugin 本地 hash / mcp·a2a probe）+ 审计翻页；plugins_root
            # 对齐 startup reconcile 的 /app/data/plugins。
            from app.application.services.extension_governance_service import (
                ExtensionGovernanceService,
            )

            app.state.extension_governance_service = ExtensionGovernanceService(
                app.state.extension_registry_read_port,
                app.state.extension_registry_write_port,
                app.state.extension_admission_port,
                DefaultExtensionProber(),
                _load_app_config,
                FileSkillRepository(settings.skills_root_dir),
                Path("/app/data/plugins"),
            )
            # D1a Task 24：Plugin 元容器安装管道单例 + saga store（mode≠off；off → None）。
            # install/uninstall 走 PluginInstallService（T21-T23 saga）；GET list 走同一
            # PluginSagaStore；reject_audit = DbPluginRejectAuditSink（关闭 T21 注入的 Protocol
            # 占位——preflight/锁内重检拒绝真实 install 时写自持 install_rejected audit）。
            # app_config_service 复用上方 reconciler-wired _install_app_config_service；
            # skill_service 局部构造（照 closure builder 最小参数集）。config_loader 无生产实现
            # （T22/T23 fake-only 注入面）→ None（Documented Limitation：见 task-24 报告）。
            from app.application.services.plugin_install_service import (
                PluginInstallService,
            )
            from app.application.services.skill_service import SkillService
            from app.application.services.skill_source_loader import SkillSourceLoader
            from app.infrastructure.external.governance.plugin_saga_store import (
                DbPluginRejectAuditSink,
                PluginSagaStore,
            )

            _plugin_saga_store = PluginSagaStore(postgres_client.session_factory)
            app.state.plugin_saga_store = _plugin_saga_store
            app.state.plugin_install_service = PluginInstallService(
                loader=SkillSourceLoader(),
                prober=DefaultExtensionProber(),
                read_port=app.state.extension_registry_read_port,
                mode=_governance_mode,
                reject_audit=DbPluginRejectAuditSink(postgres_client.session_factory),
                store=_plugin_saga_store,
                skill_service=SkillService(
                    FileSkillRepository(settings.skills_root_dir),
                    registry_write_port=app.state.extension_registry_write_port,
                    registry_read_port=app.state.extension_registry_read_port),
                app_config_service=_install_app_config_service,
                write_port=app.state.extension_registry_write_port,
                config_loader=None,
            )
            try:
                async with postgres_client.session_factory() as _s:
                    _got = (await _s.execute(
                        text("SELECT pg_try_advisory_lock(:k)"),
                        {"k": D1A_STARTUP_RECONCILE_LOCK_KEY})).scalar()
                    if _got:
                        try:
                            await _reconciler.run_startup_reconcile(
                                app_config=_app_config,
                                skill_repository=FileSkillRepository(
                                    settings.skills_root_dir),
                                read_port=app.state.extension_registry_read_port,
                                plugins_root=Path("/app/data/plugins"))
                        finally:
                            await _s.execute(
                                text("SELECT pg_advisory_unlock(:k)"),
                                {"k": D1A_STARTUP_RECONCILE_LOCK_KEY})
                    else:
                        logger.info(
                            "D1a startup reconcile skipped: another pod holds the lock")
            except Exception:
                logger.warning(
                    "D1a startup reconcile failed; continuing boot", exc_info=True)
        else:
            app.state.extension_reconciler = None
            app.state.extension_install_service = None   # D1a T19：off → 路由旧直通
            app.state.extension_governance_service = None  # D1a T20：off → 治理路由 409
            app.state.plugin_install_service = None      # D1a T24：off → plugin 路由 409
            app.state.plugin_saga_store = None           # D1a T24：off → GET /v2/plugins 409

        # lifespan分界点
        yield
    finally:
        try:
            logger.info("Manus应用正在关闭")
            idle_watchdog = getattr(app.state, "idle_watchdog", None)
            if idle_watchdog is not None:
                await idle_watchdog.stop()
            agent_svc = getattr(app.state, "agent_service", None)
            if agent_svc:
                await asyncio.wait_for(agent_svc.shutdown(), timeout=30.0)
            logger.info("Agent服务成功关闭")
        except asyncio.TimeoutError:
            logger.warning("Agent服务关闭超时, 强制关闭, 部分任务将被释放")
        except Exception as e:
            logger.error(f"Agent服务关闭期间出现错误: {str(e)}")

        # C3 PR-3c — stop the per-pod MailboxSupervisor registry BEFORE the
        # lifecycle service closes. Order: AgentService.shutdown drains
        # in-flight runners (above) → registry.stop_all cancels supervisor
        # tasks waiting on XREADGROUP / running destroy() side-effects →
        # SandboxLifecycleService.shutdown (below) tears down sandbox infra.
        # ``stop_all`` is best-effort: it logs + swallows per-slot cancel
        # failures so one stuck supervisor doesn't block teardown.
        sup_registry = getattr(app.state, "supervisor_registry", None)
        if sup_registry is not None:
            try:
                await asyncio.wait_for(sup_registry.stop_all(), timeout=10.0)
                logger.info("SupervisorRegistry 关闭成功")
            except asyncio.TimeoutError:
                logger.warning("SupervisorRegistry 关闭超时")
            except Exception as e:
                logger.warning(f"SupervisorRegistry 关闭时出错: {e}")

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

        # 停 B9 探测后台循环（Task 14）——在 flush_service.shutdown() 之前、
        # 既有 Redis/Postgres 关闭段之前（B9 先停，spec R3#7）。cancel 沿同 task
        # 传播让 in-flight tick 的 cleanup 闭环（F4）；_stop_b9_probe 内部吞异常，
        # 无 task（构造失败 fail-open 路径）时静默 no-op。
        try:
            await _stop_b9_probe(app)
            logger.info("B9 扩展探测关闭成功")
        except Exception as e:
            logger.warning(f"B9 扩展探测关闭时出错: {e}")

        # 停 B9 统计 flusher（Task 19）——先 probe 后 stats，均在既有 Redis 关闭段之前
        # （spec §9 / brief）：recorder 停收 → flusher 限时 drain ≤2s → cancel。无 stats
        # （flag off / fail-open 路径）时静默 no-op。
        try:
            await _stop_b9_stats(app)
            logger.info("B9 扩展统计关闭成功")
        except Exception as e:
            logger.warning(f"B9 扩展统计关闭时出错: {e}")

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
# D1a Task 20：§9.2 治理路由族（独立 APIRouter，off 门在 handler 层——照抄 include 区）。
app.include_router(extension_governance_router, prefix="/api")
# D1a Task 24：§9.2 Plugin 路由族（install/delete/enabled/list；off 门在 handler 层）。
app.include_router(plugin_router, prefix="/api")

logger.info("FastAPI应用程序实例已创建。")
