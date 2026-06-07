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
