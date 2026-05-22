"""Integration test fixtures — async DB session with automatic migration.

Requires: SQLALCHEMY_DATABASE_URL env var pointing to pgvector-enabled PostgreSQL.

Local: docker compose 默认库是 manus 且不映射宿主机端口（docker-compose.yml:15-31）。
集成测试**禁止**指向 dev 库（manus）—— 会污染开发数据。本地跑集成测试要起独立 test-only 容器，库名固定为 manus_test：
  1. 临时映射 postgres 端口（推荐 127.0.0.1:55432:5432 避免和 compose pg 冲突），独立起容器：
     docker run -d --rm --name actus-pg-test -p 127.0.0.1:55432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=manus_test pgvector/pgvector:pg17
  2. 设置环境变量指向可达的数据库：
     SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/manus_test

CI: ci.yml:13-22 单独启动 postgres 容器，映射 5432，库名 manus_test。
    CI 会设置 SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test

默认值使用 CI 环境（manus_test），本地开发者需显式设置环境变量。
"""

import asyncio
import os
import uuid as _uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DB_URL = os.environ.get(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test",
)


def _run_migrations() -> None:
    """Run alembic upgrade head using a sync connection.

    Mirrors main.py:70-77 startup logic. Converts asyncpg URL to psycopg2 for alembic.
    """
    sync_url = DB_URL.replace("+asyncpg", "+psycopg2", 1)
    api_root = Path(__file__).resolve().parent.parent.parent  # api/
    alembic_cfg = Config(str(api_root / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", sync_url)
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module", autouse=True)
def _migrate():
    """Ensure DB schema is up-to-date before integration tests run."""
    _run_migrations()


@pytest.fixture(scope="module")
async def async_engine():
    engine = create_async_engine(DB_URL, echo=False)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(async_engine):
    """Per-test async session with automatic rollback."""
    async_session = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        async with session.begin():
            yield session
            await session.rollback()


# ── B6 fixtures ──────────────────────────────────────────────────────────────
#
# All B6 integration tasks (T5, T10, T13-T16, T16a, T18) depend on the
# fixtures below.  They are defined here once so all sub-packages can
# request them through normal conftest resolution.


@pytest.fixture
async def async_session_factory(async_engine):
    """[CXR2-P2-4] Build a session_factory from the test async_engine.

    ``DBUnitOfWork.__init__`` expects ``session_factory: async_sessionmaker[AsyncSession]``
    (api/app/infrastructure/repositories/db_uow.py:20).
    """
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def uow_factory(async_session_factory):
    """Yields a callable that returns a fresh ``DBUnitOfWork`` bound to the test session_factory.

    Usage::

        async with uow_factory() as uow:
            assert uow.db_session is not None
    """
    from app.infrastructure.repositories.db_uow import DBUnitOfWork

    def _factory():
        return DBUnitOfWork(session_factory=async_session_factory)

    yield _factory


@pytest.fixture
async def seed_session(db_session):
    """Insert a fresh ``UserModel`` + ``SessionModel`` row; yield the ORM SessionModel.

    The yielded object exposes ``.id`` and ``.user_id`` for FK relationships.
    Teardown is handled by the enclosing ``db_session`` rollback.
    """
    from app.infrastructure.models.user import UserModel
    from app.infrastructure.models.session import SessionModel

    uid = str(_uuid.uuid4())
    sid = f"sess-b6-{_uuid.uuid4().hex[:12]}"

    user = UserModel(
        id=uid,
        username=f"b6test_{uid[:8]}",
        password_hash="x",
    )
    session_row = SessionModel(
        id=sid,
        user_id=uid,
        status="pending",
        title="b6 smoke session",
    )
    db_session.add(user)
    db_session.add(session_row)
    await db_session.flush()

    yield session_row


@pytest.fixture
async def seed_other_user_session(db_session):
    """Same as ``seed_session`` but with a *different* user — for cross-user 403 tests."""
    from app.infrastructure.models.user import UserModel
    from app.infrastructure.models.session import SessionModel

    uid = str(_uuid.uuid4())
    sid = f"sess-b6-other-{_uuid.uuid4().hex[:12]}"

    user = UserModel(
        id=uid,
        username=f"b6other_{uid[:8]}",
        password_hash="x",
    )
    session_row = SessionModel(
        id=sid,
        user_id=uid,
        status="pending",
        title="b6 other user session",
    )
    db_session.add(user)
    db_session.add(session_row)
    await db_session.flush()

    yield session_row


@pytest.fixture
def messages_at_85_percent():
    """A ``list[BaseMessage]`` whose token estimate exceeds 85 % of context_window=80_000.

    With the char estimator (1 char ≈ 0.25 tokens, safety_factor 1.15):
      20 × 16_000 chars × 0.25 × 1.15 ≈ 92_000 tokens > 68_000 (85% of 80_000).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    return [SystemMessage(content="sys")] + [
        HumanMessage(content="x" * 16_000) for _ in range(20)
    ]


@pytest.fixture
def memory_at_85_percent(messages_at_85_percent):
    """A ``Memory`` wrapping ``messages_at_85_percent`` as serialised dicts."""
    from app.domain.models.memory import Memory
    from app.domain.services.graphs.message_utils import messages_to_dicts

    return Memory(messages=messages_to_dicts(messages_at_85_percent))


@pytest.fixture
def fake_summary_llm():
    """A ``BaseChatModel`` stub whose ``ainvoke`` returns a fixed ``AIMessage``."""
    from langchain_core.messages import AIMessage

    class _FakeSummaryLLM:
        async def ainvoke(self, prompt, **kw):  # noqa: ANN001,ANN201
            return AIMessage(content="<test summary>")

    return _FakeSummaryLLM()


@pytest.fixture
async def planner_react_with_compactor(uow_factory, fake_summary_llm, seed_session):
    """Minimal ``PlannerReActFlow`` bypassing ``__init__``, wired to a real ``GradualCompactor`` + UoW.

    Uses ``PlannerReActFlow.__new__`` to skip the heavy constructor; only the
    fields that B6 tests touch are populated.

    .. WARNING:: Transaction isolation gap.

       ``seed_session`` flushes user/session rows through ``db_session``'s
       ``begin()...rollback()`` block — they are visible to ``db_session``
       but never committed.  ``uow_factory()`` opens an **independent**
       connection with its own transaction; under PostgreSQL's read-committed
       isolation it cannot see those uncommitted rows.

       Tests that exercise compaction at ``level > 0`` (which calls
       ``uow.session.save_memory(self._session_id, "react", memory)`` inside
       ``_check_overflow``) will see a 0-row UPDATE or FK violation against
       ``sessions.id``.  If your B6 test needs the persistence path, either:

       (a) commit the seed data manually before invoking the flow (use a
           separate UoW + ``await uow.db_session.commit()``), or
       (b) replace ``flow._uow_factory`` with a no-op / mock that doesn't
           hit the DB.
    """
    from app.domain.models.context_overflow_config import ContextOverflowConfig
    from app.domain.services.flows.planner_react import PlannerReActFlow
    from app.domain.services.graphs.compaction import GradualCompactor
    from app.domain.services.graphs.token_estimator import TokenEstimator

    estimator = TokenEstimator(strategy="char")
    overflow_cfg = ContextOverflowConfig(
        context_window=80_000,
        context_overflow_guard_enabled=True,
        soft_trigger_ratio=0.85,
        hard_trigger_ratio=0.95,
        target_ratio=0.65,
        token_estimator="char",
        model_name="test-model",
    )
    compactor = GradualCompactor(
        token_estimator=estimator,
        soft_trigger_ratio=overflow_cfg.soft_trigger_ratio,
        hard_trigger_ratio=overflow_cfg.hard_trigger_ratio,
        target_ratio=overflow_cfg.target_ratio,
        summary_max_chars=overflow_cfg.summary_max_chars,
        token_safety_factor=overflow_cfg.token_safety_factor,
    )

    flow = PlannerReActFlow.__new__(PlannerReActFlow)  # bypass heavy __init__
    flow._compactor = compactor
    flow._summary_llm = fake_summary_llm
    flow._uow_factory = uow_factory
    flow._session_id = seed_session.id
    flow._overflow_config = overflow_cfg
    flow._cost_callback_handler = None
    flow._last_compaction_result = None

    yield flow


# ── B3-core supervisor fixtures (db_session-keyed) ──────────────────────────
#
# Plan basis: docs/superpowers/plans/2026-05-07-b3-core-pr0-plan.md (Task 1,
# P1-1 fix).  Fixtures here depend on ``db_session`` so they live with the
# integration tree.
#
# Round-3 audit P1-NEW-1 fix: pytest only walks UP the directory tree, NOT
# sideways across sibling trees.  The non-DB-keyed counterparts (redis_client,
# asgi_client, app, sample_user_token, other_user_token, mock_runner,
# _auth_dependency_overrides) used to live in a now-deleted app-tree conftest;
# all fixtures are consolidated below in this file (line ~393 onward).


@pytest.fixture
async def sample_user(db_session):
    """Persist a synthetic ``UserModel`` for FK references.

    The ``sessions`` table has a FK on ``user_id`` → ``users.id``; supervisor
    anchor tests need a real row to satisfy that constraint.
    """
    from app.infrastructure.models.user import UserModel

    user_id = str(_uuid.uuid4())
    user = UserModel(
        id=user_id,
        username=f"b3test_{user_id[:8]}",
        password_hash="x",
    )
    db_session.add(user)
    await db_session.flush()
    yield user
    # Cleanup is handled by the enclosing ``db_session`` rollback.


@pytest.fixture
async def sample_session(db_session, sample_user):
    """Persist a synthetic ``Session`` with PR-2 supervisor defaults."""
    from app.domain.models.session import Session, SessionStatus
    from app.infrastructure.models.session import SessionModel

    sid = f"sess-b3-{_uuid.uuid4().hex[:12]}"
    orm = SessionModel(
        id=sid,
        user_id=sample_user.id,
        status=SessionStatus.RUNNING.value,
        title="b3 supervisor anchor session",
        task_id=sid,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
    )
    db_session.add(orm)
    await db_session.flush()
    yield Session(  # convert ORM → domain (domain may default supervisor fields)
        id=sid,
        user_id=sample_user.id,
        status=SessionStatus.RUNNING,
        task_id=sid,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
    )


@pytest.fixture
def make_session(db_session, sample_user):
    """Factory for sessions with arbitrary supervisor-field overrides."""

    async def _factory(**overrides):
        from app.domain.models.session import SessionStatus
        from app.infrastructure.models.session import SessionModel

        sid = overrides.pop("id", f"sess-b3-{_uuid.uuid4().hex[:12]}")
        defaults = {
            "id": sid,
            "user_id": sample_user.id,
            "status": SessionStatus.RUNNING.value,
            "title": "b3 supervisor factory session",
            "task_id": sid,
            "execution_mode": "foreground",
            "execution_phase": "running",
            "retry_budget_remaining": 3,
            "was_background": False,
        }
        defaults.update(overrides)
        orm = SessionModel(**defaults)
        db_session.add(orm)
        await db_session.flush()
        return orm

    return _factory


class MockRequest:
    """Lightweight stand-in for FastAPI Request in dependency tests."""

    def __init__(self, app, headers=None):
        self.app = app
        self.headers = headers or {}


@pytest.fixture
def asgi_client_factory(app):
    """Build httpx AsyncClient instances with optional headers."""
    import httpx

    def _factory(headers=None):
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=headers or {},
        )

    return _factory


@pytest.fixture
async def session_repo(db_session):
    """``DBSessionRepository`` for FSM/Restart/Repo/PG/FINISHING/Notif anchors.

    Used 6+ times across C-FSM-*, C-Restart-*, C-Repo-*, C-PG-*,
    C-FINISHING-1, C-Notif-*.
    """
    from app.infrastructure.repositories.db_session_repository import DBSessionRepository

    yield DBSessionRepository(db_session=db_session)


@pytest.fixture
async def notification_repo(db_session):
    """``DBMemorySystemNotificationRepository`` per spec v3 §6.8 reuse decision.

    P3-2 verified: ctor signature is ``__init__(self, db_session: AsyncSession)``
    at api/app/infrastructure/repositories/db_memory_system_notification_repository.py:29.
    """
    from app.infrastructure.repositories.db_memory_system_notification_repository import (
        DBMemorySystemNotificationRepository,
    )

    yield DBMemorySystemNotificationRepository(db_session=db_session)


async def _no_op_callback(session_id: str) -> None:
    """Default callback: matches positional signature at agent_task_runner.py:2861."""
    return None


@pytest.fixture
async def runner_factory(db_session, sample_user):
    """Factory that constructs ``AgentTaskRunner`` instances with realistic deps.

    Used by C-FINISHING-1 / C-Notif-Watchdog anchors.
    """
    from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
    from app.infrastructure.repositories.db_session_repository import (
        DBSessionRepository,
    )

    class _SameSessionUow:
        def __init__(self):
            self.db_session = db_session
            self.session = DBSessionRepository(db_session=db_session)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    class _NoopSandbox:
        async def ensure_sandbox(self) -> None:
            return None

    class _NoopTool:
        manager = None

        async def initialize(self, *_args, **_kwargs) -> None:
            return None

        async def cleanup(self) -> None:
            return None

    def _uow_factory():
        return _SameSessionUow()

    constructed: list = []

    def _factory(*, session_id: str, user_id: str | None = None, **overrides):
        from app.domain.services.agent_task_runner import AgentTaskRunner

        runner = AgentTaskRunner(
            uow_factory=_uow_factory,
            llm=object(),
            agent_config=AgentConfig(
                max_iterations=100,
                max_retries=3,
                max_search_results=10,
            ),
            mcp_config=MCPConfig(mcpServers={}),
            a2a_config=A2AConfig(a2a_servers=[]),
            session_id=session_id,
            user_id=user_id or sample_user.id,
            file_storage=object(),
            browser=object(),
            search_engine=object(),
            sandbox=_NoopSandbox(),
            **overrides,
        )
        runner._mcp_tool = _NoopTool()
        runner._a2a_tool = _NoopTool()
        runner._skill_tool = _NoopTool()
        constructed.append(runner)
        return runner

    yield _factory
    # Optional cleanup — cancel any still-running runners
    for runner in constructed:
        try:
            await runner.cancel(reason="test_teardown")
        except Exception:
            pass


@pytest.fixture
async def agent_service_with_redis(db_session, redis_client, app):
    """``AgentService`` instance wired with real DB UoW + Redis event recovery.

    PR-1 only needs the producer/recovery/route path for seq cursor anchors.
    This deliberately does not construct PR-2's ExecutionSupervisor, repositories,
    watchdog, or Lua admission components.
    """
    from app.application.services.agent_service import AgentService, _ConfigSnapshot
    from app.application.services.session_service import SessionService
    from app.domain.models.app_config import (
        A2AConfig,
        AgentConfig,
        MCPConfig,
        SkillRiskPolicy,
    )
    from app.domain.models.context_overflow_config import ContextOverflowConfig
    from app.infrastructure.external.event_recovery.redis_event_recovery import (
        RedisEventRecovery,
    )
    from app.infrastructure.repositories.db_session_repository import DBSessionRepository
    from app.domain.services.execution_supervisor import ExecutionSupervisor
    from app.domain.services.idle_watchdog import IdleWatchdog
    from app.infrastructure.external.task.redis_stream_task import RedisStreamTask
    from app.interfaces.dependencies.rate_limit import rate_limit_read
    from app.interfaces.service_dependencies import get_agent_service, get_session_service

    async def _noop_rate_limit() -> None:
        return None

    class _SameSessionUow:
        def __init__(self):
            self.db_session = db_session
            self.session = DBSessionRepository(db_session=db_session)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    def _uow_factory():
        return _SameSessionUow()

    snapshot = _ConfigSnapshot(
        llm=object(),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
        ),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        skill_risk_policy=SkillRiskPolicy(),
        overflow_config=ContextOverflowConfig(),
        summary_llm=None,
        vision_fallback_model=None,
        skill_creator_service=None,
        supports_vision=True,
        supports_pdf_input=False,
        file_understanding_config=None,
    )
    service = AgentService(
        uow_factory=_uow_factory,
        config_snapshot=snapshot,
        sandbox_cls=object,
        task_cls=RedisStreamTask,
        search_engine=object(),
        file_storage=object(),
        redis_client=redis_client,
        event_recovery=RedisEventRecovery(max_count=10000),
    )
    session_service = SessionService(uow_factory=_uow_factory)
    supervisor = ExecutionSupervisor(
        redis_client=redis_client,
        session_repository=DBSessionRepository(db_session=db_session),
    )
    service._supervisor = supervisor
    app.state.supervisor = supervisor
    app.state.idle_watchdog = IdleWatchdog(
        redis_client=redis_client,
        supervisor=supervisor,
        session_repository=DBSessionRepository(db_session=db_session),
    )
    service._idle_watchdog = app.state.idle_watchdog
    app.dependency_overrides[get_agent_service] = lambda: service
    app.dependency_overrides[get_session_service] = lambda: session_service
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(rate_limit_read, None)
        if getattr(app.state, "supervisor", None) is supervisor:
            delattr(app.state, "supervisor")
        if getattr(app.state, "idle_watchdog", None) is not None:
            delattr(app.state, "idle_watchdog")


# ── B3-core supervisor fixtures (non-DB-keyed) ──────────────────────────────
#
# Round-3 audit P1-NEW-1 fix: pytest only walks UP the directory tree, NOT
# sideways across sibling trees.  Anchors live in ``api/tests/integration/``
# so all fixtures they consume must also live in (or be importable from)
# this conftest.  Round-2's app-tree conftest (now deleted) was invisible to
# integration anchors — fixtures consolidated here.


@pytest.fixture
async def redis_client(monkeypatch):
    """Real Redis client connected to local test instance.

    Spec v3 §3.2 — supervisor:hot, supervisor:owner, supervisor:bg,
    supervisor:user, supervisor:system keys.
    """
    import redis.asyncio as redis_asyncio
    from app.domain.services.idle_watchdog import IdleWatchdog
    from app.main import app as fastapi_app
    from app.infrastructure.external.message_queue import redis_stream_message_queue

    client = redis_asyncio.from_url(
        "redis://localhost:6379/15",  # DB 15 for tests; isolate from app DB 0
        decode_responses=True,
    )
    class _RedisClientWrapper:
        def __init__(self, inner):
            self.client = inner

        def __getattr__(self, name: str):
            return getattr(self.client, name)

    wrapper = _RedisClientWrapper(client)
    monkeypatch.setattr(redis_stream_message_queue, "get_redis", lambda: wrapper)
    fastapi_app.state.idle_watchdog = IdleWatchdog(redis_client=wrapper)
    # Flush before each test for isolation
    await client.flushdb()
    yield wrapper
    await client.flushdb()
    if getattr(fastapi_app.state, "idle_watchdog", None) is not None:
        delattr(fastapi_app.state, "idle_watchdog")
    await client.aclose()


@pytest.fixture
def mock_runner():
    """Lightweight ``MagicMock`` for tests that don't need a real runner."""
    import asyncio
    from unittest.mock import MagicMock

    runner = MagicMock()
    runner.session_id = str(_uuid.uuid4())
    runner.cancel_reason = None
    runner.request_cancel = MagicMock(return_value=asyncio.sleep(0))  # awaitable no-op
    return runner


@pytest.fixture
def app():
    """FastAPI app for ASGI test client — imported from main module.

    Round-3 audit P2-NEW-1 caveat: this fixture returns the app singleton
    WITHOUT running its ``lifespan`` startup.  Tests that need ``app.state.X``
    populated by lifespan (e.g. C-Redis-2 referencing ``app.state.idle_watchdog``)
    must either install ``asgi-lifespan`` and use ``LifespanManager`` OR
    construct the supervisor/watchdog manually.  Most PR-0 anchors stay xfail
    until PR-2 ships lifespan integration.
    """
    from app.main import app as fastapi_app

    return fastapi_app


@pytest.fixture
async def asgi_client(app, _auth_dependency_overrides):
    """httpx AsyncClient for endpoint tests — uses ASGITransport + DI override.

    Round-2 audit P1-C fix: ``sample_user`` only ``flush()``-es inside the
    rolled-back integration transaction; the real ``get_current_user`` opens
    an independent DB session and would 401 against a fresh test user.
    ``_auth_dependency_overrides`` installs a JWT-subject-aware shim so
    PR-3c endpoint anchors reach the contract under test instead of bouncing
    on auth.

    Mirrors ``api/tests/integration/endpoints/conftest.py:45-60`` (asgi-lifespan
    NOT in pyproject.toml).
    """
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def _auth_dependency_overrides(app, sample_user):
    """JWT-subject-aware ``get_current_user`` shim for endpoint anchor tests.

    Round-3 audit P1-NEW-2 fix: previous override unconditionally returned
    ``sample_user``, breaking C-Auth-1 (cross-user 403 anchor) — `other_user_token`
    was silently authenticated as `sample_user`.

    Fix: decode the JWT ``sub`` claim from the ``Authorization`` header.
    - sub == sample_user.id → return real sample_user (DB-backed)
    - sub == any other UUID → return a SimpleNamespace stub with .id=sub
      (sufficient for endpoint owner-comparison: ``sess.user_id != current_user.id``
      yields 403)
    - missing/malformed token → raise HTTPException(401)

    Decode is unverified (testing only — no secret check).
    """
    try:
        from app.interfaces.dependencies.auth import get_current_user  # type: ignore
    except ImportError:
        # Auth dep not on import path; endpoint anchors will xfail naturally.
        yield
        return

    import base64
    import json
    from types import SimpleNamespace

    from fastapi import Header, HTTPException

    sample_user_id = str(sample_user.id)

    async def _override_get_current_user(authorization: str = Header(default="")):
        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            raise HTTPException(status_code=401, detail="missing token")
        try:
            # JWT structure: header.payload.signature; we read payload only.
            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        except Exception:
            raise HTTPException(status_code=401, detail="malformed token")

        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="no sub claim")

        if sub == sample_user_id:
            return sample_user.to_domain()
        # Synthetic "other" user — has .id matching JWT sub.  Used by C-Auth-1
        # to drive `sess.user_id != current_user.id` → 403.
        return SimpleNamespace(
            id=sub,
            username=f"other-{sub[:8]}",
            is_admin=lambda: False,
        )

    app.dependency_overrides[get_current_user] = _override_get_current_user
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def sample_user_token(sample_user) -> str:
    """Synthetic JWT for sample_user.  Used by C-Cancel-1 / C-MultiTab-1."""
    from core.security import create_access_token

    return create_access_token({"sub": str(sample_user.id)})


@pytest.fixture
def other_user_token() -> str:
    """JWT for a different user_id — used by C-Auth-1 (cross-user 403 anchor).

    Round-3 audit P1-NEW-2 fix: with the JWT-aware override, this token's
    ``sub`` claim resolves to a SimpleNamespace stub user (NOT sample_user),
    so the endpoint's owner check correctly returns 403.
    """
    from core.security import create_access_token

    return create_access_token({"sub": str(_uuid.uuid4())})


# ── PE-1 T20 SkillSource E2E fixture ───────────────────────────────────────
#
# Stub mirroring SkillTool's ``_tool_bindings`` shape (see
# ``api/app/domain/services/tools/skill.py:133-141``) plus the two public
# ports SkillRiskRefresher reaches into:
#   - ``resolve_skill_dir(tool_name) -> Path | None``
#   - ``refresh_risk_if_stale(tool_name) -> str | None``
#
# The fixture builds a real on-disk skill directory under tmp_path so
# ``SkillsGuard.compute_content_hash(skill_dir)`` succeeds. The fake's
# content_hash does NOT match the test's ``SkillCallMetadata.content_hash``
# ("sha256:hash1"), so SkillRiskRefresher routes through the "hash differs"
# branch → calls ``refresh_risk_if_stale`` → returns "high" → SkillSource
# returns RiskLevel.HIGH (defense in depth).


@pytest.fixture
def fake_skill_tool_with_high_risk_binding(tmp_path):
    """Stub SkillTool with a single binding registering ``myskill_run`` at
    HIGH final_risk.

    Layout::

        tmp_path/skills_root/sk_test/manifest.json   # one stable file → deterministic hash

    The skill_dir is real (so ``SkillsGuard.compute_content_hash`` works);
    refresh_risk_if_stale always returns "high" (mirrors SkillSource's
    "refreshed → risk_level=HIGH" branch).
    """
    from pathlib import Path

    from app.domain.models.skill import (
        Skill, SkillRuntimeType, SkillSourceType,
    )

    skills_root = tmp_path / "skills_root"
    skill_dir = skills_root / "sk_test"
    skill_dir.mkdir(parents=True)
    (skill_dir / "manifest.json").write_text(
        '{"name": "myskill", "tools": []}', encoding="utf-8",
    )

    skill = Skill(
        id="sk_test",
        slug="myskill",
        name="myskill",
        description="",
        version="0.1.0",
        source_type=SkillSourceType.LOCAL,
        source_ref="/tmp/myskill",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={"name": "myskill", "tools": []},
        enabled=True,
        installed_by=None,
        trust_origin="user_installed",
        # content_hash deliberately differs from the test's
        # SkillCallMetadata.content_hash so SkillRiskRefresher routes through
        # the "hash diff → refresh_risk_if_stale" path.
        scan_report={"content_hash": "sha256:hash1", "verdict": "safe"},
    )

    class _FakeSkillTool:
        """SkillTool-shaped stub for SkillRiskRefresher / SkillSource."""

        _tool_bindings: dict[str, dict[str, object]] = {
            "myskill_run": {
                "skill": skill,
                "runtime_type": SkillRuntimeType.NATIVE,
                "manifest_tool": {},
                "final_risk": "high",
                "trust_origin": "user_installed",
                "scan_verdict": "safe",
            },
        }

        def __init__(self, *, root: Path) -> None:
            self._skills_root = root

        def has_tool(self, name: str) -> bool:
            return name in self._tool_bindings

        def resolve_skill_dir(self, tool_name: str) -> Path | None:
            binding = self._tool_bindings.get(tool_name)
            if not binding:
                return None
            return self._skills_root / binding["skill"].id

        def refresh_risk_if_stale(self, tool_name: str) -> str | None:
            binding = self._tool_bindings.get(tool_name)
            if not binding:
                return None
            return binding["final_risk"]

    return _FakeSkillTool(root=skills_root)


# ── C3 PR-4 Phase H — Mailbox supervisor integration harness ────────────────
#
# Plan reference: docs/superpowers/plans/2026-05-21-c3-mailbox-control-protocol.md
# §"Phase H — Integration tests T1, T4, T7, T11" lines 4804-5113.
#
# Fixtures added here (alongside the existing ``redis_client``):
#   - ``sandbox_lifecycle_spy`` — records destroy/suspend calls; can be
#     configured to raise on destroy via ``set_destroy_side_effect``.
#   - ``child_session_in_db`` — FK-satisfied root + child session rows;
#     yields the child ORM row exposing ``.id`` / ``.parent_session_id``.
#   - ``mailbox_envelope_factory`` — synchronous builder for
#     ``MailboxEnvelope`` with sane defaults; tests override only what they need.
#   - ``full_supervisor_stack`` — yields ``(supervisor, ctx, audit_repo,
#     publisher, task)`` where ``task`` is the ``asyncio.Task`` running the
#     supervisor (codex F10 fix — the harness comment previously claimed
#     ``registry`` as the 5th tuple element, but the fixture body yields
#     the bare task; there is no SupervisorRegistry instantiation here).
#     Backed by real Postgres (via ``async_session_factory``) + real Redis
#     (via ``redis_client``) + DI-injected ``sandbox_lifecycle_spy``.
#
# Auto-deactivating skipif in ``test_mailbox_crash_recovery.py`` /
# ``test_mailbox_supervisor_lifecycle.py`` checks for these names at module
# import time and activates the previously-skipped tests as soon as the
# fixtures land.


@pytest.fixture
def sandbox_lifecycle_spy():
    """Recording spy for ``_SandboxLifecycleProtocol`` (mailbox_supervisor.py).

    Records every ``destroy(session_id, reason)`` and ``suspend(session_id)``
    call in ``.destroy_calls`` / ``.suspend_calls`` (each entry is a
    ``{"session_id": str, "reason": Any}`` dict).

    ``set_destroy_side_effect(exc_factory)`` makes every subsequent ``destroy``
    call raise ``exc_factory()`` (a zero-arg callable returning an Exception).
    Pass ``None`` to clear. The semantics deliberately do NOT limit to "next N
    calls" — T11 needs destroy to keep raising past
    ``MAILBOX_POISON_MAX_RECLAIM`` retries until poison drop kicks in.
    """

    class _SandboxLifecycleSpy:
        def __init__(self) -> None:
            self.destroy_calls: list[dict] = []
            self.suspend_calls: list[dict] = []
            self._destroy_exc_factory = None

        def set_destroy_side_effect(self, exc_factory) -> None:
            """``exc_factory`` is a zero-arg callable returning an Exception
            instance (e.g. ``lambda: SandboxLifecycleError("boom")``). Pass
            ``None`` to reset to the default no-raise behavior.
            """
            self._destroy_exc_factory = exc_factory

        async def destroy(self, session_id: str, reason) -> None:
            self.destroy_calls.append({"session_id": session_id, "reason": reason})
            if self._destroy_exc_factory is not None:
                raise self._destroy_exc_factory()

        async def suspend(self, session_id: str) -> None:
            self.suspend_calls.append({"session_id": session_id})

    return _SandboxLifecycleSpy()


@pytest.fixture
async def child_session_in_db(db_session, sample_user, full_supervisor_stack):
    """Flush a child ``SessionModel`` row whose ``parent_session_id`` matches
    the supervisor stack's root, with ``worker_type='subagent'``,
    ``subagent_control_plane='mailbox'``, ``status='running'``.

    Codex r3 [R3-2, HIGH TEST] — the fixture used to build its OWN root row
    distinct from ``full_supervisor_stack``'s root. Tests then routed
    envelopes via ``ctx.root_session_id`` (the stack's root) but used the
    child whose actual ancestor was the other orphan root. The cross-root
    guard in ``_handle_envelope`` was the only thing masking the
    inconsistency — and R3-1's publisher contract assertion (negative test
    in ``test_mailbox_cross_root_guard.py``) makes that masking explicit.

    Now ``child_session_in_db`` depends on ``full_supervisor_stack`` so the
    child is linked to the SAME root row the supervisor owns. ``ctx``-routed
    envelopes thus match the child's real ancestor chain.

    Yields the child SessionModel ORM row — exposes ``.id`` and
    ``.parent_session_id``. FK on ``sessions.parent_session_id`` is satisfied
    because ``full_supervisor_stack`` already flushed the root row.
    """
    from app.infrastructure.models.session import SessionModel

    # Inherit the root from full_supervisor_stack so the child + supervisor
    # share a coherent root_session_id.
    _supervisor, ctx, _audit_repo, _publisher, _task = full_supervisor_stack
    root_id = ctx.root_session_id
    child_id = f"child-c3h-{_uuid.uuid4().hex[:12]}"

    child_row = SessionModel(
        id=child_id,
        user_id=sample_user.id,
        parent_session_id=root_id,
        status="running",
        title="c3 ph mailbox child",
        task_id=child_id,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
        worker_type="subagent",
        subagent_control_plane="mailbox",
    )
    db_session.add(child_row)
    await db_session.flush()
    yield child_row


@pytest.fixture
def mailbox_envelope_factory():
    """Synchronous builder for ``MailboxEnvelope`` with sane defaults.

    Caller overrides any field via kwargs::

        env = mailbox_envelope_factory(
            envelope_id="01HSPYU0t1d00000000000000",
            type=MailboxEnvelopeType.RESULT_READY,
            parent_session_id=root_id,
            child_session_id=child_id,
            payload=ResultReadyPayload(...).model_dump(mode="json"),
        )
    """
    from datetime import datetime, timezone

    from app.domain.models.mailbox_envelope import (
        MailboxEnvelope,
        MailboxEnvelopeType,
        ProducerRole,
        ProgressKind,
        ProgressUpdatePayload,
        ProgressVisibility,
    )

    def _make(**overrides) -> MailboxEnvelope:
        defaults: dict = dict(
            envelope_id=f"env-{_uuid.uuid4().hex[:12]}",
            type=MailboxEnvelopeType.PROGRESS_UPDATE,
            parent_session_id=f"parent-{_uuid.uuid4().hex[:8]}",
            child_session_id=f"child-{_uuid.uuid4().hex[:8]}",
            correlation_id=f"corr-{_uuid.uuid4().hex[:12]}",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.CHILD_AGENT,
            payload=ProgressUpdatePayload(
                kind=ProgressKind.HEARTBEAT,
                visibility=ProgressVisibility.HIDDEN,
            ).model_dump(mode="json"),
        )
        defaults.update(overrides)
        return MailboxEnvelope(**defaults)

    return _make


@pytest.fixture
async def full_supervisor_stack(
    db_session,
    async_session_factory,
    redis_client,
    sandbox_lifecycle_spy,
    sample_user,
):
    """Spawn one ``MailboxSupervisor`` task bound to a fresh root session.

    Yields a tuple ``(supervisor, ctx, audit_repo, publisher, task)``:

      - ``supervisor`` — the live ``MailboxSupervisor`` instance
      - ``ctx`` — the ``SupervisorContext`` (also exposes ``root_session_id``,
        ``telemetry`` collecting stub, ``now()`` for envelope timestamps)
      - ``audit_repo`` — ``DbMailboxEnvelopeAuditRepository`` bound to the
        test ``async_session_factory``
      - ``publisher`` — ``RedisMailboxPublisher`` bound to ``redis_client``
      - ``task`` — the running ``asyncio.Task``

    Implementation notes:
      - ``redis_client`` is a thin wrapper exposing the underlying
        ``redis.asyncio.Redis``; we pass the wrapper through — the supervisor
        only invokes ``.xreadgroup`` / ``.xack`` / ``.xautoclaim`` / ``.set``
        etc., all of which the wrapper forwards via ``__getattr__``.
      - The root session is flushed BUT not committed (rolled back at
        teardown via ``db_session``). The audit repo, however, uses its own
        independent session per call — meaning audit rows it writes DO commit
        and persist across teardown. Tests assert against the audit repo's
        ``fetch_raw`` reads (which see the committed rows). The transactional
        gap is documented in ``planner_react_with_compactor``'s docstring.
      - The supervisor's tight tick intervals (``_CANCEL_CHECK_INTERVAL_S`` /
        ``_ORPHAN_CHECK_INTERVAL_S``) stay at module defaults; individual
        tests monkeypatch attributes directly when they need faster cascades.
    """
    import uuid as _uid

    from app.application.services.mailbox_supervisor import (
        MailboxSupervisor,
        SupervisorContext,
    )
    from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
        RedisMailboxPublisher,
    )
    from app.infrastructure.repositories.db_mailbox_envelope_audit_repository import (
        DbMailboxEnvelopeAuditRepository,
    )

    # Each test gets a unique root id so module-scoped DB / Redis state
    # doesn't bleed across tests.
    root_session_id = f"root-c3h-stack-{_uid.uuid4().hex[:12]}"
    pod_id = f"pod-test-{_uid.uuid4().hex[:6]}"
    instance_id = f"i{_uid.uuid4().hex[:6]}"

    # Persist the root session row so any FK lookups during audit /
    # supervisor flow find a real parent.
    #
    # codex r3 [R3-2, HIGH TEST] fixture chain — ``child_session_in_db``
    # was reworked to DEPEND on ``full_supervisor_stack``: it now derives
    # its ``parent_session_id`` from ``ctx.root_session_id`` (this same
    # ``root_session_id`` built one line above), not its own freshly-
    # minted orphan root. See ``child_session_in_db`` lines 953-956 for
    # the inverse hand-off (``ctx = full_supervisor_stack; root_id =
    # ctx.root_session_id``).
    #
    # codex r4 [R4-6, LOW DOC] — the canonical pattern for tests that
    # need a child wired up to this supervisor is ``async def test_x(
    # full_supervisor_stack, child_session_in_db, ...)``: pytest resolves
    # ``full_supervisor_stack`` first (this fixture), then
    # ``child_session_in_db`` re-injects the same fixture instance and
    # derives the child's ``parent_session_id`` from ``ctx.root_session_id``.
    # Tests should route envelopes via ``ctx.root_session_id`` (the
    # supervisor's root) and ``child_session_in_db.id`` — those two
    # always belong to the same ancestor chain after R3-2.
    from app.infrastructure.models.session import SessionModel

    root_row = SessionModel(
        id=root_session_id,
        user_id=sample_user.id,
        status="running",
        title="c3 ph supervisor stack root",
        task_id=root_session_id,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
        worker_type="root",
    )
    db_session.add(root_row)
    await db_session.flush()

    audit_repo = DbMailboxEnvelopeAuditRepository(
        session_factory=async_session_factory,
    )
    publisher = RedisMailboxPublisher(redis_client)

    # In-process callback stub — records every envelope dispatched to the
    # in-process agent service. Tests can introspect ``.received``.
    class _StubAgentCallback:
        def __init__(self) -> None:
            self.received: list = []

        async def __call__(self, envelope) -> None:
            self.received.append(envelope)

    agent_service_callback = _StubAgentCallback()

    # Telemetry collecting stub — supervisor emits e.g.
    # ``mailbox.poison_message_dropped`` / ``mailbox.cascade_*``;
    # tests assert via the recorded list of (name, data) tuples.
    class _StubTelemetry:
        def __init__(self) -> None:
            self.emitted: list[tuple[str, dict]] = []

        async def emit(self, name: str, data: dict) -> None:
            self.emitted.append((name, data))

    telemetry = _StubTelemetry()

    ctx = SupervisorContext(
        root_session_id=root_session_id,
        pod_id=pod_id,
        instance_id=instance_id,
        redis=redis_client,
        audit_repo=audit_repo,
        publisher=publisher,
        sandbox_lifecycle=sandbox_lifecycle_spy,
        agent_service_callback=agent_service_callback,
        telemetry=telemetry,
    )

    # Expose extra hooks tests need (mirror unit-test fixture shape).
    ctx.agent_callback = agent_service_callback  # type: ignore[attr-defined]
    ctx.telemetry = telemetry  # type: ignore[attr-defined]

    sup = MailboxSupervisor(
        ctx,
        block_ms=0,  # non-blocking — see PR-3a comment
        idle_poll_sleep_s=0.05,
    )
    task = asyncio.create_task(sup.run())
    # Give the supervisor a tick to ensure_group + initial xautoclaim.
    await asyncio.sleep(0.1)

    try:
        yield sup, ctx, audit_repo, publisher, task
    finally:
        try:
            await sup.stop(drain_timeout_s=2.0)
        except Exception:
            pass
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass