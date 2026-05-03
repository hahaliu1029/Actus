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
