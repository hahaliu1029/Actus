"""Integration test fixtures — async DB session with automatic migration.

Requires: SQLALCHEMY_DATABASE_URL env var pointing to pgvector-enabled PostgreSQL.

Local: docker compose 默认库是 manus 且不映射宿主机端口（docker-compose.yml:15-31）。
本地运行集成测试前需要：
  1. 临时映射 postgres 端口，例如 docker compose port publish 或单独启动：
     docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=manus pgvector/pgvector:pg17
  2. 设置环境变量指向可达的数据库：
     SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus

CI: ci.yml:13-22 单独启动 postgres 容器，映射 5432，库名 manus_test。
    CI 会设置 SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test

默认值使用 CI 环境（manus_test），本地开发者需显式设置环境变量。
"""

import os
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
