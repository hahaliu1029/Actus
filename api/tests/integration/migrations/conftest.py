"""Isolation helpers for tests that exercise historical Alembic revisions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


DB_URL = os.environ.get(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test",
)


def _alembic_config() -> Config:
    sync_url = DB_URL.replace("+asyncpg", "+psycopg2", 1)
    api_root = Path(__file__).resolve().parent.parent.parent.parent
    cfg = Config(str(api_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


def _reset_public_schema(cfg: Config) -> None:
    """Reset only the explicitly isolated ``manus_test`` database."""
    sync_url = cfg.get_main_option("sqlalchemy.url")
    if make_url(sync_url).database != "manus_test":
        raise RuntimeError(
            "historical migration tests refuse to reset a database other than "
            "manus_test"
        )
    engine = create_engine(sync_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def migration_schema_at(request, _migrate):
    """Build a historical revision from base, then restore current head.

    Forward-only migrations make ``head -> old revision`` invalid by design.
    Historical migration tests therefore rebuild the disposable ``manus_test``
    schema at their module's ``MIGRATION_TARGET`` instead of reversing through
    a forward-only boundary.
    """
    target = getattr(request.module, "MIGRATION_TARGET", None)
    if not target:
        raise RuntimeError("module using migration_schema_at lacks MIGRATION_TARGET")

    cfg = _alembic_config()
    _reset_public_schema(cfg)
    command.upgrade(cfg, target)
    try:
        yield
    finally:
        _reset_public_schema(cfg)
        command.upgrade(cfg, "head")
