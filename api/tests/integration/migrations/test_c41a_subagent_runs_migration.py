"""C4.1a PR-2 — subagent_runs migration up/down 集成测试（需 PostgreSQL；spec §7 PR-2）。

autouse `_migrate`（tests/integration/conftest.py）已把 schema 升到 head。本测试
downgrade 到 s3pr1（drop subagent_runs）→ 断言表消失 → finally upgrade 回 head。
**集成未本地跑，CI 验证。**
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

_API_ROOT = Path(__file__).resolve().parents[3]  # api/
_DB_URL = os.environ.get(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test",
)


def _alembic_cfg() -> Config:
    cfg = Config(str(_API_ROOT / "alembic.ini"))
    # alembic 用 sync 驱动（mirror conftest._run_migrations）
    cfg.set_main_option("sqlalchemy.url", _DB_URL.replace("+asyncpg", "+psycopg2", 1))
    return cfg


async def test_downgrade_drops_then_upgrade_recreates(async_engine):
    async def _table_exists() -> bool:
        async with async_engine.connect() as conn:
            r = await conn.execute(text("SELECT to_regclass('public.subagent_runs')"))
            return r.scalar() is not None

    assert await _table_exists() is True  # _migrate 已 upgrade head
    cfg = _alembic_cfg()
    try:
        command.downgrade(cfg, "s3pr1_add_session_depth_lineage")
        assert await _table_exists() is False
    finally:
        command.upgrade(cfg, "head")  # 恢复 head，勿污染其它 module
    assert await _table_exists() is True
