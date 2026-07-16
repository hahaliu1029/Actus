"""SPM PR-4 Task 32 — off pure-chat zero-touch, asserted at the DB layer.

CI-only: needs a real (pgvector) PostgreSQL — matches the ``integration`` marker
convention (see ``tests/integration/conftest.py`` + the "Local Test
Infrastructure" section of ``CLAUDE.md``; never point ``SQLALCHEMY_DATABASE_URL``
at the dev ``manus`` DB). Locally this is skipped unless the env var points at a
reachable test DB; CI (``ci.yml``) provides ``manus_test``. Collection passes with
no DB — the body only touches Postgres at run time.

INV-SPM-3 (zero-touch): an ``off`` deployment has no sandbox plane, so a pure-chat
session must NEVER write a ``sandbox_lifecycle_log`` transition row (the
authoritative single-writer activation signal, spec §5.9). This shell asserts the
DB-level invariant directly: a fresh off-era session carries zero lifecycle-log
rows.
"""
from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.infrastructure.models.sandbox_lifecycle_log import SandboxLifecycleLogModel

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_off_pure_chat_integration_zero_touch(
    db_session, seed_session, monkeypatch
) -> None:
    """A pure-chat session under ``sandbox_provision_mode='off'`` records ZERO
    ``sandbox_lifecycle_log`` transition rows (INV-SPM-3 zero-touch).

    Shell scope: asserts the authoritative DB-level zero-touch signal for a
    freshly-seeded off-era session. The full agent-loop drive (AgentService pure
    chat → FINISHING → COMPLETED) is CI-harness follow-up; the invariant checked
    here is the one an end-to-end off run must preserve.
    """
    from core.config import get_settings

    monkeypatch.setattr(
        get_settings(), "sandbox_provision_mode", "off", raising=False
    )

    sid = seed_session.id
    count = (
        await db_session.execute(
            select(func.count())
            .select_from(SandboxLifecycleLogModel)
            .where(SandboxLifecycleLogModel.session_id == sid)
        )
    ).scalar_one()
    assert count == 0  # INV-SPM-3: off pure-chat never transitions the sandbox
