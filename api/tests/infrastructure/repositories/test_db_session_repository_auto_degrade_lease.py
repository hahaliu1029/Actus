from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.event import ExecutionStatePayload, PendingExecutionEvent
from app.domain.models.session import SessionStatus
from app.infrastructure.repositories.db_session_repository import DBSessionRepository

pytestmark = pytest.mark.anyio


async def test_generic_terminal_retires_pending_execution_outbox_atomically() -> None:
    db_session = AsyncMock()
    result = MagicMock(rowcount=1)
    db_session.execute.return_value = result
    repo = DBSessionRepository(db_session=db_session)

    assert await repo.update_to_terminal(
        "session-1",
        status=SessionStatus.COMPLETED,
        terminal_reason="natural",
    ) is True

    compiled = db_session.execute.await_args.args[0].compile()
    sql = str(compiled).lower()
    assert "execution_revision=(sessions.execution_revision +" in sql
    assert "pending_execution_event" in sql
    assert None in compiled.params.values()


async def test_renew_auto_degrade_expiry_is_one_conditional_update() -> None:
    db_session = AsyncMock()
    result = MagicMock()
    expires_at = datetime(
        2026, 7, 14, 12, 0, tzinfo=timezone.utc
    )
    result.one_or_none.return_value = SimpleNamespace(
        expires_at=expires_at,
        execution_revision=7,
    )
    db_session.execute.return_value = result
    repo = DBSessionRepository(db_session=db_session)

    assert await repo.renew_auto_degrade_expiry_if_running(
        "session-1",
        expires_at=expires_at,
    ) == (expires_at, 7)

    assert db_session.execute.await_count == 1
    stmt = db_session.execute.await_args.args[0]
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
    assert sql.startswith("update sessions set")
    assert "sessions.id = 'session-1'" in sql
    assert "sessions.status = 'running'" in sql
    assert "sessions.execution_mode = 'background'" in sql
    assert "sessions.execution_phase = 'running'" in sql
    assert "sessions.background_reason = 'auto_degrade'" in sql
    assert "greatest(sessions.expires_at" in sql
    assert "returning sessions.expires_at" in sql
    assert "sessions.execution_revision" in sql
    assert "coordinator_run_id" not in sql
    assert "work_unit_id" not in sql


async def test_renew_auto_degrade_expiry_returns_false_for_stale_cas() -> None:
    db_session = AsyncMock()
    result = MagicMock()
    result.one_or_none.return_value = None
    db_session.execute.return_value = result
    repo = DBSessionRepository(db_session=db_session)

    renewed = await repo.renew_auto_degrade_expiry_if_running(
        "session-1",
        expires_at=datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )

    assert renewed is None


async def test_resume_auto_degrade_is_one_conditional_update() -> None:
    db_session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = 4
    db_session.execute.return_value = result
    repo = DBSessionRepository(db_session=db_session)

    assert await repo.resume_auto_degrade_to_foreground_if_running(
        "session-1",
        expected_execution_revision=3,
        pending_event=PendingExecutionEvent(
            payload=ExecutionStatePayload(
                execution_mode="foreground",
                execution_phase="running",
                retry_budget_remaining=3,
                execution_revision=4,
            )
        ),
    ) == 4

    stmt = db_session.execute.await_args.args[0]
    compiled = stmt.compile()
    sql = str(compiled).lower()
    assert "sessions.status =" in sql
    assert "sessions.execution_mode =" in sql
    assert "sessions.execution_phase =" in sql
    assert "sessions.background_reason =" in sql
    assert "sessions.execution_revision =" in sql
    assert "foreground" in compiled.params.values()
    assert list(compiled.params.values()).count(None) >= 3
    assert "coordinator_run_id" not in sql
    assert "work_unit_id" not in sql


async def test_expired_background_terminal_is_one_authoritative_conditional_update(
) -> None:
    db_session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = 6
    db_session.execute.return_value = result
    repo = DBSessionRepository(db_session=db_session)
    sweep_now = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)

    assert await repo.update_to_terminal_if_background_expired(
        "session-1",
        status=SessionStatus.TIMED_OUT,
        terminal_reason="watchdog_timeout",
        expires_at_lte=sweep_now,
        expected_execution_revision=5,
        pending_event=PendingExecutionEvent(
            payload=ExecutionStatePayload(
                execution_mode="background",
                execution_phase="terminated",
                retry_budget_remaining=3,
                execution_revision=6,
            )
        ),
    ) == 6

    assert db_session.execute.await_count == 1
    stmt = db_session.execute.await_args.args[0]
    compiled = stmt.compile()
    sql = str(compiled).lower()
    assert "sessions.id =" in sql
    assert "sessions.status =" in sql
    assert "sessions.execution_mode =" in sql
    assert "sessions.execution_phase in" in sql
    assert "sessions.expires_at <=" in sql
    assert "sessions.execution_revision =" in sql
    assert "timed_out" in compiled.params.values()
    assert "terminated" in compiled.params.values()


async def test_watchdog_terminal_without_event_clears_superseded_outbox() -> None:
    db_session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = 6
    db_session.execute.return_value = result
    repo = DBSessionRepository(db_session=db_session)

    assert await repo.update_to_terminal_if_background_expired(
        "session-1",
        status=SessionStatus.TIMED_OUT,
        terminal_reason="watchdog_timeout",
        expires_at_lte=datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
        expected_execution_revision=5,
        pending_event=None,
    ) == 6

    compiled = db_session.execute.await_args.args[0].compile()
    sql = str(compiled).lower()
    assert "pending_execution_event" in sql
    assert None in compiled.params.values()


async def test_claim_background_retry_returns_budget_and_execution_revision() -> None:
    db_session = AsyncMock()
    result = MagicMock()
    result.one_or_none.return_value = SimpleNamespace(
        retry_budget_remaining=2,
        execution_revision=7,
    )
    db_session.execute.return_value = result
    repo = DBSessionRepository(db_session=db_session)
    expires_at = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)

    assert await repo.claim_background_retry_from_suspend(
        "session-1",
        expires_at=expires_at,
    ) == (2, 7)

    assert db_session.execute.await_count == 1
    compiled = db_session.execute.await_args.args[0].compile()
    sql = str(compiled).lower()
    assert "execution_revision=(sessions.execution_revision +" in sql
    assert "pending_execution_event" in sql
    assert None in compiled.params.values()
    assert "returning sessions.retry_budget_remaining" in sql
    assert "sessions.execution_revision" in sql


async def test_rollback_background_retry_claim_is_execution_revision_fenced() -> None:
    db_session = AsyncMock()
    db_session.execute.return_value = MagicMock(rowcount=1)
    repo = DBSessionRepository(db_session=db_session)

    assert await repo.rollback_background_retry_claim_if_active(
        "session-1",
        expected_execution_revision=7,
        retry_budget_remaining=3,
        expires_at=None,
        suspended_reason="tool_error",
    ) is True

    assert db_session.execute.await_count == 1
    stmt = db_session.execute.await_args.args[0]
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
    assert "sessions.status = 'running'" in sql
    assert "sessions.execution_mode = 'background'" in sql
    assert "sessions.execution_phase = 'running'" in sql
    assert "sessions.execution_revision = 7" in sql
    assert "execution_revision" not in sql.split(" where ", 1)[0]
