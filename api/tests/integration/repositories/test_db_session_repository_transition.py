"""SessionRepository.transition_status atomic CAS + mode_revision +1.

Moved from tests/infrastructure/repositories/ so that pytest discovers
the integration conftest.py fixtures (session_repo, db_session, sample_user).
"""

import uuid

import pytest

from app.domain.models.session import SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
async def ensure_session(db_session, sample_user):
    """Factory: insert a SessionModel row with the requested status and return its id.

    Accepts keyword argument ``status`` (SessionStatus enum).
    Teardown is handled by the enclosing ``db_session`` rollback.
    """
    from app.infrastructure.models.session import SessionModel

    async def _factory(*, status: SessionStatus) -> str:
        sid = f"sess-pe0-{uuid.uuid4().hex[:12]}"
        orm = SessionModel(
            id=sid,
            user_id=sample_user.id,
            status=status.value,
            title="pe-0 transition test session",
            mode_revision=0,
        )
        db_session.add(orm)
        await db_session.flush()
        return sid

    return _factory


@pytest.fixture
async def ensure_committed_session(async_engine):
    """Factory: insert + COMMIT a SessionModel row; yield its id; delete on teardown.

    Unlike ``ensure_session`` (which flushes inside a rolled-back transaction),
    this fixture commits the row so that independent AsyncSession connections
    created by ``uow_factory`` can see it under read-committed isolation.

    Required by concurrent tests where each task must open its own UoW/session.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from sqlalchemy import delete
    from app.infrastructure.models.session import SessionModel
    from app.infrastructure.models.user import UserModel

    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    created_ids: list[str] = []
    committed_user_id = f"user-pe0-conc-{uuid.uuid4().hex[:12]}"
    # Use a fixture-owned committed user. Reusing sample_user here deadlocks:
    # its same PK is still uncommitted in the outer db_session transaction, so
    # PostgreSQL waits for that transaction while the test waits for INSERT.
    async with session_factory() as setup_session:
        async with setup_session.begin():
            setup_session.add(
                UserModel(
                    id=committed_user_id,
                    username=f"pe0_conc_{uuid.uuid4().hex[:12]}",
                    password_hash="x",
                )
            )

    async def _factory(*, status: SessionStatus) -> str:
        sid = f"sess-pe0-conc-{uuid.uuid4().hex[:12]}"
        async with session_factory() as setup_session:
            async with setup_session.begin():
                orm = SessionModel(
                    id=sid,
                    user_id=committed_user_id,
                    status=status.value,
                    title="pe-0 concurrent cas test session",
                    mode_revision=0,
                )
                setup_session.add(orm)
        created_ids.append(sid)
        return sid

    yield _factory

    # Teardown: delete committed rows so the test DB stays clean.
    if created_ids:
        async with session_factory() as cleanup_session:
            async with cleanup_session.begin():
                await cleanup_session.execute(
                    delete(SessionModel).where(SessionModel.id.in_(created_ids))
                )
    # Also remove the fixture-owned user row committed above.
    async with session_factory() as cleanup_session:
        async with cleanup_session.begin():
            await cleanup_session.execute(
                delete(UserModel).where(UserModel.id == committed_user_id)
            )


async def test_transition_status_cas_increments_mode_revision(session_repo, ensure_session):
    sid = await ensure_session(status=SessionStatus.RUNNING)

    before = await session_repo.get_by_id(sid)
    assert before.status is SessionStatus.RUNNING

    ok = await session_repo.transition_status(
        session_id=sid,
        from_state=SessionStatus.RUNNING,
        to_state=SessionStatus.TAKEOVER_PENDING,
    )
    assert ok is True

    after = await session_repo.get_by_id(sid)
    assert after.status is SessionStatus.TAKEOVER_PENDING
    # internal mode_revision read via repo helper (added by this task too)
    rev = await session_repo.read_mode_revision(sid)
    assert rev == 1


async def test_transition_status_cas_loser_returns_false(session_repo, ensure_session):
    sid = await ensure_session(status=SessionStatus.RUNNING)
    ok1 = await session_repo.transition_status(
        session_id=sid,
        from_state=SessionStatus.RUNNING,
        to_state=SessionStatus.TAKEOVER_PENDING,
    )
    ok2 = await session_repo.transition_status(
        session_id=sid,
        from_state=SessionStatus.RUNNING,
        to_state=SessionStatus.TAKEOVER_PENDING,
    )
    assert ok1 is True
    assert ok2 is False


async def test_transition_status_concurrent_one_winner(
    uow_factory, ensure_committed_session
):
    """Concurrent CAS: exactly one of 20 concurrent tasks wins the UPDATE.

    Each task opens its own UoW (independent AsyncSession + independent
    PostgreSQL connection).  The session row is committed to DB before
    the race starts so that read-committed isolation sees it.

    This validates the ``UPDATE … WHERE status=:from`` CAS primitive across
    genuinely concurrent transactions — not within a single rolled-back
    test transaction.
    """
    import asyncio

    sid = await ensure_committed_session(status=SessionStatus.RUNNING)

    async def _attempt() -> bool:
        # Each task gets its own UoW + AsyncSession + DB connection.
        async with uow_factory() as uow:
            return await uow.session.transition_status(
                session_id=sid,
                from_state=SessionStatus.RUNNING,
                to_state=SessionStatus.TAKEOVER_PENDING,
            )
        # uow.__aexit__ commits on success (no exception raised).

    results = await asyncio.gather(*[_attempt() for _ in range(20)])
    assert results.count(True) == 1, (
        f"Expected exactly 1 winner, got {results.count(True)}; results={results}"
    )
    assert results.count(False) == 19


# ---------------------------------------------------------------------------
# P1#1: update_status / update_to_terminal must bump mode_revision
# ---------------------------------------------------------------------------

async def test_update_status_bumps_mode_revision(session_repo, ensure_session):
    """update_status() must increment mode_revision so PE race checks see a fence."""
    sid = await ensure_session(status=SessionStatus.RUNNING)

    rev_before = await session_repo.read_mode_revision(sid)
    assert rev_before == 0

    await session_repo.update_status(sid, SessionStatus.WAITING)

    rev_after = await session_repo.read_mode_revision(sid)
    assert rev_after == 1, (
        "update_status must increment mode_revision by 1 (PE-0 race fence)"
    )


async def test_update_to_terminal_bumps_mode_revision(session_repo, ensure_session):
    """update_to_terminal() must increment mode_revision so PE race checks see a fence."""
    sid = await ensure_session(status=SessionStatus.RUNNING)

    rev_before = await session_repo.read_mode_revision(sid)
    assert rev_before == 0

    ok = await session_repo.update_to_terminal(
        session_id=sid,
        status=SessionStatus.COMPLETED,
        # PE-0 round 33 P1 fix: terminal_reason must be a value from the
        # ck_sessions_terminal_reason CheckConstraint allow-list (see
        # alembic/versions/b3p2_add_session_supervisor_columns.py:101):
        # "natural" | "user_cancel" | "server_restart" |
        # "resume_state_lost" | "watchdog_timeout". Using "test done"
        # raised IntegrityError before this assertion could run.
        terminal_reason="natural",
    )
    assert ok is True

    rev_after = await session_repo.read_mode_revision(sid)
    assert rev_after == 1, (
        "update_to_terminal must increment mode_revision by 1 (PE-0 race fence)"
    )


# ---------------------------------------------------------------------------
# Round 31 P2: transition_status must accept extra_values for atomic terminal writes
# ---------------------------------------------------------------------------

async def test_transition_status_extra_values_writes_terminal_metadata(
    session_repo, ensure_session, db_session
):
    """transition_status(extra_values=...) must merge columns into the
    SAME UPDATE as the status CAS so SSM.complete() can write
    completed_at / terminal_reason / execution_phase atomically.

    Previously SSM.complete() only updated status + mode_revision,
    leaving terminal metadata blank — downstream stats / supervisor
    recovery would see a COMPLETED row with completed_at IS NULL.
    """
    from datetime import datetime

    from app.infrastructure.models.session import SessionModel
    from sqlalchemy import select

    sid = await ensure_session(status=SessionStatus.FINISHING)

    # PE-0 round 37 P1 fix: SessionModel.completed_at is `DateTime` without
    # timezone (naive) — asyncpg rejects binding an aware datetime to a naive
    # TIMESTAMP column. Use naive datetime to match the column type and the
    # SSM.complete() / update_to_terminal() write path.
    now = datetime.now()
    ok = await session_repo.transition_status(
        session_id=sid,
        from_state=SessionStatus.FINISHING,
        to_state=SessionStatus.COMPLETED,
        extra_values={
            "completed_at": now,
            # PE-0 round 32 P2 fix: use a value from the Session
            # terminal_reason Literal allow-list (matches SSM.complete()).
            "terminal_reason": "natural",
            "execution_phase": "terminated",
        },
    )
    assert ok is True

    row = (
        await db_session.execute(
            select(
                SessionModel.status,
                SessionModel.completed_at,
                SessionModel.terminal_reason,
                SessionModel.execution_phase,
                SessionModel.mode_revision,
            ).where(SessionModel.id == sid)
        )
    ).one()
    assert row.status == SessionStatus.COMPLETED.value
    assert row.completed_at is not None, (
        "extra_values['completed_at'] must land in the same UPDATE as status"
    )
    assert row.terminal_reason == "natural"
    assert row.execution_phase == "terminated"
    assert row.mode_revision == 1, (
        "mode_revision must still be bumped by 1 even with extra_values"
    )


async def test_transition_status_extra_values_rejects_reserved_keys(
    session_repo, ensure_session
):
    """Reserved repo-owned columns (status / mode_revision / updated_at)
    must not be overridable via extra_values — guards against silent
    contract violations like passing status='ERROR' through the side door.
    """
    sid = await ensure_session(status=SessionStatus.FINISHING)

    import pytest as _pytest

    with _pytest.raises(ValueError, match="repo-owned keys"):
        await session_repo.transition_status(
            session_id=sid,
            from_state=SessionStatus.FINISHING,
            to_state=SessionStatus.COMPLETED,
            extra_values={"status": "bogus"},
        )

    with _pytest.raises(ValueError, match="repo-owned keys"):
        await session_repo.transition_status(
            session_id=sid,
            from_state=SessionStatus.FINISHING,
            to_state=SessionStatus.COMPLETED,
            extra_values={"mode_revision": 999},
        )


async def test_transition_status_extra_values_none_keeps_legacy_behavior(
    session_repo, ensure_session, db_session
):
    """Default call (extra_values omitted / None) must keep the original
    behavior: status + mode_revision + updated_at only; no spurious writes
    to terminal columns.
    """
    from app.infrastructure.models.session import SessionModel
    from sqlalchemy import select

    sid = await ensure_session(status=SessionStatus.RUNNING)

    ok = await session_repo.transition_status(
        session_id=sid,
        from_state=SessionStatus.RUNNING,
        to_state=SessionStatus.TAKEOVER_PENDING,
    )
    assert ok is True

    row = (
        await db_session.execute(
            select(
                SessionModel.completed_at,
                SessionModel.terminal_reason,
                SessionModel.execution_phase,
            ).where(SessionModel.id == sid)
        )
    ).one()
    # ensure_session inserts with phase='running' (server_default), no
    # completed_at / terminal_reason. Confirm transition_status didn't
    # touch them when extra_values is omitted.
    assert row.completed_at is None
    assert row.terminal_reason is None
    assert row.execution_phase == "running"
