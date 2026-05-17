"""DefaultSessionStateMachine — call-through to repo + CAS semantics."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.domain.models.session import SessionStatus
from app.domain.services.permission.errors import SessionModeViolation
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)


def _run(coro):
    return asyncio.run(coro)


class _FakeUow:
    """Mimics IUnitOfWork as an async context manager. SSM consumes
    `uow.session: SessionRepository` so we attach an AsyncMock there.
    Each test mutates _fake_repo.transition_status / read_status_with_revision."""

    def __init__(self, session_repo_mock):
        self.session = session_repo_mock
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


def _make_fake_uow_factory(transition_return=True, revision_return=None):
    """Create a factory + exposed fake_repo for assertions."""
    if revision_return is None:
        revision_return = (SessionStatus.RUNNING, 5)

    fake_repo = AsyncMock()
    fake_repo.read_status_with_revision = AsyncMock(return_value=revision_return)
    fake_repo.transition_status = AsyncMock(return_value=transition_return)

    def factory():
        return _FakeUow(fake_repo)

    factory._fake_repo = fake_repo
    return factory


class TestGetModeWithRevision:
    def test_passthrough(self):
        factory = _make_fake_uow_factory()
        pub = AsyncMock()

        ssm = DefaultSessionStateMachine(
            uow_factory=factory, redis=None, event_publisher=pub,
        )
        mode, rev = _run(ssm.get_mode_with_revision("s1"))
        assert mode is SessionStatus.RUNNING
        assert rev == 5


class TestRequestTakeover:
    def test_calls_transition_running_to_pending(self):
        factory = _make_fake_uow_factory(transition_return=True)
        pub = AsyncMock()

        ssm = DefaultSessionStateMachine(
            uow_factory=factory, redis=None, event_publisher=pub,
        )
        _run(ssm.request_takeover("s1", reason="user_clicked_takeover"))
        fake_repo = factory._fake_repo
        fake_repo.transition_status.assert_awaited_once()
        kwargs = fake_repo.transition_status.await_args.kwargs
        assert kwargs["from_state"] is SessionStatus.RUNNING
        assert kwargs["to_state"] is SessionStatus.TAKEOVER_PENDING

    def test_raises_when_cas_loses(self):
        factory = _make_fake_uow_factory(
            transition_return=False,
            revision_return=(SessionStatus.TAKEOVER, 6),
        )
        pub = AsyncMock()

        ssm = DefaultSessionStateMachine(
            uow_factory=factory, redis=None, event_publisher=pub,
        )
        with pytest.raises(SessionModeViolation):
            _run(ssm.request_takeover("s1", reason="x"))


class TestComplete:
    def test_only_from_finishing(self):
        factory = _make_fake_uow_factory(transition_return=True)
        pub = AsyncMock()

        ssm = DefaultSessionStateMachine(
            uow_factory=factory, redis=None, event_publisher=pub,
        )
        _run(ssm.complete("s1"))
        kwargs = factory._fake_repo.transition_status.await_args.kwargs
        assert kwargs["from_state"] is SessionStatus.FINISHING
        assert kwargs["to_state"] is SessionStatus.COMPLETED

    def test_complete_writes_terminal_metadata_atomically(self):
        """PE-0 round 31 P2: SSM.complete() must forward terminal
        metadata (completed_at / terminal_reason / execution_phase) via
        ``extra_values`` so the row is consistent with update_to_terminal.
        Future-proofs the dormant SSM.complete() path against downstream
        stats / supervisor recovery / phase filtering bugs.
        """
        from datetime import datetime

        factory = _make_fake_uow_factory(transition_return=True)
        pub = AsyncMock()

        ssm = DefaultSessionStateMachine(
            uow_factory=factory, redis=None, event_publisher=pub,
        )
        _run(ssm.complete("s1"))

        kwargs = factory._fake_repo.transition_status.await_args.kwargs
        assert "extra_values" in kwargs, (
            "complete() must forward extra_values so terminal columns are "
            "written in the same UPDATE as the status CAS"
        )
        extra = kwargs["extra_values"]
        assert extra is not None
        assert set(extra.keys()) == {
            "completed_at",
            "terminal_reason",
            "execution_phase",
        }, f"unexpected extra_values keys: {sorted(extra.keys())}"
        assert isinstance(extra["completed_at"], datetime)
        # PE-0 round 33 P1 fix: completed_at column is TIMESTAMP WITHOUT
        # TIME ZONE (see infrastructure/models/session.py:113 — DateTime
        # without `timezone=True`). asyncpg rejects aware datetime binding
        # to naive column, so SSM.complete() emits naive datetime to match
        # update_to_terminal()'s contract.
        assert extra["completed_at"].tzinfo is None, (
            "completed_at must be timezone-naive to match the "
            "TIMESTAMP WITHOUT TIME ZONE column (sessions.completed_at) — "
            "mirrors update_to_terminal()'s datetime.now() contract"
        )
        # PE-0 round 32 P2 fix: terminal_reason must be a value from the
        # Session.terminal_reason Literal allow-list. "natural" is the
        # semantic match for SSM.complete()'s FINISHING -> COMPLETED.
        assert extra["terminal_reason"] == "natural"
        assert extra["execution_phase"] == "terminated"

    def test_transition_forwards_extra_values_kwarg(self):
        """SSM.transition() must transparently forward extra_values to repo
        so SSM.complete (and any future SSM caller that needs atomic
        terminal-metadata writes) can rely on the contract.
        """
        factory = _make_fake_uow_factory(transition_return=True)
        pub = AsyncMock()

        ssm = DefaultSessionStateMachine(
            uow_factory=factory, redis=None, event_publisher=pub,
        )
        payload = {"terminal_reason": "x", "execution_phase": "terminated"}
        _run(
            ssm.transition(
                "s1",
                SessionStatus.FINISHING,
                SessionStatus.COMPLETED,
                reason="complete",
                extra_values=payload,
            )
        )
        kwargs = factory._fake_repo.transition_status.await_args.kwargs
        assert kwargs["extra_values"] == payload

    def test_transition_extra_values_defaults_to_none(self):
        """Non-terminal transitions must not pass spurious extra_values."""
        factory = _make_fake_uow_factory(transition_return=True)
        pub = AsyncMock()

        ssm = DefaultSessionStateMachine(
            uow_factory=factory, redis=None, event_publisher=pub,
        )
        _run(ssm.request_takeover("s1", reason="user"))
        kwargs = factory._fake_repo.transition_status.await_args.kwargs
        assert kwargs.get("extra_values") is None, (
            "request_takeover (non-terminal) must not inject extra_values"
        )
