"""PR-4 anchor: runner emits bg_failed_watchdog after terminal write."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.session import Session
from app.domain.models.session import SessionStatus
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)

pytestmark = pytest.mark.anyio


def _make_runner(
    *,
    was_background: bool,
    emitter: Any,
    session_state: Session | None = None,
) -> AgentTaskRunner:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "s1"
    runner._user_id = "u1"
    runner._was_background = was_background
    runner._memory_notification_emitter = emitter
    runner._uow_factory = _uow_factory_for(session_state)
    runner._session_state_machine = DefaultSessionStateMachine(uow_factory=lambda: None)
    return runner


class _SessionRepo:
    def __init__(self, session_state: Session | None) -> None:
        self._session_state = session_state
        self.status_updates: list[tuple[str, SessionStatus]] = []

    async def get_by_id(self, session_id: str) -> Session | None:
        if self._session_state is None or self._session_state.id != session_id:
            return None
        return self._session_state

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        self.status_updates.append((session_id, status))


class _Uow:
    def __init__(self, session_state: Session | None) -> None:
        self.session = _SessionRepo(session_state)

    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def _uow_factory_for(session_state: Session | None):
    def factory() -> _Uow:
        return _Uow(session_state)

    return factory


class _ResumePostprocessFlow:
    _deferred_final_state = object()
    _deferred_summaries = object()

    async def resume(self, _command: object):
        if False:
            yield None


async def test_runner_emits_bg_failed_watchdog_after_terminal_write() -> None:
    calls: list[tuple[str, Any]] = []
    emitter = AsyncMock()

    async def fake_set_terminal_status(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> None:
        calls.append(("terminal", (status, terminal_reason)))

    async def fake_emit(**kwargs: Any) -> None:
        calls.append(("emit", kwargs))

    emitter.emit.side_effect = fake_emit
    runner = _make_runner(was_background=True, emitter=emitter)
    runner._set_terminal_status = fake_set_terminal_status

    await runner._set_terminal_status_with_notifications(SessionStatus.TIMED_OUT)

    assert calls == [
        ("terminal", (SessionStatus.TIMED_OUT, None)),
        (
            "emit",
            {
                "user_id": "u1",
                "event_type": "bg_failed_watchdog",
                "payload": {"session_id": "s1"},
            },
        ),
    ]


async def test_runner_does_not_emit_bg_failed_watchdog_for_never_bg_session() -> None:
    calls: list[tuple[str, Any]] = []
    emitter = AsyncMock()

    async def fake_set_terminal_status(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> None:
        calls.append(("terminal", (status, terminal_reason)))

    runner = _make_runner(was_background=False, emitter=emitter)
    runner._set_terminal_status = fake_set_terminal_status

    await runner._set_terminal_status_with_notifications(SessionStatus.TIMED_OUT)

    assert calls == [("terminal", (SessionStatus.TIMED_OUT, None))]
    emitter.emit.assert_not_awaited()


async def test_runner_emits_bg_completed_for_completed() -> None:
    emitter = AsyncMock()

    async def fake_set_terminal_status(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> None:
        return None

    runner = _make_runner(was_background=True, emitter=emitter)
    runner._set_terminal_status = fake_set_terminal_status

    await runner._set_terminal_status_with_notifications(SessionStatus.COMPLETED)

    emitter.emit.assert_awaited_once_with(
        user_id="u1",
        event_type="bg_completed",
        payload={"session_id": "s1"},
    )


async def test_runner_skips_terminal_notification_when_write_was_noop() -> None:
    emitter = AsyncMock()

    async def fake_set_terminal_status(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> bool:
        return False

    runner = _make_runner(was_background=True, emitter=emitter)
    runner._set_terminal_status = fake_set_terminal_status

    await runner._set_terminal_status_with_notifications(SessionStatus.COMPLETED)

    emitter.emit.assert_not_awaited()


async def test_runner_emits_bg_cancelled_for_user_cancel() -> None:
    emitter = AsyncMock()

    async def fake_set_terminal_status(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> None:
        return None

    runner = _make_runner(was_background=True, emitter=emitter)
    runner._set_terminal_status = fake_set_terminal_status

    await runner._set_terminal_status_with_notifications(
        SessionStatus.COMPLETED,
        "user_cancel",
    )

    emitter.emit.assert_awaited_once_with(
        user_id="u1",
        event_type="bg_cancelled",
        payload={"session_id": "s1"},
    )


async def test_runner_uses_fresh_session_state_after_auto_degrade() -> None:
    emitter = AsyncMock()

    async def fake_set_terminal_status(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> None:
        return None

    runner = _make_runner(
        was_background=False,
        emitter=emitter,
        session_state=Session(id="s1", user_id="u1", was_background=True),
    )
    runner._set_terminal_status = fake_set_terminal_status

    await runner._set_terminal_status_with_notifications(SessionStatus.TIMED_OUT)

    emitter.emit.assert_awaited_once_with(
        user_id="u1",
        event_type="bg_failed_watchdog",
        payload={"session_id": "s1"},
    )


async def test_runner_emits_bg_retry_exhausted_when_budget_reaches_zero() -> None:
    emitter = AsyncMock()

    async def fake_set_terminal_status(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> None:
        return None

    runner = _make_runner(
        was_background=True,
        emitter=emitter,
        session_state=Session(
            id="s1",
            user_id="u1",
            was_background=True,
            retry_budget_remaining=0,
        ),
    )
    runner._set_terminal_status = fake_set_terminal_status

    await runner._set_terminal_status_with_notifications(SessionStatus.COMPLETED)

    emitter.emit.assert_awaited_once_with(
        user_id="u1",
        event_type="bg_retry_exhausted",
        payload={"session_id": "s1"},
    )


async def test_resume_postprocess_failure_emits_bg_completed() -> None:
    emitter = AsyncMock()
    runner = _make_runner(
        was_background=True,
        emitter=emitter,
        session_state=Session(id="s1", user_id="u1", was_background=True),
    )
    runner._flow = _ResumePostprocessFlow()
    runner._uow = _uow_factory_for(
        Session(id="s1", user_id="u1", was_background=True)
    )()
    runner._put_and_add_event = AsyncMock()
    runner._run_postprocess_or_cancel = AsyncMock(
        side_effect=RuntimeError("postprocess failed")
    )
    runner._build_compaction_events_if_any = MagicMock(return_value=[])
    runner._snapshot_metrics = MagicMock(return_value={})
    runner._set_terminal_status = AsyncMock()

    await runner.resume(object(), object())

    runner._set_terminal_status.assert_awaited_once_with(
        SessionStatus.COMPLETED,
        None,
    )
    emitter.emit.assert_awaited_once_with(
        user_id="u1",
        event_type="bg_completed",
        payload={"session_id": "s1"},
    )


async def test_runner_logs_and_swallows_bg_failed_watchdog_emit_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    emitter = AsyncMock()
    emitter.emit.side_effect = RuntimeError("DB down")

    async def fake_set_terminal_status(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> None:
        return None

    runner = _make_runner(was_background=True, emitter=emitter)
    runner._set_terminal_status = fake_set_terminal_status
    caplog.set_level(logging.WARNING)

    await runner._set_terminal_status_with_notifications(SessionStatus.TIMED_OUT)

    assert "bg_failed_watchdog emit failed" in caplog.text
