"""A4-2: SSM.emit_session_mode_changed — single construct-and-dispatch entry.

Builds the canonical event (assert FIELD equality, NOT byte/golden — BaseEvent
makes a fresh id/created_at at construction, event.py:71), dispatches to the
caller-owned sink exactly once, returns the event, passes mode_revision through
verbatim, and propagates sink errors (caller owns degrade)."""
import asyncio

import pytest

from app.domain.models.event import SessionModeChangedEvent
from app.domain.models.session import SessionStatus
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)
from app.domain.services.session.mode_event import (
    build_session_mode_changed_event,
)


def _run(coro):
    return asyncio.run(coro)


def _ssm() -> DefaultSessionStateMachine:
    # uow_factory is irrelevant to emit (the method never touches the repo);
    # a dummy factory keeps the ctor happy. The SSM ctor takes exactly the two
    # kwargs uow_factory + redis (A4-2 retired the SSE publisher param).
    return DefaultSessionStateMachine(uow_factory=lambda: None, redis=None)


def test_builder_builds_expected_event_fields():
    ev = build_session_mode_changed_event(
        to=SessionStatus.TAKEOVER,
        from_mode="running",
        reason="takeover_started",
        mode_revision=11,
    )
    assert isinstance(ev, SessionModeChangedEvent)
    assert ev.type == "session_mode_changed"
    assert ev.to == "takeover"  # SessionStatus normalized to .value
    assert ev.from_mode == "running"
    assert ev.reason == "takeover_started"
    assert ev.mode_revision == 11


def test_builder_accepts_str_to_and_none_from_mode():
    ev = build_session_mode_changed_event(
        to="takeover_pending",
        from_mode=None,
        reason="takeover_reopened",
        mode_revision=None,
    )
    assert ev.to == "takeover_pending"  # str passthrough (no normalization)
    assert ev.from_mode is None
    assert ev.mode_revision is None


def test_emit_dispatches_to_sink_once_and_returns_event():
    seen: list = []

    async def sink(sid, ev):
        seen.append((sid, ev))

    ssm = _ssm()
    returned = _run(
        ssm.emit_session_mode_changed(
            "s1",
            to=SessionStatus.WAITING,
            from_mode="running",
            reason="wait",
            mode_revision=7,
            sink=sink,
        )
    )
    assert len(seen) == 1
    sid, ev = seen[0]
    assert sid == "s1"
    assert ev is returned  # returns the exact event it dispatched
    assert ev.to == "waiting"
    assert ev.mode_revision == 7  # rev pass-through, verbatim (read-your-writes)


def test_emit_propagates_sink_exception():
    async def boom(sid, ev):
        raise RuntimeError("sink failed")

    ssm = _ssm()
    with pytest.raises(RuntimeError, match="sink failed"):
        _run(
            ssm.emit_session_mode_changed(
                "s1",
                to="running",
                from_mode="takeover",
                reason="takeover_ended",
                mode_revision=3,
                sink=boom,
            )
        )
