"""C2 PR-3 §8.5.3 — CoordinatorChildCancelListener unit tests.

Covers:
- subscribe creates root-scoped stream consumer group `coordinator:child:{cid}`
- consumer group does NOT collide with supervisor's `actus:mailbox-supervisor:v1`
- CANCEL_REQUEST matching child → runner.request_stop(PARENT_CANCEL)
- ready_event set after start()
- shutdown cancels the listener task
- irrelevant envelope skipped (no request_stop)
- pre-subscribe race documented (spec §8.5.3 race table, v1 accepts TIMED_OUT)
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.application.services.coordinator_child_cancel_listener import (
    CoordinatorChildCancelListener,
)


def _mk_subscriber_with_consume(envelopes: list[dict] | None = None) -> AsyncMock:
    """Build an AsyncMock subscriber whose .consume yields envelopes once."""
    sub = AsyncMock()
    payloads = envelopes or []

    async def fake_consume(*, predicate, **_kwargs):
        for env in payloads:
            if await predicate(env):
                yield env

    sub.consume = fake_consume
    return sub


@pytest.mark.anyio
async def test_subscribes_root_scoped_stream() -> None:
    sub = AsyncMock()
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=MagicMock(),
    )
    await listener.start()
    sub.subscribe.assert_awaited_once()
    kwargs = sub.subscribe.await_args.kwargs
    assert kwargs["stream_key"] == "actus:child:root1:mailbox"
    assert kwargs["consumer_group"] == "coordinator:child:c1"
    assert kwargs["consumer_name"] == "c1-listener"
    await listener.shutdown()


@pytest.mark.anyio
async def test_consumer_group_isolated_from_supervisor() -> None:
    sub = AsyncMock()
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=MagicMock(),
    )
    await listener.start()
    cg = sub.subscribe.await_args.kwargs["consumer_group"]
    assert "supervisor" not in cg
    assert cg.startswith("coordinator:child:")
    await listener.shutdown()


@pytest.mark.anyio
async def test_cancel_request_calls_request_stop_with_parent_cancel() -> None:
    sub = _mk_subscriber_with_consume([
        {"child_session_id": "c1", "type": "CANCEL_REQUEST", "envelope_id": "env-1"},
    ])
    runner = MagicMock()
    runner.request_stop = MagicMock()
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=runner,
    )
    await listener.start()
    await listener._listen_loop_one_iteration()
    from app.application.services.coordinator_child_runner import StopReason
    runner.request_stop.assert_called_once_with(StopReason.PARENT_CANCEL)
    await listener.shutdown()


@pytest.mark.anyio
async def test_ready_event_before_and_after_start() -> None:
    sub = AsyncMock()
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=MagicMock(),
    )
    assert not listener.ready_event.is_set()
    await listener.start()
    assert listener.ready_event.is_set()
    await listener.shutdown()


@pytest.mark.anyio
async def test_shutdown_cancels_listening_task() -> None:
    sub = AsyncMock()

    async def never_yield(*, predicate, **_kwargs):
        await asyncio.sleep(60)
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    sub.consume = never_yield
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=MagicMock(),
    )
    await listener.start()
    assert listener._task is not None
    assert not listener._task.done()
    await listener.shutdown(timeout=0.5)
    assert listener._task.done()


@pytest.mark.anyio
async def test_irrelevant_envelope_skipped() -> None:
    sub = _mk_subscriber_with_consume([
        {"child_session_id": "other-child", "type": "CANCEL_REQUEST"},
        {"child_session_id": "c1", "type": "PROGRESS_UPDATE"},
    ])
    runner = MagicMock()
    runner.request_stop = MagicMock()
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=runner,
    )
    await listener.start()
    await listener._listen_loop_one_iteration()
    runner.request_stop.assert_not_called()
    await listener.shutdown()


@pytest.mark.anyio
async def test_only_matching_child_cancel_dispatches() -> None:
    sub = _mk_subscriber_with_consume([
        {"child_session_id": "c2", "type": "CANCEL_REQUEST"},
        {"child_session_id": "c1", "type": "PROGRESS_UPDATE"},
        {"child_session_id": "c1", "type": "CANCEL_REQUEST", "envelope_id": "e-cancel"},
    ])
    runner = MagicMock()
    runner.request_stop = MagicMock()
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=runner,
    )
    await listener.start()
    await listener._listen_loop_one_iteration()
    from app.application.services.coordinator_child_runner import StopReason
    runner.request_stop.assert_called_once_with(StopReason.PARENT_CANCEL)
    await listener.shutdown()


@pytest.mark.anyio
async def test_listener_task_fatal_exception_is_logged_not_silent(caplog) -> None:
    """[r5 P1-3 fix] If the listener task dies with an unhandled exception,
    the done-callback must log it. Without this, the listener would die
    silently and subsequent CANCEL_REQUEST envelopes would never reach the
    runner."""
    import logging
    sub = AsyncMock()

    async def boom_consume(*, predicate, **_kwargs):
        raise RuntimeError("subscriber blew up")
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    sub.consume = boom_consume
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=MagicMock(),
    )
    caplog.set_level(logging.ERROR, logger="app.application.services.coordinator_child_cancel_listener")
    await listener.start()
    # Let the task fail.
    await asyncio.sleep(0.05)
    assert listener._task is not None
    assert listener._task.done()
    assert any(
        "died with unhandled exception" in rec.message
        for rec in caplog.records
    ), f"expected fatal log; got {[r.message for r in caplog.records]}"


def test_pre_subscribe_race_contract_pinned_in_module_docstring() -> None:
    """[r7 P2 fix] spec §8.5.3 race table — CANCEL before subscribe →
    supervisor backstop + child wallclock budget watchdog trip TIMED_OUT.

    v1 accepts this race; PR-4 hardens with a startup fence but cannot fully
    eliminate. This test pins the design decision to the module docstring so
    a silent refactor that removes the doc + the backstop expectation fails
    here (replacing the prior tautological ``assert True``)."""
    import app.application.services.coordinator_child_cancel_listener as listener_mod

    doc = listener_mod.__doc__ or ""
    # Three load-bearing claims must appear in the module docstring:
    assert "Pre-subscribe race" in doc, (
        "module docstring must call out the pre-subscribe race contract"
    )
    assert "backstop" in doc or "watchdog" in doc, (
        "module docstring must reference the watchdog backstop"
    )
    assert "TIMED_OUT" in doc, (
        "module docstring must reference the TIMED_OUT outcome path"
    )
