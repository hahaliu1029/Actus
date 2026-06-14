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
    """[C2b budget §3-9 R4#1] The race contract CHANGED: dispatch now
    pre-creates this listener's consumer group, so a CANCEL_REQUEST published
    before subscribe is retained as group backlog — the race is CLOSED, not
    "accepted with TIMED_OUT backstop". This test pins the NEW contract to
    the module docstring so a silent revert of either the doc or the
    pre-creation rationale fails here."""
    import app.application.services.coordinator_child_cancel_listener as listener_mod

    doc = listener_mod.__doc__ or ""
    # Load-bearing claims of the post-split contract:
    assert "Pre-subscribe race" in doc, (
        "module docstring must call out the pre-subscribe race contract"
    )
    assert "CLOSED" in doc, (
        "module docstring must state the race is CLOSED by group pre-creation"
    )
    assert "pre-creates" in doc, (
        "module docstring must credit dispatch group pre-creation"
    )
    assert "BUSYGROUP-idempotent" in doc, (
        "module docstring must pin the idempotent re-subscribe behavior"
    )
    assert "watchdog" in doc, (
        "module docstring must still reference the (now-live) watchdog brake"
    )
    assert "TIMED_OUT" not in doc, (
        "old accepted-race wording must be gone — the backstop story changed"
    )


class _RecordingSubscriber:
    """Concrete subscriber that records subscribe + destroy_group calls.

    AsyncMock auto-creates any attribute; we need explicit tracking so the
    NOGROUP-skip-path test ("destroy_group MUST NOT be called") can assert a
    NEGATIVE invariant. Mirrors the orchestrator's ``_FakeSubscriber`` pattern.
    """

    def __init__(
        self,
        *,
        subscribe_error: BaseException | None = None,
        destroy_group_error: BaseException | None = None,
        consume_never_returns: bool = True,
    ) -> None:
        self._subscribe_error = subscribe_error
        self._destroy_group_error = destroy_group_error
        self._consume_never_returns = consume_never_returns
        self.subscribe_calls: list[dict] = []
        self.destroy_group_calls: list[dict] = []

    async def subscribe(
        self, *, stream_key: str, consumer_group: str,
        consumer_name: str, start_id: str = "$",
    ) -> None:
        self.subscribe_calls.append({
            "stream_key": stream_key,
            "consumer_group": consumer_group,
            "consumer_name": consumer_name,
            "start_id": start_id,
        })
        if self._subscribe_error is not None:
            raise self._subscribe_error

    async def consume(self, *, predicate, **_kwargs):
        if self._consume_never_returns:
            await asyncio.Event().wait()
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    async def destroy_group(
        self, *, stream_key: str, consumer_group: str,
    ) -> None:
        self.destroy_group_calls.append({
            "stream_key": stream_key,
            "consumer_group": consumer_group,
        })
        if self._destroy_group_error is not None:
            raise self._destroy_group_error


class TestConsumerGroupCleanup:
    """[Round 7 P2] Listener MUST destroy its per-listener consumer group in
    shutdown() so dead groups don't accumulate under the long-lived root
    mailbox stream.

    Invariants pinned here:
      1. destroy_group fires once during shutdown() after start() succeeded.
      2. destroy_group skipped when start() was never called (no group).
      3. destroy_group skipped when subscribe failed inside start().
      4. destroy_group failure is logged + swallowed (shutdown stays idempotent).
      5. Destroy fires AFTER the listener task is drained (so any in-flight
         XACK completes before the group is torn down).
    """

    @pytest.mark.anyio
    async def test_destroy_group_called_in_shutdown(self) -> None:
        sub = _RecordingSubscriber()
        listener = CoordinatorChildCancelListener(
            subscriber=sub, root_session_id="root1", child_session_id="c1",
            runner=MagicMock(),
        )
        await listener.start()
        await listener.shutdown()
        assert len(sub.destroy_group_calls) == 1
        assert sub.destroy_group_calls[0] == {
            "stream_key": "actus:child:root1:mailbox",
            "consumer_group": "coordinator:child:c1",
        }

    @pytest.mark.anyio
    async def test_destroy_group_skipped_when_never_started(self) -> None:
        """start() never called → ``_subscribed=False`` → no destroy attempt."""
        sub = _RecordingSubscriber()
        listener = CoordinatorChildCancelListener(
            subscriber=sub, root_session_id="root1", child_session_id="c1",
            runner=MagicMock(),
        )
        # shutdown() before start() — should be a no-op for destroy_group.
        await listener.shutdown()
        assert sub.destroy_group_calls == []

    @pytest.mark.anyio
    async def test_destroy_group_skipped_when_subscribe_failed(self) -> None:
        """subscribe inside start() raised → no group ever created → destroy
        MUST be skipped to avoid spurious NOGROUP roundtrip."""
        sub = _RecordingSubscriber(
            subscribe_error=RuntimeError("redis NOGROUP-equivalent"),
        )
        listener = CoordinatorChildCancelListener(
            subscriber=sub, root_session_id="root1", child_session_id="c1",
            runner=MagicMock(),
        )
        with pytest.raises(RuntimeError, match="redis NOGROUP-equivalent"):
            await listener.start()
        await listener.shutdown()
        # subscribe was attempted; destroy_group MUST be skipped.
        assert sub.subscribe_calls != []
        assert sub.destroy_group_calls == []

    @pytest.mark.anyio
    async def test_destroy_group_failure_logged_not_raised(
        self, caplog,
    ) -> None:
        """If destroy_group raises (Redis hiccup), shutdown() MUST log a
        WARNING and return normally — caller assumes idempotent cleanup."""
        import logging
        caplog.set_level(
            logging.WARNING,
            logger="app.application.services.coordinator_child_cancel_listener",
        )
        sub = _RecordingSubscriber(
            destroy_group_error=RuntimeError(
                "simulated Redis WRONGTYPE on xgroup_destroy",
            ),
        )
        listener = CoordinatorChildCancelListener(
            subscriber=sub, root_session_id="root1", child_session_id="c1",
            runner=MagicMock(),
        )
        await listener.start()
        # shutdown() MUST NOT raise.
        await listener.shutdown()
        assert len(sub.destroy_group_calls) == 1
        warn_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "destroy_group" in r.message
            and "failed" in r.message
        ]
        assert warn_records, (
            "expected at least one WARNING 'destroy_group ... failed' log;"
            f" got: {[(r.levelname, r.message) for r in caplog.records]}"
        )
        assert warn_records[0].exc_info is not None, (
            "destroy_group failure log must include exc_info for traceback"
        )

    @pytest.mark.anyio
    async def test_destroy_group_called_after_task_drain(self) -> None:
        """[ordering invariant] The destroy_group call MUST happen AFTER the
        listener task has been cancelled + drained — otherwise an in-flight
        XACK / XREADGROUP could race against XGROUP DESTROY and emit a stray
        NOGROUP. We verify this by recording observed state at destroy time."""
        sub = _RecordingSubscriber()
        listener = CoordinatorChildCancelListener(
            subscriber=sub, root_session_id="root1", child_session_id="c1",
            runner=MagicMock(),
        )

        observed_task_done: list[bool] = []
        original_destroy = sub.destroy_group

        async def _destroy_recording_task_state(**kwargs):
            observed_task_done.append(
                listener._task is None or listener._task.done()
            )
            await original_destroy(**kwargs)

        sub.destroy_group = _destroy_recording_task_state  # type: ignore[assignment]

        await listener.start()
        await listener.shutdown()
        # destroy_group was called and at that moment the listener task was
        # already done (drained).
        assert observed_task_done == [True], (
            "destroy_group must fire AFTER task drain; observed_task_done="
            f"{observed_task_done}"
        )
