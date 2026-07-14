"""C2 PR-3 §7.5 — CoordinatorTerminalEnvelopeWaiter direct unit tests.

Covers (r6 P2-2 fix): worker_node tests mock the waiter; this file pins the
waiter's own semantics — stream key derivation, waiter consumer group, terminal
predicate, asyncio.wait_for timeout, envelope round-trip via MailboxEnvelope.model_validate.

Plan ref: This file is a R6 follow-up addition beyond the original Task 3.5.5
file list in ``docs/superpowers/plans/2026-05-25-c2-coordinator-task-runner-plan.md``.
The waiter is shipped by Task 3.5.5; codex R6 flagged missing direct tests as
P2 and this file closes that gap. The plan should be amended in a follow-up to
include this test file in Task 3.5.5's deliverable list.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

from pydantic import ValidationError

from app.application.services.coordinator_terminal_envelope_waiter import (
    CoordinatorTerminalEnvelopeWaiter,
)
from app.domain.models.mailbox_envelope import (
    CancelAckPayload,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.infrastructure.external.mailbox.redis_mailbox_subscriber import (
    RedisMailboxSubscriber,
)


def _env_dict(env_type: MailboxEnvelopeType, child_sid: str, payload: dict) -> dict:
    """Construct a wire-shape envelope dict (what subscriber yields)."""
    role = (
        ProducerRole.CHILD_AGENT.value
        if env_type in (MailboxEnvelopeType.RESULT_READY, MailboxEnvelopeType.CANCEL_ACK)
        else ProducerRole.PARENT_AGENT.value
    )
    return {
        "envelope_id": "e-1",
        "type": env_type.value,
        "parent_session_id": "root1",
        "child_session_id": child_sid,
        "correlation_id": "corr-r1",
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "producer_role": role,
        "payload": payload,
        "reclaim_count": 0,
    }


def _mk_subscriber(envelopes: list[dict]) -> AsyncMock:
    sub = AsyncMock()

    async def fake_consume(*, predicate, **_kwargs):
        for env in envelopes:
            if await predicate(env):
                yield env

    sub.consume = fake_consume
    sub.subscribe = AsyncMock()
    return sub


@pytest.mark.anyio
async def test_subscribe_uses_root_scoped_stream_and_waiter_group() -> None:
    """[r6 P2-2] Stream key is actus:child:{root}; group is coordinator:waiter:{cid}."""
    payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    sub = _mk_subscriber([_env_dict(MailboxEnvelopeType.RESULT_READY, "c1", payload)])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    await waiter.await_terminal(
        child_session_id="c1", root_session_id="root1",
        cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=1.0,
    )
    sub.subscribe.assert_awaited_once()
    kwargs = sub.subscribe.await_args.kwargs
    assert kwargs["stream_key"] == "actus:child:root1:mailbox"
    assert kwargs["consumer_group"] == "coordinator:waiter:c1"
    assert kwargs["consumer_name"] == "waiter-c1"


@pytest.mark.anyio
async def test_filters_terminal_types_only() -> None:
    """[r6 P2-2] Non-terminal envelope types (e.g. PROGRESS_UPDATE) MUST be skipped."""
    progress_env = {
        "envelope_id": "e-progress", "type": "PROGRESS_UPDATE",
        "parent_session_id": "p1", "child_session_id": "c1",
        "correlation_id": "corr",
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "producer_role": "child_agent",
        "payload": {"kind": "heartbeat", "visibility": "hidden"},
    }
    ok_payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    terminal_env = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)
    sub = _mk_subscriber([progress_env, terminal_env])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    env = await waiter.await_terminal(
        child_session_id="c1", root_session_id="root1",
        cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=1.0,
    )
    # Only the terminal envelope was yielded — progress was filtered.
    assert env.type == MailboxEnvelopeType.RESULT_READY


@pytest.mark.anyio
async def test_filters_other_child_sessions() -> None:
    """[r6 P2-2] Terminal envelope addressed to a DIFFERENT child MUST be skipped."""
    ok_payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    other_child = _env_dict(MailboxEnvelopeType.RESULT_READY, "OTHER-CHILD", ok_payload)
    my_child = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)
    sub = _mk_subscriber([other_child, my_child])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    env = await waiter.await_terminal(
        child_session_id="c1", root_session_id="root1",
        cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=1.0,
    )
    assert env.child_session_id == "c1"


@pytest.mark.anyio
async def test_default_none_awaits_delayed_terminal_without_wait_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No deadline means a direct consume await, not a hidden fixed timeout."""
    ok_payload = ResultReadyPayload(
        summary="delayed", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    terminal = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)
    sub = AsyncMock()
    sub.subscribe = AsyncMock()
    sub.destroy_group = AsyncMock()

    async def delayed_consume(*, predicate, **_kwargs):
        await asyncio.sleep(0.02)
        if await predicate(terminal):
            yield terminal

    async def forbidden_wait_for(*_args, **_kwargs):
        raise AssertionError("timeout=None must not call asyncio.wait_for")

    sub.consume = delayed_consume
    monkeypatch.setattr(
        "app.application.services.coordinator_terminal_envelope_waiter."
        "asyncio.wait_for",
        forbidden_wait_for,
    )

    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    env = await waiter.await_terminal(
        child_session_id="c1",
        root_session_id="root1",
        cancel_event=asyncio.Event(),
        coordinator_run_id="corr-r1",
    )

    assert env.payload["summary"] == "delayed"
    sub.destroy_group.assert_awaited_once()


@pytest.mark.anyio
async def test_zero_awaits_delayed_terminal_without_wait_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero is the public unlimited sentinel, identical to ``None``."""
    ok_payload = ResultReadyPayload(
        summary="delayed", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    terminal = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)
    sub = AsyncMock()
    sub.subscribe = AsyncMock()
    sub.destroy_group = AsyncMock()

    async def delayed_consume(*, predicate, **_kwargs):
        await asyncio.sleep(0.02)
        if await predicate(terminal):
            yield terminal

    async def forbidden_wait_for(*_args, **_kwargs):
        raise AssertionError("timeout=0 must not call asyncio.wait_for")

    sub.consume = delayed_consume
    monkeypatch.setattr(
        "app.application.services.coordinator_terminal_envelope_waiter."
        "asyncio.wait_for",
        forbidden_wait_for,
    )

    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    env = await waiter.await_terminal(
        child_session_id="c1",
        root_session_id="root1",
        cancel_event=asyncio.Event(),
        coordinator_run_id="corr-r1",
        timeout=0,
    )

    assert env.payload["summary"] == "delayed"
    sub.destroy_group.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_timeout",
    [-1, float("nan"), float("inf"), float("-inf")],
)
async def test_invalid_timeout_rejected_before_subscribe(
    invalid_timeout: float,
) -> None:
    sub = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)

    with pytest.raises(ValueError, match="timeout"):
        await waiter.await_terminal(
                child_session_id="c1",
                root_session_id="root1",
                cancel_event=asyncio.Event(),
                coordinator_run_id="corr-r1",
                timeout=invalid_timeout,
        )

    sub.subscribe.assert_not_awaited()
    sub.destroy_group.assert_not_awaited()


@pytest.mark.anyio
async def test_timeout_raises_when_no_terminal_arrives() -> None:
    """[r6 P2-2] asyncio.wait_for fires after timeout when subscriber yields nothing."""
    sub = AsyncMock()

    async def never_yield(*, predicate, **_kwargs):
        # Sleep forever — emulate stream with no terminal envelope.
        await asyncio.sleep(60)
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    sub.consume = never_yield
    sub.subscribe = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    with pytest.raises(asyncio.TimeoutError) as ei:
        await waiter.await_terminal(
            child_session_id="c1", root_session_id="root1",
            cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=0.05,
        )
    assert "c1" in str(ei.value)


@pytest.mark.anyio
async def test_envelope_validates_to_mailbox_envelope_model() -> None:
    """[r6 P2-2] Returned envelope is a validated MailboxEnvelope (Pydantic model),
    not a raw dict — caller can read .type, .payload, .child_session_id, etc."""
    ok_payload = ResultReadyPayload(
        summary="done", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    sub = _mk_subscriber([_env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    env = await waiter.await_terminal(
        child_session_id="c1", root_session_id="root1",
        cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=1.0,
    )
    # Validated as MailboxEnvelope — outcome field reachable via payload dict.
    assert env.payload["outcome"] == "success"
    assert env.producer_role == ProducerRole.CHILD_AGENT


class _OneBatchRedis:
    """Redis protocol fake; the subscriber and waiter remain production code."""

    def __init__(self, entries: list[tuple[str, dict[bytes, bytes]]]) -> None:
        self._entries = entries
        self._delivered = False
        self._blocked = asyncio.Event()
        self.xack_calls: list[tuple[str, str, str]] = []
        self.xgroup_create_calls: list[dict[str, Any]] = []
        self.xgroup_destroy_calls: list[tuple[str, str]] = []

    async def xgroup_create(self, **kwargs: Any) -> None:
        self.xgroup_create_calls.append(kwargs)

    async def xreadgroup(self, **_kwargs: Any) -> list[Any]:
        if not self._delivered:
            self._delivered = True
            return [("actus:child:root1:mailbox", self._entries)]
        await self._blocked.wait()
        return []

    async def xack(
        self, stream_key: str, consumer_group: str, msg_id: str,
    ) -> None:
        self.xack_calls.append((stream_key, consumer_group, msg_id))

    async def xgroup_destroy(
        self, stream_key: str, consumer_group: str,
    ) -> None:
        self.xgroup_destroy_calls.append((stream_key, consumer_group))


@pytest.mark.anyio
async def test_real_subscriber_yields_identity_match_before_deep_validation() -> None:
    """Malformed expected terminal is ACKed once, then fails deep validation.

    RedisMailboxSubscriber deliberately swallows predicate exceptions. The
    waiter's predicate must therefore remain a total raw identity filter so a
    matching malformed terminal is yielded and validated by ``_consume_one``.
    """
    valid_payload = ResultReadyPayload(
        summary="wrong child", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    wrong_identity = _env_dict(
        MailboxEnvelopeType.RESULT_READY, "other-child", valid_payload,
    )
    malformed_match = _env_dict(
        MailboxEnvelopeType.RESULT_READY,
        "c1",
        {"outcome": "not_a_real_outcome"},
    )
    entries = [
        ("1-0", {b"envelope": json.dumps(["not", "a", "mapping"]).encode()}),
        ("2-0", {b"envelope": json.dumps({}).encode()}),
        ("3-0", {b"envelope": json.dumps(wrong_identity).encode()}),
        ("4-0", {b"envelope": json.dumps(malformed_match).encode()}),
    ]
    redis = _OneBatchRedis(entries)
    liveness = AsyncMock()

    async def never_stale(_child_session_id: str) -> None:
        await asyncio.Future()

    liveness.await_stale = never_stale
    persisted = AsyncMock(return_value=None)
    orphan = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=RedisMailboxSubscriber(redis=redis),  # type: ignore[arg-type]
        liveness_service=liveness,
        persisted_terminal_reader=persisted,
        orphan_reconciler=orphan,
    )

    with pytest.raises(ValidationError):
        await waiter.await_terminal(
            child_session_id="c1",
            root_session_id="root1",
            cancel_event=asyncio.Event(),
            coordinator_run_id="corr-r1",
            timeout=0.05,
        )

    assert [call[2] for call in redis.xack_calls] == [
        "1-0", "2-0", "3-0", "4-0",
    ]
    assert redis.xgroup_destroy_calls == [(
        "actus:child:root1:mailbox", "coordinator:waiter:c1",
    )]
    persisted.assert_not_awaited()
    orphan.assert_not_awaited()


class _RecordingSubscriber:
    """In-memory subscriber that records subscribe + destroy_group calls.

    Mirrors the orchestrator's ``_FakeSubscriber`` testing pattern (track call
    history for invariant assertions). Distinct from the ``_mk_subscriber``
    AsyncMock factory above because we need to:
      1. Assert ``destroy_group`` was called exactly once with the right args.
      2. Inject a ``destroy_group_error`` for the failure-path test.
      3. Optionally make ``subscribe`` raise to validate the
         ``subscribed=False`` skip path.

    AsyncMock auto-creates any attribute as a coroutine which makes negative
    assertions ("was NOT called") tricky; this concrete fake is explicit.
    """

    def __init__(
        self,
        envelopes: list[dict],
        *,
        consume_never_returns: bool = False,
        subscribe_error: BaseException | None = None,
        destroy_group_error: BaseException | None = None,
    ) -> None:
        self._envelopes = envelopes
        self._consume_never_returns = consume_never_returns
        self._subscribe_error = subscribe_error
        self._destroy_group_error = destroy_group_error
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
        for env in self._envelopes:
            if await predicate(env):
                yield env
        if self._consume_never_returns:
            await asyncio.Event().wait()

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
    """[Round 7 P2] Waiter MUST destroy its per-waiter consumer group in the
    await_terminal() finally so dead groups don't accumulate under the
    long-lived root mailbox stream.

    Invariants pinned here:
      1. destroy_group fires once on terminal-envelope return path.
      2. destroy_group fires once on the asyncio.wait_for timeout path.
      3. destroy_group fires once on arbitrary exception inside consume.
      4. destroy_group fires once when caller cancellation interrupts the wait.
      5. destroy_group is NOT called when subscribe itself raised (no group
         to destroy → no spurious NOGROUP roundtrip).
      6. destroy_group failure is logged + swallowed (caller-facing exception,
         e.g. TimeoutError, propagates unmodified).
    """

    @pytest.mark.anyio
    async def test_destroy_group_called_after_terminal_envelope_returns(
        self,
    ) -> None:
        ok_payload = ResultReadyPayload(
            summary="ok", outcome=ResultReadyOutcome.SUCCESS,
        ).model_dump(mode="python")
        sub = _RecordingSubscriber(
            [_env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)],
        )
        waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
        env = await waiter.await_terminal(
            child_session_id="c1", root_session_id="root1",
            cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=1.0,
        )
        assert env.type == MailboxEnvelopeType.RESULT_READY
        assert len(sub.destroy_group_calls) == 1
        assert sub.destroy_group_calls[0] == {
            "stream_key": "actus:child:root1:mailbox",
            "consumer_group": "coordinator:waiter:c1",
        }

    @pytest.mark.anyio
    async def test_destroy_group_called_after_timeout(self) -> None:
        sub = _RecordingSubscriber([], consume_never_returns=True)
        waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
        with pytest.raises(asyncio.TimeoutError):
            await waiter.await_terminal(
                child_session_id="c1", root_session_id="root1",
                cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=0.05,
            )
        # Timeout path STILL hits the finally → destroy_group fires.
        assert len(sub.destroy_group_calls) == 1
        assert sub.destroy_group_calls[0] == {
            "stream_key": "actus:child:root1:mailbox",
            "consumer_group": "coordinator:waiter:c1",
        }

    @pytest.mark.anyio
    async def test_destroy_group_called_on_consume_exception(self) -> None:
        """An unexpected error mid-consume must NOT leak the consumer group."""

        class _BoomSubscriber(_RecordingSubscriber):
            async def consume(self, *, predicate, **_kwargs):
                raise RuntimeError("simulated subscriber crash")
                if False:  # pragma: no cover
                    yield  # type: ignore[unreachable]

        sub = _BoomSubscriber([])
        waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
        with pytest.raises(RuntimeError, match="simulated subscriber crash"):
            await waiter.await_terminal(
                child_session_id="c1", root_session_id="root1",
                cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=1.0,
            )
        assert len(sub.destroy_group_calls) == 1

    @pytest.mark.anyio
    async def test_destroy_group_called_once_when_wait_is_cancelled(self) -> None:
        """Caller cancellation must tear down the group without masking cancel."""
        sub = _RecordingSubscriber([], consume_never_returns=True)
        waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
        task = asyncio.create_task(waiter.await_terminal(
            child_session_id="c1",
            root_session_id="root1",
            cancel_event=asyncio.Event(),
            coordinator_run_id="corr-r1",
        ))
        while not sub.subscribe_calls:
            await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert sub.destroy_group_calls == [{
            "stream_key": "actus:child:root1:mailbox",
            "consumer_group": "coordinator:waiter:c1",
        }]

    @pytest.mark.anyio
    async def test_destroy_group_skipped_when_subscribe_failed(self) -> None:
        """If subscribe itself raised, no group was created — destroy_group
        MUST NOT be called (would emit a spurious NOGROUP roundtrip)."""
        sub = _RecordingSubscriber(
            [], subscribe_error=RuntimeError("redis NOGROUP-equivalent"),
        )
        waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
        with pytest.raises(RuntimeError, match="redis NOGROUP-equivalent"):
            await waiter.await_terminal(
                child_session_id="c1", root_session_id="root1",
                cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=1.0,
            )
        # subscribe was attempted (and failed); destroy_group MUST be skipped.
        assert sub.subscribe_calls != []
        assert sub.destroy_group_calls == []

    @pytest.mark.anyio
    async def test_destroy_group_failure_logged_not_raised(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If destroy_group raises in the finally, the waiter MUST log a
        WARNING and let the caller-facing return value propagate unmodified."""
        import logging
        caplog.set_level(
            logging.WARNING,
            logger="app.application.services.coordinator_terminal_envelope_waiter",
        )
        ok_payload = ResultReadyPayload(
            summary="ok", outcome=ResultReadyOutcome.SUCCESS,
        ).model_dump(mode="python")
        sub = _RecordingSubscriber(
            [_env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)],
            destroy_group_error=RuntimeError(
                "simulated Redis WRONGTYPE on xgroup_destroy",
            ),
        )
        waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
        env = await waiter.await_terminal(
            child_session_id="c1", root_session_id="root1",
            cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=1.0,
        )
        # Terminal envelope still returns normally; destroy_group error
        # is swallowed.
        assert env.type == MailboxEnvelopeType.RESULT_READY
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
    async def test_destroy_group_failure_on_timeout_preserves_timeout(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """[hardening] When BOTH the consume path times out AND destroy_group
        raises in the finally, the caller still sees ``asyncio.TimeoutError``
        — the cleanup error must not mask the primary exception."""
        import logging
        caplog.set_level(
            logging.WARNING,
            logger="app.application.services.coordinator_terminal_envelope_waiter",
        )
        sub = _RecordingSubscriber(
            [],
            consume_never_returns=True,
            destroy_group_error=RuntimeError("destroy boom"),
        )
        waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
        with pytest.raises(asyncio.TimeoutError):
            await waiter.await_terminal(
                child_session_id="c1", root_session_id="root1",
                cancel_event=asyncio.Event(), coordinator_run_id="corr-r1", timeout=0.05,
            )
        assert len(sub.destroy_group_calls) == 1
        assert any(
            "destroy_group" in r.message and "failed" in r.message
            for r in caplog.records
        )


@pytest.mark.anyio
async def test_terminal_wins_and_stale_loser_is_cancelled_and_drained() -> None:
    payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    sub = _RecordingSubscriber([
        _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", payload),
    ])
    stale_cancelled = asyncio.Event()

    async def await_stale(_child_id: str):
        try:
            await asyncio.Future()
        finally:
            stale_cancelled.set()

    liveness = AsyncMock()
    liveness.await_stale = await_stale
    orphan = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=sub,
        liveness_service=liveness,
        orphan_reconciler=orphan,
        persisted_terminal_reader=AsyncMock(return_value=None),
    )

    result = await waiter.await_terminal(
        child_session_id="c1",
        root_session_id="root1",
        cancel_event=asyncio.Event(),
        coordinator_run_id="corr-r1",
    )

    assert result.type == MailboxEnvelopeType.RESULT_READY
    assert stale_cancelled.is_set()
    orphan.assert_not_awaited()
    assert len(sub.destroy_group_calls) == 1


@pytest.mark.anyio
async def test_stale_first_reconciles_once_then_waits_for_authoritative_terminal() -> None:
    payload = ResultReadyPayload(
        summary="eventual", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    terminal = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", payload)
    sub = _RecordingSubscriber([])

    async def delayed_consume(*, predicate, **_kwargs):
        await asyncio.sleep(0.01)
        if await predicate(terminal):
            yield terminal

    sub.consume = delayed_consume
    liveness = AsyncMock()
    liveness.await_stale = AsyncMock(return_value=None)
    orphan = AsyncMock()
    persisted = AsyncMock(return_value=None)
    waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=sub,
        liveness_service=liveness,
        orphan_reconciler=orphan,
        persisted_terminal_reader=persisted,
    )

    result = await waiter.await_terminal(
        child_session_id="c1",
        root_session_id="root1",
        cancel_event=asyncio.Event(),
        coordinator_run_id="corr-r1",
    )

    assert result.payload["summary"] == "eventual"
    persisted.assert_awaited_once()
    orphan.assert_awaited_once_with(
        child_session_id="c1",
        root_session_id="root1",
        coordinator_run_id="corr-r1",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("redis liveness read failed"),
        asyncio.TimeoutError("redis liveness timeout"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
        asyncio.CancelledError("stale read cancelled"),
    ],
)
async def test_stale_failure_propagates_without_persisted_or_orphan_side_effect(
    error: BaseException,
) -> None:
    terminal_cancelled = asyncio.Event()
    sub = _RecordingSubscriber([])

    async def never_consume(**_kwargs):
        try:
            await asyncio.Future()
        finally:
            terminal_cancelled.set()
        if False:  # pragma: no cover - keeps this an async generator
            yield {}

    sub.consume = never_consume
    liveness = AsyncMock()
    liveness.await_stale = AsyncMock(side_effect=error)
    persisted = AsyncMock(return_value=_env_dict(
        MailboxEnvelopeType.CANCEL_ACK,
        "c1",
        CancelAckPayload(final_state="cancelled").model_dump(mode="python"),
    ))
    orphan = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=sub,
        liveness_service=liveness,
        orphan_reconciler=orphan,
        persisted_terminal_reader=persisted,
    )

    with pytest.raises(type(error)) as caught:
        await waiter.await_terminal(
            child_session_id="c1",
            root_session_id="root1",
            cancel_event=asyncio.Event(),
            coordinator_run_id="corr-r1",
            timeout=1.0,
        )

    assert caught.value is error
    persisted.assert_not_awaited()
    orphan.assert_not_awaited()
    assert terminal_cancelled.is_set()
    assert len(sub.destroy_group_calls) == 1


@pytest.mark.anyio
async def test_stale_same_tick_prefers_persisted_terminal_without_orphan() -> None:
    payload = ResultReadyPayload(
        summary="persisted", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    persisted_envelope = _env_dict(
        MailboxEnvelopeType.RESULT_READY, "c1", payload,
    )
    sub = _RecordingSubscriber([], consume_never_returns=True)
    liveness = AsyncMock()
    liveness.await_stale = AsyncMock(return_value=None)
    orphan = AsyncMock()
    persisted = AsyncMock(return_value=persisted_envelope)
    waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=sub,
        liveness_service=liveness,
        orphan_reconciler=orphan,
        persisted_terminal_reader=persisted,
    )

    result = await waiter.await_terminal(
        child_session_id="c1",
        root_session_id="root1",
        cancel_event=asyncio.Event(),
        coordinator_run_id="corr-r1",
    )

    assert result.payload["summary"] == "persisted"
    orphan.assert_not_awaited()
    assert len(sub.destroy_group_calls) == 1


@pytest.mark.anyio
async def test_stream_rejects_wrong_root_alongside_other_wrong_identity() -> None:
    payload = ResultReadyPayload(
        summary="valid", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    valid = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", payload)
    valid["parent_session_id"] = "root1"
    wrong_type = dict(valid, type=MailboxEnvelopeType.PROGRESS_UPDATE.value)
    wrong_type["payload"] = {"kind": "heartbeat", "visibility": "hidden"}
    wrong_child = dict(valid, child_session_id="other-child")
    wrong_root = dict(valid, parent_session_id="other-root")
    wrong_run = dict(valid, correlation_id="other-run")
    sub = _RecordingSubscriber([
        wrong_type, wrong_child, wrong_root, wrong_run, valid,
    ])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)

    result = await waiter.await_terminal(
        child_session_id="c1",
        root_session_id="root1",
        cancel_event=asyncio.Event(),
        coordinator_run_id="corr-r1",
        timeout=1.0,
    )

    assert result.parent_session_id == "root1"
    assert result.payload["summary"] == "valid"


@pytest.mark.anyio
@pytest.mark.parametrize("as_model", [False, True])
@pytest.mark.parametrize(
    "bad_update",
    [
        {"type": MailboxEnvelopeType.PROGRESS_UPDATE.value,
         "payload": {"kind": "heartbeat", "visibility": "hidden"}},
        {"child_session_id": "other-child"},
        {"parent_session_id": "other-root"},
        {"correlation_id": "other-run"},
    ],
)
async def test_persisted_terminal_wrong_identity_fails_closed(
    bad_update: dict[str, object],
    as_model: bool,
) -> None:
    payload = ResultReadyPayload(
        summary="persisted", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    persisted_value = _env_dict(
        MailboxEnvelopeType.RESULT_READY, "c1", payload,
    )
    persisted_value["parent_session_id"] = "root1"
    persisted_value.update(bad_update)
    if as_model:
        persisted_value = MailboxEnvelope.model_validate(persisted_value)

    sub = _RecordingSubscriber([], consume_never_returns=True)
    liveness = AsyncMock()
    liveness.await_stale = AsyncMock(return_value=None)
    orphan = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=sub,
        liveness_service=liveness,
        orphan_reconciler=orphan,
        persisted_terminal_reader=AsyncMock(return_value=persisted_value),
    )

    with pytest.raises(ValueError, match="terminal envelope"):
        await waiter.await_terminal(
            child_session_id="c1",
            root_session_id="root1",
            cancel_event=asyncio.Event(),
            coordinator_run_id="corr-r1",
            timeout=1.0,
        )

    orphan.assert_not_awaited()


@pytest.mark.anyio
async def test_terminal_arriving_during_persisted_read_wins_without_orphan() -> None:
    payload = ResultReadyPayload(
        summary="stream won", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    terminal = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", payload)
    terminal["parent_session_id"] = "root1"
    persisted_started = asyncio.Event()
    release_persisted = asyncio.Event()
    terminal_ready = asyncio.Event()
    terminal_yielded = asyncio.Event()
    sub = _RecordingSubscriber([])

    async def consume(*, predicate, **_kwargs):
        await terminal_ready.wait()
        if await predicate(terminal):
            terminal_yielded.set()
            yield terminal

    async def read_persisted(**_kwargs):
        persisted_started.set()
        await release_persisted.wait()
        return None

    sub.consume = consume
    liveness = AsyncMock()
    liveness.await_stale = AsyncMock(return_value=None)
    orphan = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=sub,
        liveness_service=liveness,
        orphan_reconciler=orphan,
        persisted_terminal_reader=read_persisted,
    )
    task = asyncio.create_task(waiter.await_terminal(
        child_session_id="c1",
        root_session_id="root1",
        cancel_event=asyncio.Event(),
        coordinator_run_id="corr-r1",
    ))

    await persisted_started.wait()
    terminal_ready.set()
    await terminal_yielded.wait()
    await asyncio.sleep(0)
    release_persisted.set()
    result = await task

    assert result.payload["summary"] == "stream won"
    orphan.assert_not_awaited()


@pytest.mark.anyio
async def test_terminal_failure_during_persisted_read_propagates_before_orphan() -> None:
    error = RuntimeError("terminal subscriber failed")
    persisted_started = asyncio.Event()
    release_persisted = asyncio.Event()
    terminal_ready = asyncio.Event()
    terminal_failed = asyncio.Event()
    sub = _RecordingSubscriber([])

    async def consume(**_kwargs):
        await terminal_ready.wait()
        terminal_failed.set()
        raise error
        if False:  # pragma: no cover - async generator shape
            yield {}

    async def read_persisted(**_kwargs):
        persisted_started.set()
        await release_persisted.wait()
        return None

    sub.consume = consume
    liveness = AsyncMock()
    liveness.await_stale = AsyncMock(return_value=None)
    orphan = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(
        subscriber=sub,
        liveness_service=liveness,
        orphan_reconciler=orphan,
        persisted_terminal_reader=read_persisted,
    )
    task = asyncio.create_task(waiter.await_terminal(
        child_session_id="c1",
        root_session_id="root1",
        cancel_event=asyncio.Event(),
        coordinator_run_id="corr-r1",
    ))

    await persisted_started.wait()
    terminal_ready.set()
    await terminal_failed.wait()
    await asyncio.sleep(0)
    release_persisted.set()
    with pytest.raises(RuntimeError) as caught:
        await task

    assert caught.value is error
    orphan.assert_not_awaited()
