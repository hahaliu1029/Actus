"""C2 PR-6 §11.3-§11.4 + §11.8 — CoordinatorRunOrchestrator full path.

Covers:
- ``should_trigger_sibling_cancel`` 5-case invariant matrix (§11.8 r15)
- observer_loop + cancel_watcher concurrent operation via asyncio.wait
  FIRST_COMPLETED
- ResultReadyOutcome routing
- NeedsAuthorizationDetails.reason='exploration_proposal' carve-out
- backward-compat: subscriber=None → PR-3 parent-cancel-only behavior is
  preserved (the PR-3 skeleton tests cover that path; here we pin only the
  new PR-6 semantics)
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import pytest
from unittest.mock import AsyncMock

from app.application.services.coordinator_envelope_factory import (
    CoordinatorEnvelopeFactory,
)
from app.application.services.coordinator_run_orchestrator import (
    CoordinatorRunOrchestrator,
    should_trigger_sibling_cancel,
)
from app.domain.models.mailbox_envelope import (
    CancelAckPayload,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.needs_authorization_details import (
    NeedsAuthorizationDetails,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _result_ready_env(
    outcome: ResultReadyOutcome,
    *,
    child_session_id: str = "c1",
    needs_auth: Optional[NeedsAuthorizationDetails] = None,
) -> MailboxEnvelope:
    payload = ResultReadyPayload(
        summary="x",
        outcome=outcome,
        needs_authorization_details=needs_auth,
    )
    return CoordinatorEnvelopeFactory().make_result_ready(
        parent_session_id="p1",
        child_session_id=child_session_id,
        correlation_id="r1",
        payload=payload,
    )


def _cancel_ack_env(*, child_session_id: str = "c1") -> MailboxEnvelope:
    return CoordinatorEnvelopeFactory().make_cancel_ack(
        parent_session_id="p1",
        child_session_id=child_session_id,
        correlation_id="r1",
        payload=CancelAckPayload(final_state="cancelled"),
    )


class _FakeSubscriber:
    """In-memory MailboxSubscriber for unit tests.

    ``subscribe`` records args; ``consume`` yields the pre-loaded
    envelope dicts in order. After the last envelope, ``consume`` sleeps
    forever (to mimic the live polling loop that never returns naturally)
    UNLESS ``exhaust_then_return`` is set — then it returns and the
    observer_loop exits, simulating "all envelopes drained".
    """

    def __init__(
        self,
        envelopes: list[MailboxEnvelope],
        *,
        exhaust_then_return: bool = False,
        delay_between: float = 0.0,
    ) -> None:
        # Serialize to JSON-shape dicts (mimics Redis wire roundtrip — enums
        # become strings); orchestrator's _deserialize re-parses via
        # MailboxEnvelope.model_validate, which normalizes enums back via the
        # _validate_payload_matches_type model_validator.
        self._dicts = [env.model_dump(mode="json") for env in envelopes]
        self._exhaust_then_return = exhaust_then_return
        self._delay = delay_between
        self.subscribe_calls: list[dict[str, Any]] = []
        # [Round 6 P2] Track destroy_group calls so tests can assert the
        # orchestrator cleans up its per-run consumer group in finally.
        self.destroy_group_calls: list[dict[str, str]] = []
        # Optional injected error for destroy_group (used by failure-path
        # tests to verify orchestrator logs + swallows).
        self.destroy_group_error: Optional[BaseException] = None

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

    async def consume(  # type: ignore[override]
        self, *, stream_key: str, consumer_group: str,
        consumer_name: str,
        predicate: Callable[[dict[str, Any]], Awaitable[bool]],
        max_iterations: Optional[int] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        for env_dict in self._dicts:
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            if await predicate(env_dict):
                yield env_dict
        if not self._exhaust_then_return:
            # Mimic live polling: never return naturally; let asyncio.wait
            # cancel us.
            await asyncio.Event().wait()

    async def destroy_group(
        self, *, stream_key: str, consumer_group: str,
    ) -> None:
        self.destroy_group_calls.append({
            "stream_key": stream_key,
            "consumer_group": consumer_group,
        })
        if self.destroy_group_error is not None:
            raise self.destroy_group_error


# ── Predicate matrix (§11.8 — pure unit) ─────────────────────────────────────


class TestPredicateInvariantMatrix:
    """[spec §11.8 r15] should_trigger_sibling_cancel 5-case matrix."""

    def test_success_does_not_trigger(self) -> None:
        env = _result_ready_env(ResultReadyOutcome.SUCCESS)
        assert should_trigger_sibling_cancel(env) is False

    @pytest.mark.parametrize("outcome", [
        ResultReadyOutcome.FAILED,
        ResultReadyOutcome.TIMED_OUT,
        ResultReadyOutcome.CANCELLED,
    ])
    def test_hard_terminal_triggers(self, outcome: ResultReadyOutcome) -> None:
        env = _result_ready_env(outcome)
        assert should_trigger_sibling_cancel(env) is True

    def test_needs_auth_exploration_proposal_does_not_trigger(self) -> None:
        env = _result_ready_env(
            ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_auth=NeedsAuthorizationDetails(reason="exploration_proposal"),
        )
        assert should_trigger_sibling_cancel(env) is False

    @pytest.mark.parametrize("reason", [
        "out_of_tool_allowlist",
        "out_of_path_lease",
        "op_mismatch",
        "hard_blocked",
        "budget_exhausted",
        "lease_expired",
        "revision_drift",
    ])
    def test_needs_auth_other_reason_triggers(self, reason: str) -> None:
        env = _result_ready_env(
            ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_auth=NeedsAuthorizationDetails(reason=reason),
        )
        assert should_trigger_sibling_cancel(env) is True

    def test_cancel_ack_does_not_trigger(self) -> None:
        env = _cancel_ack_env()
        assert should_trigger_sibling_cancel(env) is False

    def test_after_wire_roundtrip_string_outcome_still_handled(self) -> None:
        """[robustness] Wire round-trip via mode='json' dumps outcome to str.

        The predicate must accept both enum (post-deserialize) and string
        (pre-deserialize) — defensive given producer-vs-consumer dual mode.
        """
        env = _result_ready_env(ResultReadyOutcome.FAILED)
        json_dict = env.model_dump(mode="json")
        assert isinstance(json_dict["payload"]["outcome"], str)
        roundtripped = MailboxEnvelope.model_validate(json_dict)
        # after model_validate the validator normalizes payload back; outcome
        # becomes the enum value (per _validate_payload_matches_type)
        assert should_trigger_sibling_cancel(roundtripped) is True


# ── Integration: observer + cancel watcher concurrency ───────────────────────


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


pytestmark = pytest.mark.anyio


class TestSiblingCancelFanOut:
    """[spec §11.3] observer sees terminal → fans out CANCEL_REQUEST to siblings."""

    async def test_failed_outcome_cancels_remaining_pending(self) -> None:
        publisher = AsyncMock()
        env_fail = _result_ready_env(
            ResultReadyOutcome.FAILED, child_session_id="c1",
        )
        subscriber = _FakeSubscriber([env_fail], exhaust_then_return=True)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2", "wu3"],
            child_session_ids={"wu1": "c1", "wu2": "c2", "wu3": "c3"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        # wu1 (the FAILED one) is NOT cancelled — already terminal.
        # wu2 + wu3 receive CANCEL_REQUEST.
        assert publisher.publish.await_count == 2
        cancelled_child_ids = {
            call.args[0].child_session_id
            for call in publisher.publish.await_args_list
        }
        assert cancelled_child_ids == {"c2", "c3"}
        for call in publisher.publish.await_args_list:
            env = call.args[0]
            assert env.type == MailboxEnvelopeType.CANCEL_REQUEST
            assert "sibling_terminal_failed" in env.payload["reason"]

    async def test_success_does_not_cancel_others(self) -> None:
        publisher = AsyncMock()
        env_ok = _result_ready_env(
            ResultReadyOutcome.SUCCESS, child_session_id="c1",
        )
        subscriber = _FakeSubscriber([env_ok], exhaust_then_return=True)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        publisher.publish.assert_not_called()

    async def test_exploration_proposal_does_not_cancel(self) -> None:
        publisher = AsyncMock()
        env = _result_ready_env(
            ResultReadyOutcome.NEEDS_AUTHORIZATION,
            child_session_id="c1",
            needs_auth=NeedsAuthorizationDetails(reason="exploration_proposal"),
        )
        subscriber = _FakeSubscriber([env], exhaust_then_return=True)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        publisher.publish.assert_not_called()

    async def test_needs_auth_other_reason_cancels_siblings(self) -> None:
        publisher = AsyncMock()
        env = _result_ready_env(
            ResultReadyOutcome.NEEDS_AUTHORIZATION,
            child_session_id="c1",
            needs_auth=NeedsAuthorizationDetails(reason="out_of_path_lease"),
        )
        subscriber = _FakeSubscriber([env], exhaust_then_return=True)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        assert publisher.publish.await_count == 1
        cancelled = publisher.publish.await_args_list[0].args[0]
        assert cancelled.child_session_id == "c2"
        assert "needs_authorization" in cancelled.payload["reason"]

    async def test_cancel_ack_does_not_cascade(self) -> None:
        publisher = AsyncMock()
        env = _cancel_ack_env(child_session_id="c1")
        subscriber = _FakeSubscriber([env], exhaust_then_return=True)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        publisher.publish.assert_not_called()


class TestObserverExitConditions:
    """Observer terminates cleanly under various conditions."""

    async def test_all_pending_resolved_returns_naturally(self) -> None:
        publisher = AsyncMock()
        env1 = _result_ready_env(ResultReadyOutcome.SUCCESS, child_session_id="c1")
        env2 = _result_ready_env(ResultReadyOutcome.SUCCESS, child_session_id="c2")
        subscriber = _FakeSubscriber([env1, env2], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=2.0,
        )
        publisher.publish.assert_not_called()

    async def test_timeout_with_no_terminal_no_cancel(self) -> None:
        publisher = AsyncMock()
        subscriber = _FakeSubscriber([], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()  # never set
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1"],
            child_session_ids={"wu1": "c1"},
            cancel_event=cancel_event, timeout_seconds=0.05,
        )
        publisher.publish.assert_not_called()


class TestParentCancelStillWorksWithSubscriber:
    """[spec §11.3] cancel_event takes a parallel path independent of observer."""

    async def test_parent_cancel_event_fires_cancel_to_all_pending(self) -> None:
        publisher = AsyncMock()
        subscriber = _FakeSubscriber([], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()

        async def fire_cancel() -> None:
            await asyncio.sleep(0.02)
            cancel_event.set()

        asyncio.create_task(fire_cancel())
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        assert publisher.publish.await_count == 2
        for call in publisher.publish.await_args_list:
            env = call.args[0]
            assert env.type == MailboxEnvelopeType.CANCEL_REQUEST
            assert env.payload["reason"] == "parent_cancel"


class TestEmitEventHook:
    """[PR-8 prep] If emit_event callable is wired, orchestrator notifies on
    sibling-cancel decisions so downstream SSE surface can render them."""

    async def test_emit_event_called_on_sibling_cancel(self) -> None:
        # [C2 PR-8 §13 Task 8.4] Orchestrator now emits a typed
        # CoordinatorSiblingCancelEvent (was a placeholder dict in PR-6).
        from app.domain.models.event import CoordinatorSiblingCancelEvent

        publisher = AsyncMock()
        emitted: list[Any] = []

        async def emit(event: Any) -> None:
            emitted.append(event)

        env = _result_ready_env(
            ResultReadyOutcome.FAILED, child_session_id="c1",
        )
        subscriber = _FakeSubscriber([env], exhaust_then_return=True)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
            emit_event=emit,
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, CoordinatorSiblingCancelEvent)
        assert ev.coordinator_run_id == "r1"
        assert ev.triggered_by_work_unit_id == "wu1"
        assert ev.triggered_by_outcome == ResultReadyOutcome.FAILED
        assert ev.cancelled_work_unit_ids == ["wu2"]

    async def test_emit_event_not_called_on_success_terminal(self) -> None:
        publisher = AsyncMock()
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        env = _result_ready_env(
            ResultReadyOutcome.SUCCESS, child_session_id="c1",
        )
        subscriber = _FakeSubscriber([env], exhaust_then_return=True)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
            emit_event=emit,
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        assert emitted == []


class TestSubscriberOptionalForPR3Compat:
    """PR-3 contract preservation — subscriber=None falls back to parent-cancel
    only. This is here to pin the API surface; the PR-3 skeleton tests
    (test_coordinator_run_orchestrator_skeleton.py) cover the full behavior.
    """

    async def test_no_subscriber_still_works_for_parent_cancel(self) -> None:
        publisher = AsyncMock()
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            parent_session_id="p1", coordinator_run_id="r1",
            # mailbox_subscriber omitted
        )
        cancel_event = asyncio.Event()
        cancel_event.set()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1"],
            child_session_ids={"wu1": "c1"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        assert publisher.publish.await_count == 1


class TestPublishDedup:
    """[spec §11.5] Once a wu has been cancelled, re-fire from a subsequent
    path (observer then watcher, or two observer events) must not
    double-publish."""

    async def test_observer_then_watcher_no_double_publish(self) -> None:
        publisher = AsyncMock()
        env = _result_ready_env(
            ResultReadyOutcome.FAILED, child_session_id="c1",
        )
        subscriber = _FakeSubscriber([env], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()

        async def fire_after() -> None:
            await asyncio.sleep(0.05)
            cancel_event.set()

        asyncio.create_task(fire_after())
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        # wu1 had terminal, so wu2 cancelled by observer. Watcher should
        # NOT re-publish for wu2.
        assert publisher.publish.await_count == 1


class _FailingSubscriber(_FakeSubscriber):
    """Subscriber whose ``subscribe`` raises (simulates Redis NOGROUP/ACL/
    connectivity failure). ``consume`` is inherited but never invoked because
    the orchestrator must NOT spawn the observer when subscribe failed."""

    async def subscribe(  # type: ignore[override]
        self, *, stream_key: str, consumer_group: str,
        consumer_name: str, start_id: str = "$",
    ) -> None:
        self.subscribe_calls.append({
            "stream_key": stream_key,
            "consumer_group": consumer_group,
            "consumer_name": consumer_name,
            "start_id": start_id,
        })
        raise RuntimeError("redis NOGROUP-equivalent")


class TestSubscribeFailure:
    """[regression] When subscribe fails (any non-BUSYGROUP exception), the
    orchestrator must NOT spawn the observer (whose consume would silently
    exit on NOGROUP and trip asyncio.wait FIRST_COMPLETED, cancelling the
    watcher before parent_cancel could fire). Only the watcher is spawned,
    so parent cancel is still honoured."""

    async def test_subscribe_failure_still_honors_parent_cancel(self) -> None:
        publisher = AsyncMock()
        subscriber = _FailingSubscriber([], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()

        async def fire_cancel() -> None:
            await asyncio.sleep(0.02)
            cancel_event.set()

        asyncio.create_task(fire_cancel())
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        # Parent cancel still delivered to all pending wu via the watcher.
        assert publisher.publish.await_count == 2
        for call in publisher.publish.await_args_list:
            env = call.args[0]
            assert env.type == MailboxEnvelopeType.CANCEL_REQUEST
            assert env.payload["reason"] == "parent_cancel"
        # subscribe was attempted exactly once.
        assert len(subscriber.subscribe_calls) == 1


class TestObserverErrorSurfacing:
    """[codex R2 P1-6] Programming errors inside the observer loop must
    surface in logs at ERROR level instead of being silently swallowed
    by the catch-all + asyncio.wait FIRST_COMPLETED.

    The original behavior caught any ``Exception``, logged it, and let
    the observer task exit naturally — which tripped ``asyncio.wait``,
    cancelled the watcher, and made parent_cancel disappear. The fix
    is two-part:

    1. The observer's catch-all re-raises after logging so the task
       completes with an exception (visible via ``task.exception()``).
    2. ``_run_with_observer`` inspects gather results after
       ``return_exceptions=True`` and logs any non-CancelledError
       task exception at ERROR with full traceback.
    """

    class _ProgrammingErrorSubscriber:
        """Subscriber whose ``consume`` raises an ``AttributeError`` (the
        canonical 'programming error' shape: typo / missing attr / dict
        key bug). Real observer code would only crash this way if the
        envelope shape changed under it."""

        def __init__(self) -> None:
            self.subscribe_calls: list[dict[str, Any]] = []

        async def subscribe(
            self, *, stream_key: str, consumer_group: str,
            consumer_name: str, start_id: str = "$",
        ) -> None:
            self.subscribe_calls.append({"stream_key": stream_key})

        async def consume(  # type: ignore[override]
            self, *, stream_key: str, consumer_group: str,
            consumer_name: str,
            predicate: Callable[[dict[str, Any]], Awaitable[bool]],
            max_iterations: Optional[int] = None,
        ) -> AsyncIterator[dict[str, Any]]:
            # Empty async-generator body that raises immediately is a
            # syntactic quirk — wrap with an unreachable yield so this
            # remains a generator function.
            raise AttributeError("simulated typo: env_dict.recipient")
            yield  # pragma: no cover — keeps this as an async generator

    async def test_observer_programming_error_logged_at_error(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An AttributeError inside observer_loop must reach the
        post-gather inspection and be logged at ERROR (not swallowed)."""
        import logging
        caplog.set_level(
            logging.ERROR,
            logger="app.application.services.coordinator_run_orchestrator",
        )

        publisher = AsyncMock()
        subscriber = self._ProgrammingErrorSubscriber()
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        # Caller-facing run() must NOT raise even though the observer
        # task ended with an exception.
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1"],
            child_session_ids={"wu1": "c1"},
            cancel_event=cancel_event, timeout_seconds=0.5,
        )

        # The programming error must be visible in logs at ERROR level.
        error_records = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR
            and "task" in r.message
            and "exception" in r.message
        ]
        assert error_records, (
            "expected at least one ERROR-level 'task ... ended with "
            "exception' log; got: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )
        # exc_info attached so the traceback survives — pytest's
        # caplog stores it on the LogRecord.
        rec = error_records[0]
        assert rec.exc_info is not None, (
            "post-gather inspection must log with exc_info so the "
            "traceback is captured; got exc_info=None"
        )

    async def test_observer_cancelled_error_not_logged_as_error(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The 'loser' task cancellation (CancelledError from
        ``task.cancel()`` in the finally) is the normal asyncio.wait
        outcome and MUST NOT be logged at ERROR."""
        import logging
        caplog.set_level(
            logging.ERROR,
            logger="app.application.services.coordinator_run_orchestrator",
        )

        publisher = AsyncMock()
        # Empty subscriber, never returns naturally → observer is
        # cancelled when the watcher wins on cancel_event.set().
        subscriber = _FakeSubscriber([], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()

        async def fire_cancel() -> None:
            await asyncio.sleep(0.02)
            cancel_event.set()

        asyncio.create_task(fire_cancel())
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1"],
            child_session_ids={"wu1": "c1"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )

        # No ERROR-level "task ... ended with exception" records — the
        # CancelledError path is the expected loser outcome.
        spurious = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR
            and "ended with" in r.message
            and "exception" in r.message
        ]
        assert spurious == [], (
            "CancelledError loser must not log at ERROR; got: "
            f"{[(r.levelname, r.message) for r in spurious]}"
        )


class TestPublishRollback:
    """[invariant] When a publish in ``_publish_cancel_to_pending`` fails for
    one wu, that wu must be rolled back out of ``_published`` so a subsequent
    path (watcher after observer failure, retry, etc.) is allowed to re-issue
    the cancel. Successful publishes stay in ``_published`` (dedup intact)."""

    async def test_publish_failure_rolls_back_dedup(self) -> None:
        publisher = AsyncMock()
        # publish() call order (within _fan_out_sibling_cancel) is sorted by
        # wu_id, so the calls fire as: wu2 (raise), wu3 (ok).
        publisher.publish.side_effect = [RuntimeError("transient"), None]
        env = _result_ready_env(
            ResultReadyOutcome.FAILED, child_session_id="c1",
        )
        subscriber = _FakeSubscriber([env], exhaust_then_return=True)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2", "wu3"],
            child_session_ids={"wu1": "c1", "wu2": "c2", "wu3": "c3"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        # wu1 (terminal) excluded; wu2 publish raised → rolled back; wu3 ok.
        assert "wu3" in orch._published
        assert "wu2" not in orch._published
        # wu1 was the terminal source, never published as a cancel target.
        assert "wu1" not in orch._published


class TestCancelledErrorDuringPublishRollsBackDedup:
    """[Round 3 P1-3] If ``asyncio.CancelledError`` (a ``BaseException``
    subclass on Py3.12, NOT ``Exception``) fires while ``publisher.publish``
    is suspended, the wu_id MUST be discarded from ``_published`` so a
    retry path can re-issue the CANCEL_REQUEST.

    The bug was: an ``except Exception`` rollback misses ``CancelledError``,
    leaving the wu_id in ``_published`` while no envelope was ever delivered
    → watcher / sibling-cancel observes "already published" → skips →
    child keeps running.
    """

    async def test_publish_cancel_to_pending_cancelled_error_rolls_back(
        self,
    ) -> None:
        """``CancelledError`` mid-publish discards the wu_id AND re-raises
        the cancellation (so the orchestrator actually aborts instead of
        silently treating it as a "skipped wu_id")."""
        publisher = AsyncMock()
        publisher.publish.side_effect = asyncio.CancelledError()
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            parent_session_id="p1",
            coordinator_run_id="r1",
        )
        with pytest.raises(asyncio.CancelledError):
            await orch._publish_cancel_to_pending(
                wu_ids=["wu1"],
                child_session_ids={"wu1": "c1"},
                coordinator_run_id="r1",
                reason="parent_cancel",
            )
        # dedup rolled back so a downstream retry can publish.
        assert "wu1" not in orch._published

    async def test_publish_cancel_to_pending_runtime_error_still_swallows(
        self,
    ) -> None:
        """Regression guard for the PR-3 / round-2 behavior: an ordinary
        ``Exception`` during publish must NOT propagate — it must roll back
        dedup and continue to the next wu_id. (This is what
        ``TestPublishRollback.test_publish_failure_rolls_back_dedup`` already
        covers end-to-end through ``run()``; here we pin the helper itself.)
        """
        publisher = AsyncMock()
        publisher.publish.side_effect = [RuntimeError("transient"), None]
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            parent_session_id="p1",
            coordinator_run_id="r1",
        )
        # No exception escapes — Exception is swallowed + logged.
        published = await orch._publish_cancel_to_pending(
            wu_ids=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            coordinator_run_id="r1",
            reason="parent_cancel",
        )
        # wu1 rolled back (its publish raised RuntimeError);
        # wu2 stayed (success).
        assert "wu1" not in orch._published
        assert "wu2" in orch._published
        assert published == ["wu2"]


class TestConsumerGroupCleanup:
    """[Round 6 P2] Orchestrator MUST destroy its per-run consumer group in
    the run() finally block so dead groups don't accumulate as XPENDING /
    group-metadata entries under a long-lived root session's mailbox stream.

    Five-property invariant covered here:
      1. destroy_group fires once on natural drain (all pending resolved).
      2. destroy_group fires once on parent-cancel exit.
      3. destroy_group fires once on the no-progress timeout path.
      4. destroy_group is NOT called when subscribe failed (no group to destroy).
      5. destroy_group failure is logged + swallowed (run completion intact).
    """

    async def test_destroy_group_called_on_natural_drain(self) -> None:
        """Observer drained all pending → destroy_group called once with
        the exact stream_key + consumer_group the orchestrator subscribed."""
        publisher = AsyncMock()
        env = _result_ready_env(
            ResultReadyOutcome.SUCCESS, child_session_id="c1",
        )
        subscriber = _FakeSubscriber([env], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1"],
            child_session_ids={"wu1": "c1"},
            cancel_event=cancel_event, timeout_seconds=2.0,
        )
        assert len(subscriber.destroy_group_calls) == 1
        assert subscriber.destroy_group_calls[0] == {
            "stream_key": "actus:child:root1:mailbox",
            "consumer_group": "coordinator:r1",
        }

    async def test_destroy_group_called_on_parent_cancel_exit(self) -> None:
        """Parent-cancel watcher wins → destroy_group still fires in finally."""
        publisher = AsyncMock()
        subscriber = _FakeSubscriber([], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()

        async def fire_cancel() -> None:
            await asyncio.sleep(0.02)
            cancel_event.set()

        asyncio.create_task(fire_cancel())
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        assert len(subscriber.destroy_group_calls) == 1
        assert subscriber.destroy_group_calls[0] == {
            "stream_key": "actus:child:root1:mailbox",
            "consumer_group": "coordinator:r1",
        }

    async def test_destroy_group_called_on_timeout(self) -> None:
        """asyncio.wait timeout path also reaches the finally block."""
        publisher = AsyncMock()
        subscriber = _FakeSubscriber([], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()  # never set
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1"],
            child_session_ids={"wu1": "c1"},
            cancel_event=cancel_event, timeout_seconds=0.05,
        )
        assert len(subscriber.destroy_group_calls) == 1
        assert subscriber.destroy_group_calls[0] == {
            "stream_key": "actus:child:root1:mailbox",
            "consumer_group": "coordinator:r1",
        }

    async def test_destroy_group_skipped_when_subscribe_failed(self) -> None:
        """If subscribe failed (subscribed=False), no consumer group was
        ever created — destroy_group MUST NOT be called (otherwise we'd
        emit a NOGROUP roundtrip on every failed-subscribe run).
        """
        publisher = AsyncMock()
        subscriber = _FailingSubscriber([], exhaust_then_return=False)
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        cancel_event.set()
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1"],
            child_session_ids={"wu1": "c1"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        # subscribe was attempted (and failed); destroy_group MUST be skipped.
        assert subscriber.subscribe_calls != []
        assert subscriber.destroy_group_calls == []

    async def test_destroy_group_failure_logged_not_raised(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If destroy_group raises (Redis connectivity hiccup, etc.), the
        orchestrator MUST log a warning + continue — the run completion
        path cannot be broken by a best-effort cleanup error.
        """
        import logging
        caplog.set_level(
            logging.WARNING,
            logger="app.application.services.coordinator_run_orchestrator",
        )

        publisher = AsyncMock()
        env = _result_ready_env(
            ResultReadyOutcome.SUCCESS, child_session_id="c1",
        )
        subscriber = _FakeSubscriber([env], exhaust_then_return=False)
        subscriber.destroy_group_error = RuntimeError(
            "simulated Redis WRONGTYPE on xgroup_destroy",
        )
        orch = CoordinatorRunOrchestrator(
            publisher=publisher,
            envelope_factory=CoordinatorEnvelopeFactory(),
            mailbox_subscriber=subscriber,
            parent_session_id="p1", coordinator_run_id="r1",
        )
        cancel_event = asyncio.Event()
        # run() MUST NOT raise.
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1"],
            child_session_ids={"wu1": "c1"},
            cancel_event=cancel_event, timeout_seconds=1.0,
        )
        # destroy_group was attempted exactly once.
        assert len(subscriber.destroy_group_calls) == 1
        # A WARNING with the diagnostic message was emitted with exc_info.
        warn_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "destroy_group failed" in r.message
        ]
        assert warn_records, (
            "expected at least one WARNING 'destroy_group failed' log; got: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )
        assert warn_records[0].exc_info is not None, (
            "destroy_group failure log must include exc_info for traceback"
        )
