"""C2 PR-3 §6/§7 — CoordinatorEnvelopeFactory unit tests."""
from __future__ import annotations

from app.application.services.coordinator_envelope_factory import (
    CoordinatorEnvelopeFactory,
)
from app.domain.models.mailbox_envelope import (
    CancelAckPayload,
    CancelPolicy,
    CoordinatorBudgetSnapshot,
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyOutcome,
    ResultReadyPayload,
)


class TestSpawnRequest:
    def test_make_spawn_request_minimal(self) -> None:
        f = CoordinatorEnvelopeFactory()
        env = f.make_spawn_request(
            parent_session_id="p1", child_session_id="c1",
            correlation_id="corr1",
            coordinator_run_id="p1:abcd1234abcd1234:a1",
            work_unit_id="abcd1234abcd1234.a1.0",
            spawn_manifest_ref="minio://m",
            spawn_manifest_sha256="sha",
        )
        assert env.type == MailboxEnvelopeType.SPAWN_REQUEST
        assert env.parent_session_id == "p1"
        assert env.child_session_id == "c1"
        assert env.producer_role == ProducerRole.PARENT_AGENT
        assert env.payload["agent_kind"] == "coordinator_step"
        assert env.payload["coordinator_context"]["coordinator_run_id"] \
            == "p1:abcd1234abcd1234:a1"
        assert env.payload["coordinator_context"]["budget"]["max_tool_calls"] == 25

    def test_make_spawn_request_with_explicit_budget(self) -> None:
        f = CoordinatorEnvelopeFactory()
        budget = CoordinatorBudgetSnapshot(
            max_tool_calls=10, max_token_cost_usd=0.1, max_wallclock_seconds=60,
        )
        env = f.make_spawn_request(
            parent_session_id="p", child_session_id="c", correlation_id="cor",
            coordinator_run_id="r", work_unit_id="w",
            spawn_manifest_ref="ref", spawn_manifest_sha256="sha",
            budget=budget, task_prompt="explore",
        )
        assert env.payload["coordinator_context"]["budget"]["max_tool_calls"] == 10
        assert env.payload["task_prompt"] == "explore"

    def test_unique_envelope_ids(self) -> None:
        f = CoordinatorEnvelopeFactory()
        a = f.make_spawn_request(
            parent_session_id="p", child_session_id="c", correlation_id="x",
            coordinator_run_id="r", work_unit_id="w",
            spawn_manifest_ref="m", spawn_manifest_sha256="s",
        )
        b = f.make_spawn_request(
            parent_session_id="p", child_session_id="c", correlation_id="x",
            coordinator_run_id="r", work_unit_id="w",
            spawn_manifest_ref="m", spawn_manifest_sha256="s",
        )
        assert a.envelope_id != b.envelope_id


class TestCancelRequest:
    def test_make_cancel_request_default_policy(self) -> None:
        f = CoordinatorEnvelopeFactory()
        env = f.make_cancel_request(
            parent_session_id="p1", child_session_id="c1",
            correlation_id="run-r1",
            reason="parent_cancel",
        )
        assert env.type == MailboxEnvelopeType.CANCEL_REQUEST
        assert env.producer_role == ProducerRole.PARENT_AGENT
        assert env.payload["reason"] == "parent_cancel"
        assert env.payload["policy"] == CancelPolicy.REQUEST_CANCEL.value

    def test_make_cancel_request_terminate_policy(self) -> None:
        f = CoordinatorEnvelopeFactory()
        env = f.make_cancel_request(
            parent_session_id="p1", child_session_id="c1",
            correlation_id="r",
            reason="orphan_timeout",
            policy=CancelPolicy.TERMINATE,
        )
        assert env.payload["policy"] == CancelPolicy.TERMINATE.value


class TestResultReady:
    def test_make_result_ready_carries_child_producer_role(self) -> None:
        f = CoordinatorEnvelopeFactory()
        payload = ResultReadyPayload(
            summary="ok", outcome=ResultReadyOutcome.SUCCESS,
        )
        env = f.make_result_ready(
            parent_session_id="p1", child_session_id="c1",
            correlation_id="r", payload=payload,
        )
        assert env.type == MailboxEnvelopeType.RESULT_READY
        assert env.producer_role == ProducerRole.CHILD_AGENT
        assert env.payload["outcome"] == "success"

    def test_make_result_ready_with_timed_out(self) -> None:
        f = CoordinatorEnvelopeFactory()
        payload = ResultReadyPayload(
            summary="budget", outcome=ResultReadyOutcome.TIMED_OUT,
        )
        env = f.make_result_ready(
            parent_session_id="p", child_session_id="c",
            correlation_id="r", payload=payload,
        )
        assert env.payload["outcome"] == "timed_out"


class TestCancelAck:
    def test_make_cancel_ack(self) -> None:
        f = CoordinatorEnvelopeFactory()
        payload = CancelAckPayload(final_state="cancelled")
        env = f.make_cancel_ack(
            parent_session_id="p", child_session_id="c",
            correlation_id="r", payload=payload,
        )
        assert env.type == MailboxEnvelopeType.CANCEL_ACK
        assert env.producer_role == ProducerRole.CHILD_AGENT
        assert env.payload["final_state"] == "cancelled"
