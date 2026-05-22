"""Unit tests for MailboxEnvelope wire contract (spec §4.1 + §4.2 + §15.7)."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    CancelPolicy,
    ProgressKind,
    ProgressVisibility,
    SpawnAckPayload,
    ProgressUpdatePayload,
    ResultReadyOutcome,
    CancelRequestPayload,
    ApprovalRequestPayload,
    ApprovalResponsePayload,
    SUBAGENT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
    APPROVAL_PAYLOAD_MAX_BYTES,
    MAILBOX_POISON_MAX_RECLAIM,
)


def _now():
    return datetime.now(tz=timezone.utc)


def _envelope_kwargs(**overrides):
    base = dict(
        envelope_id="01HSPYU0000000000000000001",
        type=MailboxEnvelopeType.PROGRESS_UPDATE,
        parent_session_id="parent-session-1",
        child_session_id="child-session-1",
        correlation_id="01HSPYU0000000000000000002",
        emitted_at=_now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload={"kind": "heartbeat", "visibility": "hidden"},
    )
    base.update(overrides)
    return base


def test_envelope_minimal_required_fields_accepted():
    env = MailboxEnvelope(**_envelope_kwargs())
    assert env.envelope_id == "01HSPYU0000000000000000001"
    assert env.type == MailboxEnvelopeType.PROGRESS_UPDATE
    assert env.producer_role == ProducerRole.CHILD_AGENT


def test_envelope_missing_producer_role_raises():
    kwargs = _envelope_kwargs()
    kwargs.pop("producer_role")
    with pytest.raises(ValidationError) as exc:
        MailboxEnvelope(**kwargs)
    assert "producer_role" in str(exc.value)


def test_envelope_missing_envelope_id_raises():
    kwargs = _envelope_kwargs()
    kwargs.pop("envelope_id")
    with pytest.raises(ValidationError):
        MailboxEnvelope(**kwargs)


def test_envelope_is_immutable_after_construction():
    env = MailboxEnvelope(**_envelope_kwargs())
    with pytest.raises(ValidationError):
        env.envelope_id = "different"  # pydantic v2 frozen=True


def test_cancel_policy_excludes_abandon():
    members = {p.value for p in CancelPolicy}
    assert members == {"TERMINATE", "REQUEST_CANCEL"}


def test_progress_kind_full_enumeration():
    assert {k.value for k in ProgressKind} == {
        "heartbeat", "tool_started", "tool_finished", "message", "metric",
    }


def test_producer_role_full_enumeration():
    assert {r.value for r in ProducerRole} == {
        "parent_agent", "child_agent", "supervisor",
        "supervisor_echo", "external_publisher",
    }


def test_constants_match_spec_values():
    assert SUBAGENT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS == 15
    assert APPROVAL_PAYLOAD_MAX_BYTES == 65536
    assert MAILBOX_POISON_MAX_RECLAIM == 5


def test_spawn_ack_payload_distinguishes_accept_vs_sandbox_ready():
    p = SpawnAckPayload(accepted=True, child_session_id="c1", sandbox_ready=False)
    assert p.sandbox_ready is False
    assert p.sandbox_ready_at is None


def test_progress_update_payload_requires_kind_and_visibility():
    with pytest.raises(ValidationError):
        ProgressUpdatePayload()  # type: ignore[call-arg]
    p = ProgressUpdatePayload(kind=ProgressKind.HEARTBEAT, visibility=ProgressVisibility.HIDDEN)
    assert p.kind == ProgressKind.HEARTBEAT


def test_result_ready_outcome_enumeration():
    assert {o.value for o in ResultReadyOutcome} == {"success", "failed", "cancelled"}


def test_approval_request_correlation_id_required():
    with pytest.raises(ValidationError):
        ApprovalRequestPayload(
            tool_name="t", tool_args_snapshot={}, risk_tier="low", rationale="r",
            tool_call_id="tc-1", requested_at=_now(), timeout_seconds=60,
        )  # type: ignore[call-arg]


def test_approval_response_decided_by_required():
    with pytest.raises(ValidationError):
        ApprovalResponsePayload(
            correlation_id="01HSPYU0000000000000000002", approved=False,
        )  # type: ignore[call-arg]


def test_cancel_request_payload_rejects_force_field():
    """Spec §4.2.1 + §15.1 — `force: bool` removed; only `policy: CancelPolicy`."""
    p = CancelRequestPayload(reason="user_request", policy=CancelPolicy.REQUEST_CANCEL)
    assert p.policy == CancelPolicy.REQUEST_CANCEL
    with pytest.raises(ValidationError):
        CancelRequestPayload(reason="x", policy=CancelPolicy.TERMINATE, force=True)  # type: ignore[call-arg]


def test_envelope_rejects_cancel_request_with_deprecated_force_field():
    """Spec §15.1: `force: bool` removed; only `policy: CancelPolicy`. The
    envelope-level validator must catch the deprecated field before dispatch."""
    with pytest.raises(ValidationError):
        MailboxEnvelope(**_envelope_kwargs(
            type=MailboxEnvelopeType.CANCEL_REQUEST,
            payload={"reason": "x", "policy": "TERMINATE", "force": True},
        ))


def test_cancel_request_payload_wire_schema_frozen_at_reason_policy():
    """codex r6 [R6-2, HIGH CONTRACT] — CancelRequestPayload wire schema is
    frozen at C3 ship at ``{reason, policy}``. The R2-6 ``destroy_reason``
    field was moved off the wire to a supervisor-private side-table; the
    payload model must reject any attempt to set it (``extra="forbid"``).
    """
    # Positive: schema accepts the canonical fields.
    p = CancelRequestPayload(reason="user_cancel", policy=CancelPolicy.TERMINATE)
    assert set(p.model_dump().keys()) == {"reason", "policy"}

    # Negative: destroy_reason is no longer part of the payload schema.
    with pytest.raises(ValidationError):
        CancelRequestPayload(  # type: ignore[call-arg]
            reason="orphan_timeout",
            policy=CancelPolicy.TERMINATE,
            destroy_reason="orphan_timeout",
        )

    # Envelope-level validator also rejects the extra field so a hostile
    # producer cannot publish a CANCEL_REQUEST with destroy_reason on the wire.
    with pytest.raises(ValidationError):
        MailboxEnvelope(**_envelope_kwargs(
            type=MailboxEnvelopeType.CANCEL_REQUEST,
            payload={
                "reason": "x",
                "policy": "TERMINATE",
                "destroy_reason": "orphan_timeout",
            },
        ))


def test_envelope_rejects_approval_request_missing_tool_call_id():
    """tool_call_id is required on ApprovalRequestPayload; envelope must
    surface the violation even though it stores payload as dict[str, Any]."""
    with pytest.raises(ValidationError):
        MailboxEnvelope(**_envelope_kwargs(
            type=MailboxEnvelopeType.APPROVAL_REQUEST,
            payload={
                "tool_name": "t",
                "tool_args_snapshot": {},
                "risk_tier": "low",
                "rationale": "r",
                "correlation_id": "corr-1",
                # tool_call_id missing
                "requested_at": _now().isoformat(),
                "timeout_seconds": 60,
            },
        ))


def test_envelope_rejects_progress_update_with_extra_field():
    """Typo'd / unknown payload field must fail at envelope boundary."""
    with pytest.raises(ValidationError):
        MailboxEnvelope(**_envelope_kwargs(
            type=MailboxEnvelopeType.PROGRESS_UPDATE,
            payload={
                "kind": "heartbeat",
                "visibility": "hidden",
                "unknown_typo_field": "x",
            },
        ))


def test_envelope_accepts_valid_cancel_request_payload():
    env = MailboxEnvelope(**_envelope_kwargs(
        type=MailboxEnvelopeType.CANCEL_REQUEST,
        payload={"reason": "user_request", "policy": "REQUEST_CANCEL"},
    ))
    assert env.type == MailboxEnvelopeType.CANCEL_REQUEST


def test_envelope_accepts_valid_approval_request_payload():
    env = MailboxEnvelope(**_envelope_kwargs(
        type=MailboxEnvelopeType.APPROVAL_REQUEST,
        payload={
            "tool_name": "shell_execute",
            "tool_args_snapshot": {"cmd": "ls"},
            "risk_tier": "medium",
            "rationale": "agent requested shell",
            "correlation_id": "01HSPYU0000000000000000003",
            "tool_call_id": "tc-1",
            "requested_at": _now().isoformat(),
            "timeout_seconds": 60,
        },
    ))
    assert env.type == MailboxEnvelopeType.APPROVAL_REQUEST


def test_envelope_validator_coerces_approved_string_to_bool():
    """codex round 15 P2: pydantic coerces 'false' → False; envelope must
    store the coerced value so downstream code sees a real bool.

    Without the write-back in ``_validate_payload_matches_type``,
    ``env.payload["approved"]`` would still be the truthy string ``"false"``
    even after validation succeeds — handlers would treat a deny as an
    approve.
    """
    env = MailboxEnvelope(**_envelope_kwargs(
        type=MailboxEnvelopeType.APPROVAL_RESPONSE,
        payload={
            "correlation_id": "01HSPYU0000000000000000003",
            "approved": "false",
            "decided_by": "user",
        },
    ))
    assert env.payload["approved"] is False


def test_envelope_validator_materializes_default_agent_kind():
    """codex round 15 P2: SPAWN_REQUEST.agent_kind defaults to 'research'.

    Without the write-back, ``env.payload.get("agent_kind")`` would still
    be ``None`` even though the typed payload model materialized the default.
    """
    env = MailboxEnvelope(**_envelope_kwargs(
        type=MailboxEnvelopeType.SPAWN_REQUEST,
        payload={"task_prompt": "do x"},
    ))
    assert env.payload["agent_kind"] == "research"


def test_envelope_validator_preserves_explicit_agent_kind():
    """codex round 15 P2: explicit agent_kind survives the normalization
    write-back unchanged."""
    env = MailboxEnvelope(**_envelope_kwargs(
        type=MailboxEnvelopeType.SPAWN_REQUEST,
        payload={"task_prompt": "do x", "agent_kind": "general"},
    ))
    assert env.payload["agent_kind"] == "general"
