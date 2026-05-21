"""Mailbox envelope wire contract (C3 spec §4 + C1 ADR §6.3 + §15.7).

Imported by:
  - infrastructure/external/mailbox/redis_mailbox_publisher.py (publisher)
  - application/services/mailbox_supervisor.py                 (consumer)
  - domain/services/agent_task_runner.py                        (child publishers)
  - application/services/subagent_research_service.py           (SPAWN_REQUEST)

Domain layer constraint: pydantic + stdlib only. No FastAPI/SQLAlchemy/Redis.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Final, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MailboxEnvelopeType(str, Enum):
    """C1 ADR §6.3 frozen — 10 types, no in-place renames."""

    SPAWN_REQUEST = "SPAWN_REQUEST"
    SPAWN_ACK = "SPAWN_ACK"
    PROGRESS_UPDATE = "PROGRESS_UPDATE"
    RESULT_READY = "RESULT_READY"
    APPROVAL_REQUEST = "APPROVAL_REQUEST"
    APPROVAL_RESPONSE = "APPROVAL_RESPONSE"
    CANCEL_REQUEST = "CANCEL_REQUEST"
    CANCEL_ACK = "CANCEL_ACK"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    HANDOFF_REQUEST = "HANDOFF_REQUEST"


class ProducerRole(str, Enum):
    """R2 P1.2 — required field; supervisor uses this to gate last_seen update."""

    PARENT_AGENT = "parent_agent"
    CHILD_AGENT = "child_agent"
    SUPERVISOR = "supervisor"
    SUPERVISOR_ECHO = "supervisor_echo"
    EXTERNAL_PUBLISHER = "external_publisher"


class CancelPolicy(str, Enum):
    """Spec §4.2.1 + §15.1 — ABANDON intentionally absent (reserved-rejected)."""

    TERMINATE = "TERMINATE"
    REQUEST_CANCEL = "REQUEST_CANCEL"


class ProgressKind(str, Enum):
    HEARTBEAT = "heartbeat"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    MESSAGE = "message"
    METRIC = "metric"


class ProgressVisibility(str, Enum):
    HIDDEN = "hidden"
    DEBUG = "debug"
    USER = "user"


class ResultReadyOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApprovalDecidedBy(str, Enum):
    USER = "user"
    AUTO_POLICY = "auto_policy"
    TIMEOUT = "timeout"


class SpawnRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    agent_kind: Literal["research", "general"] = "research"
    task_prompt: str
    parent_correlation_id: Optional[str] = None
    quota_token: Optional[str] = None


class SpawnAckPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accepted: bool
    child_session_id: str
    sandbox_ready: bool
    sandbox_ready_at: Optional[datetime] = None
    reason: Optional[str] = None


class ProgressUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: ProgressKind
    visibility: ProgressVisibility
    step_id: Optional[str] = None
    status: Optional[Literal["running", "blocked", "awaiting_io"]] = None
    partial_summary: Optional[str] = None
    percent: Optional[float] = None
    phase: Optional[Literal["idle", "in_tool", "finalizing"]] = None
    tool_call_id: Optional[str] = None


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_type: Literal["file", "url", "memory_chunk", "skill_ref"]
    ref: str
    description: Optional[str] = None


class CostAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_usd: float = 0.0
    tool_call_count: int = 0


class ResultReadyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    summary: str
    outcome: ResultReadyOutcome
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    cost_summary: CostAggregate = Field(default_factory=CostAggregate)


class CancelRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reason: str
    policy: CancelPolicy


class CancelAckPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    final_state: Literal["cancelled", "force_terminated", "completed"]
    summary: Optional[str] = None


class ApprovalRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tool_name: str
    tool_args_snapshot: dict[str, Any]
    risk_tier: Literal["low", "medium", "high", "critical"]
    rationale: str
    correlation_id: str
    tool_call_id: str
    requested_at: datetime
    timeout_seconds: int


class ApprovalResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    correlation_id: str
    approved: bool
    decided_by: ApprovalDecidedBy
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None


class DependencyBlockedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    blocked_on: str
    reason: str
    suggested_retry_after_seconds: Optional[int] = None


class HandoffRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    handoff_target: Literal["human", "sibling_agent", "external_workflow"]
    reason: str
    context_summary: str


_PAYLOAD_MODELS: dict[MailboxEnvelopeType, type[BaseModel]] = {
    MailboxEnvelopeType.SPAWN_REQUEST: SpawnRequestPayload,
    MailboxEnvelopeType.SPAWN_ACK: SpawnAckPayload,
    MailboxEnvelopeType.PROGRESS_UPDATE: ProgressUpdatePayload,
    MailboxEnvelopeType.RESULT_READY: ResultReadyPayload,
    MailboxEnvelopeType.CANCEL_REQUEST: CancelRequestPayload,
    MailboxEnvelopeType.CANCEL_ACK: CancelAckPayload,
    MailboxEnvelopeType.APPROVAL_REQUEST: ApprovalRequestPayload,
    MailboxEnvelopeType.APPROVAL_RESPONSE: ApprovalResponsePayload,
    MailboxEnvelopeType.DEPENDENCY_BLOCKED: DependencyBlockedPayload,
    MailboxEnvelopeType.HANDOFF_REQUEST: HandoffRequestPayload,
}


class MailboxEnvelope(BaseModel):
    """Wire contract — C1 ADR §6.3 frozen + R2 P1.2 producer_role."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    envelope_id: str
    type: MailboxEnvelopeType
    parent_session_id: str
    child_session_id: str
    correlation_id: str
    emitted_at: datetime
    producer_role: ProducerRole
    payload: dict[str, Any]
    reclaim_count: int = 0

    @model_validator(mode="after")
    def _validate_payload_matches_type(self) -> "MailboxEnvelope":
        """C3 PR-1 (codex round 11 P2 + round 15 P2): enforce typed payload schema
        AND normalize the dict back to the envelope so handlers downstream see
        coerced types (e.g. "false" → False) and materialized defaults
        (SPAWN_REQUEST.agent_kind → "research").

        Without the write-back, ``model_validate`` silently discards the parsed
        model and ``env.payload`` keeps the original on-wire shape — pydantic
        would coerce ``approved="false"`` to ``False`` during validation but
        ``env.payload["approved"]`` would still be the truthy string ``"false"``,
        and ``SPAWN_REQUEST`` without ``agent_kind`` would pass validation but
        ``env.payload.get("agent_kind")`` would still be ``None`` instead of
        the default ``"research"``.

        Frozen model — bypass via ``object.__setattr__`` to write the normalized
        dict. ``mode="python"`` keeps Python types intact (datetime stays
        datetime, enum stays enum); the persistence boundary (e.g.
        ``upsert_processing``) converts to JSONB-safe via ``to_jsonable_python``.
        """
        payload_model = _PAYLOAD_MODELS.get(self.type)
        if payload_model is None:
            raise ValueError(f"no payload schema registered for type {self.type}")
        parsed = payload_model.model_validate(self.payload)
        object.__setattr__(self, "payload", parsed.model_dump(mode="python"))
        return self


# Frozen constants — spec §4.3 + §15.4
SUBAGENT_SPAWN_ACK_TIMEOUT_SECONDS: int = 10
SUBAGENT_SPAWN_SANDBOX_READY_TIMEOUT_SECONDS: int = 60
SUBAGENT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS: int = 15
SUBAGENT_PROGRESS_STALE_AFTER_SECONDS: int = 90
SUBAGENT_RESULT_READY_TIMEOUT_SECONDS: int = 600
SUBAGENT_HANDOFF_REQUEST_TIMEOUT_SECONDS: int = 1800

CHILD_CANCEL_ACK_TIMEOUT_MS: int = 30_000
CANCEL_AUTO_ESCALATE_TO_TERMINATE: bool = True

APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS: int = 300
APPROVAL_REQUEST_TIMEOUT_MAX_SECONDS: int = 900
APPROVAL_TIMEOUT_TERMINAL_BEHAVIOR: Final[str] = "deny"
APPROVAL_PAYLOAD_MAX_BYTES: int = 65_536

MAILBOX_STREAM_MAXLEN_APPROX: int = 10_000
MAILBOX_XREADGROUP_COUNT: int = 32
MAILBOX_XREADGROUP_BLOCK_MS: int = 1_000
MAILBOX_XAUTOCLAIM_PERIODIC_INTERVAL_SECONDS: int = 30
MAILBOX_PEL_IDLE_MS_FOR_CLAIM: int = 60_000
MAILBOX_POISON_MAX_RECLAIM: int = 5

PROCESSING_STALE_THRESHOLD_SECONDS: int = 300

CHILD_TO_PARENT_TYPES: frozenset[MailboxEnvelopeType] = frozenset(
    {
        MailboxEnvelopeType.SPAWN_ACK,
        MailboxEnvelopeType.PROGRESS_UPDATE,
        MailboxEnvelopeType.RESULT_READY,
        MailboxEnvelopeType.CANCEL_ACK,
        MailboxEnvelopeType.APPROVAL_REQUEST,
        MailboxEnvelopeType.DEPENDENCY_BLOCKED,
        MailboxEnvelopeType.HANDOFF_REQUEST,
    }
)

MAILBOX_STREAM_KEY_TEMPLATE: str = "actus:child:{root_session_id}:mailbox"
MAILBOX_CONSUMER_GROUP_NAME: str = "actus:mailbox-supervisor:v1"
MAILBOX_DEDUP_KEY_TEMPLATE: str = "actus:mailbox:dedup:{parent_session_id}:{envelope_id}"
MAILBOX_DEDUP_TTL_SECONDS: int = 86_400
