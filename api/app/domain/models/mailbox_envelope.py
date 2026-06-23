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

from app.domain.models.needs_authorization_details import NeedsAuthorizationDetails
from app.domain.models.patch_manifest import PatchManifest


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
    TIMED_OUT = "timed_out"  # C2 PR-3 §6.2 — wallclock budget / force_terminate
    NEEDS_AUTHORIZATION = "needs_authorization"  # C2 PR-3 §6.2 — child stopped on user approval gate


class ApprovalDecidedBy(str, Enum):
    USER = "user"
    AUTO_POLICY = "auto_policy"
    TIMEOUT = "timeout"


class CoordinatorBudgetSnapshot(BaseModel):
    """C2 PR-3 §6.2 — per-child budget snapshot inside SpawnRequest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_tool_calls: int
    max_token_cost_usd: float
    max_wallclock_seconds: int


class CoordinatorChildContext(BaseModel):
    """C2 PR-3 §6.2 — coordinator-specific spawn payload addendum.

    Only present when agent_kind == "coordinator_step" (enforced by
    ``SpawnRequestPayload._coordinator_context_iff_coordinator_step``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    coordinator_run_id: str
    work_unit_id: str
    parent_session_id: str
    spawn_manifest_ref: str
    spawn_manifest_sha256: str
    session_mode_revision: int
    budget: CoordinatorBudgetSnapshot


class SpawnRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # C2 PR-3 §6.2 — agent_kind widened with "coordinator_step"; defaults preserved.
    agent_kind: Literal["research", "general", "coordinator_step"] = "research"
    task_prompt: str
    parent_correlation_id: Optional[str] = None
    quota_token: Optional[str] = None
    # C2 PR-3 §6.2 — coordinator addendum; iff agent_kind == "coordinator_step".
    coordinator_context: Optional[CoordinatorChildContext] = None

    @model_validator(mode="after")
    def _coordinator_context_iff_coordinator_step(self) -> "SpawnRequestPayload":
        if self.agent_kind == "coordinator_step":
            if self.coordinator_context is None:
                raise ValueError(
                    "coordinator_context required when agent_kind=coordinator_step"
                )
        else:
            if self.coordinator_context is not None:
                raise ValueError(
                    "coordinator_context only valid for agent_kind=coordinator_step"
                )
        return self


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
    # [C2 PR-4 §6.2] coordinator child writes; tied to outcome by the
    # `_outcome_field_matrix` validator below.
    patch_manifest: Optional[PatchManifest] = None
    # [C2-full S2 §3.2 C1] MinIO ref to a full PatchManifest JSON, used when
    # the inline manifest would exceed the envelope-store 64KB whitelist
    # ceiling (shell-mode children can produce large patch sets). SUCCESS-only
    # and MUTUALLY EXCLUSIVE with inline ``patch_manifest`` — AT MOST one of
    # the two carries the write set (the model layer permits a both-None
    # SUCCESS; the write-phase "a manifest MUST be present/resolvable"
    # invariant is enforced at the graph layer — see ``_outcome_field_matrix``
    # below). worker_node / the rehydrate builder resolve the ref via
    # artifact_storage.get_bytes before the reducer runs.
    patch_manifest_ref: Optional[str] = None
    # [C2 PR-4 §6.2 + r14 P1-2] structured grievance; required when
    # outcome=NEEDS_AUTHORIZATION. Free-text rationale lives in
    # ``needs_authorization_details.proposed_write_plan.rationale_ref``
    # (MinIO), not inline, per spec §11 envelope store minimality.
    needs_authorization_details: Optional[NeedsAuthorizationDetails] = None

    @model_validator(mode="after")
    def _outcome_field_matrix(self) -> "ResultReadyPayload":
        """[C2 PR-4 r3 P2#3] outcome ↔ optional-fields matrix:

        - NEEDS_AUTHORIZATION   ⇒ needs_authorization_details MUST be set
                                  (the reducer / orchestrator routes on it)
        - non-NEEDS_AUTHORIZATION ⇒ needs_authorization_details MUST be None
                                  (carrying it on SUCCESS would let a producer
                                  smuggle an authorization request through a
                                  completion envelope, confusing the reducer)
        - patch_manifest is allowed on SUCCESS only (other outcomes have no
          meaningful write set to apply)
        - [S2 §3.2 C1] patch_manifest_ref is allowed on SUCCESS only and is
          mutually exclusive with inline patch_manifest (at most one carries
          the write set). This model layer is phase-agnostic, so it permits a
          both-None SUCCESS; the write-phase "a resolvable manifest MUST be
          present" invariant is enforced downstream at the graph layer
          (``worker_node`` / rehydrate demotion to FAILED), not here.
        """
        if self.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION:
            if self.needs_authorization_details is None:
                raise ValueError(
                    "outcome=NEEDS_AUTHORIZATION requires "
                    "needs_authorization_details"
                )
        else:
            if self.needs_authorization_details is not None:
                raise ValueError(
                    f"outcome={self.outcome.value} forbids "
                    "needs_authorization_details (only NEEDS_AUTHORIZATION "
                    "may carry it)"
                )
        if self.patch_manifest is not None and self.outcome != ResultReadyOutcome.SUCCESS:
            raise ValueError(
                f"outcome={self.outcome.value} forbids patch_manifest "
                "(only SUCCESS may carry one)"
            )
        if self.patch_manifest_ref is not None and self.outcome != ResultReadyOutcome.SUCCESS:
            raise ValueError(
                f"outcome={self.outcome.value} forbids patch_manifest_ref "
                "(only SUCCESS may carry one)"
            )
        if self.patch_manifest is not None and self.patch_manifest_ref is not None:
            raise ValueError(
                "patch_manifest and patch_manifest_ref are mutually exclusive "
                "(at most one may carry the write set)"
            )
        return self


class CancelRequestPayload(BaseModel):
    """Spec §7.6 — cancel request payload.

    Wire schema is frozen at C3 ship: ``{reason, policy}``. NO other
    fields belong here.

    codex r6 [R6-2, HIGH CONTRACT] — earlier rounds (R2-6, R3-6) added an
    optional ``destroy_reason`` to thread ``DestroyReason.ORPHAN_TIMEOUT``
    from orphan/poison cascades into the TERMINATE handler. That broke
    the frozen wire schema (external consumers, spec-conformance tests,
    type-checkers all saw a new field) and forced an R3-6 producer_role
    guard to defuse the hostile-override attack the field opened. R6-2
    restores the spec by moving the override to an in-process side-table
    on ``MailboxSupervisor`` (keyed by synthetic envelope_id, populated
    by ``_emit_cascade_terminate`` BEFORE publish, consumed/popped by
    ``CancelRequestHandler._terminate_outcome``). The side-table is
    supervisor-private; external producers cannot reach it, so the
    R3-6 guard is moot and the wire stays clean.
    """

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
