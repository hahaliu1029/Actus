"""DefaultPermissionEngine — the concrete decision facade.

Owns ApprovalStateWriter calls (R5 CS4 single caller). Reads from
ApprovalStateReader (Stage P.1) + ConfirmationQueue + SessionStateMachine.get_mode_with_revision.
Policy lookup goes through uow.user_tool_approval_policy via the
per-call uow_factory pattern (PE-0 adds this slot to IUnitOfWork —
see C-R5-P1). Emits decision_recorder events on every stage transition.

No FastAPI / SQLAlchemy imports — domain layer constraint.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from app.domain.models.approval_grant import ApprovalDecision
from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import (
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
    ToolOutcome,
)
from app.domain.services.permission.confirmation_queue import (
    ConfirmationDetail,
    ConfirmationQueue,
)
from app.domain.services.permission.context import (
    EvaluationContext,
    PreflightResumeResult,
    ResumeSignal,
)
from app.domain.services.permission.engine import PermissionEngine
from app.domain.services.permission.errors import (
    PolicyConflict,
    SessionModeViolation,
    UnsupportedSource,
    WriterIntegrityError,
)
from app.domain.services.permission.escalation_provider import EscalationProvider
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskLevel

if TYPE_CHECKING:
    from app.domain.services.permission.sources.base import PermissionSource

# Module-level constants for session mode sets
_ALLOWED_LIVE_MODES = {SessionStatus.RUNNING, SessionStatus.WAITING}
_TAKEOVER_DENY_MODES = {SessionStatus.TAKEOVER_PENDING, SessionStatus.TAKEOVER}
_LIFECYCLE_TERMINAL_MODES = {
    SessionStatus.PENDING,
    SessionStatus.FINISHING,
    SessionStatus.COMPLETED,
    SessionStatus.TIMED_OUT,
}


class DefaultPermissionEngine(PermissionEngine):
    """Rule-pattern matching is delegated to `ApprovalStateReader.check`,
    which consults `approval_grants` only — the legacy `tool_approval_rules`
    fallback was retired in PE-4d1 (the table was dropped in PE-4d2). The
    engine holds no repository directly, keeping the domain service free of
    infra imports.

    Policy lookup is delegated to a per-call UoW via the new
    `uow.user_tool_approval_policy` slot (added to IUnitOfWork in PE-0 —
    see C-R5-P1). The engine is per-task scoped (built inside
    `AgentService._create_task` — see C-R2-P0-1); the UoW per-call
    pattern avoids capturing a stale AsyncSession.

    Stage P.2 escalation providers (SmartApprove) are registered at
    construction time.
    """

    def __init__(
        self,
        *,
        uow_factory: Any,  # per-call UoW for policy lookup (C-R5-P1)
        writer: Any,
        queue: ConfirmationQueue,
        session_machine: Any,
        reader: Any,
        escalation_registry: Mapping[str, EscalationProvider],
        # PE-1 §2.4: per-source RiskAssessment derivation. Required at PE-1;
        # default {} only to keep PE-0 unit tests that pre-date the field
        # working without surgery. Production wiring MUST pass a populated
        # mapping (validate_pe_source_registry enforces this at DI time).
        sources: Mapping[str, "PermissionSource"] | None = None,
        decision_recorder: Optional[Callable[..., None]] = None,
        confirmation_timeout_seconds: int = 300,  # P2#3: configurable deadline
    ) -> None:
        self._uow_factory = uow_factory
        self._writer = writer
        self._queue = queue
        self._ssm = session_machine
        self._reader = reader
        self._escalation_registry: Mapping[str, EscalationProvider] = escalation_registry
        # PE-1 §2.4: per-source adapters; resolved per evaluate() call.
        self._sources: Mapping[str, "PermissionSource"] = sources or {}
        self._decision_recorder = decision_recorder or (lambda *a, **kw: None)
        self._confirmation_timeout_seconds: int = confirmation_timeout_seconds
        # [C2 PR-2 §5.4] ChildScopeGate — pure stateless 4-way intersection.
        # evaluate() prologue invokes it when ctx.child_permission_context is non-None.
        from app.domain.services.permission.child_scope_gate import ChildScopeGate
        self._child_scope_gate = ChildScopeGate()

    def register_source(self, name: str, source: "PermissionSource") -> None:
        """Post-construction source injection (PE-1 §2.6).

        Required because some sources (notably SkillSource) depend on
        runtime objects (skill_tool) that are only available after the
        engine has been built. Wiring sequence:
            1. build_permission_engine(..., sources={"native": NativeSource()})
            2. construct task_runner / acquire skill_tool
            3. pe.register_source("skill", SkillSource(refresher, redis))
            4. validate_pe_source_registry(pe._sources)

        Single-write per name — repeated calls raise ValueError so we
        never silently swap evaluators mid-flight.
        """
        if not isinstance(self._sources, dict):
            # Convert frozen Mapping → dict so we can mutate. After-first-
            # evaluate mutation is the caller's concern (this method is for
            # DI wiring, not hot-path swapping); INV-1b CI static scan
            # ensures no source impl writes to writer/queue/SSM so a swap
            # between registrations cannot corrupt those.
            self._sources = dict(self._sources)
        if name in self._sources:
            raise ValueError(
                f"source '{name}' already registered "
                "(register_source is single-write per name)"
            )
        self._sources[name] = source

    # ------------------------------------------------------------------ helpers

    async def _get_policy(self, user_id: str, tool_name: str) -> Any:
        """Per-call policy lookup via UoW.user_tool_approval_policy (C-R5-P1)."""
        async with self._uow_factory() as uow:
            row = await uow.user_tool_approval_policy.get(user_id, tool_name)
            return row.policy if row else None

    def _record_decision(
        self,
        name: str,
        outcome: str,
        *,
        reason: Optional[str] = None,
        attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit a decision trace event; never raises (observability must not break correctness)."""
        try:
            self._decision_recorder(name, outcome, reason=reason, attrs=attrs or {})
        except Exception:
            return

    @staticmethod
    def _cid(call: ToolCallSpec) -> str:
        return f"{call.session_id}:{call.tool_call_id}"

    @staticmethod
    def _cid_hash(cid: Optional[str]) -> Optional[str]:
        if cid is None:
            return None
        return hashlib.sha256(cid.encode("utf-8")).hexdigest()[:16]

    def _build_attrs(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
        stage: str,
        confirmation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "tool_name": call.tool_name,
            "tool_call_id": call.tool_call_id,
            "session_id": call.session_id,
            "tool_args_hash": call.arg_digest,
            "decision_stage": stage,
            "tool_source": call.tool_source,
            "session_mode": ctx.session_mode.value,
            "confirmation_id_hash": self._cid_hash(confirmation_id),
        }

    def _build_approve_decision(
        self,
        call: ToolCallSpec,
        *,
        scope: str,
        confirmation_id: Optional[str] = None,
        source_type: str = "user_click",
        risk_level_override: Optional[str] = None,
    ) -> ApprovalDecision:
        """Build an ApprovalDecision for approve effect with full field set.

        Scope must be 'session' or 'always' — 'once' is not a persistent grant.
        source_type defaults to 'user_click' for human confirmation paths;
        pass 'smart_approve' when the decision came from SmartApprove (P2#2 fix).
        risk_level_override: when provided (e.g. from ConfirmationDetail.risk_level
        read back from the queue during commit_resume), takes precedence over
        call.risk_assessment which may be None for reconstructed ToolCallSpecs.
        """
        if scope not in ("session", "always"):
            raise ValueError(
                f"persistent grant scope must be session/always, got {scope!r}"
            )
        if scope == "session":
            sid_for_grant = call.session_id
            expires = datetime.now(timezone.utc) + timedelta(hours=24)
        else:  # always
            sid_for_grant = None
            expires = None
        actual_risk = (
            risk_level_override
            if risk_level_override is not None
            else (
                call.risk_assessment.final_level.name.lower()
                if call.risk_assessment is not None
                else "unknown"
            )
        )
        return ApprovalDecision(
            user_id=call.user_id,
            session_id=sid_for_grant,
            tool_name=call.tool_name,
            tool_source=call.tool_source,  # type: ignore[arg-type]
            arg_digest=call.arg_digest or "",
            primary_arg=call.primary_arg or "",
            dir_arg=call.dir_arg or "",
            scope=scope,  # type: ignore[arg-type]
            effect="approve",
            source_type=source_type,  # type: ignore[arg-type]
            confirmation_id=confirmation_id,
            expires_at=expires,
            risk_level=actual_risk,
        )

    def _build_deny_decision(
        self,
        call: ToolCallSpec,
        *,
        scope: str,
        confirmation_id: Optional[str] = None,
        source_type: str = "user_click",
        risk_level_override: Optional[str] = None,
    ) -> ApprovalDecision:
        """Build an ApprovalDecision for deny effect with full field set.

        Scope must be 'session' or 'always' — 'once' goes through write_audit_only.
        risk_level_override: when provided (e.g. from ConfirmationDetail.risk_level
        read back from the queue during commit_resume), takes precedence over
        call.risk_assessment which may be None for reconstructed ToolCallSpecs.
        """
        if scope not in ("session", "always"):
            raise ValueError(
                f"persistent deny scope must be session/always, got {scope!r}"
            )
        if scope == "session":
            sid_for_grant = call.session_id
            expires = datetime.now(timezone.utc) + timedelta(hours=24)
        else:
            sid_for_grant = None
            expires = None
        actual_risk = (
            risk_level_override
            if risk_level_override is not None
            else (
                call.risk_assessment.final_level.name.lower()
                if call.risk_assessment is not None
                else "unknown"
            )
        )
        return ApprovalDecision(
            user_id=call.user_id,
            session_id=sid_for_grant,
            tool_name=call.tool_name,
            tool_source=call.tool_source,  # type: ignore[arg-type]
            arg_digest=call.arg_digest or "",
            primary_arg=call.primary_arg or "",
            dir_arg=call.dir_arg or "",
            scope=scope,  # type: ignore[arg-type]
            effect="deny",
            source_type=source_type,  # type: ignore[arg-type]
            confirmation_id=confirmation_id,
            expires_at=expires,
            risk_level=actual_risk,
        )

    # ------------------------------------------------------------------ evaluate

    async def evaluate(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
    ) -> ToolOutcome:
        """Stage S / P.1 / P.2 decision facade.

        Sequence (per plan preamble + corrections C-P0-2/4/5/10, C-P1-1, C-R2-NEW-P2):
          1. Lifecycle terminal → SessionModeViolation (410)
          2. TAKEOVER mode → ephemeral Denied (no grant)
          3. Other disallowed mode → SessionModeViolation
          4. Policy resolution via _get_policy
          5. Persistent DENY policy → writer.write(deny) + Denied
          6. Stage P.1 reader.check → allow/deny/no_match
          7. Risk gate: NONE/LOW + AUTO/None → AllowSuccess
          8. Stage P.2: SmartApprove with PRE + POST mode_revision recheck (C-P0-10)
          9. Fall through → enqueue ConfirmationDetail + return Asked
        """
        # [C2 PR-2 §5.4] Prologue: child scope gate BEFORE any source loop / writer touch.
        # INV-1b/2/3 safe — gate is pure function. None child_ctx => root session => skip.
        #
        # COLD CODE (PR-2): no production caller currently sets
        # ctx.child_permission_context; PR-3 ChildAgentRunnerFactory.build is
        # responsible for plumbing it through. Until then this branch is
        # unreachable at runtime — but the contract here is what PR-3+ tests
        # will exercise. Do NOT remove this branch even though it's currently
        # dead; PR-3 needs it. See ChildScopeGate module docstring for
        # PR-3 acceptance criteria (pre-PE bypass + replay revalidation).
        cctx = ctx.child_permission_context
        if cctx is not None:
            from app.domain.services.permission.child_scope_gate import (
                ScopeDecision,
                extract_target_path,
            )
            from app.domain.services.permission.child_scope_violation import (
                ChildScopeViolation,
            )
            decision = await self._child_scope_gate.check_in_scope(call, ctx, cctx)
            if decision != ScopeDecision.IN_SCOPE:
                # Use shared helper so violation.target_path matches the path the
                # gate consulted for lease lookup (filepath canonical, path fallback).
                target_path = extract_target_path(call)
                self._record_decision(
                    "permission_engine.child_scope_violation",
                    "deny",
                    reason=decision.value,
                    attrs=self._build_attrs(call, ctx, "child_scope_prologue"),
                )
                raise ChildScopeViolation(
                    decision,
                    tool_name=call.tool_name,
                    target_path=target_path,
                )

        # 1. Lifecycle terminal modes
        if ctx.session_mode in _LIFECYCLE_TERMINAL_MODES:
            self._record_decision(
                "permission_engine.lifecycle_violation",
                "deny",
                reason="lifecycle_terminal",
                attrs=self._build_attrs(call, ctx, "session_mode_check"),
            )
            raise SessionModeViolation(
                f"tool call not allowed in mode={ctx.session_mode.value}"
            )

        # 2. TAKEOVER ephemeral deny — no grant, no audit
        if ctx.session_mode in _TAKEOVER_DENY_MODES:
            self._record_decision(
                "permission_engine.session_takeover_deny",
                "deny",
                reason="session_in_takeover",
                attrs=self._build_attrs(call, ctx, "session_mode_check"),
            )
            return Denied(
                content="会话处于接管状态，工具调用被拒",
                reason=DecisionReason(
                    type="approval_policy",
                    code="session_in_takeover",
                    message=f"session in {ctx.session_mode.value} — agent must yield",
                ),
            )

        # 3. Other disallowed modes (defensive)
        if ctx.session_mode not in _ALLOWED_LIVE_MODES:
            raise SessionModeViolation(
                f"tool call not allowed in mode={ctx.session_mode.value}"
            )

        # 4. Policy resolution (per-call UoW, C-R5-P1)
        policy = await self._get_policy(call.user_id, call.tool_name)
        self._record_decision(
            "permission_engine.policy_get",
            "policy_resolved",
            reason=getattr(policy, "name", None) or "none",
            attrs=self._build_attrs(call, ctx, "policy_get"),
        )

        # 5. Persistent DENY policy → persistent deny grant (NOT write_audit_only)
        if policy is not None and policy.name == "DENY":
            try:
                await self._writer.write(
                    self._build_deny_decision(call, scope="session")
                )
            except Exception as exc:
                raise WriterIntegrityError(str(exc)) from exc
            self._record_decision(
                "permission_engine.policy_deny",
                "deny",
                reason="policy_deny",
                attrs=self._build_attrs(call, ctx, "policy_get"),
            )
            return Denied(
                content="tool 已被用户策略禁用",
                reason=DecisionReason(
                    type="approval_policy",
                    code="policy_deny",
                    message="user policy DENY",
                ),
            )

        # 5.5. PE-1 §2.4 — source-uniform RiskAssessment derivation.
        # NativeSource is passthrough; SkillSource recomputes (Risk #1).
        # We resolve source here so that the assessment used by steps 6-9
        # is always the canonical one for this tool_source.
        source = self._sources.get(call.tool_source)
        if source is None:
            self._record_decision(
                "permission_engine.unsupported_source",
                "deny",
                reason=f"unregistered_source:{call.tool_source}",
                attrs=self._build_attrs(call, ctx, "source_dispatch"),
            )
            raise UnsupportedSource(call.tool_source)
        source_assessment = await source.assess_risk(call)
        # Override caller pre-fill (Risk #1 hard rule). Frozen dataclass →
        # dataclasses.replace creates a new instance; subsequent code reads
        # `call.risk_assessment` as the post-source value.
        call = dataclasses.replace(call, risk_assessment=source_assessment)

        # 6. Stage P.1 — ApprovalStateReader grant lookup (C-P0-2)
        verdict = await self._reader.check(
            user_id=call.user_id,
            session_id=call.session_id,
            tool_name=call.tool_name,
            arg_digest=call.arg_digest or "",
            primary_arg=call.primary_arg or "",
            dir_arg=call.dir_arg or "",
        )
        if verdict == "allow":
            # ASK policy overrides existing approve grants: the user explicitly
            # wants to be asked even if they previously approved.  Deny grants
            # remain safety-critical and must not be re-opened by ASK.
            if policy is not None and policy.name == "ASK":
                # Fall through to Asked enqueue below; skip AllowSuccess short-circuit.
                pass
            else:
                self._record_decision(
                    "permission_engine.stage_p1_grant_hit",
                    "allow",
                    reason="session_grant_hit",
                    attrs=self._build_attrs(call, ctx, "stage_p1_reader"),
                )
                return AllowSuccess(
                    content="auto-allowed by prior grant",
                    data={"via": "session_grant"},
                )
        if verdict == "deny":
            self._record_decision(
                "permission_engine.stage_p1_deny_hit",
                "deny",
                reason="prior_deny_grant",
                attrs=self._build_attrs(call, ctx, "stage_p1_reader"),
            )
            return Denied(
                content="prior deny grant present",
                reason=DecisionReason(
                    type="approval_policy",
                    code="prior_deny_grant",
                    message="user previously denied this arg_digest",
                ),
            )
        # "no_match" — fall through

        # 7. Risk gate (C-R2-NEW-P2 + C-P1-1):
        # AUTO / no-policy MUST NOT short-circuit when risk_level >= MEDIUM.
        # RiskLevel enum: NONE=0 / LOW=1 / MEDIUM=2 / HIGH=3 (no CRITICAL).
        risk_level_enum = (
            call.risk_assessment.final_level
            if call.risk_assessment is not None
            else RiskLevel.NONE
        )
        is_dangerous = risk_level_enum in (RiskLevel.MEDIUM, RiskLevel.HIGH)

        # 7a. Pure AUTO / no explicit policy + safe risk → AllowSuccess (no grant)
        if (policy is None or policy.name == "AUTO") and not is_dangerous:
            self._record_decision(
                "permission_engine.policy_auto",
                "allow",
                reason="policy_auto_low_risk",
                attrs=self._build_attrs(call, ctx, "decision_final"),
            )
            return AllowSuccess(
                content="auto-allowed by policy",
                data={
                    "via": "policy_rule",
                    "policy": (policy.name if policy else "default"),
                },
            )

        # 7b. AUTO / None + dangerous risk, OR ASK policy:
        # ASK policy means the user ALWAYS wants human confirmation — skip
        # SmartApprove entirely and go straight to enqueue (P1#1 fix).
        # Only AUTO / no-policy with dangerous risk enters Stage P.2.
        if policy is not None and policy.name == "ASK":
            # Fall through immediately to Asked enqueue below.
            provider = None
        else:
            provider = self._escalation_registry.get("smart_approve")
        if provider is not None:
            # Pre-provider recheck (C-P0-10): catch TAKEOVER race that happened
            # between ctx snapshot and entry into Stage P.2.
            # P1#2: also check the actual mode, not just revision.  Some
            # update_status paths (e.g. db_session_repository.update_status used
            # for simple RUNNING→TAKEOVER transitions without a full INV-4 hard
            # migration) do not always bump mode_revision, so revision equality
            # alone is not sufficient to detect TAKEOVER/terminal transitions.
            mode_pre, rev_pre = await self._ssm.get_mode_with_revision(call.session_id)
            if rev_pre != ctx.session_mode_revision:
                raise PolicyConflict("session_mode_changed_during_evaluate")
            if mode_pre in _TAKEOVER_DENY_MODES or mode_pre in _LIFECYCLE_TERMINAL_MODES:
                raise PolicyConflict("session_mode_changed_during_evaluate")

            smart_outcome = await provider.resolve(call, ctx, None)

            # Post-provider recheck: catch TAKEOVER race during LLM call.
            # P1#2: same fix — check both revision and mode after the LLM call.
            mode_post, rev_post = await self._ssm.get_mode_with_revision(call.session_id)
            if rev_post != rev_pre:
                raise PolicyConflict("session_mode_changed_during_evaluate")
            if mode_post in _TAKEOVER_DENY_MODES or mode_post in _LIFECYCLE_TERMINAL_MODES:
                raise PolicyConflict("session_mode_changed_during_evaluate")

            if isinstance(smart_outcome, AllowSuccess):
                try:
                    await self._writer.write(
                        self._build_approve_decision(
                            call, scope="session", source_type="smart_approve"
                        )
                    )
                except Exception as exc:
                    raise WriterIntegrityError(str(exc)) from exc
                return smart_outcome

            if isinstance(smart_outcome, Denied):
                # SmartApprove LLM explicitly said deny.
                # Round 38 P2: do NOT persist as session-scoped grant —
                # ApprovalStateReader.check intentionally does NOT surface
                # session_deny grants (see approval_state_reader.py:104 "Priority 4:
                # session_deny 不 surface — design doc §Open Questions 1"). Writing
                # one is therefore a no-op for future invocations and just wastes
                # DB/audit storage. Record the decision for the audit trail and
                # return Denied. Future calls with the same args will re-trigger
                # SmartApprove (intentional: LLM may update its decision based on
                # new context).
                self._record_decision(
                    "permission_engine.smart_approve_deny",
                    "deny",
                    reason="smart_approve_llm_denied",
                    attrs=self._build_attrs(call, ctx, "stage_p2_smart_deny"),
                )
                return smart_outcome

            # smart_outcome is Asked — fall through to enqueue

        # 8. Asked — enqueue ConfirmationDetail (dedupe via queue.read)
        cid = self._cid(call)
        existing_pending = await self._queue.read(call.session_id, call.tool_call_id)
        if existing_pending is None:
            risk_level_str = (
                call.risk_assessment.final_level.name.lower()
                if call.risk_assessment is not None
                else "unknown"
            )
            matched_patterns = (
                list(call.risk_assessment.matched_patterns)
                if call.risk_assessment is not None
                else []
            )
            await self._queue.store(
                ConfirmationDetail(
                    session_id=call.session_id,
                    tool_call_id=call.tool_call_id,
                    user_id=call.user_id,
                    tool_name=call.tool_name,
                    tool_args=dict(call.tool_args),
                    risk_level=risk_level_str,
                    arg_digest=call.arg_digest or "",
                    primary_arg=call.primary_arg or "",
                    dir_arg=call.dir_arg,
                    matched_patterns=matched_patterns,
                    deadline_ts=datetime.now(timezone.utc).timestamp() + self._confirmation_timeout_seconds,
                )
            )
        self._record_decision(
            "permission_engine.policy_ask_enqueued",
            "ask",
            reason="ask_policy_user_confirmation_required",
            attrs=self._build_attrs(call, ctx, "decision_final", confirmation_id=cid),
        )
        # PE-1 §2.4 step 9: per-source reason.type dispatch.
        # Skill source historically uses risk_enforce (matches the legacy
        # react_graph.py:2243-2395 R3 path) with the LLM/risk_assessor's
        # risk_reason as the user-facing message. Native uses approval_policy
        # with the canned "user confirmation required" message (preserves
        # the legacy native confirmation UX).
        if call.tool_source == "skill":
            reason_type = "risk_enforce"
            reason_message = (
                (call.risk_assessment.risk_reason if call.risk_assessment is not None else None)
                or "user confirmation required"
            )
        else:
            reason_type = "approval_policy"
            reason_message = "user confirmation required"
        reason_code = (
            f"ask:{call.risk_assessment.final_level.name.lower()}"
            if call.risk_assessment is not None
            else "ask:once"
        )
        return Asked(
            content=f"Confirm {call.tool_name}",
            reason=DecisionReason(
                type=reason_type,
                code=reason_code,
                message=reason_message,
            ),
            confirmation_id=cid,
        )

    # ------------------------------------------------------------------ preflight_resume

    async def preflight_resume(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
        resume_signal: ResumeSignal,
    ) -> PreflightResumeResult:
        """Claim-only entry point: atomically marks the pending ConfirmationQueue
        entry as 'processing' with a 16-byte hex claim_nonce.

        Does NOT write any grant. Returns PreflightResumeResult with the
        claim_nonce + processing_started_at + detail for commit_resume to
        validate ownership.

        Raises:
            PolicyConflict("no_pending_confirmation") — nothing in queue.
            PolicyConflict("arg_digest_mismatch") — ToolCallSpec drifted.
            PolicyConflict("approval_already_claimed") — CAS lost (another worker).
        """
        detail = await self._queue.read(call.session_id, call.tool_call_id)
        if detail is None:
            self._record_decision(
                "permission_engine.preflight_no_pending",
                "deny",
                reason="no_pending_confirmation",
                attrs=self._build_attrs(call, ctx, "preflight"),
            )
            raise PolicyConflict("no_pending_confirmation")

        if (call.arg_digest or "") != (detail.arg_digest or ""):
            self._record_decision(
                "permission_engine.preflight_arg_mismatch",
                "deny",
                reason="arg_digest_mismatch",
                attrs=self._build_attrs(call, ctx, "preflight"),
            )
            raise PolicyConflict("arg_digest_mismatch")

        # Round 36 P1#1: capture the SSM mode_revision AT preflight time so
        # commit_resume can detect a RUNNING(X) -> TAKEOVER(Y) -> RUNNING(Z)
        # round-trip that happened between preflight and the EvaluationContext
        # read used by the existing round-34 check (ctx carries the post-
        # interrupt revision, which already equals the SSM value by the time
        # commit_resume runs — invisible to the round-34 comparison).
        # Best-effort: SSM failure here is not fatal; commit_resume still
        # enforces the ctx-based check from round 34 and the mode guards from
        # P1#3, so the missing field only weakens the race window detection.
        preflight_mode_rev: int | None = None
        try:
            _, preflight_mode_rev = await self._ssm.get_mode_with_revision(
                call.session_id
            )
        except Exception:
            self._record_decision(
                "permission_engine.preflight_ssm_unavailable",
                "warn",
                reason="ssm_get_mode_failed",
                attrs=self._build_attrs(call, ctx, "preflight"),
            )

        claim_nonce = secrets.token_hex(16)
        started = datetime.now(timezone.utc)
        ok = await self._queue.mark_processing_if_pending(
            call.session_id,
            call.tool_call_id,
            claim_nonce=claim_nonce,
            processing_started_at=started,
        )
        if not ok:
            self._record_decision(
                "permission_engine.preflight_already_claimed",
                "deny",
                reason="approval_already_claimed",
                attrs=self._build_attrs(call, ctx, "preflight"),
            )
            raise PolicyConflict("approval_already_claimed")

        # Round 36 P1#1: persist the captured revision under the winning claim.
        # Done AFTER the CAS so a losing concurrent preflight never overwrites
        # the winner's session_mode_revision field.
        if preflight_mode_rev is not None:
            try:
                await self._queue.set_preflight_mode_revision(
                    call.session_id,
                    call.tool_call_id,
                    preflight_mode_rev,
                )
            except Exception:
                # Best-effort: if persisting the rev fails, commit_resume falls
                # back to the round-34 ctx-based comparison + mode guards.  We
                # intentionally do NOT roll back the claim — the user-facing
                # confirmation is already in flight and the worst case is a
                # weaker race-detection guard, not a correctness failure.
                self._record_decision(
                    "permission_engine.preflight_rev_persist_failed",
                    "warn",
                    reason="redis_hset_failed",
                    attrs=self._build_attrs(call, ctx, "preflight"),
                )

        # Reflect the persisted value on the returned detail so callers / tests
        # observe the round-trip without needing a fresh queue.read.  Cheap
        # mutation (dataclass is not frozen).
        if preflight_mode_rev is not None:
            detail.session_mode_revision_at_preflight = preflight_mode_rev

        self._record_decision(
            "permission_engine.preflight_claimed",
            "ask",
            reason="claim_acquired",
            attrs=self._build_attrs(
                call, ctx, "preflight", confirmation_id=resume_signal.confirmation_id
            ),
        )
        return PreflightResumeResult(
            claim_nonce=claim_nonce,
            processing_started_at=started,
            detail=detail,
        )

    # ------------------------------------------------------------------ commit_resume

    async def commit_resume(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
        resume_signal: ResumeSignal,
        claim_nonce: str,
    ) -> ToolOutcome:
        """Graph-layer entry: validates claim nonce, writes grant (or audit-only
        for once-deny), cleans up queue, returns final ToolOutcome.

        Critical paths (per plan + corrections C-P1-3/4/5):
          1. No queue entry            → PolicyConflict("no_pending_confirmation")
          2. Nonce mismatch            → PolicyConflict("claim_nonce_mismatch")
          3. arg_digest mismatch       → PolicyConflict("arg_digest_mismatch")  [C-P1-5]
          4. approve + scope="once"    → no grant, AllowSuccess, cleanup  [C-P1-3]
          5. approve + scope=session/always → writer.write(approve) + AllowSuccess + cleanup
          6. deny + scope="once"       → write_audit_only (NEVER writer.write)
          7. deny + scope=session/always → writer.write(deny) (NOT write_audit_only)
          All writer calls wrapped in try/except → WriterIntegrityError.
        """
        detail = await self._queue.read(call.session_id, call.tool_call_id)
        if detail is None:
            raise PolicyConflict("no_pending_confirmation")

        if detail.claim_nonce != claim_nonce:
            self._record_decision(
                "permission_engine.commit_nonce_mismatch",
                "deny",
                reason="claim_nonce_mismatch",
                attrs=self._build_attrs(
                    call, ctx, "commit_resume",
                    confirmation_id=resume_signal.confirmation_id,
                ),
            )
            raise PolicyConflict("claim_nonce_mismatch")

        # C-P1-5: recheck arg_digest after nonce check, before acting on action
        if (call.arg_digest or "") != (detail.arg_digest or ""):
            self._record_decision(
                "permission_engine.commit_arg_mismatch",
                "deny",
                reason="arg_digest_mismatch",
                attrs=self._build_attrs(call, ctx, "commit_resume"),
            )
            raise PolicyConflict("arg_digest_mismatch")

        # P1#3: Re-validate session mode before writing any grants/audit.
        # The session may have transitioned to TAKEOVER or a terminal state
        # after preflight_resume succeeded but before commit_resume is reached.
        # This re-check mirrors the evaluate() mode guards to prevent replay
        # in a session that is no longer in a live execution mode.
        current_mode, current_rev = await self._ssm.get_mode_with_revision(call.session_id)

        # Round 36 P1#1: compare the preflight-captured revision (persisted in
        # the queue by preflight_resume) against the current SSM revision.
        # The round-34 P1#1 check below only sees ctx.session_mode_revision,
        # which is the rev the interrupt_helper read just before commit — by
        # then any race-induced bump from RUNNING(X) -> TAKEOVER(Y) -> RUNNING(Z)
        # has already settled and ctx == current both equal Z, so the round-34
        # comparison cannot fire.  The preflight-time rev (= X) catches that
        # full round-trip.  We do this BEFORE the ctx comparison so that the
        # narrower preflight-window detection wins when both are off.
        detail_preflight_rev = detail.session_mode_revision_at_preflight
        if (
            detail_preflight_rev is not None
            and current_rev != detail_preflight_rev
        ):
            await self._queue.cleanup(call.session_id, call.tool_call_id)
            self._record_decision(
                "permission_engine.commit_preflight_revision_drift",
                "deny",
                reason="session_mode_changed_during_resume",
                attrs=self._build_attrs(
                    call, ctx, "commit_resume",
                    confirmation_id=resume_signal.confirmation_id,
                ),
            )
            raise PolicyConflict("session_mode_changed_during_resume")

        # P1#1 (round 34): also compare revision to detect round-trip races.
        # ctx.session_mode_revision was captured by the HTTP boundary right
        # before calling preflight_resume → commit_resume. If revision moved
        # by the time we re-read here, a transition happened — refuse to commit
        # even if mode "looks" identical (e.g. RUNNING → TAKEOVER → RUNNING
        # round-trip where the value comparison on mode alone would miss it).
        if current_rev != ctx.session_mode_revision:
            await self._queue.cleanup(call.session_id, call.tool_call_id)
            self._record_decision(
                "permission_engine.commit_revision_drift",
                "deny",
                reason="session_mode_changed_during_commit",
                attrs=self._build_attrs(
                    call, ctx, "commit_resume",
                    confirmation_id=resume_signal.confirmation_id,
                ),
            )
            raise PolicyConflict("session_mode_changed_during_commit")

        if current_mode in _LIFECYCLE_TERMINAL_MODES:
            # P2#3: cleanup queue before raising so the confirmation item is
            # not stuck in "processing" forever (preflight already marked it
            # processing; the sweeper skips processing entries).
            await self._queue.cleanup(call.session_id, call.tool_call_id)
            self._record_decision(
                "permission_engine.commit_mode_terminal",
                "deny",
                reason="commit_session_terminal",
                attrs=self._build_attrs(call, ctx, "commit_resume",
                                        confirmation_id=resume_signal.confirmation_id),
            )
            raise SessionModeViolation(
                f"commit_resume rejected: session is in terminal mode={current_mode.value}"
            )
        if current_mode in _TAKEOVER_DENY_MODES:
            # P2#3: cleanup queue before returning Denied so the confirmation
            # item does not remain stuck in "processing" state.
            await self._queue.cleanup(call.session_id, call.tool_call_id)
            self._record_decision(
                "permission_engine.commit_mode_takeover",
                "deny",
                reason="commit_session_in_takeover",
                attrs=self._build_attrs(call, ctx, "commit_resume",
                                        confirmation_id=resume_signal.confirmation_id),
            )
            return Denied(
                content="会话已进入接管模式，工具执行被拒绝",
                reason=DecisionReason(
                    type="approval_policy",
                    code="session_in_takeover",
                    message=f"session in {current_mode.value} at commit time — agent must yield",
                ),
            )
        if current_mode not in _ALLOWED_LIVE_MODES:
            # P2#3: cleanup queue before raising for the unknown-mode branch too.
            await self._queue.cleanup(call.session_id, call.tool_call_id)
            self._record_decision(
                "permission_engine.commit_mode_unknown",
                "deny",
                reason="commit_session_disallowed_mode",
                attrs=self._build_attrs(call, ctx, "commit_resume",
                                        confirmation_id=resume_signal.confirmation_id),
            )
            raise SessionModeViolation(
                f"commit_resume rejected: session mode={current_mode.value} not allowed"
            )

        action = resume_signal.action
        scope = resume_signal.grant_scope

        if action == "approve":
            # C-P1-3: approve + scope="once" → no persistent grant written, just allow.
            # P1#1 (round-22): but DO write an audit log for traceability — parity with
            # legacy once-approve and PE's own deny-once path (which already writes
            # write_audit_only).  Without this, all approve-once decisions are invisible
            # in the approval audit trail, breaking the approval-chain audit requirement.
            if scope == "once":
                try:
                    await self._writer.write_audit_only(
                        user_id=call.user_id,
                        session_id=call.session_id,
                        tool_name=call.tool_name,
                        tool_args=dict(call.tool_args),
                        risk_level=detail.risk_level,
                        action="approve",
                        scope="once",
                        approved_by="user",
                    )
                except Exception as exc:
                    # Mirror the deny-once error path: cleanup before raising so the
                    # queue entry does not remain in a dangling 'processing' state.
                    try:
                        await self._queue.cleanup(call.session_id, call.tool_call_id)
                    except Exception as _rb_err:
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "commit_resume approve-once cleanup (after audit write failure) "
                            "failed for %s:%s: %s",
                            call.session_id, call.tool_call_id, _rb_err,
                        )
                    raise WriterIntegrityError(str(exc)) from exc
                # asyncio.shield cleanup (consistent with deny-once success path)
                await asyncio.shield(self._queue.cleanup(call.session_id, call.tool_call_id))
                self._record_decision(
                    "permission_engine.commit_approve_once",
                    "allow",
                    reason="user_approved_once",
                    attrs=self._build_attrs(
                        call, ctx, "commit_resume",
                        confirmation_id=resume_signal.confirmation_id,
                    ),
                )
                return AllowSuccess(
                    content="user_approved_once",
                    data={"via": "user_click", "scope": "once"},
                )

            # approve + scope=session/always → persistent grant (C-P1-4: pass confirmation_id)
            # P2#2: pass detail.risk_level as override so audit records use the real
            # risk level stored in the queue (call.risk_assessment is None for
            # reconstructed ToolCallSpecs in commit_resume).
            try:
                await self._writer.write(
                    self._build_approve_decision(
                        call,
                        scope=scope,
                        confirmation_id=resume_signal.confirmation_id,
                        risk_level_override=detail.risk_level,
                    )
                )
            except Exception as exc:
                # P2#2 (round-7): When the writer fails the graph has already
                # received _build_resume_error_command (pending_ask_* cleared +
                # tool call added to completed_tool_call_prefix) and moved on to
                # the next node.  Rolling back the queue to 'pending' would leave
                # a dangling re-claimable entry whose session_id/tool_call_id no
                # longer maps to a live interrupt — a subsequent /resume would
                # re-enter commit_resume for a call the graph has already passed.
                # Instead: cleanup the queue so the entry is gone and the tool
                # call failure is communicated via the ToolMessage error content
                # already written by _build_resume_error_command.  The LLM will
                # observe the error and decide whether to retry.
                # cleanup failure is logged but must not shadow the writer error.
                try:
                    await self._queue.cleanup(call.session_id, call.tool_call_id)
                except Exception as _rb_err:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "commit_resume approve cleanup (after writer failure) failed for %s:%s: %s",
                        call.session_id, call.tool_call_id, _rb_err,
                    )
                raise WriterIntegrityError(str(exc)) from exc

            # P1#1: shield cleanup so it completes even if the outer task is cancelled
            # (e.g. HTTP/SSE disconnect fires asyncio.CancelledError between write and
            # cleanup).  The grant is already persisted; cleanup must not be skipped
            # or the queue entry stays re-claimable → duplicate approval risk.
            await asyncio.shield(self._queue.cleanup(call.session_id, call.tool_call_id))
            self._record_decision(
                "permission_engine.commit_approve",
                "allow",
                reason="user_approved",
                attrs=self._build_attrs(
                    call, ctx, "commit_resume",
                    confirmation_id=resume_signal.confirmation_id,
                ),
            )
            return AllowSuccess(
                content="user_approved",
                data={"via": "user_click", "scope": scope},
            )

        # action == "deny"
        if scope == "once":
            # R5 CS4: write_audit_only is the ONLY path allowed for scope=once.
            # writer.write for once scope would violate the single-writer contract.
            # P2#2: use detail.risk_level from the queue (call.risk_assessment is
            # None for reconstructed ToolCallSpecs in commit_resume).
            try:
                await self._writer.write_audit_only(
                    user_id=call.user_id,
                    session_id=call.session_id,
                    tool_name=call.tool_name,
                    tool_args=dict(call.tool_args),
                    risk_level=detail.risk_level,
                    action="deny",
                    scope="once",
                    approved_by="user",
                )
            except Exception as exc:
                # P2#2 (round-7): Same as approve path — cleanup, not mark_pending.
                # write_audit_only failure means the audit record was not written,
                # but the graph has already moved on (_build_resume_error_command).
                # Cleanup the queue entry so it does not remain in a dangling state.
                try:
                    await self._queue.cleanup(call.session_id, call.tool_call_id)
                except Exception as _rb_err:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "commit_resume deny-once cleanup (after writer failure) failed for %s:%s: %s",
                        call.session_id, call.tool_call_id, _rb_err,
                    )
                raise WriterIntegrityError(str(exc)) from exc
        else:
            # deny + scope=session/always → persistent deny grant (NOT write_audit_only)
            # P2#2: pass detail.risk_level as override so audit records use the real
            # risk level stored in the queue (call.risk_assessment is None for
            # reconstructed ToolCallSpecs in commit_resume).
            try:
                await self._writer.write(
                    self._build_deny_decision(
                        call,
                        scope=scope,
                        confirmation_id=resume_signal.confirmation_id,
                        risk_level_override=detail.risk_level,
                    )
                )
            except Exception as exc:
                # P2#2 (round-7): Same as approve path — cleanup, not mark_pending.
                try:
                    await self._queue.cleanup(call.session_id, call.tool_call_id)
                except Exception as _rb_err:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "commit_resume deny cleanup (after writer failure) failed for %s:%s: %s",
                        call.session_id, call.tool_call_id, _rb_err,
                    )
                raise WriterIntegrityError(str(exc)) from exc

        # P1#1: shield cleanup so it completes even if the outer task is cancelled
        # (e.g. HTTP/SSE disconnect fires asyncio.CancelledError between write and
        # cleanup).  The grant/audit is already persisted; cleanup must not be skipped
        # or the queue entry stays re-claimable → duplicate deny replay risk.
        await asyncio.shield(self._queue.cleanup(call.session_id, call.tool_call_id))
        self._record_decision(
            "permission_engine.commit_deny",
            "deny",
            reason="user_denied",
            attrs=self._build_attrs(
                call, ctx, "commit_resume",
                confirmation_id=resume_signal.confirmation_id,
            ),
        )
        return Denied(
            content="user_denied",
            reason=DecisionReason(
                type="approval_policy",
                code=f"deny:{scope}",
                message="user clicked deny",
            ),
        )

    # ------------------------------------------------------------------ cleanup_pending_confirmation

    async def cleanup_pending_confirmation(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> None:
        """Public facade over _queue.cleanup for interrupt_helper error paths.

        Called when commit_resume was never reached (e.g. SSM failure) or when
        commit_resume raised a PolicyConflict that exited early without cleanup
        (claim_nonce_mismatch / arg_digest_mismatch branches). Swallows all
        exceptions so cleanup failures do not shadow the original error.
        """
        try:
            await self._queue.cleanup(session_id, tool_call_id)
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).exception(
                "cleanup_pending_confirmation: queue.cleanup failed for %s:%s",
                session_id, tool_call_id,
            )
