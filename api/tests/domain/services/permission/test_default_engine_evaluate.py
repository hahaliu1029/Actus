"""DefaultPermissionEngine.evaluate — branch coverage."""

import asyncio
import dataclasses
from unittest.mock import AsyncMock

from app.domain.models.user_tool_approval_policy import ApprovalPolicy
from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import (
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
)
from app.domain.services.permission.context import EvaluationContext
from app.domain.services.permission.default_engine import DefaultPermissionEngine
from app.domain.services.permission.errors import (
    PolicyConflict,
    SessionModeViolation,
)
from app.domain.services.permission.sources import NativeSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel

import pytest


def _run(coro):
    return asyncio.run(coro)


def _none_ra() -> RiskAssessment:
    """RiskAssessment for a 'no risk' native call.

    PE-1 step 5.5 routes every call through a PermissionSource; NativeSource
    is a passthrough that raises if ``risk_assessment is None`` (it expects
    tool_node to have already run RiskAssessor). Pre-fill with a NONE-level
    assessment so existing PE-0 unit tests that don't care about risk still
    exercise the policy / grant / queue branches.
    """
    return RiskAssessment(
        tool_name="file_write",
        tool_args={"path": "/x"},
        static_level=RiskLevel.NONE,
        dynamic_level=RiskLevel.NONE,
        final_level=RiskLevel.NONE,
        risk_reason="safe",
        matched_patterns=[],
        suggested_alternative=None,
        primary_arg="",
        dir_arg=None,
        arg_digest="adg",
    )


def _call(tcid: str = "tc1") -> ToolCallSpec:
    return ToolCallSpec(
        tool_name="file_write",
        tool_args={"path": "/x"},
        tool_source="native",
        user_id="u1",
        session_id="s1",
        arg_digest="adg",
        tool_call_id=tcid,
        risk_assessment=_none_ra(),
    )


class _FakeUowForPE:
    """Mimics IUnitOfWork with `user_tool_approval_policy` attached
    (PE-0 added this slot to UoW — C-R5-P1)."""

    def __init__(self, policy_row):
        self.user_tool_approval_policy = AsyncMock()
        self.user_tool_approval_policy.get = AsyncMock(return_value=policy_row)
        self.commit = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


def _make_engine(*, policy="auto", grant=None, smart=None, queue_existing=None):
    if policy is None:
        policy_row = None
    else:
        # Real UserToolApprovalPolicy model has `policy: ApprovalPolicy`
        policy_row = AsyncMock()
        policy_row.policy = ApprovalPolicy[policy.upper()]

    def uow_factory():
        return _FakeUowForPE(policy_row)

    writer = AsyncMock()
    writer.write = AsyncMock(return_value=("decision-1", True))
    writer.write_audit_only = AsyncMock(return_value="audit-1")
    queue = AsyncMock()
    queue.read = AsyncMock(return_value=queue_existing)
    queue.store = AsyncMock(return_value=None)
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(
        return_value=(SessionStatus.RUNNING, 7),
    )
    reader = AsyncMock()
    # Real ApprovalStateReader.check returns "allow" | "deny" | "no_match"
    # (see C-P0-2). The grant fixture is a verdict string here, not an object.
    reader.check = AsyncMock(
        return_value=grant if isinstance(grant, str) else "no_match"
    )
    smart_provider = AsyncMock()
    smart_provider.name = "smart_approve"
    smart_provider.resolve = (
        AsyncMock(return_value=smart)
        if smart
        else AsyncMock(return_value=AllowSuccess(content="ok", data={}))
    )
    engine = DefaultPermissionEngine(
        uow_factory=uow_factory,
        writer=writer,
        queue=queue,
        session_machine=ssm,
        reader=reader,
        escalation_registry={"smart_approve": smart_provider},
        sources={"native": NativeSource()},
    )
    return engine, writer, queue, ssm, reader, smart_provider


def _ctx(mode: SessionStatus = SessionStatus.RUNNING, rev: int = 7) -> EvaluationContext:
    return EvaluationContext(session_mode=mode, session_mode_revision=rev)


def test_evaluate_auto_policy_returns_allow_no_writer_call():
    engine, writer, *_ = _make_engine(policy="auto")

    async def _run_test():
        out = await engine.evaluate(_call(), _ctx())
        assert isinstance(out, AllowSuccess)
        writer.write.assert_not_awaited()
        writer.write_audit_only.assert_not_awaited()

    _run(_run_test())


def test_evaluate_deny_policy_writes_persistent_deny_grant_not_audit_only():
    """codex round-2 NEW-P0: persistent deny MUST use writer.write(effect=DENY),
    NOT write_audit_only (write_audit_only locked to scope=once only)."""
    engine, writer, *_ = _make_engine(policy="deny")

    async def _run_test():
        out = await engine.evaluate(_call(), _ctx())
        assert isinstance(out, Denied)
        writer.write.assert_awaited_once()
        writer.write_audit_only.assert_not_awaited()

    _run(_run_test())


def test_evaluate_session_in_takeover_returns_denied_ephemeral_no_grant():
    """codex round-3 fix 16: TAKEOVER deny is ephemeral — no grant, no audit."""
    engine, writer, *_ = _make_engine(policy="auto")

    async def _run_test():
        out = await engine.evaluate(_call(), _ctx(mode=SessionStatus.TAKEOVER))
        assert isinstance(out, Denied)
        assert out.reason.code == "session_in_takeover"
        writer.write.assert_not_awaited()
        writer.write_audit_only.assert_not_awaited()

    _run(_run_test())


def test_evaluate_session_finishing_raises_410():
    engine, *_ = _make_engine(policy="auto")

    async def _run_test():
        with pytest.raises(SessionModeViolation):
            await engine.evaluate(_call(), _ctx(mode=SessionStatus.FINISHING))

    _run(_run_test())


def test_evaluate_ask_policy_stores_queue_and_returns_asked_with_confirmation_id():
    """ASK policy with no SmartApprove provider falls straight through to
    ConfirmationQueue enqueue and returns Asked with a stable confirmation_id."""
    # Build engine WITHOUT a smart_approve provider in the registry so the
    # engine skips Stage P.2 and goes straight to Asked enqueue.

    def uow_factory():
        policy_row = AsyncMock()
        policy_row.policy = ApprovalPolicy["ASK"]
        return _FakeUowForPE(policy_row)

    writer = AsyncMock()
    writer.write = AsyncMock(return_value=("decision-1", True))
    writer.write_audit_only = AsyncMock(return_value="audit-1")
    queue = AsyncMock()
    queue.read = AsyncMock(return_value=None)  # no existing pending
    queue.store = AsyncMock(return_value=None)
    reader = AsyncMock()
    reader.check = AsyncMock(return_value="no_match")
    engine = DefaultPermissionEngine(
        uow_factory=uow_factory,
        writer=writer,
        queue=queue,
        session_machine=AsyncMock(),
        reader=reader,
        escalation_registry={},  # no smart_approve provider
        sources={"native": NativeSource()},
    )

    async def _run_test():
        out = await engine.evaluate(_call(tcid="tc-ask"), _ctx())
        assert isinstance(out, Asked)
        assert out.confirmation_id == "s1:tc-ask"
        queue.store.assert_awaited_once()
        writer.write.assert_not_awaited()

    _run(_run_test())


def test_evaluate_existing_session_grant_short_circuits():
    """R-D' idempotency: existing session grant on this arg_digest -> Allow
    for AUTO/None policy.

    ApprovalStateReader.check (real interface, C-P0-2) returns
    Literal["allow", "deny", "no_match"]. The fixture passes the verdict
    string directly; _make_engine wires it onto reader.check.

    Note: ASK policy intentionally does NOT short-circuit on a prior allow grant
    — that behaviour is covered by test_evaluate_ask_policy_overrides_existing_approve_grant.
    """
    engine, writer, queue, *_ = _make_engine(policy="auto", grant="allow")

    async def _run_test():
        out = await engine.evaluate(_call(), _ctx())
        assert isinstance(out, AllowSuccess)
        queue.store.assert_not_awaited()
        writer.write.assert_not_awaited()

    _run(_run_test())


def test_evaluate_race_post_smart_detects_revision_change():
    """codex P1-4 + round-5 NEW-P1: mode_revision change during slow
    Stage P.2 raises PolicyConflict('session_mode_changed_during_evaluate').

    Uses AUTO policy + HIGH risk to reach Stage P.2 (SmartApprove path).
    ASK policy now bypasses Stage P.2 entirely (P1#1 fix), so we use AUTO here.
    """
    engine, writer, queue, ssm, reader, smart = _make_engine(policy="auto")
    # First read (pre-provider recheck) returns same rev=7;
    # Second read (post-provider) returns rev=9 — race detected.
    ssm.get_mode_with_revision = AsyncMock(
        side_effect=[
            (SessionStatus.RUNNING, 7),  # pre-provider recheck: OK
            (SessionStatus.RUNNING, 9),  # post-provider: revision moved
        ]
    )
    smart.resolve = AsyncMock(return_value=AllowSuccess(content="ok", data={}))

    async def _run_test():
        # Force the engine into Stage P.2 via HIGH risk (AUTO policy + dangerous).
        with pytest.raises(PolicyConflict) as exc:
            await engine.evaluate(_call_with_risk(RiskLevel.HIGH), _ctx(rev=7))
        assert "session_mode_changed_during_evaluate" in str(exc.value)

    _run(_run_test())


# ---------------------------------------------------------------------------
# Helpers for risk-gate tests (C-P1-1)
# ---------------------------------------------------------------------------

def _make_risk_assessment(level: RiskLevel) -> RiskAssessment:
    """Build a minimal but fully-typed RiskAssessment for a given final level."""
    return RiskAssessment(
        tool_name="file_write",
        tool_args={"path": "/x"},
        static_level=level,
        dynamic_level=RiskLevel.NONE,
        final_level=level,
        risk_reason=f"test: {level.name}",
        matched_patterns=[],
        suggested_alternative=None,
        primary_arg="/x",
        dir_arg=None,
        arg_digest="aabbccdd00112233",
    )


def _call_with_risk(level: RiskLevel) -> ToolCallSpec:
    """Return a ToolCallSpec carrying a RiskAssessment at the given level."""
    return dataclasses.replace(_call(), risk_assessment=_make_risk_assessment(level))


# ---------------------------------------------------------------------------
# Test 1 (C-P0-10): pre-provider recheck catches revision drift
# ---------------------------------------------------------------------------

def test_evaluate_race_pre_smart_detects_revision_change():
    """codex C-P0-10: pre-provider revision recheck catches transitions that
    happened between ctx snapshot and entry into Stage P.2.

    Uses AUTO policy + HIGH risk to reach Stage P.2.
    ASK policy now bypasses Stage P.2 entirely (P1#1 fix), so we use AUTO here.
    """
    engine, writer, queue, ssm, reader, smart = _make_engine(policy="auto")
    # PRE recheck returns rev=99 — already drifted from ctx.rev=7.
    # POST recheck should never be reached.
    ssm.get_mode_with_revision = AsyncMock(
        side_effect=[
            (SessionStatus.RUNNING, 99),  # PRE recheck — already drifted
            (SessionStatus.RUNNING, 99),  # POST (defensive; should not reach)
        ]
    )
    smart.resolve = AsyncMock(return_value=AllowSuccess(content="ok", data={}))

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await engine.evaluate(_call_with_risk(RiskLevel.HIGH), _ctx(rev=7))
        assert "session_mode_changed_during_evaluate" in str(exc.value)
        # Provider must NOT have been called — pre-recheck stopped execution first.
        smart.resolve.assert_not_awaited()

    _run(_run_test())


# ---------------------------------------------------------------------------
# P1#2: mode recheck catches TAKEOVER even when revision did NOT change
# ---------------------------------------------------------------------------

def test_evaluate_race_pre_smart_takeover_mode_no_revision_bump():
    """P1#2: PRE recheck sees TAKEOVER mode with same revision → PolicyConflict.

    Some update_status paths do not increment mode_revision for simple mode
    transitions.  The pre-provider recheck must inspect the mode itself, not
    only the revision, so a RUNNING→TAKEOVER transition that did not bump
    revision is still caught.

    Uses AUTO policy + HIGH risk to reach Stage P.2.
    ASK policy now bypasses Stage P.2 entirely (P1#1 fix).
    """
    engine, writer, queue, ssm, reader, smart = _make_engine(policy="auto")
    # SAME revision (7) as ctx, but mode changed to TAKEOVER — revision alone
    # would miss this transition.
    ssm.get_mode_with_revision = AsyncMock(
        side_effect=[
            (SessionStatus.TAKEOVER, 7),  # PRE recheck — mode changed, rev same
            (SessionStatus.TAKEOVER, 7),  # POST (defensive; should not reach)
        ]
    )
    smart.resolve = AsyncMock(return_value=AllowSuccess(content="ok", data={}))

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await engine.evaluate(_call_with_risk(RiskLevel.HIGH), _ctx(rev=7))
        assert "session_mode_changed_during_evaluate" in str(exc.value)
        # Provider must NOT have been called — pre-recheck stopped execution first.
        smart.resolve.assert_not_awaited()

    _run(_run_test())


def test_evaluate_race_post_smart_takeover_mode_no_revision_bump():
    """P1#2: POST recheck sees TAKEOVER mode with same revision → PolicyConflict.

    Same as above but the TAKEOVER transition happens during the SmartApprove
    LLM call (post-provider recheck path).  Revision did not change, but mode
    moved to TAKEOVER_PENDING.

    Uses AUTO policy + HIGH risk to reach Stage P.2.
    ASK policy now bypasses Stage P.2 entirely (P1#1 fix).
    """
    engine, writer, queue, ssm, reader, smart = _make_engine(policy="auto")
    ssm.get_mode_with_revision = AsyncMock(
        side_effect=[
            (SessionStatus.RUNNING, 7),       # PRE recheck: OK
            (SessionStatus.TAKEOVER_PENDING, 7),  # POST: mode changed, rev same
        ]
    )
    smart.resolve = AsyncMock(return_value=AllowSuccess(content="ok", data={}))

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await engine.evaluate(_call_with_risk(RiskLevel.HIGH), _ctx(rev=7))
        assert "session_mode_changed_during_evaluate" in str(exc.value)

    _run(_run_test())


def test_evaluate_race_pre_smart_terminal_mode_no_revision_bump():
    """P1#2: PRE recheck sees FINISHING (terminal) mode with same revision → PolicyConflict.

    Uses AUTO policy + HIGH risk to reach Stage P.2.
    ASK policy now bypasses Stage P.2 entirely (P1#1 fix).
    """
    engine, writer, queue, ssm, reader, smart = _make_engine(policy="auto")
    ssm.get_mode_with_revision = AsyncMock(
        side_effect=[
            (SessionStatus.FINISHING, 7),  # PRE recheck — terminal mode, rev same
            (SessionStatus.FINISHING, 7),  # POST (defensive)
        ]
    )
    smart.resolve = AsyncMock(return_value=AllowSuccess(content="ok", data={}))

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await engine.evaluate(_call_with_risk(RiskLevel.HIGH), _ctx(rev=7))
        assert "session_mode_changed_during_evaluate" in str(exc.value)
        smart.resolve.assert_not_awaited()

    _run(_run_test())


# ---------------------------------------------------------------------------
# Tests 2-4 (C-P1-1): risk-gate branches with AUTO / None policy
# ---------------------------------------------------------------------------

def test_evaluate_no_policy_low_risk_auto_allows():
    """C-P1-1: AUTO/None policy + LOW risk -> AllowSuccess short-circuit (no Stage P.2)."""
    engine, writer, queue, ssm, reader, smart = _make_engine(policy=None)
    call = _call_with_risk(RiskLevel.LOW)

    async def _run_test():
        out = await engine.evaluate(call, _ctx())
        assert isinstance(out, AllowSuccess)
        writer.write.assert_not_awaited()
        writer.write_audit_only.assert_not_awaited()
        queue.store.assert_not_awaited()
        # Smart provider must NOT have been called — risk gate short-circuited.
        smart.resolve.assert_not_awaited()

    _run(_run_test())


def test_evaluate_no_policy_medium_risk_enqueues_asked():
    """C-P1-1: AUTO/None policy + MEDIUM risk MUST go through Stage P.2 -> Asked."""
    engine, writer, queue, ssm, reader, smart = _make_engine(policy=None)
    # SmartApprove escalates to Asked (fall-through after recheck)
    smart.resolve = AsyncMock(
        return_value=Asked(
            content="needs user",
            reason=DecisionReason(
                type="smart_approve", code="escalated", message="escalated"
            ),
            confirmation_id="s1:tc1",
        )
    )
    call = _call_with_risk(RiskLevel.MEDIUM)

    async def _run_test():
        out = await engine.evaluate(call, _ctx())
        assert isinstance(out, Asked)
        # Stage P.2 must have run (not short-circuited by risk gate).
        smart.resolve.assert_awaited_once()
        # Confirmation must have been enqueued.
        queue.store.assert_awaited_once()

    _run(_run_test())


def test_evaluate_no_policy_high_risk_enqueues_asked():
    """C-P1-1: AUTO/None policy + HIGH risk MUST go through Stage P.2 -> Asked."""
    engine, writer, queue, ssm, reader, smart = _make_engine(policy=None)
    smart.resolve = AsyncMock(
        return_value=Asked(
            content="needs user",
            reason=DecisionReason(
                type="smart_approve", code="escalated", message="escalated"
            ),
            confirmation_id="s1:tc1",
        )
    )
    call = _call_with_risk(RiskLevel.HIGH)

    async def _run_test():
        out = await engine.evaluate(call, _ctx())
        assert isinstance(out, Asked)
        # Stage P.2 must have run.
        smart.resolve.assert_awaited_once()
        # Confirmation must have been enqueued.
        queue.store.assert_awaited_once()

    _run(_run_test())


# ---------------------------------------------------------------------------
# P2#3: ConfirmationQueue deadline_ts uses configured confirmation_timeout_seconds
# ---------------------------------------------------------------------------

def test_evaluate_uses_configured_timeout_for_deadline_ts():
    """P2#3: DefaultPermissionEngine must use confirmation_timeout_seconds (not
    hardcoded 300) when computing the deadline_ts stored in ConfirmationQueue.

    We build an engine with confirmation_timeout_seconds=60, trigger an ASK
    path, and assert that queue.store was called with a deadline_ts that is
    ~60 seconds from now — not the default 300.
    """
    import time

    def uow_factory():
        policy_row = AsyncMock()
        policy_row.policy = ApprovalPolicy["ASK"]
        return _FakeUowForPE(policy_row)

    writer = AsyncMock()
    writer.write = AsyncMock(return_value=("decision-1", True))
    writer.write_audit_only = AsyncMock(return_value="audit-1")
    queue = AsyncMock()
    queue.read = AsyncMock(return_value=None)
    queue.store = AsyncMock(return_value=None)
    reader = AsyncMock()
    reader.check = AsyncMock(return_value="no_match")

    engine = DefaultPermissionEngine(
        uow_factory=uow_factory,
        writer=writer,
        queue=queue,
        session_machine=AsyncMock(),
        reader=reader,
        escalation_registry={},  # no smart_approve — straight to Asked
        sources={"native": NativeSource()},
        confirmation_timeout_seconds=60,  # non-default
    )

    before_ts = time.time()

    async def _run_test():
        out = await engine.evaluate(_call(tcid="tc-timeout"), _ctx())
        assert isinstance(out, Asked)
        queue.store.assert_awaited_once()
        # Inspect the ConfirmationDetail passed to queue.store
        store_call_args = queue.store.call_args
        detail = store_call_args[0][0]  # positional first arg
        after_ts = time.time()
        # deadline_ts should be ~60s ahead, not ~300s
        assert before_ts + 50 <= detail.deadline_ts <= after_ts + 70, (
            f"deadline_ts {detail.deadline_ts!r} is not ~60s ahead of now "
            f"(before={before_ts:.1f}, after={after_ts:.1f}). "
            "confirmation_timeout_seconds=60 was not respected (P2#3)."
        )
        # Ensure it is clearly NOT the hardcoded 300s window
        assert detail.deadline_ts < before_ts + 200, (
            "deadline_ts looks like it uses the hardcoded 300s — expected 60s (P2#3)"
        )

    _run(_run_test())


# ---------------------------------------------------------------------------
# P1#1: ASK policy must bypass SmartApprove and go directly to Asked enqueue
# ---------------------------------------------------------------------------

def test_evaluate_ask_policy_bypasses_smart_approve_directly_enqueues_asked():
    """P1#1: When user policy is ASK and smart_approve is enabled (registered),
    pe.evaluate must NOT call smart_provider.resolve — it should skip Stage P.2
    entirely and go straight to ConfirmationQueue enqueue, returning Asked.

    Regression: before fix, ASK + smart_approve_enabled could let SmartApprove
    LLM auto-approve/deny on behalf of a user who explicitly wanted confirmation.
    """
    call = _call_with_risk(RiskLevel.HIGH)  # HIGH risk ensures we would normally hit P.2

    engine, writer, queue, ssm, reader, smart = _make_engine(
        policy="ask",
        smart=Asked(
            content="escalated",
            reason=DecisionReason(type="smart_approve", code="escalated", message="x"),
            confirmation_id="s1:tc1",
        ),
    )
    # SmartApprove would return AllowSuccess if called — this is the wrong path.
    smart.resolve = AsyncMock(
        return_value=AllowSuccess(content="smart-auto-approved", data={})
    )

    async def _run_test():
        out = await engine.evaluate(call, _ctx())
        # Must return Asked (user confirmation required), NOT AllowSuccess.
        assert isinstance(out, Asked), (
            f"Expected Asked but got {type(out).__name__!r} — "
            "ASK policy must bypass SmartApprove (P1#1)"
        )
        # SmartApprove must never have been called.
        smart.resolve.assert_not_awaited()
        # ConfirmationQueue must have been asked to store.
        queue.store.assert_awaited_once()
        # No grant must have been written.
        writer.write.assert_not_awaited()

    _run(_run_test())


# ---------------------------------------------------------------------------
# round-21 P2: ASK policy must override existing approve grant (not short-circuit)
# ---------------------------------------------------------------------------

def test_evaluate_ask_policy_overrides_existing_approve_grant():
    """round-21 P2: policy=ASK + reader.check returns 'allow' (prior approve grant)
    must NOT return AllowSuccess — it must fall through to enqueue Asked.

    Regression: before fix, Stage P.1 short-circuited on verdict='allow' before
    inspecting the policy, so ASK was silently bypassed by older approve grants.
    """
    # ASK policy, reader returns "allow" (simulate a previously-approved grant)
    engine, writer, queue, *_ = _make_engine(policy="ask", grant="allow")

    async def _run_test():
        out = await engine.evaluate(_call(), _ctx())
        # Must return Asked, not AllowSuccess — ASK overrides approve grant.
        assert isinstance(out, Asked), (
            f"Expected Asked but got {type(out).__name__!r} — "
            "ASK policy must not be short-circuited by a prior approve grant (round-21 P2)"
        )
        # Confirmation must have been enqueued.
        queue.store.assert_awaited_once()
        # No grant must have been written (confirm not auto-approved).
        writer.write.assert_not_awaited()

    _run(_run_test())


def test_evaluate_ask_policy_respects_existing_deny_grant():
    """round-21 P2: policy=ASK + reader.check returns 'deny' (prior deny grant)
    must still return Denied — ASK does NOT override deny grants (safety-critical).

    Deny grants represent a safety decision and must not be weakened by ASK.
    """
    # ASK policy, reader returns "deny" (simulate a previously-denied grant)
    engine, writer, queue, *_ = _make_engine(policy="ask", grant="deny")

    async def _run_test():
        out = await engine.evaluate(_call(), _ctx())
        # Must return Denied — deny grant is respected even with ASK policy.
        assert isinstance(out, Denied), (
            f"Expected Denied but got {type(out).__name__!r} — "
            "ASK policy must still respect a prior deny grant (round-21 P2)"
        )
        assert out.reason.code == "prior_deny_grant"
        # Confirmation must NOT have been enqueued — denied before reaching ask.
        queue.store.assert_not_awaited()
        # No grant must have been written.
        writer.write.assert_not_awaited()

    _run(_run_test())


# ---------------------------------------------------------------------------
# P2#2: SmartApprove approve must write grant with source_type="smart_approve"
# ---------------------------------------------------------------------------

def test_evaluate_smart_approve_writes_grant_with_smart_approve_source():
    """P2#2: When SmartApprove approves a call, the persisted grant must carry
    source_type='smart_approve', NOT 'user_click'.

    Before fix, _build_approve_decision always hardcoded source_type='user_click',
    so the audit record was indistinguishable from a human click.  The deny branch
    already passed source_type='smart_approve'; approve must be consistent.
    """
    call = _call_with_risk(RiskLevel.HIGH)

    engine, writer, queue, ssm, reader, smart = _make_engine(policy="auto")
    # SmartApprove returns AllowSuccess to simulate LLM auto-approve.
    smart.resolve = AsyncMock(return_value=AllowSuccess(content="sa-ok", data={}))

    async def _run_test():
        out = await engine.evaluate(call, _ctx())
        assert isinstance(out, AllowSuccess)
        # writer.write must have been called exactly once.
        writer.write.assert_awaited_once()
        decision = writer.write.call_args[0][0]  # positional arg 0: ApprovalDecision
        assert decision.source_type == "smart_approve", (
            f"Expected source_type='smart_approve' but got {decision.source_type!r} — "
            "SmartApprove approve grant attribution is wrong (P2#2)"
        )

    _run(_run_test())


# ---------------------------------------------------------------------------
# Round 38 P2: SmartApprove deny must NOT persist a session-scoped deny grant
# ---------------------------------------------------------------------------

def test_evaluate_smart_approve_deny_does_not_persist_session_grant():
    """Round 38 P2: When SmartApprove returns Denied, the engine MUST NOT
    persist a session-scoped deny grant via writer.write.

    Rationale: ApprovalStateReader.check intentionally does NOT surface
    session_deny grants (see approval_state_reader.py:104 — "Priority 4:
    session_deny 不 surface").  Writing one is therefore a no-op for future
    invocations and just wastes DB/audit storage.

    The engine must instead record a decision trace event for the audit trail
    and return the Denied outcome.  Subsequent calls with the same args will
    re-trigger SmartApprove (intentional: the LLM may update its decision
    based on new context).
    """
    call = _call_with_risk(RiskLevel.HIGH)

    # Build engine with a custom decision_recorder so we can assert the
    # smart_approve_deny event was emitted.
    recorded_events: list[tuple[str, str, str | None]] = []

    def _capture_decision(name, outcome, *, reason=None, attrs=None):
        recorded_events.append((name, outcome, reason))

    def uow_factory():
        policy_row = AsyncMock()
        policy_row.policy = ApprovalPolicy["AUTO"]
        return _FakeUowForPE(policy_row)

    writer = AsyncMock()
    writer.write = AsyncMock(return_value=("decision-1", True))
    writer.write_audit_only = AsyncMock(return_value="audit-1")
    queue = AsyncMock()
    queue.read = AsyncMock(return_value=None)
    queue.store = AsyncMock(return_value=None)
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(
        return_value=(SessionStatus.RUNNING, 7),
    )
    reader = AsyncMock()
    reader.check = AsyncMock(return_value="no_match")
    smart_provider = AsyncMock()
    smart_provider.name = "smart_approve"
    smart_provider.resolve = AsyncMock(
        return_value=Denied(
            content="sa-denied",
            reason=DecisionReason(
                type="smart_approve",
                code="llm_deny",
                message="llm decided to deny",
            ),
        )
    )

    engine = DefaultPermissionEngine(
        uow_factory=uow_factory,
        writer=writer,
        queue=queue,
        session_machine=ssm,
        reader=reader,
        escalation_registry={"smart_approve": smart_provider},
        sources={"native": NativeSource()},
        decision_recorder=_capture_decision,
    )

    async def _run_test():
        out = await engine.evaluate(call, _ctx())
        # Outcome must be Denied (the LLM's decision is returned).
        assert isinstance(out, Denied), (
            f"Expected Denied but got {type(out).__name__!r}"
        )
        assert out.reason.code == "llm_deny"
        # writer.write MUST NOT have been called — no persistent session_deny
        # grant should be written (round 38 P2).
        writer.write.assert_not_awaited()
        writer.write_audit_only.assert_not_awaited()
        # Confirmation queue must not have been touched.
        queue.store.assert_not_awaited()
        # SmartApprove must have actually been called (sanity check).
        smart_provider.resolve.assert_awaited_once()
        # Decision trace must include the smart_approve_deny event.
        event_names = [name for name, *_ in recorded_events]
        assert "permission_engine.smart_approve_deny" in event_names, (
            f"Expected decision trace 'permission_engine.smart_approve_deny' "
            f"but got events: {event_names!r}"
        )
        deny_event = next(
            ev for ev in recorded_events
            if ev[0] == "permission_engine.smart_approve_deny"
        )
        # outcome="deny", reason="smart_approve_llm_denied"
        assert deny_event[1] == "deny"
        assert deny_event[2] == "smart_approve_llm_denied"

    _run(_run_test())
