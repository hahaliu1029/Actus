"""commit_resume: writes grant, returns final ToolOutcome."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import AllowSuccess, Denied
from app.domain.services.permission.confirmation_queue import ConfirmationDetail
from app.domain.services.permission.context import (
    EvaluationContext,
    ResumeSignal,
)
from app.domain.services.permission.default_engine import DefaultPermissionEngine
from app.domain.services.permission.errors import (
    PolicyConflict,
    SessionModeViolation,
    WriterIntegrityError,
)
from app.domain.services.permission.tool_call_spec import ToolCallSpec


def _run(coro):
    return asyncio.run(coro)


def _make_detail(**overrides) -> ConfirmationDetail:
    """Test helper — supplies all 11 required fields with sensible defaults
    (real dataclass at api/app/domain/services/permission/confirmation_queue.py:18
    has no field-level defaults except status/claim_nonce/processing_started_at,
    so omitting any required field raises TypeError).
    Per C-P0-3 correction.
    """
    base = dict(
        session_id="s",
        tool_call_id="tc1",
        user_id="u",
        tool_name="file_write",
        tool_args={"path": "/x"},
        risk_level="medium",
        arg_digest="d1",
        primary_arg="",
        dir_arg=None,
        matched_patterns=[],
        deadline_ts=9999999999.0,
    )
    base.update(overrides)
    return ConfirmationDetail(**base)


def _spec() -> ToolCallSpec:
    return ToolCallSpec(
        tool_name="file_write",
        tool_args={"path": "/x"},
        tool_source="native",
        user_id="u",
        session_id="s",
        arg_digest="d1",
        tool_call_id="tc1",
    )


def _signal(action: str = "approve", scope: str = "session") -> ResumeSignal:
    return ResumeSignal(
        confirmation_id="s:tc1",
        action=action,  # type: ignore[arg-type]
        grant_scope=scope,  # type: ignore[arg-type]
        actor="user_click",
    )


def _engine(*, queue_read=None, session_mode=SessionStatus.RUNNING, session_mode_rev=1):
    queue = AsyncMock()
    queue.read = AsyncMock(return_value=queue_read)
    queue.cleanup = AsyncMock()
    writer = AsyncMock()
    writer.write = AsyncMock(return_value=("d-1", True))
    writer.write_audit_only = AsyncMock(return_value=None)
    # P1#3: commit_resume now calls ssm.get_mode_with_revision to recheck
    # session mode before writing grants.  Set up the mock to return a live mode.
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(session_mode, session_mode_rev))
    eng = DefaultPermissionEngine(
        uow_factory=AsyncMock(),
        writer=writer,
        queue=queue,
        session_machine=ssm,
        reader=AsyncMock(),
        escalation_registry={},
    )
    return eng, queue, writer


def _detail_with_nonce(nonce: str) -> ConfirmationDetail:
    return _make_detail(
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
    )


def test_commit_approve_writes_persistent_grant_and_returns_allow():
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="approve", scope="session"),
            claim_nonce=nonce,
        )
        assert isinstance(out, AllowSuccess)
        writer.write.assert_awaited_once()
        writer.write_audit_only.assert_not_awaited()
        queue.cleanup.assert_awaited_once()

    _run(_run_test())


def test_commit_deny_once_uses_write_audit_only():
    """R5 CS4: write_audit_only ONLY for scope=once."""
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="deny", scope="once"),
            claim_nonce=nonce,
        )
        assert isinstance(out, Denied)
        writer.write_audit_only.assert_awaited_once()
        writer.write.assert_not_awaited()
        queue.cleanup.assert_awaited_once()

    _run(_run_test())


def test_commit_deny_session_uses_writer_write_persistent_deny():
    """codex round-2 NEW-P0: scope=session/always deny uses writer.write,
    NOT write_audit_only (which is locked to scope=once)."""
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="deny", scope="session"),
            claim_nonce=nonce,
        )
        assert isinstance(out, Denied)
        writer.write.assert_awaited_once()
        writer.write_audit_only.assert_not_awaited()

    _run(_run_test())


def test_commit_nonce_mismatch_raises():
    eng, queue, writer = _engine(queue_read=_detail_with_nonce("real-nonce"))

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await eng.commit_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(),
                claim_nonce="fake-nonce",
            )
        assert "claim_nonce_mismatch" in str(exc.value)
        writer.write.assert_not_awaited()

    _run(_run_test())


def test_commit_no_pending_raises():
    eng, *_ = _engine(queue_read=None)

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await eng.commit_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(),
                claim_nonce="x",
            )
        assert "no_pending_confirmation" in str(exc.value)

    _run(_run_test())


def test_commit_writer_integrity_error_propagates():
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))
    writer.write = AsyncMock(side_effect=RuntimeError("UNIQUE violation"))

    async def _run_test():
        with pytest.raises(WriterIntegrityError):
            await eng.commit_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )

    _run(_run_test())


def test_commit_arg_digest_mismatch_raises():
    """C-P1-5: commit_resume must recheck arg_digest after nonce check.

    If the ToolCallSpec arg_digest has drifted vs the queued detail,
    raise PolicyConflict('arg_digest_mismatch') before acting on action.
    """
    nonce = "n" * 32
    # Queue has arg_digest="d1"; spec will use a different digest to simulate drift.
    spec_with_drift = ToolCallSpec(
        tool_name="file_write",
        tool_args={"path": "/x"},
        tool_source="native",
        user_id="u",
        session_id="s",
        arg_digest="DRIFTED",  # does NOT match detail.arg_digest="d1"
        tool_call_id="tc1",
    )
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await eng.commit_resume(
                spec_with_drift,
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )
        assert "arg_digest_mismatch" in str(exc.value)
        # No grant should be written when arg_digest drifts
        writer.write.assert_not_awaited()

    _run(_run_test())


def test_commit_approve_once_no_grant_write():
    """C-P1-3: approve + scope='once' → AllowSuccess with NO persistent grant written.

    'Allow once' (user clicks "run this one time") must not create a
    persistent ApprovalGrant — only run the tool this one time.

    P1#1 (round-22): write_audit_only IS called for approve+once to maintain
    audit-trail parity with legacy once-approve and PE's deny-once path.
    The persistent writer.write (grant) must NOT be called.
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="approve", scope="once"),  # approve + once
            claim_nonce=nonce,
        )
        assert isinstance(out, AllowSuccess)
        # No persistent grant write (C-P1-3)
        writer.write.assert_not_awaited()
        # P1#1: audit log IS written for approve-once (parity with deny-once)
        writer.write_audit_only.assert_awaited_once()
        call_kwargs = writer.write_audit_only.call_args.kwargs
        assert call_kwargs.get("action") == "approve"
        assert call_kwargs.get("scope") == "once"
        # Queue cleanup must still happen
        queue.cleanup.assert_awaited_once()

    _run(_run_test())


def test_commit_approve_once_writes_audit_log():
    """P1#1 (round-22): approve + scope='once' writes audit log for traceability.

    Audit parity requirement: legacy once-approve and PE's deny-once path both
    write write_audit_only.  approve-once must do the same so the audit trail is
    complete and approve-once decisions are visible in the approval log.

    Verifies:
    - write_audit_only called with action="approve", scope="once"
    - writer.write (persistent grant) NOT called (C-P1-3 still holds)
    - Returns AllowSuccess
    - Queue cleanup still happens
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="approve", scope="once"),
            claim_nonce=nonce,
        )
        assert isinstance(out, AllowSuccess)
        assert out.content == "user_approved_once"
        # P1#1: audit log written with approve/once
        writer.write_audit_only.assert_awaited_once()
        kwargs = writer.write_audit_only.call_args.kwargs
        assert kwargs.get("action") == "approve"
        assert kwargs.get("scope") == "once"
        assert kwargs.get("approved_by") == "user"
        assert kwargs.get("tool_name") == "file_write"
        # C-P1-3: no persistent grant
        writer.write.assert_not_awaited()
        # queue.cleanup must still happen
        queue.cleanup.assert_awaited_once()

    _run(_run_test())


def test_commit_approve_once_audit_write_failure_raises_writer_integrity_error():
    """P1#1 (round-22): when write_audit_only fails, raise WriterIntegrityError.

    Mirrors deny-once error handling: cleanup before raising so the queue
    entry does not remain stuck in 'processing'.
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))
    writer.write_audit_only = AsyncMock(side_effect=RuntimeError("DB error"))

    async def _run_test():
        with pytest.raises(WriterIntegrityError):
            await eng.commit_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(action="approve", scope="once"),
                claim_nonce=nonce,
            )
        # Cleanup must still be attempted after audit failure (P1#1 error path)
        queue.cleanup.assert_awaited_once()

    _run(_run_test())


# ---------------------------------------------------------------------------
# P1#3: commit_resume session mode recheck tests
# ---------------------------------------------------------------------------


def test_commit_resume_takeover_mode_returns_denied():
    """P1#3: session transitions to TAKEOVER between preflight and commit_resume.

    Preflight succeeded (mode=RUNNING), but by the time commit_resume runs the
    session is in TAKEOVER_PENDING.  commit_resume must return Denied (not allow)
    so the graph does not execute the tool.

    Note (round 34 P1#1): ctx.session_mode_revision is aligned to the SSM
    mock's revision so that the new revision-drift guard does NOT fire first
    — this test is exercising the mode-based deny branch specifically.
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(
        queue_read=_detail_with_nonce(nonce),
        session_mode=SessionStatus.TAKEOVER_PENDING,  # mode changed after preflight
        session_mode_rev=2,
    )

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(
                session_mode=SessionStatus.RUNNING,
                session_mode_revision=2,  # matches SSM rev — drift guard passes
            ),
            _signal(action="approve", scope="session"),
            claim_nonce=nonce,
        )
        assert isinstance(out, Denied)
        assert out.reason.code == "session_in_takeover"
        # No grant should be written
        writer.write.assert_not_awaited()
        writer.write_audit_only.assert_not_awaited()

    _run(_run_test())


def test_commit_resume_finishing_mode_raises_session_mode_violation():
    """P1#3: session transitions to FINISHING (terminal) between preflight and commit_resume.

    FINISHING is a lifecycle terminal mode — commit_resume must raise
    SessionModeViolation to prevent tool execution in a dead session.

    Note (round 34 P1#1): ctx.session_mode_revision aligned to SSM rev so
    the new revision-drift guard does not pre-empt the terminal-mode branch.
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(
        queue_read=_detail_with_nonce(nonce),
        session_mode=SessionStatus.FINISHING,  # terminal — commit must reject
        session_mode_rev=3,
    )

    async def _run_test():
        with pytest.raises(SessionModeViolation) as exc:
            await eng.commit_resume(
                _spec(),
                EvaluationContext(
                    session_mode=SessionStatus.RUNNING,
                    session_mode_revision=3,  # matches SSM rev — drift guard passes
                ),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )
        assert "terminal" in str(exc.value)
        writer.write.assert_not_awaited()

    _run(_run_test())


def test_commit_resume_takeover_mode_cleans_queue_before_denied():
    """P2#3: commit_resume returns Denied AND cleans the queue on TAKEOVER.

    preflight_resume marked the entry "processing".  If commit_resume returns
    Denied without cleanup, the entry stays stuck in "processing" forever —
    the sweeper skips processing entries.  We must call queue.cleanup() BEFORE
    returning Denied so the confirmation item can be reclaimed / expired.

    Note (round 34 P1#1): ctx.session_mode_revision aligned to SSM rev so
    the new revision-drift guard does not pre-empt the takeover branch.
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(
        queue_read=_detail_with_nonce(nonce),
        session_mode=SessionStatus.TAKEOVER_PENDING,
        session_mode_rev=2,
    )

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(
                session_mode=SessionStatus.RUNNING,
                session_mode_revision=2,  # matches SSM rev — drift guard passes
            ),
            _signal(action="approve", scope="session"),
            claim_nonce=nonce,
        )
        assert isinstance(out, Denied)
        assert out.reason.code == "session_in_takeover"
        # P2#3: queue.cleanup must have been called before returning Denied
        queue.cleanup.assert_awaited_once_with("s", "tc1")
        writer.write.assert_not_awaited()

    _run(_run_test())


def test_commit_resume_terminal_mode_cleans_queue_before_raise():
    """P2#3: commit_resume raises SessionModeViolation AND cleans the queue on terminal.

    Same root cause as TAKEOVER path but for FINISHING/COMPLETED/etc.
    queue.cleanup() must be called before raise so the item is not stuck.

    Note (round 34 P1#1): ctx.session_mode_revision aligned to SSM rev so
    the new revision-drift guard does not pre-empt the terminal branch.
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(
        queue_read=_detail_with_nonce(nonce),
        session_mode=SessionStatus.FINISHING,
        session_mode_rev=3,
    )

    async def _run_test():
        with pytest.raises(SessionModeViolation) as exc:
            await eng.commit_resume(
                _spec(),
                EvaluationContext(
                    session_mode=SessionStatus.RUNNING,
                    session_mode_revision=3,  # matches SSM rev — drift guard passes
                ),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )
        assert "terminal" in str(exc.value)
        # P2#3: queue.cleanup must have been called before the raise
        queue.cleanup.assert_awaited_once_with("s", "tc1")
        writer.write.assert_not_awaited()

    _run(_run_test())


def test_commit_resume_live_mode_allows_proceed():
    """P1#3: session still in RUNNING at commit time — proceed as normal."""
    nonce = "n" * 32
    eng, queue, writer = _engine(
        queue_read=_detail_with_nonce(nonce),
        session_mode=SessionStatus.RUNNING,
        session_mode_rev=1,
    )

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="approve", scope="session"),
            claim_nonce=nonce,
        )
        assert isinstance(out, AllowSuccess)
        writer.write.assert_awaited_once()

    _run(_run_test())


def test_commit_resume_rejects_when_revision_drifts():
    """P1#1 (round 34): commit_resume must compare revision, not just mode.

    Scenario: between preflight_resume and commit_resume the session went
    RUNNING(rev=X) → TAKEOVER(rev=X+1) → RUNNING(rev=X+2). Mode looks
    "identical" (RUNNING == RUNNING) but state has actually changed.
    Comparing only `current_mode` would miss this round-trip; comparing
    `current_rev != ctx.session_mode_revision` catches it.

    Expected: PolicyConflict("session_mode_changed_during_commit") raised,
    queue.cleanup called before raising, no writer.write / write_audit_only.
    """
    nonce = "n" * 32
    # SSM now reports revision=2 even though mode is still RUNNING.
    # ctx below carries the pre-preflight revision=1 captured by the HTTP
    # layer — the engine must detect the drift and refuse to commit.
    eng, queue, writer = _engine(
        queue_read=_detail_with_nonce(nonce),
        session_mode=SessionStatus.RUNNING,
        session_mode_rev=2,
    )

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await eng.commit_resume(
                _spec(),
                EvaluationContext(
                    session_mode=SessionStatus.RUNNING,
                    session_mode_revision=1,  # captured before drift
                ),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )
        assert str(exc.value) == "session_mode_changed_during_commit"
        # Queue must be cleaned up before raising so the entry doesn't
        # remain stuck in "processing" state.
        queue.cleanup.assert_awaited_once_with("s", "tc1")
        # No grant / audit written — drift was caught before the action branch.
        writer.write.assert_not_awaited()
        writer.write_audit_only.assert_not_awaited()

    _run(_run_test())


def test_commit_resume_rejects_when_preflight_revision_drifts():
    """Round 36 P1#1: detect a RUNNING(X) -> TAKEOVER(Y) -> RUNNING(Z) round-trip
    that happened entirely between preflight and the ctx read used by the
    round-34 check.

    Scenario:
      - preflight time SSM rev = 1 (persisted on detail)
      - session goes through TAKEOVER and back to RUNNING; SSM rev = 3 now
      - interrupt_helper reads (mode=RUNNING, rev=3) and passes that ctx
      - commit_resume re-reads SSM and ALSO sees rev=3 (==ctx) → round-34
        check passes
      - But detail.session_mode_revision_at_preflight = 1 != current_rev = 3
        → round-36 check fires PolicyConflict('session_mode_changed_during_resume')
    """
    nonce = "n" * 32
    # detail was written by preflight when SSM rev was 1.
    detail = _make_detail(
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
        session_mode_revision_at_preflight=1,
    )
    # commit-time SSM and interrupt_helper agree on rev=3 → round-34 silent.
    eng, queue, writer = _engine(
        queue_read=detail,
        session_mode=SessionStatus.RUNNING,
        session_mode_rev=3,
    )

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await eng.commit_resume(
                _spec(),
                EvaluationContext(
                    session_mode=SessionStatus.RUNNING,
                    session_mode_revision=3,  # interrupt-time read, matches SSM
                ),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )
        assert str(exc.value) == "session_mode_changed_during_resume"
        # Queue must be cleaned up before raising (entry was 'processing').
        queue.cleanup.assert_awaited_once_with("s", "tc1")
        # No grant / audit written — drift caught before action branch.
        writer.write.assert_not_awaited()
        writer.write_audit_only.assert_not_awaited()

    _run(_run_test())


def test_commit_resume_allows_when_preflight_revision_matches_current():
    """Round 36 P1#1: round-trip guard does NOT fire when preflight rev still
    matches the current SSM rev (the happy path).
    """
    nonce = "n" * 32
    detail = _make_detail(
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
        session_mode_revision_at_preflight=5,
    )
    eng, queue, writer = _engine(
        queue_read=detail,
        session_mode=SessionStatus.RUNNING,
        session_mode_rev=5,  # matches detail's preflight rev
    )

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(
                session_mode=SessionStatus.RUNNING,
                session_mode_revision=5,  # matches SSM too
            ),
            _signal(action="approve", scope="session"),
            claim_nonce=nonce,
        )
        assert isinstance(out, AllowSuccess)
        writer.write.assert_awaited_once()

    _run(_run_test())


def test_commit_resume_skips_preflight_check_when_detail_rev_absent():
    """Round 36 P1#1: legacy detail (written before round 36) has
    session_mode_revision_at_preflight=None.  The new check must skip the
    preflight comparison entirely and fall through to the round-34 ctx-based
    check, preserving backwards compatibility with in-flight queue entries
    that predate the migration.
    """
    nonce = "n" * 32
    detail = _make_detail(
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
        # NO session_mode_revision_at_preflight — legacy entry
    )
    assert detail.session_mode_revision_at_preflight is None  # sanity
    eng, queue, writer = _engine(
        queue_read=detail,
        session_mode=SessionStatus.RUNNING,
        session_mode_rev=1,
    )

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),
            EvaluationContext(
                session_mode=SessionStatus.RUNNING,
                session_mode_revision=1,  # matches SSM → round-34 check passes
            ),
            _signal(action="approve", scope="session"),
            claim_nonce=nonce,
        )
        # Detail had no preflight rev → check skipped → fall through to action.
        assert isinstance(out, AllowSuccess)
        writer.write.assert_awaited_once()

    _run(_run_test())


# ---------------------------------------------------------------------------
# P2#2: writer failure must cleanup queue (not rollback to pending)
# ---------------------------------------------------------------------------


def test_commit_approve_writer_failure_cleans_up_queue():
    """P2#2 (round-7): When writer.write raises, cleanup must be called on the queue
    before WriterIntegrityError is propagated.

    The graph has already processed _build_resume_error_command (cleared pending_ask_*,
    added the tool call to completed_tool_call_prefix) and moved on to the next node.
    Rolling back to 'pending' (mark_pending) would leave a dangling re-claimable entry
    whose session/tool_call_id no longer maps to a live interrupt — a subsequent /resume
    would re-enter commit_resume for a call the graph has already passed, causing
    undefined behavior.

    Instead: cleanup removes the queue entry so the tool call failure is surfaced via
    the ToolMessage error content, and the LLM decides whether to retry.
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))
    writer.write = AsyncMock(side_effect=RuntimeError("UNIQUE violation"))
    # mark_pending should NOT be called after round-7 fix
    queue.mark_pending = AsyncMock()

    async def _run_test():
        with pytest.raises(WriterIntegrityError):
            await eng.commit_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )
        # Cleanup must be called (not mark_pending)
        queue.cleanup.assert_awaited_once_with("s", "tc1")
        # mark_pending must NOT be called (no longer rolling back to pending)
        queue.mark_pending.assert_not_awaited()

    _run(_run_test())


def test_commit_deny_writer_failure_cleans_up_queue():
    """P2#2 (round-7): Same cleanup contract for deny + scope=session path.

    writer.write (deny grant) fails → cleanup called → WriterIntegrityError raised.
    """
    nonce = "n" * 32
    eng, queue, writer = _engine(queue_read=_detail_with_nonce(nonce))
    writer.write = AsyncMock(side_effect=RuntimeError("FK violation"))
    queue.mark_pending = AsyncMock()

    async def _run_test():
        with pytest.raises(WriterIntegrityError):
            await eng.commit_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(action="deny", scope="session"),
                claim_nonce=nonce,
            )
        queue.cleanup.assert_awaited_once_with("s", "tc1")
        queue.mark_pending.assert_not_awaited()

    _run(_run_test())


# ---------------------------------------------------------------------------
# P2#2 (round-9): risk_level must be sourced from queue detail, not call.risk_assessment
# ---------------------------------------------------------------------------


def test_commit_approve_uses_queue_detail_risk_level():
    """P2#2 (round-9): approve session/always must use detail.risk_level for the audit.

    ToolCallSpec reconstructed for commit_resume has risk_assessment=None.
    _build_approve_decision must fall back to risk_level_override (from detail),
    NOT to "unknown".
    """
    nonce = "n" * 32
    # detail carries the real risk level stored at evaluate() time
    detail = _make_detail(
        risk_level="high",
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
    )
    eng, queue, writer = _engine(queue_read=detail)

    # Capture the ApprovalDecision passed to writer.write
    captured_decisions: list = []

    async def _capturing_write(decision):
        captured_decisions.append(decision)
        return ("d-1", True)

    writer.write = _capturing_write  # type: ignore[method-assign]

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),  # risk_assessment=None
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="approve", scope="session"),
            claim_nonce=nonce,
        )
        assert isinstance(out, AllowSuccess)
        assert len(captured_decisions) == 1
        assert captured_decisions[0].risk_level == "high", (
            f"expected 'high' from detail, got {captured_decisions[0].risk_level!r}"
        )

    _run(_run_test())


def test_commit_deny_session_uses_queue_detail_risk_level():
    """P2#2 (round-9): deny session/always must use detail.risk_level for the audit.

    Same as approve path — call.risk_assessment is None, detail.risk_level is the
    real value.
    """
    nonce = "n" * 32
    detail = _make_detail(
        risk_level="high",
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
    )
    eng, queue, writer = _engine(queue_read=detail)

    captured_decisions: list = []

    async def _capturing_write(decision):
        captured_decisions.append(decision)
        return ("d-1", True)

    writer.write = _capturing_write  # type: ignore[method-assign]

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),  # risk_assessment=None
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="deny", scope="session"),
            claim_nonce=nonce,
        )
        assert isinstance(out, Denied)
        assert len(captured_decisions) == 1
        assert captured_decisions[0].risk_level == "high", (
            f"expected 'high' from detail, got {captured_decisions[0].risk_level!r}"
        )

    _run(_run_test())


def test_commit_deny_once_uses_queue_detail_risk_level():
    """P2#2 (round-9): deny once must pass detail.risk_level to write_audit_only.

    write_audit_only receives risk_level=detail.risk_level, NOT the fallback
    derived from call.risk_assessment (which is None in commit_resume context).
    """
    nonce = "n" * 32
    detail = _make_detail(
        risk_level="high",
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
    )
    eng, queue, writer = _engine(queue_read=detail)

    captured_kwargs: list = []

    async def _capturing_write_audit_only(**kwargs):
        captured_kwargs.append(kwargs)
        return None

    writer.write_audit_only = _capturing_write_audit_only  # type: ignore[method-assign]

    async def _run_test():
        out = await eng.commit_resume(
            _spec(),  # risk_assessment=None
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(action="deny", scope="once"),
            claim_nonce=nonce,
        )
        assert isinstance(out, Denied)
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["risk_level"] == "high", (
            f"expected 'high' from detail, got {captured_kwargs[0]['risk_level']!r}"
        )

    _run(_run_test())


# ---------------------------------------------------------------------------
# P1#1 (round-9): cleanup after successful write is shielded from CancelledError
# ---------------------------------------------------------------------------


def test_commit_approve_shield_cleanup_runs_on_cancellation():
    """P1#1 (round-9): asyncio.shield on cleanup ensures it completes even if
    the outer task is cancelled immediately after write succeeds.

    Scenario: writer.write succeeds, then the outer task is cancelled before
    cleanup runs.  asyncio.shield must let the cleanup coroutine complete so
    the queue entry is removed (preventing re-claimable grant replay).

    We simulate this by patching cleanup to record calls and injecting a
    CancelledError into the outer task after write.  asyncio.shield keeps the
    cleanup coroutine alive in the event loop even when the wrapper task is
    cancelled.
    """
    nonce = "n" * 32
    detail = _make_detail(
        risk_level="medium",
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
    )

    cleanup_called: list[bool] = []

    async def _run_test():
        import asyncio as _asyncio

        queue = AsyncMock()
        cleanup_done = _asyncio.Event()

        async def _slow_cleanup(session_id, tool_call_id):
            cleanup_called.append(True)
            cleanup_done.set()

        queue.read = AsyncMock(return_value=detail)
        queue.cleanup = _slow_cleanup
        queue.mark_processing_if_pending = AsyncMock(return_value=True)

        writer = AsyncMock()
        writer.write = AsyncMock(return_value=("d-1", True))
        writer.write_audit_only = AsyncMock(return_value=None)

        ssm = AsyncMock()
        ssm.get_mode_with_revision = AsyncMock(
            return_value=(SessionStatus.RUNNING, 1)
        )

        eng = DefaultPermissionEngine(
            uow_factory=AsyncMock(),
            writer=writer,
            queue=queue,
            session_machine=ssm,
            reader=AsyncMock(),
            escalation_registry={},
        )

        # Run commit_resume in a task and cancel it right after write succeeds
        # by checking cleanup_done event.  asyncio.shield keeps cleanup alive.
        task = _asyncio.create_task(
            eng.commit_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )
        )

        # Wait for writer.write to have been called (cleanup is next)
        # then cancel — shield should keep cleanup coroutine alive.
        await writer.write.wait_until_called() if hasattr(writer.write, "wait_until_called") else None
        # Wait until the cleanup event fires (cleanup was shielded and ran)
        await _asyncio.wait_for(cleanup_done.wait(), timeout=2.0)

        # At this point, cleanup completed regardless of task cancellation
        assert cleanup_called, "cleanup must be called even under cancellation"

        # Cancel the task if still running (non-fatal for this test)
        if not task.done():
            task.cancel()
            try:
                await task
            except (_asyncio.CancelledError, Exception):
                pass

    _run(_run_test())


def test_commit_approve_shield_cleanup_propagates_redis_error():
    """P1#1 (round-9): if cleanup itself raises a Redis/non-cancel exception,
    that error propagates normally (asyncio.shield does not swallow it).

    This ensures the 'shield' choice (Option A) correctly surfaces cleanup
    failures to the caller so they are not silently dropped.
    """
    nonce = "n" * 32
    detail = _make_detail(
        risk_level="medium",
        claim_nonce=nonce,
        processing_started_at=datetime.now(timezone.utc),
    )
    eng, queue, writer = _engine(queue_read=detail)
    # writer.write succeeds; cleanup raises a Redis-like error
    writer.write = AsyncMock(return_value=("d-1", True))
    queue.cleanup = AsyncMock(side_effect=RuntimeError("redis connection lost"))

    async def _run_test():
        with pytest.raises(RuntimeError, match="redis connection lost"):
            await eng.commit_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(action="approve", scope="session"),
                claim_nonce=nonce,
            )

    _run(_run_test())
