"""preflight_resume: claim only, no grant write."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.domain.models.session import SessionStatus
from app.domain.services.permission.confirmation_queue import ConfirmationDetail
from app.domain.services.permission.context import (
    EvaluationContext,
    ResumeSignal,
)
from app.domain.services.permission.default_engine import DefaultPermissionEngine
from app.domain.services.permission.errors import PolicyConflict
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


def _engine(
    *,
    queue_read=None,
    claim_ok: bool = True,
    ssm_mode_rev: tuple | None = None,
    ssm_raises: bool = False,
):
    """Round 36 P1#1: ``ssm_mode_rev`` controls what the SSM returns from
    ``get_mode_with_revision`` so tests can exercise the new preflight rev
    persistence branch.  ``ssm_raises=True`` simulates a transient SSM
    error so the best-effort skip branch is covered.
    """
    queue = AsyncMock()
    queue.read = AsyncMock(return_value=queue_read)
    queue.mark_processing_if_pending = AsyncMock(return_value=claim_ok)
    queue.set_preflight_mode_revision = AsyncMock()
    writer = AsyncMock()
    ssm = AsyncMock()
    if ssm_raises:
        ssm.get_mode_with_revision = AsyncMock(
            side_effect=RuntimeError("ssm offline")
        )
    elif ssm_mode_rev is not None:
        ssm.get_mode_with_revision = AsyncMock(return_value=ssm_mode_rev)
    eng = DefaultPermissionEngine(
        uow_factory=AsyncMock(),
        writer=writer,
        queue=queue,
        session_machine=ssm,
        reader=AsyncMock(),
        escalation_registry={},
    )
    return eng, queue, writer


def test_preflight_no_pending_raises():
    eng, *_ = _engine(queue_read=None)

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await eng.preflight_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(),
            )
        assert "no_pending_confirmation" in str(exc.value)

    _run(_run_test())


def test_preflight_arg_digest_mismatch_raises():
    detail = _make_detail(arg_digest="DIFFERENT")
    eng, *_ = _engine(queue_read=detail)

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await eng.preflight_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(),
            )
        assert "arg_digest_mismatch" in str(exc.value)

    _run(_run_test())


def test_preflight_claim_loser_raises_already_claimed():
    detail = _make_detail()  # arg_digest="d1" matches _spec()
    eng, *_ = _engine(queue_read=detail, claim_ok=False)

    async def _run_test():
        with pytest.raises(PolicyConflict) as exc:
            await eng.preflight_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
                _signal(),
            )
        assert "approval_already_claimed" in str(exc.value)

    _run(_run_test())


def test_preflight_success_returns_nonce_and_does_not_write_grant():
    detail = _make_detail()  # arg_digest="d1" matches _spec()
    eng, queue, writer = _engine(queue_read=detail, claim_ok=True)

    async def _run_test():
        res = await eng.preflight_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(),
        )
        assert len(res.claim_nonce) == 32  # secrets.token_hex(16) → 32 hex chars
        assert isinstance(res.processing_started_at, datetime)
        writer.write.assert_not_awaited()
        queue.mark_processing_if_pending.assert_awaited_once()

    _run(_run_test())


# ---------------------------------------------------------------------------
# Round 36 P1#1: preflight_resume persists SSM mode_revision to queue
# ---------------------------------------------------------------------------


def test_preflight_persists_mode_revision_to_queue_detail():
    """Round 36 P1#1: preflight_resume captures SSM mode_revision and writes it
    to the queue via set_preflight_mode_revision so commit_resume can detect
    a TAKEOVER round-trip that happened between preflight and commit.
    """
    detail = _make_detail()  # arg_digest="d1" matches _spec()
    eng, queue, writer = _engine(
        queue_read=detail,
        claim_ok=True,
        ssm_mode_rev=(SessionStatus.RUNNING, 7),
    )

    async def _run_test():
        res = await eng.preflight_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=7),
            _signal(),
        )
        # set_preflight_mode_revision must be called with the SSM rev (7)
        queue.set_preflight_mode_revision.assert_awaited_once_with("s", "tc1", 7)
        # Returned detail also reflects the persisted rev (callers / tests can
        # observe without a fresh queue.read).
        assert res.detail.session_mode_revision_at_preflight == 7

    _run(_run_test())


def test_preflight_ssm_failure_is_best_effort_and_still_claims():
    """Round 36 P1#1: if SSM read fails, preflight still acquires the claim
    and does NOT call set_preflight_mode_revision (best-effort branch).
    commit_resume will fall back to the round-34 ctx-based check.
    """
    detail = _make_detail()
    eng, queue, writer = _engine(
        queue_read=detail,
        claim_ok=True,
        ssm_raises=True,
    )

    async def _run_test():
        res = await eng.preflight_resume(
            _spec(),
            EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=1),
            _signal(),
        )
        # CAS still succeeds → caller receives a valid claim_nonce
        assert len(res.claim_nonce) == 32
        queue.mark_processing_if_pending.assert_awaited_once()
        # SSM raised → persist branch skipped, detail untouched
        queue.set_preflight_mode_revision.assert_not_awaited()
        assert res.detail.session_mode_revision_at_preflight is None

    _run(_run_test())


def test_preflight_does_not_persist_rev_when_claim_lost():
    """Round 36 P1#1: a losing CAS must not stomp on the winning preflight's
    persisted rev.  We persist the rev AFTER mark_processing_if_pending so
    the path is unreachable on claim loss.
    """
    detail = _make_detail()
    eng, queue, writer = _engine(
        queue_read=detail,
        claim_ok=False,  # CAS loses
        ssm_mode_rev=(SessionStatus.RUNNING, 5),
    )

    async def _run_test():
        import pytest as _pytest
        with _pytest.raises(PolicyConflict):
            await eng.preflight_resume(
                _spec(),
                EvaluationContext(session_mode=SessionStatus.RUNNING, session_mode_revision=5),
                _signal(),
            )
        # Losing CAS must NOT call the persist helper — otherwise we would
        # overwrite the winner's already-persisted rev.
        queue.set_preflight_mode_revision.assert_not_awaited()

    _run(_run_test())
