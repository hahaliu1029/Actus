import dataclasses
from datetime import datetime, timezone

import pytest

from app.domain.models.session import SessionStatus
from app.domain.services.permission.context import (
    EvaluationContext,
    PreflightResumeResult,
    ResumeSignal,
)


# C-P0-3: use _make_detail helper to supply all 11 required fields
def _make_detail(**overrides):
    """Test helper — supplies all 11 required fields with sensible defaults
    (real dataclass at api/app/domain/services/permission/confirmation_queue.py:14
    has no field-level defaults except status, so omitting any required
    field raises TypeError)."""
    from app.domain.services.permission.confirmation_queue import ConfirmationDetail

    base = dict(
        session_id="s_test",
        tool_call_id="tc_test",
        user_id="u_test",
        tool_name="file_write",
        tool_args={"path": "/x"},
        risk_level="medium",
        arg_digest="ad_test",
        primary_arg="",
        dir_arg=None,
        matched_patterns=[],
        deadline_ts=9999999999.0,
    )
    base.update(overrides)
    return ConfirmationDetail(**base)


def test_evaluation_context_required_fields():
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=42,
    )
    assert ctx.session_mode is SessionStatus.RUNNING
    assert ctx.session_mode_revision == 42
    assert ctx.retry_count == 0
    assert ctx.prior_outcomes == ()


def test_evaluation_context_is_frozen():
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING, session_mode_revision=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.retry_count = 99


def test_resume_signal_action_is_literal_approve_or_deny():
    sig = ResumeSignal(
        confirmation_id="s1:tc1",
        action="approve",
        grant_scope="session",
        actor="user_click",
    )
    assert sig.action == "approve"

    deny = ResumeSignal(
        confirmation_id="s1:tc2", action="deny",
        grant_scope="once", actor="user_click",
    )
    assert deny.action == "deny"


def test_preflight_result_carries_nonce_and_timestamp():
    detail = _make_detail(session_id="s1", tool_call_id="tc1",
                          tool_name="file_write",
                          arg_digest="abc123",
                          deadline_ts=datetime.now(timezone.utc).timestamp() + 300)
    now = datetime.now(timezone.utc)
    res = PreflightResumeResult(
        claim_nonce="deadbeef" * 4,
        processing_started_at=now,
        detail=detail,
    )
    assert len(res.claim_nonce) == 32   # 16 bytes hex
    assert res.detail is detail
    assert res.processing_started_at is now
