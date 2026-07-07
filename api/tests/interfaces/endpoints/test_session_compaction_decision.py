"""B11 §8: unit tests for the /compact request decision (pure fn, no DB)."""
from __future__ import annotations

import pytest

from app.domain.models.session import SessionStatus
from app.interfaces.endpoints.session_compaction_routes import (
    _compaction_request_decision,
)


def test_completed_and_timed_out_allowed():
    for st in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT):
        assert (
            _compaction_request_decision(
                st, manual_enabled=True, overflow_guard_enabled=True
            )
            is None
        )


@pytest.mark.parametrize(
    "status,reason",
    [
        (SessionStatus.PENDING, "no_compactable_history"),
        (SessionStatus.RUNNING, "run_active"),
        (SessionStatus.FINISHING, "run_active"),
        (SessionStatus.WAITING, "waiting_resume_unsupported"),
        (SessionStatus.TAKEOVER, "takeover_active"),
        (SessionStatus.TAKEOVER_PENDING, "takeover_active"),
    ],
)
def test_non_compactable_statuses_rejected(status, reason):
    assert (
        _compaction_request_decision(
            status, manual_enabled=True, overflow_guard_enabled=True
        )
        == reason
    )


def test_flag_off_rejected_regardless_of_status():
    assert (
        _compaction_request_decision(
            SessionStatus.COMPLETED, manual_enabled=False, overflow_guard_enabled=True
        )
        == "manual_compaction_disabled"
    )


def test_overflow_guard_off_rejected():
    assert (
        _compaction_request_decision(
            SessionStatus.COMPLETED, manual_enabled=True, overflow_guard_enabled=False
        )
        == "overflow_guard_disabled"
    )
