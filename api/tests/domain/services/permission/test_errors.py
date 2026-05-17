import pytest

from app.domain.services.permission.errors import (
    EscalationProviderError,
    EscalationTimeout,
    EscalationUnavailable,
    PermissionError,
    PolicyConflict,
    SessionModeViolation,
    WriterIntegrityError,
)


def test_subclass_hierarchy():
    assert issubclass(PolicyConflict, PermissionError)
    assert issubclass(SessionModeViolation, PermissionError)
    assert issubclass(WriterIntegrityError, PermissionError)
    assert issubclass(EscalationProviderError, PermissionError)
    assert issubclass(EscalationTimeout, EscalationProviderError)
    assert issubclass(EscalationUnavailable, EscalationProviderError)


def test_policy_conflict_carries_code():
    err = PolicyConflict("approval_already_claimed")
    assert "approval_already_claimed" in str(err)


def test_session_mode_violation_can_be_raised():
    with pytest.raises(SessionModeViolation):
        raise SessionModeViolation("not allowed in FINISHING")
