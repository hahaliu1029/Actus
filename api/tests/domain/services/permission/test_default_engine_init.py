"""DefaultPermissionEngine constructor — all 7 dependencies required."""

from unittest.mock import AsyncMock

import pytest

from app.domain.services.permission.default_engine import DefaultPermissionEngine
from app.domain.services.permission.smart_approve_provider import SmartApproveProvider


def _make():
    return DefaultPermissionEngine(
        uow_factory=AsyncMock(),
        writer=AsyncMock(),
        queue=AsyncMock(),
        session_machine=AsyncMock(),
        reader=AsyncMock(),
        escalation_registry={
            "smart_approve": SmartApproveProvider(AsyncMock(), timeout_seconds=1.0),
        },
        decision_recorder=lambda *a, **kw: None,
    )


def test_constructs_with_required_deps():
    engine = _make()
    assert engine is not None


def test_missing_decision_recorder_falls_back_to_noop():
    engine = DefaultPermissionEngine(
        uow_factory=AsyncMock(),
        writer=AsyncMock(),
        queue=AsyncMock(),
        session_machine=AsyncMock(),
        reader=AsyncMock(),
        escalation_registry={},
        decision_recorder=None,
    )
    # noop decision recorder must not raise when called
    engine._record_decision("permission_engine.test", "allow")  # type: ignore[attr-defined]
