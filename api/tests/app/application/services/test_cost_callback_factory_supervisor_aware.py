"""B3-core PR-3b: supervisor-aware cost callback factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from app.application.services.cost_callback_factory import (
    build_cost_callback_handler,
    build_supervisor_aware_callback_handler,
)
from app.domain.services.cost_callback_handler import (
    CostCallbackHandler,
    SupervisorAwareCallbackHandler,
)


def test_old_builder_still_returns_base_cost_handler() -> None:
    handler = build_cost_callback_handler(
        session_id="session-1",
        user_id="user-1",
        uow_factory=lambda: AsyncMock(),
    )

    assert isinstance(handler, CostCallbackHandler)
    assert not isinstance(handler, SupervisorAwareCallbackHandler)


def test_supervisor_aware_builder_wires_supervisor() -> None:
    supervisor = AsyncMock()

    handler = build_supervisor_aware_callback_handler(
        supervisor=supervisor,
        session_id="session-1",
        user_id="user-1",
        uow_factory=lambda: AsyncMock(),
    )

    assert isinstance(handler, SupervisorAwareCallbackHandler)
    assert isinstance(handler, CostCallbackHandler)
    assert handler.session_id == "session-1"
    assert handler.user_id == "user-1"
    assert handler._supervisor is supervisor
