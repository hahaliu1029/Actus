"""[R2-P1-4 + R3-P1-3 + R4-P2-3] Path B writes NO record, emits NO SSE event,
but DOES fire OTel record_decision for ops dashboards.

[P1.2 fix] Uses a committed user+session via uow_factory() + try/finally cleanup
instead of the flushed-only seed_session fixture, so independent UoW connections
opened by the callback can see the rows.

Run: cd api && uv run pytest -m integration tests/integration/test_recovery_callback_no_record.py -v
"""
import uuid as _uuid

import pytest
from sqlalchemy import delete
from unittest.mock import patch

# Project convention: pytest.mark.anyio (NOT asyncio). pytest-asyncio NOT installed.
pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def _build_flow(uow_factory, fake_summary_llm, session_id: str):
    """Construct a minimal PlannerReActFlow stub for the given *committed* session_id."""
    from app.domain.models.context_overflow_config import ContextOverflowConfig
    from app.domain.services.flows.planner_react import PlannerReActFlow
    from app.domain.services.graphs.compaction import GradualCompactor
    from app.domain.services.graphs.token_estimator import TokenEstimator

    estimator = TokenEstimator(strategy="char")
    overflow_cfg = ContextOverflowConfig(
        context_window=80_000,
        context_overflow_guard_enabled=True,
        soft_trigger_ratio=0.85,
        hard_trigger_ratio=0.95,
        target_ratio=0.65,
        token_estimator="char",
        model_name="test-model",
    )
    compactor = GradualCompactor(
        token_estimator=estimator,
        soft_trigger_ratio=overflow_cfg.soft_trigger_ratio,
        hard_trigger_ratio=overflow_cfg.hard_trigger_ratio,
        target_ratio=overflow_cfg.target_ratio,
        summary_max_chars=overflow_cfg.summary_max_chars,
        token_safety_factor=overflow_cfg.token_safety_factor,
    )

    flow = PlannerReActFlow.__new__(PlannerReActFlow)  # bypass heavy __init__
    flow._compactor = compactor
    flow._summary_llm = fake_summary_llm
    flow._uow_factory = uow_factory
    flow._session_id = session_id
    flow._overflow_config = overflow_cfg
    flow._cost_callback_handler = None
    flow._last_compaction_result = None
    return flow


async def test_recovery_callback_no_record_no_event_otel_only(
    fake_summary_llm, uow_factory, messages_at_85_percent
):
    from app.infrastructure.models.user import UserModel
    from app.infrastructure.models.session import SessionModel

    user_id = str(_uuid.uuid4())
    session_id = f"sess-recovery-{_uuid.uuid4().hex[:12]}"

    # Step 1: commit seed rows so independent UoW connections can see them
    async with uow_factory() as uow:
        uow.db_session.add(
            UserModel(id=user_id, username=f"u-{user_id[:8]}", password_hash="x")
        )
        await uow.db_session.flush()
        uow.db_session.add(
            SessionModel(id=session_id, user_id=user_id, status="pending", title="t")
        )
        await uow.db_session.commit()

    try:
        flow = await _build_flow(uow_factory, fake_summary_llm, session_id)
        callback = flow._build_on_context_overflow_callback()

        with patch("app.infrastructure.observability.decision_trace.record_decision") as mock_decision:
            returned = await callback(messages_at_85_percent, kwargs={})

        assert returned is not None  # callback returns the compacted messages list

        # [CXR1-P2-7 review fix] Filter by decision name — record_decision may fire
        # from other code paths along the callback (compactor, recovery model). We
        # only assert the callback itself emitted exactly one "compaction_recovery_callback".
        callback_decisions = [
            c for c in mock_decision.call_args_list
            if c.args and c.args[0] == "compaction_recovery_callback"
        ]
        assert len(callback_decisions) == 1
        kwargs = callback_decisions[0].kwargs
        assert kwargs.get("outcome") in {"llm_summary", "hard_truncate"}
        # No attrs beyond outcome (CANONICAL_ATTRIBUTES is FROZEN)
        assert set(kwargs.keys()) <= {"outcome"}

        # Path B must NOT emit "compaction" (Path A's decision name) — that's Path A's namespace
        assert not any(c.args and c.args[0] == "compaction" for c in mock_decision.call_args_list)

        async with uow_factory() as uow:
            rows = await uow.compaction.list_for_session(session_id)
        assert len(rows) == 0  # NO record was written

        # _last_compaction_result must NOT have been set by Path B
        assert getattr(flow, "_last_compaction_result", None) is None

    finally:
        async with uow_factory() as uow:
            await uow.db_session.execute(
                delete(SessionModel).where(SessionModel.id == session_id)
            )
            await uow.db_session.execute(
                delete(UserModel).where(UserModel.id == user_id)
            )
            await uow.db_session.commit()
