"""End-to-end Path A: trigger Level 2, assert record + matching compaction_id.

[P1.2 fix] Uses a committed user+session via uow_factory() + try/finally cleanup
instead of the flushed-only seed_session fixture, so independent UoW connections
opened by _check_overflow can see the rows.

Run: cd api && uv run pytest -m integration tests/integration/test_check_overflow_writes_record.py -v
"""
import uuid as _uuid

import pytest
from sqlalchemy import delete

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


async def test_check_overflow_writes_record_and_populates_compaction_id(
    fake_summary_llm, uow_factory, memory_at_85_percent
):
    from app.infrastructure.models.user import UserModel
    from app.infrastructure.models.session import SessionModel
    from app.infrastructure.models.conversation_compaction import ConversationCompactionModel

    user_id = str(_uuid.uuid4())
    session_id = f"sess-writes-{_uuid.uuid4().hex[:12]}"

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
        result = await flow._check_overflow(memory_at_85_percent)

        assert result is not None
        assert result.level_applied >= 2
        assert result.compaction_id is not None
        assert len(result.compaction_id) == 16

        async with uow_factory() as uow:
            rows = await uow.compaction.list_for_session(session_id)
        assert len(rows) == 1
        assert rows[0].compaction_id == result.compaction_id

    finally:
        async with uow_factory() as uow:
            await uow.db_session.execute(
                delete(ConversationCompactionModel).where(
                    ConversationCompactionModel.session_id == session_id
                )
            )
            await uow.db_session.execute(
                delete(SessionModel).where(SessionModel.id == session_id)
            )
            await uow.db_session.execute(
                delete(UserModel).where(UserModel.id == user_id)
            )
            await uow.db_session.commit()


async def test_check_overflow_emits_path_a_otel_decision_after_commit(
    fake_summary_llm, uow_factory, memory_at_85_percent
):
    """[CXR1-P2-6 + P1.2 fix] Path A fires record_decision('compaction', outcome=primary_kind) after commit."""
    from unittest.mock import patch
    from app.infrastructure.models.user import UserModel
    from app.infrastructure.models.session import SessionModel
    from app.infrastructure.models.conversation_compaction import ConversationCompactionModel

    user_id = str(_uuid.uuid4())
    session_id = f"sess-otel-{_uuid.uuid4().hex[:12]}"

    # Step 1: commit seed rows
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
        with patch("app.infrastructure.observability.decision_trace.record_decision") as mock:
            await flow._check_overflow(memory_at_85_percent)

        # At least one call must be name="compaction"; outcome is one of the 2 atomic kinds
        compaction_calls = [c for c in mock.call_args_list if c.args and c.args[0] == "compaction"]
        assert compaction_calls, "expected record_decision('compaction', outcome=...) after Path A commit"
        outcome = compaction_calls[0].kwargs.get("outcome")
        assert outcome in {"llm_summary", "hard_truncate"}
        # No attrs beyond outcome (CANONICAL_ATTRIBUTES is FROZEN per [R2-P1-5])
        assert set(compaction_calls[0].kwargs.keys()) <= {"outcome"}

    finally:
        async with uow_factory() as uow:
            await uow.db_session.execute(
                delete(ConversationCompactionModel).where(
                    ConversationCompactionModel.session_id == session_id
                )
            )
            await uow.db_session.execute(
                delete(SessionModel).where(SessionModel.id == session_id)
            )
            await uow.db_session.execute(
                delete(UserModel).where(UserModel.id == user_id)
            )
            await uow.db_session.commit()
