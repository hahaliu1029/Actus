"""[R2-P1-3 + CXR1-P2-5] End-to-end Path A invariant: commit failure rolls back
both save_memory AND record_compaction, AND no CompactionEvent ever yields.

Design note — transaction isolation
------------------------------------
The ``planner_react_with_compactor`` fixture seeds user/session rows through
``db_session``'s rollback-managed ``begin()`` block.  Those rows are visible
inside the same connection but **never committed**, so an independent UoW
(``uow_factory()``) opened by ``_check_overflow`` cannot see them.
``save_memory`` issues an UPDATE with rowcount-check and would raise
``ValueError("会话 not found")`` at execute-time, *before* our patched
``commit()`` ever fires.

To exercise the commit-raise path end-to-end we manage our own connection:
  1. Commit seed user+session rows through UoW #1.
  2. Build a ``PlannerReActFlow`` stub (bypassing __init__) wired to UoW #2.
  3. Patch ``AsyncSession.commit`` to raise once.
  4. Call ``flow._check_overflow(memory)``  →  must raise.
  5. Assert: no compaction row committed (rollback), memory unchanged in DB,
     ``_last_compaction_result.compaction_id`` is None.
  6. Clean up committed seed rows in a finally block.

Run: cd api && uv run pytest -m integration tests/integration/test_check_overflow_commit_raise.py -v
"""
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError

# Project convention: pytest.mark.anyio (NOT asyncio). pytest-asyncio NOT installed.
pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def _build_flow(uow_factory, session_id: str, fake_summary_llm):
    """Construct a minimal PlannerReActFlow stub for the given session_id.

    Mirrors the ``planner_react_with_compactor`` fixture exactly but wires to
    a provided *committed* session_id so independent UoW connections can see it.
    """
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


async def test_check_overflow_commit_failure_rolls_back_both_writes_and_no_event(
    uow_factory, fake_summary_llm
):
    """Path A invariant: commit failure → no compaction record, no memory change, no result.

    (a) No compaction record was committed (record_compaction rolled back).
    (b) save_memory rolled back — DB memory unchanged from pre-state snapshot.
    (c) _last_compaction_result not set with a populated compaction_id.
    """
    from app.infrastructure.models.user import UserModel
    from app.infrastructure.models.session import SessionModel
    from app.domain.models.memory import Memory
    from app.domain.services.graphs.message_utils import messages_to_dicts
    from langchain_core.messages import HumanMessage, SystemMessage
    from sqlalchemy import delete

    uid = str(uuid.uuid4())
    sid = f"sess-16a-{uuid.uuid4().hex[:12]}"

    # ── 1. Commit seed rows so independent UoW connections can see them ─────
    async with uow_factory() as uow_seed:
        user = UserModel(id=uid, username=f"b6test16a_{uid[:8]}", password_hash="x")
        session_row = SessionModel(
            id=sid, user_id=uid, status="pending", title="16a test session"
        )
        uow_seed.db_session.add(user)
        await uow_seed.db_session.flush()
        uow_seed.db_session.add(session_row)
        await uow_seed.db_session.commit()  # explicit: make rows visible to other connections

    try:
        # ── 2. Snapshot pre-state of memory in DB ───────────────────────────
        async with uow_factory() as uow_pre:
            before_memory = await uow_pre.session.get_memory(sid, "react")

        # ── 3. Build messages that exceed 85% threshold ──────────────────────
        msgs = [SystemMessage(content="sys")] + [
            HumanMessage(content="x" * 16_000) for _ in range(20)
        ]
        memory = Memory(messages=messages_to_dicts(msgs))

        # ── 4. Build flow stub wired to committed session ────────────────────
        flow = await _build_flow(uow_factory, sid, fake_summary_llm)

        # ── 5. Patch commit to raise — fires when Path A reaches the
        #       explicit `await uow.db_session.commit()` at planner_react.py:754
        async def _failing_commit(self):  # noqa: ANN001
            raise IntegrityError("synthetic-16a", None, Exception("forced"))

        with patch(
            "sqlalchemy.ext.asyncio.AsyncSession.commit",
            new=_failing_commit,
        ):
            with patch(
                "app.infrastructure.observability.decision_trace.record_decision"
            ) as mock_decision:
                with pytest.raises(Exception):
                    await flow._check_overflow(memory)

        # ── 6. Assertion (a): no compaction record committed (rollback) ──────
        async with uow_factory() as uow_post:
            rows = await uow_post.compaction.list_for_session(sid)
        assert len(rows) == 0, (
            f"Expected 0 compaction rows after commit failure, got {len(rows)}"
        )

        # ── 7. Assertion (b): save_memory rolled back — memory unchanged ─────
        async with uow_factory() as uow_post2:
            after_memory = await uow_post2.session.get_memory(sid, "react")

        before_msgs = before_memory.messages if before_memory else []
        after_msgs = after_memory.messages if after_memory else []
        assert after_msgs == before_msgs, (
            "save_memory was NOT rolled back: memory changed despite commit failure"
        )

        # ── 8. Assertion (c): _last_compaction_result.compaction_id is None ──
        # replace(result, compaction_id=...) at planner_react.py:758 runs AFTER
        # the commit that raised, so compaction_id must not be set.
        last = getattr(flow, "_last_compaction_result", None)
        assert last is None or last.compaction_id is None, (
            f"Expected _last_compaction_result.compaction_id=None, "
            f"got compaction_id={last.compaction_id if last else 'attr missing'}"
        )

        # ── 9. Assertion (d): Path A OTel must NOT fire when commit raises ────
        # record_decision('compaction') fires AFTER the commit succeeds; if
        # commit raised, the record was rolled back and OTel must not emit.
        compaction_decisions = [
            c for c in mock_decision.call_args_list
            if c.args and c.args[0] == "compaction"
        ]
        assert len(compaction_decisions) == 0, (
            "Path A OTel must not fire when commit raises — record was rolled back"
        )

    finally:
        # ── 9. Clean up committed seed rows ──────────────────────────────────
        async with uow_factory() as uow_cleanup:
            await uow_cleanup.db_session.execute(
                delete(SessionModel).where(SessionModel.id == sid)
            )
            await uow_cleanup.db_session.execute(
                delete(UserModel).where(UserModel.id == uid)
            )
            await uow_cleanup.db_session.commit()
