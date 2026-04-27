"""B4 M0 post-audit: cost persister propagates commit failures.

``DBUnitOfWork.__aexit__`` deliberately swallows commit failures (SSE-
disconnect handling relies on that silencing). That is the right default
for *streaming* writes but wrong for the cost ledger: a swallowed commit
error would leave the cost row missing from the DB while
``CostCallbackHandler._persist_safely`` thinks the write succeeded —
silent undercount exactly like the audit flagged.

The factory commits explicitly and re-raises on failure; the handler's
failure counter then ticks and a degraded-marker row gets written so
aggregation reports ``partial``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.application.services.cost_callback_factory import (
    build_cost_callback_handler,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeDbSession:
    """Captures execute calls; raises on commit to simulate a commit failure."""

    def __init__(self, *, commit_raises: Exception | None = None) -> None:
        self.executed: list[Any] = []
        self.committed: bool = False
        self.rolled_back: bool = False
        self._commit_raises = commit_raises

    async def execute(self, stmt: Any) -> Any:
        self.executed.append(stmt)
        return MagicMock(scalars=MagicMock(return_value=MagicMock(all=lambda: [])))

    async def commit(self) -> None:
        if self._commit_raises is not None:
            raise self._commit_raises
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        pass


class _FakeUoW:
    """Mimics ``DBUnitOfWork``'s ``__aexit__`` contract (swallows commit/rollback)."""

    def __init__(self, db_session: _FakeDbSession) -> None:
        self.db_session = db_session

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Mimic DBUnitOfWork: try commit, swallow any error.
        try:
            if exc_type:
                await self.db_session.rollback()
            else:
                await self.db_session.commit()
        except Exception:
            pass
        await self.db_session.close()


async def test_factory_persister_rerasises_commit_failure() -> None:
    """Explicit commit inside the persister must propagate, not be eaten by UoW."""
    fake_session = _FakeDbSession(commit_raises=RuntimeError("commit lost"))

    def factory() -> _FakeUoW:
        return _FakeUoW(fake_session)

    handler = build_cost_callback_handler(
        session_id="s", user_id="u", uow_factory=factory
    )

    from datetime import datetime, timezone
    from decimal import Decimal
    from uuid import uuid4

    from app.domain.models.cost_record import CostRecord, CostStatus

    record = CostRecord(
        id=str(uuid4()),
        session_id="s",
        user_id="u",
        run_id=str(uuid4()),
        node_name="planner",
        step_ix=0,
        attempt_ix=0,
        model="gpt-4o",
        provider="openai_official",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        total_usd=Decimal("0.001"),
        pricing_version="v1",
        cost_status=CostStatus.ACTUAL,
        created_at=datetime.now(timezone.utc),
    )

    # Direct invocation of the persister closure — the real handler wraps
    # this in ``_persist_safely`` which catches the re-raised error.
    with pytest.raises(RuntimeError, match="commit lost"):
        await handler._persister(record)

    # Rollback was invoked on failure (session hygiene).
    assert fake_session.rolled_back is True, (
        "On commit failure the persister must rollback the pending state "
        "before re-raising so the UoW's __aexit__ cleanup stays clean."
    )


async def test_handler_counts_factory_commit_failure_and_writes_marker() -> None:
    """End-to-end: commit failure → persist_failure_count ticks + degraded marker."""
    commit_counter = {"n": 0}

    class _RecordingSession(_FakeDbSession):
        async def commit(self) -> None:
            commit_counter["n"] += 1
            if commit_counter["n"] == 1:
                raise RuntimeError("commit lost once")
            self.committed = True

    def factory() -> _FakeUoW:
        return _FakeUoW(_RecordingSession())

    handler = build_cost_callback_handler(
        session_id="s2", user_id="u", uow_factory=factory
    )

    from uuid import uuid4

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="hi",
                            usage_metadata={
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                            },
                        )
                    )
                ]
            ]
        ),
        run_id=run_id,
    )
    await handler.flush_pending()

    assert handler.persist_failure_count == 1, (
        "Commit failure must be seen as a persist failure so the degraded "
        "marker fires and aggregation can report partial. Got "
        f"persist_failure_count={handler.persist_failure_count}."
    )
    # At least 2 commit attempts: primary (failed) + marker (succeeded).
    # The real DBUnitOfWork's ``__aexit__`` also calls commit() on the
    # success path (SQLAlchemy treats the redundant commit as idempotent),
    # so the marker path actually sees 2 commit() calls. Total is 3; we
    # assert >= 2 to keep the test robust to either variant.
    assert commit_counter["n"] >= 2
