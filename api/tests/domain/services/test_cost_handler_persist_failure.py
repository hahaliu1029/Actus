"""B4 M0 post-audit: persist-failure → aggregate flips to ``partial``.

Without this, a failed DB write silently disappears from the ledger and
``GET /cost`` keeps reporting ``cost_status=actual`` even though one or
more records are missing — exactly the "silent undercount" the audit
flagged.

Strategy: on persister exception, the handler writes a best-effort
degraded-marker row (``cost_status=unknown``, zeroed tokens/amount).
Aggregation's existing actual+unknown → partial rubric then surfaces the
incident. A running counter is also exposed for callers that need a
session-local signal.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.application.services.cost_aggregation_service import (
    CostAggregationService,
)
from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.services.cost_callback_handler import CostCallbackHandler

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_llm_result(usage: dict | None):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    return LLMResult(
        generations=[
            [ChatGeneration(message=AIMessage(content="hi", usage_metadata=usage))]
        ]
    )


class _FailFirstPersister:
    """Raises on the first insert, succeeds on every subsequent one.

    Models a DB blip where the primary write fails but a retry can land.
    """

    def __init__(self) -> None:
        self.rows: List[CostRecord] = []
        self.calls: int = 0

    async def __call__(self, record: CostRecord) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated db blip")
        self.rows.append(record)


class _InMemoryRepo:
    def __init__(self, rows: list[CostRecord]) -> None:
        self._rows = rows

    async def find_by_session(self, session_id: str) -> list[CostRecord]:
        return [r for r in self._rows if r.session_id == session_id]


async def test_persist_failure_yields_degraded_marker() -> None:
    """One LLM call, primary persist fails → marker lands with cost_status=unknown."""
    persister = _FailFirstPersister()
    handler = CostCallbackHandler(
        session_id="sess-degraded", user_id="u", persister=persister
    )

    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(
        _make_llm_result(
            {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}
        ),
        run_id=run_id,
    )
    await handler.flush_pending()

    assert persister.calls == 2
    assert handler.persist_failure_count == 1
    assert len(persister.rows) == 1

    marker = persister.rows[0]
    assert marker.cost_status == CostStatus.UNKNOWN
    assert marker.total_usd == Decimal(0)
    assert marker.input_tokens == 0
    assert marker.output_tokens == 0
    assert marker.node_name == "persist_degraded"
    assert marker.run_id == str(run_id), (
        "Marker must keep the original run_id so DB-level uniqueness prevents "
        "double-counting if the original write partially landed."
    )


async def test_mixed_success_and_persist_failure_aggregates_as_partial() -> None:
    """Two LLM calls: first write fails (marker lands unknown), second succeeds.

    This is the exact audit scenario: "2 次 LLM 调用中 1 次成本丢失 显示成可信的 actual".
    With the fix, aggregation returns ``partial`` instead of ``actual``.
    """
    persister = _FailFirstPersister()
    handler = CostCallbackHandler(
        session_id="sess-mix", user_id="u", persister=persister
    )

    # Call 1: primary persist fails → marker (unknown) lands.
    rid1 = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=rid1,
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(
        _make_llm_result(
            {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}
        ),
        run_id=rid1,
    )

    # Call 2: no failure — lands as actual.
    rid2 = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=rid2,
        metadata={"langgraph_node": "executor_node", "langgraph_step": 1},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(
        _make_llm_result(
            {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}
        ),
        run_id=rid2,
    )
    await handler.flush_pending()

    # _FailFirstPersister fails call #1 only; we expect:
    #   call 1: primary (fails) → marker (ok)   = 2 persister calls
    #   call 2: primary (ok)                    = 1 persister call
    assert persister.calls == 3
    assert handler.persist_failure_count == 1
    # Two landed rows: the unknown marker + the successful actual
    assert len(persister.rows) == 2
    statuses = {r.cost_status for r in persister.rows}
    assert statuses == {CostStatus.UNKNOWN, CostStatus.ACTUAL}

    svc = CostAggregationService(repository=_InMemoryRepo(persister.rows))
    agg = await svc.get_aggregate("sess-mix")
    assert agg.cost_status == CostStatus.PARTIAL, (
        "A session where one LLM call's cost row got lost and the next "
        "succeeded must report cost_status=partial — never actual. "
        f"Got: {agg.cost_status!r}"
    )
    assert agg.has_partial_records is True


async def test_single_call_persist_failure_aggregates_as_partial() -> None:
    """Session with exactly ONE LLM call whose write failed must report partial.

    Without the degraded-marker override in aggregation, an unknown-only
    session (record_count=1, cost_status=unknown) reads as clean "nothing
    happened" instead of "we know a row went missing". The audit pinned this
    as a real ledger gap.
    """
    persister = _FailFirstPersister()
    handler = CostCallbackHandler(
        session_id="sess-single-fail", user_id="u", persister=persister
    )
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(
        _make_llm_result(
            {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}
        ),
        run_id=run_id,
    )
    await handler.flush_pending()

    # Exactly one row landed (the marker).
    assert len(persister.rows) == 1
    assert persister.rows[0].node_name == "persist_degraded"

    svc = CostAggregationService(repository=_InMemoryRepo(persister.rows))
    agg = await svc.get_aggregate("sess-single-fail")
    assert agg.record_count == 1
    assert agg.cost_status == CostStatus.PARTIAL, (
        "Single-call session whose cost row failed to persist must still "
        "report cost_status=partial (not unknown) — the degraded marker "
        "is the signal that the ledger is known incomplete."
    )
    assert agg.has_partial_records is True


async def test_persist_failure_survives_even_if_marker_also_fails() -> None:
    """If the retry also fails, don't raise — the LLM call must stay unblocked."""
    calls: List[int] = []

    async def always_fails(_record: CostRecord) -> None:
        calls.append(0)
        raise RuntimeError("db down hard")

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=always_fails
    )
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node"},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(
        _make_llm_result({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
        run_id=run_id,
    )
    await handler.flush_pending()

    assert len(calls) == 2
    assert handler.persist_failure_count == 1
