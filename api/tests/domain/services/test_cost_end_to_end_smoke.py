"""B4 M0 Phase Z: end-to-end smoke for the cost pipeline.

Exercises the full chain in-memory (no real DB, no real LLM):

    LangGraph node → LLM ainvoke → on_chat_model_start (cache metadata) →
    on_llm_end (build CostRecord) → repository.insert → aggregation.get_aggregate

If every Phase in isolation passes but this one fails, the integration
between them is broken. This test is the M0 definition of "the pipeline
works end-to-end."
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, List, Optional
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.application.services.cost_aggregation_service import CostAggregationService
from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.services.cost_callback_handler import CostCallbackHandler


def _make_llm_result(usage_metadata: dict | None = None) -> LLMResult:
    msg = AIMessage(content="ok", usage_metadata=usage_metadata)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeChatModel(BaseChatModel):
    """BaseChatModel that returns a fixed AIMessage with provider-style usage_metadata."""

    usage_metadata: dict | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-smoke"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(content="ok", usage_metadata=self.usage_metadata)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    @property
    def _identifying_params(self) -> dict[str, Any]:
        # provider_id must be set alongside model — the audit removed
        # cross-provider pricing fallbacks, so get_price(model) alone
        # returns None and cost_status collapses to UNKNOWN.
        return {"model": "gpt-4o", "provider_id": "openai_official"}


class _InMemoryRepo:
    def __init__(self) -> None:
        self.rows: List[CostRecord] = []

    async def insert(self, record: CostRecord) -> None:
        self.rows.append(record)

    async def find_by_session(self, session_id: str) -> List[CostRecord]:
        return [r for r in self.rows if r.session_id == session_id]


class _State(TypedDict):
    messages: list[BaseMessage]


async def test_graph_invoke_produces_cost_record_and_aggregate() -> None:
    repo = _InMemoryRepo()
    fake_llm = _FakeChatModel(
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
        }
    )
    handler = CostCallbackHandler(
        session_id="sess-e2e",
        user_id="user-e2e",
        persister=repo.insert,
    )

    async def planner_node(state: _State, config: RunnableConfig) -> _State:
        response = await fake_llm.ainvoke(state["messages"], config=config)
        return {"messages": state["messages"] + [response]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("planner", planner_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", END)
    graph = builder.compile()

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"callbacks": [handler]},
    )
    await handler.flush_pending()

    # 1. Repo now holds exactly one row
    assert len(repo.rows) == 1, f"expected 1 row, got {len(repo.rows)}"

    row = repo.rows[0]
    assert row.session_id == "sess-e2e"
    assert row.user_id == "user-e2e"
    assert row.node_name == "planner", (
        f"langgraph_node must flow from Pregel metadata into CostRecord.node_name; "
        f"got {row.node_name!r}"
    )
    assert row.model == "gpt-4o"
    assert row.cost_status == CostStatus.ACTUAL
    assert row.input_tokens == 1000
    assert row.output_tokens == 500
    assert row.total_usd == Decimal("0.0075")

    # 2. Aggregation rolls up to a one-node, all-actual session
    svc = CostAggregationService(repository=repo)
    agg = await svc.get_aggregate("sess-e2e")

    assert agg.record_count == 1
    assert agg.total_usd == Decimal("0.0075")
    assert agg.cost_status == CostStatus.ACTUAL
    assert agg.by_node == {"planner": Decimal("0.0075")}
    assert agg.by_model == {"gpt-4o": Decimal("0.0075")}
    assert agg.by_provider == {"openai_official": Decimal("0.0075")}
    assert agg.has_partial_records is False


async def test_graph_invoke_with_no_usage_marks_unknown() -> None:
    """Provider omits usage → CostRecord marked unknown (not estimated).

    Post-audit: ``estimated`` implies a heuristic we never implemented;
    ``unknown`` is the honest label until real estimation lands.
    """
    repo = _InMemoryRepo()
    fake_llm = _FakeChatModel(usage_metadata=None)
    handler = CostCallbackHandler(
        session_id="sess-est",
        user_id="user-e2e",
        persister=repo.insert,
    )

    async def planner_node(state: _State, config: RunnableConfig) -> _State:
        response = await fake_llm.ainvoke(state["messages"], config=config)
        return {"messages": state["messages"] + [response]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("planner", planner_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", END)
    graph = builder.compile()

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"callbacks": [handler]},
    )
    await handler.flush_pending()

    assert len(repo.rows) == 1
    assert repo.rows[0].cost_status == CostStatus.UNKNOWN
    assert repo.rows[0].total_usd == Decimal(0)

    svc = CostAggregationService(repository=repo)
    agg = await svc.get_aggregate("sess-est")
    assert agg.cost_status == CostStatus.UNKNOWN
    assert agg.has_partial_records is False


class TestE2EDegradedMarkerVisible:
    """B4 Issue 1D: end-to-end visibility of the degraded sentinel.

    Spec: docs/superpowers/specs/2026-04-27-b4-1d-finishing-drain-design.md §5

    Positive case: drain timeout + healthy marker persister → marker row
    lands in DB → aggregation reports cost_status=partial.

    Negative case: drain timeout + unhealthy marker persister → marker
    dropped → aggregation reports cost_status=actual. This pins the
    documented best-effort trade-off (§2 Non-goals + §8 ADR).
    """

    async def test_drain_timeout_with_healthy_marker_reports_partial(
        self,
    ) -> None:
        from app.application.services.cost_aggregation_service import (
            CostAggregationService,
        )

        # Use the same captured-records fixture pattern as elsewhere in
        # this file: a list-backed persister stands in for the DB.
        captured: List[CostRecord] = []

        async def persist(record: CostRecord) -> None:
            captured.append(record)

        handler = CostCallbackHandler(
            session_id="sess-Q", user_id="user-Q", persister=persist
        )

        # Drive one happy LLM call to land an ACTUAL row, then dispatch
        # the marker (mirroring what _set_terminal_status would do on a
        # drain timeout with healthy persister).
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(usage_metadata={
                "input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500
            }),
            run_id=run_id,
        )
        await handler.flush_pending()
        ok = await handler.write_session_degraded_marker(reason="drain_timeout")
        assert ok is True

        # Aggregation: build a fake repo backed by the captured list.
        class _ListRepo:
            async def find_by_session(self, session_id: str) -> List[CostRecord]:
                return [r for r in captured if r.session_id == session_id]

        service = CostAggregationService(repository=_ListRepo())
        agg = await service.get_aggregate("sess-Q")

        assert agg.cost_status == CostStatus.PARTIAL, (
            "Aggregation must surface partial when a persist_degraded row "
            "is present alongside actual rows."
        )

        # The marker row carries sentinel labels (model="session_degraded_marker",
        # provider="internal") so it surfaces in by_model / by_provider as a
        # recognizable internal entry — but MUST NOT introduce empty keys
        # (which would happen if the marker used model="" / provider="").
        assert "" not in (agg.by_model or {}), (
            "by_model must not contain an empty-string key — would leak "
            "as {'': '0'} in GET /cost. Marker rows must carry non-empty "
            "sentinel labels (see _build_session_degraded_record)."
        )
        assert "" not in (agg.by_provider or {}), (
            "by_provider must not contain an empty-string key (same "
            "reason as by_model)."
        )

    async def test_marker_dropped_under_unhealthy_persister_reports_actual(
        self,
    ) -> None:
        """Spec §5 negative case: drain timeout AND marker persister
        exceeds soft bound → marker write returns False, no marker row in
        DB, aggregation reports cost_status=actual.

        This pins the documented best-effort trade-off (§2 Non-goals: 'Hard
        guarantee of degraded marker presence'). A future implementer must
        not silently upgrade this to a strict guarantee — that would
        require a non-DB fallback channel (rejected; see §8 ADR).
        """
        from app.application.services.cost_aggregation_service import (
            CostAggregationService,
        )

        captured: List[CostRecord] = []
        block_event = asyncio.Event()  # never set → marker persist hangs

        async def persist(record: CostRecord) -> None:
            # Successful for cost rows, hanging for marker rows. Distinguish
            # by node_name (markers carry "persist_degraded").
            if record.node_name == "persist_degraded":
                await block_event.wait()
            captured.append(record)

        handler = CostCallbackHandler(
            session_id="sess-N", user_id="user-N", persister=persist
        )

        # Land one ACTUAL row.
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(usage_metadata={
                "input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500
            }),
            run_id=run_id,
        )
        await handler.flush_pending()
        # Marker write hangs past _MARKER_WRITE_TIMEOUT_SECONDS → False.
        ok = await handler.write_session_degraded_marker(reason="drain_timeout")
        assert ok is False  # documented best-effort trade-off

        class _ListRepo:
            async def find_by_session(self, session_id: str) -> List[CostRecord]:
                return [r for r in captured if r.session_id == session_id]

        service = CostAggregationService(repository=_ListRepo())
        agg = await service.get_aggregate("sess-N")

        # Negative trade-off: aggregation reports ACTUAL because no
        # persist_degraded row reached the DB.
        assert agg.cost_status == CostStatus.ACTUAL, (
            "When marker write fails, aggregation reports actual — this is "
            "the documented best-effort trade-off (§2 Non-goals + §8 ADR). "
            "If you're tempted to 'fix' this assertion, you actually need a "
            "non-DB fallback channel; please re-read the ADR first."
        )

        # Cleanup: release the parked marker task so the event loop is
        # quiet at teardown.
        block_event.set()
        for _ in range(20):
            await asyncio.sleep(0.05)
