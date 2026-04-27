"""B4 M0 post-audit: on_llm_error with partial stream data is billed.

LangChain hands ``on_llm_error`` a merged partial ``LLMResult`` via
``kwargs["response"]`` when a stream errors after some tokens have
already been emitted (see langchain_core/language_models/chat_models.py
lines ~712-722). Providers frequently charge for those tokens — SSE
disconnect, mid-stream upstream error, client cancel all fall into this
bucket. Before this fix, the handler silently dropped the pending entry
and wrote NO row, so the ledger lost the cost.

After fix: write a ``cost_status=UNKNOWN`` row whose tokens/cost reflect
the partial usage, mixing with downstream ``actual`` rows in the
aggregate as ``partial``.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.services.cost_callback_handler import CostCallbackHandler

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _partial_response(
    content: str = "partial...",
    usage: dict | None = None,
) -> LLMResult:
    msg = AIMessage(content=content, usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


async def _start_pending(handler: CostCallbackHandler, run_id) -> None:
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )


async def test_error_with_partial_usage_writes_unknown_record_with_real_cost() -> None:
    """Mid-stream error + usage block → UNKNOWN record w/ actual token counts + cost."""
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    run_id = uuid4()
    await _start_pending(handler, run_id)

    response = _partial_response(
        content="half a sentence be",
        usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    await handler.on_llm_error(
        RuntimeError("upstream 503"),
        run_id=run_id,
        response=response,
    )
    await handler.flush_pending()

    assert len(captured) == 1, (
        "Partial stream + usage must result in a CostRecord so the "
        "ledger reflects provider-billed tokens. Got "
        f"{len(captured)} rows."
    )
    rec = captured[0]
    assert rec.cost_status == CostStatus.UNKNOWN, (
        "Mid-flight error must tag the row UNKNOWN so the session "
        "aggregate flips to partial; got cost_status=%r" % rec.cost_status
    )
    assert rec.input_tokens == 100
    assert rec.output_tokens == 50
    # gpt-4o @ openai_official: 100 input @ 2.5/M + 50 output @ 10/M
    # = 250/1M + 500/1M = 0.00025 + 0.00050 = 0.00075
    assert rec.total_usd == Decimal("0.00075")
    assert run_id not in handler.pending_keys()


async def test_error_with_partial_content_no_usage_writes_unknown_zero() -> None:
    """Partial content but no usage metadata → UNKNOWN with zero tokens."""
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    run_id = uuid4()
    await _start_pending(handler, run_id)

    response = _partial_response(content="some tokens leaked", usage=None)
    await handler.on_llm_error(
        RuntimeError("client disconnected"),
        run_id=run_id,
        response=response,
    )
    await handler.flush_pending()

    assert len(captured) == 1, (
        "Partial content must still produce a row (UNKNOWN + zero tokens) "
        "so the session is visibly degraded."
    )
    rec = captured[0]
    assert rec.cost_status == CostStatus.UNKNOWN
    assert rec.input_tokens == 0
    assert rec.output_tokens == 0


async def test_error_with_no_response_does_not_write_anything() -> None:
    """Early failure (connection error before any response): pop pending, no row."""
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    run_id = uuid4()
    await _start_pending(handler, run_id)

    await handler.on_llm_error(ConnectionError("boom"), run_id=run_id)
    await handler.flush_pending()

    assert len(captured) == 0, (
        "Early failure (no response kwarg) must not synthesize a row — "
        "nothing streamed means nothing to account for."
    )
    assert run_id not in handler.pending_keys()


async def test_error_with_partial_tool_calls_only_writes_unknown() -> None:
    """Streaming tool-call delta but empty content → still a partial payload.

    Providers bill tokens while emitting tool-call chunks; an empty
    ``content`` string must NOT cause the handler to treat this as an
    empty shell.
    """
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    run_id = uuid4()
    await _start_pending(handler, run_id)

    tool_call_only = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "search", "args": {"q": "hi"}}
        ],
    )
    response = LLMResult(
        generations=[[ChatGeneration(message=tool_call_only)]]
    )
    await handler.on_llm_error(
        RuntimeError("disconnect after tool call"),
        run_id=run_id,
        response=response,
    )
    await handler.flush_pending()

    assert len(captured) == 1, (
        "Partial tool-call payload (content='' + tool_calls=[...]) must "
        "still produce a CostRecord — _response_has_partial_payload needs "
        "to look past msg.content."
    )
    rec = captured[0]
    assert rec.cost_status == CostStatus.UNKNOWN


async def test_error_with_reasoning_only_additional_kwargs_writes_unknown() -> None:
    """``additional_kwargs`` (reasoning_content etc.) is billable output.

    Thinking-capable providers (Anthropic, DeepSeek Reasoner, o1-family)
    emit reasoning tokens into ``additional_kwargs``; ``content`` stays
    empty until the visible answer ships. A mid-flight error after only
    reasoning has streamed must not silently vanish from the ledger.
    """
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    run_id = uuid4()
    await _start_pending(handler, run_id)

    reasoning_only = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "let me think..."},
    )
    response = LLMResult(
        generations=[[ChatGeneration(message=reasoning_only)]]
    )
    await handler.on_llm_error(
        RuntimeError("disconnect during thinking"),
        run_id=run_id,
        response=response,
    )
    await handler.flush_pending()

    assert len(captured) == 1, (
        "Reasoning-only partial (content='' + additional_kwargs has "
        "content) must still produce a CostRecord."
    )
    assert captured[0].cost_status == CostStatus.UNKNOWN


async def test_error_with_empty_response_shell_does_not_write() -> None:
    """LLMResult with no content + no usage is an empty shell → skip."""
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    run_id = uuid4()
    await _start_pending(handler, run_id)

    empty_response = _partial_response(content="", usage=None)
    await handler.on_llm_error(
        RuntimeError("early crash"),
        run_id=run_id,
        response=empty_response,
    )
    await handler.flush_pending()

    assert len(captured) == 0
    assert run_id not in handler.pending_keys()
