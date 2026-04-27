"""B4 M0 post-audit: dict responses also populate usage_metadata.

``_agenerate`` explicitly supports the dict path (proxies / test doubles
routinely hand back a dict instead of a pydantic SDK model). Before this
fix, ``getattr(dict_response, "usage", None)`` always returned None and
the resulting CostRecord collapsed to ``cost_status=unknown``. The
audit-reproducer was: ``ainvoke`` + real CostCallbackHandler + dict
response with a valid usage block → CostRecord reports zero tokens.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def model() -> ActusResponsesModel:
    return ActusResponsesModel(
        base_url="https://api.test.com/v1",
        api_key="sk-test",
        model_name="test-model",
        temperature=0.5,
        max_tokens=1024,
    )


def _dict_response(
    text: str = "ok",
    input_tokens: int = 42,
    output_tokens: int = 8,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> dict[str, Any]:
    """A dict-shaped Responses API payload, as some proxies return."""
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cached_tokens is not None:
        usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
    if reasoning_tokens is not None:
        usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": usage,
    }


async def test_dict_response_populates_basic_usage(
    model: ActusResponsesModel,
) -> None:
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(
        return_value=_dict_response(input_tokens=42, output_tokens=8)
    )

    with patch.object(model, "_get_client", return_value=mock_client):
        result = await model._agenerate([HumanMessage(content="hi")])

    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    assert msg.usage_metadata is not None, (
        "dict response with a usage block must populate usage_metadata. "
        "getattr(dict, 'usage') returns None — the adapter needs a "
        "mapping-aware accessor."
    )
    assert msg.usage_metadata["input_tokens"] == 42
    assert msg.usage_metadata["output_tokens"] == 8
    assert msg.usage_metadata["total_tokens"] == 50


async def test_dict_response_surfaces_cached_and_reasoning_details(
    model: ActusResponsesModel,
) -> None:
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(
        return_value=_dict_response(
            input_tokens=100, output_tokens=50, cached_tokens=60, reasoning_tokens=30
        )
    )

    with patch.object(model, "_get_client", return_value=mock_client):
        result = await model._agenerate([HumanMessage(content="hi")])

    msg = result.generations[0].message
    assert msg.usage_metadata is not None
    input_details = msg.usage_metadata.get("input_token_details") or {}
    output_details = msg.usage_metadata.get("output_token_details") or {}
    assert input_details.get("cache_read") == 60
    assert output_details.get("reasoning") == 30


async def test_dict_response_end_to_end_via_cost_handler(
    model: ActusResponsesModel,
) -> None:
    """Public ``ainvoke`` + real CostCallbackHandler — the audit reproducer."""
    from app.domain.models.cost_record import CostRecord, CostStatus
    from app.domain.services.cost_callback_handler import CostCallbackHandler

    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="sess-dict", user_id="u", persister=persist
    )

    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(
        return_value=_dict_response(input_tokens=1000, output_tokens=500)
    )

    with patch.object(model, "_get_client", return_value=mock_client):
        await model.ainvoke(
            [HumanMessage(content="hi")],
            config={"callbacks": [handler]},
        )
    await handler.flush_pending()

    assert len(captured) == 1
    rec = captured[0]
    assert rec.input_tokens == 1000, (
        f"dict-response end-to-end must surface tokens; got {rec.input_tokens}. "
        "If this is 0, _resp_get / _agenerate's dict path regressed."
    )
    assert rec.output_tokens == 500
    # cost_status may be UNKNOWN because the fixture uses a test-only
    # model name not in PRICING_TABLE. What this test locks is that
    # tokens are captured; pricing is orthogonal (covered by the
    # registry-coverage gate).
    assert rec.cost_status in (CostStatus.ACTUAL, CostStatus.UNKNOWN)
