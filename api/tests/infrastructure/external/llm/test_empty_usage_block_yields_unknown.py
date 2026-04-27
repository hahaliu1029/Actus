"""B4 M0 post-audit: empty usage blocks are rejected → CostRecord = unknown, not actual $0.

The audit reproducer: providers sometimes hand back a ``usage`` payload
that's technically non-None but has no token counters (``{}`` / bare
``SimpleNamespace()`` / every counter is None). The old extractors
defaulted each field to 0 and returned
``{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}``. The
handler then marked the row ``cost_status=actual, total_usd=0`` — a
silent undercount that looks like a clean "free" call.

Fix: extractors require at least one authoritative counter to be
present. Otherwise they return ``None``, the handler stamps
``cost_status=unknown``, and aggregation surfaces the gap honestly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.domain.services.cost_callback_handler import CostCallbackHandler
from app.domain.models.cost_record import CostRecord, CostStatus
from app.infrastructure.external.llm.actus_chat_model import (
    ActusChatModel,
    _extract_usage_metadata,
)
from app.infrastructure.external.llm.actus_responses_model import (
    ActusResponsesModel,
    _extract_responses_usage_metadata,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Pure extractor tests — tightest proof of the fix
# ---------------------------------------------------------------------------


def test_chat_extractor_returns_none_for_empty_object() -> None:
    assert _extract_usage_metadata(SimpleNamespace()) is None


def test_chat_extractor_returns_none_when_all_counters_are_none() -> None:
    usage = SimpleNamespace(
        prompt_tokens=None, completion_tokens=None, total_tokens=None
    )
    assert _extract_usage_metadata(usage) is None


def test_chat_extractor_accepts_real_zero_output() -> None:
    """A provider legitimately returning 'zero output' is ACTUAL, not unknown."""
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=0, total_tokens=5)
    meta = _extract_usage_metadata(usage)
    assert meta is not None
    assert meta["input_tokens"] == 5
    assert meta["output_tokens"] == 0


def test_responses_extractor_returns_none_for_empty_dict() -> None:
    assert _extract_responses_usage_metadata({}) is None


def test_responses_extractor_returns_none_for_empty_object() -> None:
    assert _extract_responses_usage_metadata(SimpleNamespace()) is None


def test_responses_extractor_returns_none_when_all_counters_are_none() -> None:
    assert (
        _extract_responses_usage_metadata(
            {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        )
        is None
    )


def test_responses_extractor_accepts_real_zero_output() -> None:
    meta = _extract_responses_usage_metadata(
        {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}
    )
    assert meta is not None
    assert meta["output_tokens"] == 0


# ---------------------------------------------------------------------------
# End-to-end: audit reproducer — empty usage → CostRecord.cost_status=unknown
# ---------------------------------------------------------------------------


def _chat_response_with_usage(usage_obj: Any):
    message = SimpleNamespace(role="assistant", content="hi", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        id="chatcmpl-x", choices=[choice], model="test", usage=usage_obj
    )


def _responses_payload_with_usage(usage: Any) -> dict:
    return {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
        ],
        "usage": usage,
    }


async def test_chat_empty_usage_yields_unknown_cost_record() -> None:
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    model = ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="sk",
        model_name="gpt-4o",
        supports_response_format=True,
    )
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_chat_response_with_usage(SimpleNamespace())
    )
    with patch.object(model, "_get_client", return_value=mock_client):
        await model.ainvoke(
            [HumanMessage(content="hi")],
            config={"callbacks": [handler]},
        )
    await handler.flush_pending()

    assert len(captured) == 1
    rec = captured[0]
    assert rec.cost_status == CostStatus.UNKNOWN, (
        "Chat Completions with empty usage must mark CostRecord unknown; "
        f"got {rec.cost_status!r}. An empty usage block faking ACTUAL $0 "
        "is the silent-undercount the audit flagged."
    )
    assert rec.input_tokens == 0
    assert rec.output_tokens == 0
    assert rec.total_usd == 0


async def test_responses_empty_dict_usage_yields_unknown_cost_record() -> None:
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    model = ActusResponsesModel(
        base_url="https://api.test.com/v1",
        api_key="sk",
        model_name="test-model",
    )
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(
        return_value=_responses_payload_with_usage({})
    )
    with patch.object(model, "_get_client", return_value=mock_client):
        await model.ainvoke(
            [HumanMessage(content="hi")],
            config={"callbacks": [handler]},
        )
    await handler.flush_pending()

    assert len(captured) == 1
    assert captured[0].cost_status == CostStatus.UNKNOWN


async def test_responses_empty_object_usage_yields_unknown_cost_record() -> None:
    captured: list[CostRecord] = []

    async def persist(rec: CostRecord) -> None:
        captured.append(rec)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    model = ActusResponsesModel(
        base_url="https://api.test.com/v1",
        api_key="sk",
        model_name="test-model",
    )
    response = SimpleNamespace(
        model_dump=lambda: {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ]
        },
        usage=SimpleNamespace(),
    )
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=response)
    with patch.object(model, "_get_client", return_value=mock_client):
        await model.ainvoke(
            [HumanMessage(content="hi")],
            config={"callbacks": [handler]},
        )
    await handler.flush_pending()

    assert len(captured) == 1
    assert captured[0].cost_status == CostStatus.UNKNOWN
