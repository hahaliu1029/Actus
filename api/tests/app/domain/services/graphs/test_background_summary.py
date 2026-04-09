"""Test run_background_summary standalone function."""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessageChunk

from app.domain.services.graphs.background_summary import run_background_summary

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _async_iter(items):
    for item in items:
        yield item


async def test_run_background_summary_streams_and_finalizes():
    """Summary function must emit partial=True chunks then partial=False final."""
    chunks = [
        AIMessageChunk(content="Hello"),
        AIMessageChunk(content=" world"),
    ]

    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter(chunks))

    events = []
    async def on_event(evt):
        events.append(evt)

    result = await run_background_summary([], mock_llm, on_event)

    assert result == "Hello world"
    partials = [e for e in events if e.partial]
    finals = [e for e in events if not e.partial]
    assert len(partials) == 2
    assert len(finals) == 1
    assert finals[0].message == "Hello world"


async def test_run_background_summary_empty_response():
    """If LLM returns no content, result is None."""
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([AIMessageChunk(content="")]))

    events = []
    async def on_event(evt):
        events.append(evt)

    result = await run_background_summary([], mock_llm, on_event)

    assert result is None
    assert len(events) == 0
