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


async def test_run_background_summary_parses_json_and_extracts_attachments():
    """When LLM returns valid JSON {message, attachments}, final event must use parsed text + File objects."""
    import json
    json_output = json.dumps({
        "message": "任务已完成。详细报告请查看附件。",
        "attachments": ["/home/ubuntu/report.md", "/home/ubuntu/data.csv"]
    })
    chunks = [AIMessageChunk(content=json_output)]

    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter(chunks))

    events = []
    async def on_event(evt):
        events.append(evt)

    result = await run_background_summary([], mock_llm, on_event)

    # Result should be the parsed message text, not raw JSON
    assert result == "任务已完成。详细报告请查看附件。"

    # Final event (partial=False) should have parsed text + File attachments
    finals = [e for e in events if not e.partial]
    assert len(finals) == 1
    final = finals[0]
    assert final.message == "任务已完成。详细报告请查看附件。"
    assert len(final.attachments) == 2
    assert final.attachments[0].filepath == "/home/ubuntu/report.md"
    assert final.attachments[0].filename == "report.md"
    assert final.attachments[1].filepath == "/home/ubuntu/data.csv"
    assert final.attachments[1].filename == "data.csv"


async def test_run_background_summary_non_json_uses_raw_text():
    """When LLM returns non-JSON text, final event uses raw text with no attachments."""
    chunks = [AIMessageChunk(content="这是一段纯文本总结，没有JSON格式。")]

    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter(chunks))

    events = []
    async def on_event(evt):
        events.append(evt)

    result = await run_background_summary([], mock_llm, on_event)

    assert result == "这是一段纯文本总结，没有JSON格式。"
    finals = [e for e in events if not e.partial]
    assert len(finals) == 1
    assert finals[0].attachments == []  # no attachments for non-JSON
