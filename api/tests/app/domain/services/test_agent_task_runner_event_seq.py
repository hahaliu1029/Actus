from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.event import MessageEvent
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_put_and_add_event_stamps_seq_on_real_output_path() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "sess-1"
    runner._event_seq_client = MagicMock()
    runner._event_seq_client.incr = AsyncMock(return_value=7)
    runner._event_seq_client.expire = AsyncMock()
    runner._event_seq_ttl_seconds = 86400

    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=False)
    uow.session = MagicMock()
    uow.session.add_event = AsyncMock()
    runner._uow = uow

    task = MagicMock()
    task.output_stream = MagicMock()
    task.output_stream.put = AsyncMock(return_value="1000-7")

    event = MessageEvent(role="assistant", message="seq-bearing")

    await runner._put_and_add_event(task, event, persist=True)

    runner._event_seq_client.incr.assert_awaited_once_with("session:seq:sess-1")
    runner._event_seq_client.expire.assert_not_awaited()
    payload = task.output_stream.put.await_args.args[0]
    assert '"seq":7' in payload
    assert event.seq == 7
    assert event.id == "1000-7"
    uow.session.add_event.assert_awaited_once_with("sess-1", event)
