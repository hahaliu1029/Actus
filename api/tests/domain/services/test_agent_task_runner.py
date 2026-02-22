from app.domain.models.event import MessageEvent
from app.domain.models.file import File
from app.domain.services.agent_task_runner import AgentTaskRunner


class _DummyInputStream:
    def __init__(self, event_json: str) -> None:
        self._event_json = event_json

    async def pop(self) -> tuple[str, str]:
        return ("1-0", self._event_json)


class _DummyTask:
    def __init__(self, event_json: str) -> None:
        self.input_stream = _DummyInputStream(event_json)


async def test_pop_event_returns_parsed_event_instance() -> None:
    event_json = MessageEvent(
        role="user",
        message="hello",
        attachments=[File(id="file-1")],
    ).model_dump_json()
    task = _DummyTask(event_json)

    event = await AgentTaskRunner._pop_event(task)

    assert isinstance(event, MessageEvent)
    assert event.id == "1-0"
    assert [attachment.id for attachment in event.attachments] == ["file-1"]
