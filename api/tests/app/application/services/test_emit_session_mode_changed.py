"""A4-0: _emit_session_mode_changed helper — live put + DB persist, degrade on
put failure, server-fixed payload."""
import json

import pytest

from app.application.services.agent_service import AgentService
from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self) -> None:
        self.add_event_calls: list = []

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))


class _Uow:
    def __init__(self) -> None:
        self.session = _SessionRepo()

    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, *exc) -> None:
        return None


class _OutputStream:
    def __init__(self, *, fail: bool = False) -> None:
        self.payloads: list[str] = []
        self._fail = fail

    async def put(self, payload: str) -> str:
        if self._fail:
            raise RuntimeError("stream closed")
        self.payloads.append(payload)
        return "evt-1"


class _Task:
    def __init__(self, *, fail: bool = False) -> None:
        self.output_stream = _OutputStream(fail=fail)


def _make_service(uow: _Uow) -> AgentService:
    return AgentService(
        uow_factory=lambda: uow,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
    )


async def test_live_site_puts_to_stream_and_persists() -> None:
    uow = _Uow()
    service = _make_service(uow)
    task = _Task()

    await service._emit_session_mode_changed(
        "s1", to="takeover", reason="takeover_started",
        mode_revision=12, from_mode="running", task=task,
    )

    payload = json.loads(task.output_stream.payloads[0])
    assert payload["type"] == "session_mode_changed"
    assert payload["to"] == "takeover"
    assert payload["mode_revision"] == 12
    # Always persisted for replay.
    assert len(uow.session.add_event_calls) == 1
    assert uow.session.add_event_calls[0][1].to == "takeover"


async def test_db_only_site_persists_without_stream() -> None:
    uow = _Uow()
    service = _make_service(uow)

    await service._emit_session_mode_changed(
        "s1", to="takeover_pending", reason="takeover_reopened",
        mode_revision=3, from_mode=None, task=None,
    )

    assert len(uow.session.add_event_calls) == 1


async def test_stream_put_failure_degrades_to_db_only() -> None:
    uow = _Uow()
    service = _make_service(uow)
    task = _Task(fail=True)

    # Must not raise — degrade to DB-only.
    await service._emit_session_mode_changed(
        "s1", to="running", reason="takeover_ended",
        mode_revision=5, from_mode="takeover", task=task,
    )

    assert task.output_stream.payloads == []
    assert len(uow.session.add_event_calls) == 1
