"""A4-2: AgentService._sse_or_db_sink — the demoted caller-owned sink for
SSM.emit_session_mode_changed on the HTTP-takeover paths. Live put + DB persist,
degrade on put failure, always persist. (Construction now lives in the SSM /
mode_event builder; this sink only dispatches a pre-built event.)"""
import json

import pytest

from app.application.services.agent_service import AgentService
from app.domain.models.event import SessionModeChangedEvent
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


def _event(
    *, to="takeover", from_mode="running", reason="takeover_started", rev=12
) -> SessionModeChangedEvent:
    return SessionModeChangedEvent(
        to=to, from_mode=from_mode, reason=reason, mode_revision=rev
    )


async def test_sink_puts_to_stream_and_persists() -> None:
    uow = _Uow()
    service = _make_service(uow)
    task = _Task()

    sink = service._sse_or_db_sink(task)
    await sink("s1", _event(rev=12))

    payload = json.loads(task.output_stream.payloads[0])
    assert payload["type"] == "session_mode_changed"
    assert payload["to"] == "takeover"
    assert payload["mode_revision"] == 12
    # Always persisted for replay.
    assert len(uow.session.add_event_calls) == 1
    assert uow.session.add_event_calls[0][1].to == "takeover"


async def test_db_only_sink_persists_without_stream() -> None:
    uow = _Uow()
    service = _make_service(uow)

    sink = service._sse_or_db_sink(None)
    await sink(
        "s1",
        _event(to="takeover_pending", from_mode=None, reason="takeover_reopened", rev=3),
    )

    assert len(uow.session.add_event_calls) == 1


async def test_sink_stream_put_failure_degrades_to_db_only() -> None:
    uow = _Uow()
    service = _make_service(uow)
    task = _Task(fail=True)

    # Must not raise — degrade to DB-only.
    sink = service._sse_or_db_sink(task)
    await sink("s1", _event(to="running", from_mode="takeover", reason="takeover_ended", rev=5))

    assert task.output_stream.payloads == []
    assert len(uow.session.add_event_calls) == 1
