from __future__ import annotations

import pytest

from app.domain.models.session import SessionStatus
from app.domain.services.idle_watchdog import IdleWatchdog

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _Repo:
    async def distinct_user_ids_with_running_bg(self) -> list[str]:
        return ["user-1"]


class _Supervisor:
    def __init__(self) -> None:
        self.terminate_calls: list[dict] = []

    async def list_background_slot_user_ids(self) -> list[str]:
        return []

    async def sweep_expired(self, *, user_id: str) -> list[str]:
        assert user_id == "user-1"
        return ["session-expired"]

    async def terminate(self, **kwargs) -> None:
        self.terminate_calls.append(kwargs)


async def test_sweep_expired_forwards_notification_emitter_to_terminate() -> None:
    supervisor = _Supervisor()
    emitter = object()
    watchdog = IdleWatchdog(
        redis_client=object(),
        supervisor=supervisor,
        session_repository=_Repo(),
        notification_emitter=emitter,
    )

    await watchdog._sweep_expired_once()

    assert supervisor.terminate_calls == [
        {
            "session_id": "session-expired",
            "user_id": "user-1",
            "terminal_reason": "watchdog_timeout",
            "status": SessionStatus.TIMED_OUT,
            "notification_emitter": emitter,
        }
    ]
