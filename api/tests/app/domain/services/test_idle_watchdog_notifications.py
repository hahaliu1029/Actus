from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models.session import SessionStatus
from app.domain.repositories.session_repository import BgSessionRow
from app.domain.services.idle_watchdog import IdleWatchdog

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _Repo:
    def __init__(self, rows: list[BgSessionRow] | None = None) -> None:
        self._rows = rows or []

    async def distinct_user_ids_with_running_bg(self) -> list[str]:
        return ["user-1"]

    async def find_running_background(self) -> list[BgSessionRow]:
        return self._rows


class _Supervisor:
    def __init__(self, *, inflight_counts: tuple[int, int] = (0, 0)) -> None:
        self.terminate_calls: list[dict] = []
        self.suspend_calls: list[dict] = []
        self.inflight_counts = inflight_counts
        self.inflight_count_calls: list[str] = []
        self.sweep_now: datetime | None = None
        self.global_reconcile_now: datetime | None = None

    async def reconcile_global_background_memberships(
        self,
        *,
        reconcile_now: datetime,
    ) -> dict[str, int]:
        self.global_reconcile_now = reconcile_now
        return {"cleaned": 0, "repaired": 0, "skipped": 0}

    async def list_background_slot_user_ids(self) -> list[str]:
        return []

    async def sweep_expired(
        self,
        *,
        user_id: str,
        sweep_now: datetime,
    ) -> list[str]:
        assert user_id == "user-1"
        self.sweep_now = sweep_now
        return ["session-expired"]

    async def terminate_expired_background(self, **kwargs) -> None:
        self.terminate_calls.append(kwargs)

    async def suspend_idle(self, **kwargs) -> None:
        self.suspend_calls.append(kwargs)

    async def get_inflight_counts(self, *, session_id: str) -> tuple[int, int]:
        self.inflight_count_calls.append(session_id)
        return self.inflight_counts


class _Redis:
    def __init__(self, *, last_activity_at: float | None) -> None:
        self._last_activity_at = last_activity_at

    async def hget(self, key: str, field: str):
        assert key == "supervisor:hot:session-idle"
        assert field == "last_activity_at"
        if self._last_activity_at is None:
            return None
        return f"{self._last_activity_at:.6f}"


class _TouchRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, *, mapping: dict[str, str]) -> None:
        self.hashes.setdefault(key, {}).update(mapping)

    async def expire(self, key: str, seconds: int) -> None:
        assert key == "supervisor:hot:session-idle"
        assert seconds == 300

    async def hget(self, key: str, field: str) -> str | None:
        return self.hashes.get(key, {}).get(field)


async def test_sweep_expired_uses_same_timestamp_for_redis_and_pg_cas() -> None:
    supervisor = _Supervisor()
    emitter = object()
    watchdog = IdleWatchdog(
        redis_client=object(),
        supervisor=supervisor,
        session_repository=_Repo(),
        notification_emitter=emitter,
    )

    await watchdog._sweep_expired_once()

    assert supervisor.sweep_now is not None
    assert supervisor.global_reconcile_now == supervisor.sweep_now
    assert supervisor.terminate_calls == [
        {
            "session_id": "session-expired",
            "user_id": "user-1",
            "sweep_now": supervisor.sweep_now,
            "notification_emitter": emitter,
        }
    ]


async def test_scan_once_suspends_stale_running_background_session() -> None:
    stale_activity = datetime.now(timezone.utc).timestamp() - 10
    supervisor = _Supervisor()
    watchdog = IdleWatchdog(
        redis_client=_Redis(last_activity_at=stale_activity),
        supervisor=supervisor,
        session_repository=_Repo(
            [
                BgSessionRow(
                    session_id="session-idle",
                    task_id="task-1",
                    user_id="user-1",
                    status=SessionStatus.RUNNING,
                )
            ]
        ),
        idle_timeout_seconds=1,
    )

    await watchdog._scan_once()

    assert supervisor.suspend_calls == [
        {"session_id": "session-idle", "user_id": "user-1"}
    ]


async def test_heartbeat_touch_still_prevents_180_second_idle_suspend() -> None:
    redis = _TouchRedis()
    supervisor = _Supervisor()
    watchdog = IdleWatchdog(
        redis_client=redis,
        supervisor=supervisor,
        session_repository=_Repo(
            [
                BgSessionRow(
                    session_id="session-idle",
                    task_id="task-1",
                    user_id="user-1",
                    status=SessionStatus.RUNNING,
                )
            ]
        ),
        idle_timeout_seconds=180,
    )

    await watchdog.touch_activity(session_id="session-idle")
    await watchdog._scan_once()

    assert supervisor.suspend_calls == []


@pytest.mark.parametrize("inflight_counts", [(1, 0), (-1, 1), (1, -1)])
async def test_scan_once_skips_stale_background_session_with_inflight_work(
    inflight_counts: tuple[int, int],
) -> None:
    stale_activity = datetime.now(timezone.utc).timestamp() - 10
    supervisor = _Supervisor(inflight_counts=inflight_counts)
    watchdog = IdleWatchdog(
        redis_client=_Redis(last_activity_at=stale_activity),
        supervisor=supervisor,
        session_repository=_Repo(
            [
                BgSessionRow(
                    session_id="session-idle",
                    task_id="task-1",
                    user_id="user-1",
                    status=SessionStatus.RUNNING,
                )
            ]
        ),
        idle_timeout_seconds=1,
    )

    await watchdog._scan_once()

    assert supervisor.inflight_count_calls == ["session-idle"]
    assert supervisor.suspend_calls == []
