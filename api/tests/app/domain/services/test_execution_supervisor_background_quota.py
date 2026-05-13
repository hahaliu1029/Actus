from __future__ import annotations

import pytest

from app.domain.services.execution_supervisor import ExecutionSupervisor

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Redis:
    def __init__(self) -> None:
        self.values = {
            "supervisor:system:bg_count": b"3",
        }
        self.hash_lengths = {
            "supervisor:user:u1": 2,
        }

    async def get(self, key: str) -> bytes | str | None:
        return self.values.get(key)

    async def hlen(self, key: str) -> int:
        return self.hash_lengths.get(key, 0)


async def test_get_background_quota_reads_redis_counts_and_limits() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_Redis(),
        session_repository=object(),
        max_system_bg=7,
        max_user_bg=5,
    )

    quota = await supervisor.get_background_quota("u1")

    assert quota == {
        "system_used": 3,
        "system_limit": 7,
        "user_used": 2,
        "user_limit": 5,
    }


async def test_get_background_quota_treats_missing_counts_as_zero() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_Redis(),
        session_repository=object(),
        max_system_bg=7,
        max_user_bg=5,
    )
    supervisor._redis.values.clear()

    quota = await supervisor.get_background_quota("new-user")

    assert quota["system_used"] == 0
    assert quota["user_used"] == 0
