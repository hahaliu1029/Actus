"""D1a §3.6 startup reconcile advisory lock 单跑者语义（需 PostgreSQL）。

两个并发 ``pg_try_advisory_lock(74520011)`` 会话——先到者 True、后到者 False（单跑者，
lifespan 层据此 skip 第二个 pod 的 reconcile）；先到者 unlock 后，后到者可重获。

advisory lock 是 **连接（会话）级**，故用两条独立 ``async_engine.connect()`` 连接模拟
两个 pod；同连接 lock/unlock 保证锁-解锁亲和性。

**集成未本地跑，CI 验证。**

Run:
    cd api && uv run pytest tests/integration/governance/test_startup_reconcile_lock.py -v
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from app.application.services.extension_reconciler import (
    D1A_STARTUP_RECONCILE_LOCK_KEY,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_second_pod_blocked_until_unlock(async_engine) -> None:
    key = D1A_STARTUP_RECONCILE_LOCK_KEY
    async with async_engine.connect() as c1, async_engine.connect() as c2:
        # 先到 pod 拿到锁
        got1 = (await c1.execute(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar()
        assert got1 is True

        # 并发第二个 pod 拿不到（单跑者语义 → lifespan skip 其 reconcile）
        got2 = (await c2.execute(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar()
        assert got2 is False

        # 先到 pod 解锁后，第二个 pod 可重获（下一轮启动可跑）
        await c1.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
        got2b = (await c2.execute(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar()
        assert got2b is True

        # 清理：释放第二把锁（避免污染同连接池后续用例）
        await c2.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
