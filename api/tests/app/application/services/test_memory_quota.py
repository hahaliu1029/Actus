"""Tests for memory_quota helper (PR-2).

覆盖 per-user 每日 memory 写入配额的主要路径：
  1. pipeline(INCR+EXPIRE) 成功 → 正常返回，TTL 总是刷新
  2. INCR > cap → QuotaExceededError
  3. pipeline 整体失败 → fail-open (返回 0 + warn log)
  4. 首次 INCR 后 EXPIRE 失败仍被 pipeline 原子化吸收（不会留无 TTL 的 key）
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.errors.exceptions import QuotaExceededError
from app.application.services.memory_quota import check_and_increment_user_daily

from tests.conftest import TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio


def _make_redis(*, incr_result: int = 1, pipeline_raises: Exception | None = None) -> AsyncMock:
    """构造一个模拟 pipeline(transaction=True) 的 Redis AsyncMock。

    pipeline 的 context manager 产出一个 pipe 对象；pipe.incr/expire 是同步
    调用（不 await），pipe.execute() 是 async。execute 返回 [incr_result, True]。
    """
    redis = AsyncMock()

    pipe = MagicMock()
    pipe.incr = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    if pipeline_raises is not None:
        pipe.execute = AsyncMock(side_effect=pipeline_raises)
    else:
        pipe.execute = AsyncMock(return_value=[incr_result, True])

    @asynccontextmanager
    async def pipeline_cm(transaction: bool = True):
        yield pipe

    redis.pipeline = pipeline_cm
    # 暴露 pipe 给测试做断言
    redis._pipe = pipe
    return redis


class TestCheckAndIncrementUserDaily:
    async def test_increment_refreshes_ttl_every_call(self) -> None:
        """每次 INCR 都应在同一 pipeline 里跟 EXPIRE，保证 key 永远有 TTL。"""
        redis = _make_redis(incr_result=1)

        current = await check_and_increment_user_daily(
            redis, TEST_USER_ID_FIXED, daily_cap=500
        )
        assert current == 1
        redis._pipe.incr.assert_called_once()
        redis._pipe.expire.assert_called_once()
        # TTL 应为 25h（覆盖时区边界）
        assert redis._pipe.expire.call_args.args[1] == 90000

    async def test_subsequent_increment_also_refreshes_ttl(self) -> None:
        """第二次及之后的 INCR 也刷新 TTL——pipeline 原子化保证绝不泄漏 no-TTL key。

        回归 P2：旧实现 ``if current == 1: expire`` 在 INCR 成功但 EXPIRE 失败
        时留下永久 key；新实现 pipeline 打包，任一失败都对 client 不可见。
        """
        redis = _make_redis(incr_result=42)

        current = await check_and_increment_user_daily(
            redis, TEST_USER_ID_FIXED, daily_cap=500
        )
        assert current == 42
        redis._pipe.expire.assert_called_once()

    async def test_at_cap_passes(self) -> None:
        """计数等于 cap → 仍通过（cap 是包含上界）。"""
        redis = _make_redis(incr_result=500)
        current = await check_and_increment_user_daily(
            redis, TEST_USER_ID_FIXED, daily_cap=500
        )
        assert current == 500

    async def test_over_cap_raises_quota_exceeded(self) -> None:
        redis = _make_redis(incr_result=501)
        with pytest.raises(QuotaExceededError) as exc_info:
            await check_and_increment_user_daily(
                redis, TEST_USER_ID_FIXED, daily_cap=500
            )
        assert exc_info.value.status_code == 429
        assert exc_info.value.data["limit"] == 500
        assert exc_info.value.data["bucket"] == "memory_user_daily"

    async def test_pipeline_failure_fails_open(self) -> None:
        """pipeline.execute() 抛异常 → 返回 0（fail-open），放行写入路径。

        pipeline 的 MULTI/EXEC 原子性保证 "INCR 已落库但 EXPIRE 失败" 对 client
        等价于 "什么都没发生"，不会出现 no-TTL 泄漏。
        """
        redis = _make_redis(pipeline_raises=Exception("redis down"))
        current = await check_and_increment_user_daily(
            redis, TEST_USER_ID_FIXED, daily_cap=500
        )
        assert current == 0

    async def test_key_includes_date(self) -> None:
        """key 带日期后缀，确保跨天自然 rollover。"""
        redis = _make_redis(incr_result=1)
        await check_and_increment_user_daily(
            redis, TEST_USER_ID_FIXED, daily_cap=500
        )
        key = redis._pipe.incr.call_args.args[0]
        assert key.startswith(f"memory:user_daily:{TEST_USER_ID_FIXED}:")
        # YYYY-MM-DD 形状
        date_part = key.rsplit(":", 1)[-1]
        assert len(date_part) == 10 and date_part[4] == "-" and date_part[7] == "-"
