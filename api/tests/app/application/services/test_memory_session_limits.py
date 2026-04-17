"""Tests for memory_session_limits helper (PR-3).

memory_save 工具会在 Redis 里维护 per-session 写入计数——20/session 硬
上限，避免 Agent 在一次任务里胡乱保存把 memory 打爆。与 memory_quota 的
per-user 每日上限正交：session 维度主要防 "同一轮对话" 内的 flood，daily
quota 防跨 session 汇总的滥用。

覆盖：
  1. pipeline(INCR+EXPIRE) 成功 → 正常返回，TTL 总是刷新
  2. INCR > cap → QuotaExceededError，bucket=memory_session_save
  3. pipeline 整体失败 → fail-open (返回 0 + warn log)
  4. key 形状包含 session_id（确保不同 session 不共享 counter）
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.errors.exceptions import QuotaExceededError
from app.application.services.memory_session_limits import (
    check_and_increment_session_save,
    refund_session_save,
)

pytestmark = pytest.mark.anyio

_TEST_SESSION_ID = "sess-01-abcdef"


def _make_redis(*, incr_result: int = 1, pipeline_raises: Exception | None = None) -> AsyncMock:
    """构造模拟 pipeline(transaction=True) 的 Redis AsyncMock。

    与 test_memory_quota._make_redis 对称——两者走同一种 pipeline 语义，分开
    helper 只是为了测试隔离，不想让 session 测试间接依赖 quota 测试的辅助。
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
    redis._pipe = pipe
    return redis


class TestCheckAndIncrementSessionSave:
    async def test_increment_refreshes_ttl_every_call(self) -> None:
        redis = _make_redis(incr_result=1)

        current = await check_and_increment_session_save(
            redis, _TEST_SESSION_ID, cap=20
        )
        assert current == 1
        redis._pipe.incr.assert_called_once()
        redis._pipe.expire.assert_called_once()
        # session counter 也是 25h TTL，与 user_daily 对齐（跨时区边界）
        assert redis._pipe.expire.call_args.args[1] == 90000

    async def test_at_cap_passes(self) -> None:
        """计数等于 cap → 通过（cap 是包含上界，第 20 次允许）。"""
        redis = _make_redis(incr_result=20)
        current = await check_and_increment_session_save(
            redis, _TEST_SESSION_ID, cap=20
        )
        assert current == 20

    async def test_over_cap_raises_quota_exceeded(self) -> None:
        redis = _make_redis(incr_result=21)
        with pytest.raises(QuotaExceededError) as exc_info:
            await check_and_increment_session_save(
                redis, _TEST_SESSION_ID, cap=20
            )
        assert exc_info.value.status_code == 429
        assert exc_info.value.data["limit"] == 20
        assert exc_info.value.data["bucket"] == "memory_session_save"

    async def test_pipeline_failure_fails_open(self) -> None:
        """Redis 挂了 → fail-open 放行（返回 0 + warn log）。

        与 memory_quota 对称：前置 rate limiter 会在 Redis 全挂时先抛 503，
        走到这里时 Redis 可用性基本已验证过；fail-closed 只会把单点故障放大。
        """
        redis = _make_redis(pipeline_raises=Exception("redis down"))
        current = await check_and_increment_session_save(
            redis, _TEST_SESSION_ID, cap=20
        )
        assert current == 0

    async def test_key_includes_session_id(self) -> None:
        """key 必须带 session_id 否则两个 session 会共享 counter。"""
        redis = _make_redis(incr_result=1)
        await check_and_increment_session_save(
            redis, _TEST_SESSION_ID, cap=20
        )
        key = redis._pipe.incr.call_args.args[0]
        assert _TEST_SESSION_ID in key
        assert key.startswith("memory:session_save:")


class TestRefundSessionSave:
    async def test_refund_calls_decr(self) -> None:
        redis = AsyncMock()
        await refund_session_save(redis, _TEST_SESSION_ID)
        redis.decr.assert_awaited_once()
        key = redis.decr.call_args.args[0]
        assert _TEST_SESSION_ID in key

    async def test_refund_swallows_redis_failure(self) -> None:
        """refund 失败不应传染主流程——仅 warn log，函数正常返回。"""
        redis = AsyncMock()
        redis.decr.side_effect = Exception("redis down")
        # 不应抛异常
        await refund_session_save(redis, _TEST_SESSION_ID)
