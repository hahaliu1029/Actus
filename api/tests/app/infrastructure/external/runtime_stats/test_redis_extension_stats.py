"""B9 Task 19 — RedisExtensionStats（队列/flusher/key builder）单元测试。

不引入 fakeredis：用手搓的 fake redis client + fake pipeline（对齐 brief Step 1
"用 MagicMock/AsyncMock pipeline" 的口径），以便逐参数断言 HINCRBY/HSET 调用。
attribution 注册表由 conftest 的 ``_clear_extension_attribution`` autouse fixture
在每个测试前后清空；本文件用 ``register_extension_tool`` 手动注册归因。
"""
from __future__ import annotations

import inspect
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.external.extension_stats import ExtensionStatsData
from app.domain.services.tools.extension_attribution import register_extension_tool
from app.infrastructure.external.runtime_stats.redis_extension_stats import (
    STATS_FLUSH_BATCH_SIZE,
    STATS_QUEUE_MAXSIZE,
    RedisExtensionStats,
    build_stats_key,
)

pytestmark = pytest.mark.anyio


class _FakePipeline:
    """记录 hincrby/hset/delete 调用、async execute() 可注入抛错的假 pipeline。"""

    def __init__(self, *, raise_on_execute: bool = False) -> None:
        self.hincrby_calls: list[tuple] = []
        self.hset_calls: list[tuple] = []
        self.delete_calls: list[tuple] = []
        self.executed = 0
        self._raise_on_execute = raise_on_execute

    def hincrby(self, key, field, amount=1):
        self.hincrby_calls.append((key, field, amount))
        return self

    def hset(self, key, field=None, value=None, mapping=None):
        self.hset_calls.append((key, field, value, mapping))
        return self

    def delete(self, *keys):
        self.delete_calls.append(keys)
        return self

    async def execute(self):
        self.executed += 1
        if self._raise_on_execute:
            raise RuntimeError("boom pipeline execute")
        return []


class _FakeRedis:
    """最小假 redis client：pipeline() 返回单例 _FakePipeline；hgetall 走 AsyncMock。"""

    def __init__(self, *, raise_on_execute: bool = False) -> None:
        self.pipe = _FakePipeline(raise_on_execute=raise_on_execute)
        self.hgetall = AsyncMock(return_value={})

    def pipeline(self, transaction: bool = True):
        return self.pipe


async def _drain_once(stats: RedisExtensionStats) -> None:
    """跑一轮 flusher 的批量落盘（不起后台循环，直接调私有单批 helper）。"""
    await stats._flush_once()  # noqa: SLF001 — 测试直接驱动单批 drain


# ---------------------------------------------------------------------------
# 1. 未归因工具直接丢弃，不入队
# ---------------------------------------------------------------------------
async def test_record_unresolved_tool_dropped_no_queue():
    stats = RedisExtensionStats(_FakeRedis())
    stats.record("some_unregistered_tool", success=True, latency_ms=12.0)
    assert stats._queue.qsize() == 0
    assert stats.dropped_count == 0  # 未归因不计入 queue-full 丢弃


# ---------------------------------------------------------------------------
# 2. 队列满：丢弃 + 计数器自增
# ---------------------------------------------------------------------------
async def test_record_queue_full_drops_with_counter(monkeypatch):
    register_extension_tool("mcp_tool_a", "mcp", "server-a")
    stats = RedisExtensionStats(_FakeRedis())
    # 把有界队列缩到 2 —— 第 3 条 record 触发 QueueFull 丢弃。
    import asyncio

    monkeypatch.setattr(stats, "_queue", asyncio.Queue(maxsize=2))
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    assert stats._queue.qsize() == 2
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)  # 溢出
    assert stats._queue.qsize() == 2
    assert stats.dropped_count == 1


# ---------------------------------------------------------------------------
# 3. flusher 批量聚合 → HINCRBY / HSET
# ---------------------------------------------------------------------------
async def test_flush_batch_aggregates_hincrby_and_hset():
    register_extension_tool("mcp_tool_a", "mcp", "server-a")
    fake = _FakeRedis()
    stats = RedisExtensionStats(fake)
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    stats.record("mcp_tool_a", success=False, latency_ms=1.0)  # 1 次失败

    await _drain_once(stats)

    key = build_stats_key("mcp", "server-a")
    # 同 key 聚合后：call_count +3、success_count +2、failure_count +1
    hincrby_by_field = {(k, f): amt for (k, f, amt) in fake.pipe.hincrby_calls}
    assert hincrby_by_field[(key, "call_count")] == 3
    assert hincrby_by_field[(key, "success_count")] == 2
    assert hincrby_by_field[(key, "failure_count")] == 1
    # HSET 至少写 last_active_at + last_success_at + last_failure_at（本批既有成功又有失败）
    hset_fields = set()
    for (hkey, field, value, mapping) in fake.pipe.hset_calls:
        if mapping:
            hset_fields.update(mapping.keys())
        elif field is not None:
            hset_fields.add(field)
    assert {"last_active_at", "last_success_at", "last_failure_at"} <= hset_fields
    assert fake.pipe.executed == 1


# ---------------------------------------------------------------------------
# 4. key builder：敌意 id 哈希，格式 + 幂等
# ---------------------------------------------------------------------------
def test_key_builder_hashes_hostile_ids():
    hostile = "a" * 5000 + ":\n:evil::server\nname"
    key = build_stats_key("mcp", hostile)
    assert re.fullmatch(r"ext_stats:mcp:[0-9a-f]{16}", key)
    # 同 id 幂等
    assert build_stats_key("mcp", hostile) == key
    # 不同 id → 不同 key（极大概率）
    assert build_stats_key("mcp", hostile + "x") != key


# ---------------------------------------------------------------------------
# 5. Redis 落盘异常：flusher 不抛（warn + 丢批）
# ---------------------------------------------------------------------------
async def test_redis_failure_flush_does_not_raise():
    register_extension_tool("mcp_tool_a", "mcp", "server-a")
    fake = _FakeRedis(raise_on_execute=True)
    stats = RedisExtensionStats(fake)
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    # 不得抛
    await _drain_once(stats)
    assert fake.pipe.executed == 1  # 尝试过 execute 但被吞


# ---------------------------------------------------------------------------
# 6. read_many：任一 Redis 异常整体抛（供上层降级 redis_unavailable）
# ---------------------------------------------------------------------------
async def test_read_many_failure_raises_for_degrade():
    fake = _FakeRedis()
    fake.hgetall = AsyncMock(side_effect=RuntimeError("redis down"))
    stats = RedisExtensionStats(fake)
    with pytest.raises(RuntimeError):
        await stats.read_many([("mcp", "server-a")])


async def test_read_many_returns_parsed_data():
    fake = _FakeRedis()
    key = build_stats_key("mcp", "server-a")

    async def _hgetall(k):
        if k == key:
            return {
                "call_count": "5",
                "success_count": "4",
                "failure_count": "1",
            }
        return {}

    fake.hgetall = AsyncMock(side_effect=_hgetall)
    stats = RedisExtensionStats(fake)
    out = await stats.read_many([("mcp", "server-a")])
    data = out[("mcp", "server-a")]
    assert isinstance(data, ExtensionStatsData)
    assert data.call_count == 5
    assert data.success_count == 4
    assert data.failure_count == 1


# ---------------------------------------------------------------------------
# 7. delete_key：入同一队列的 DEL 变体（fire-and-forget）
# ---------------------------------------------------------------------------
async def test_delete_key_enqueued():
    fake = _FakeRedis()
    stats = RedisExtensionStats(fake)
    stats.delete_key("mcp", "server-a")
    assert stats._queue.qsize() == 1
    # drain 后 pipeline 发出 DEL
    await _drain_once(stats)
    key = build_stats_key("mcp", "server-a")
    assert any(key in call for call in fake.pipe.delete_calls)


# ---------------------------------------------------------------------------
# 8. shutdown：拒收新记录 + 限时 drain
# ---------------------------------------------------------------------------
async def test_shutdown_rejects_new_records_and_drains():
    register_extension_tool("mcp_tool_a", "mcp", "server-a")
    fake = _FakeRedis()
    stats = RedisExtensionStats(fake)
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)

    await stats.shutdown()

    # shutdown 后 record 拒收
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    assert stats._closed is True
    # 已入队的记录被 drain（execute 至少调用一次）
    assert fake.pipe.executed >= 1
    # 拒收后队列不再增长
    assert stats._queue.qsize() == 0


# ---------------------------------------------------------------------------
# 9. record 绝不 await（INV-B9-6 结构门）
# ---------------------------------------------------------------------------
def test_record_never_awaits():
    assert not inspect.iscoroutinefunction(RedisExtensionStats.record)
    assert not inspect.iscoroutinefunction(RedisExtensionStats.delete_key)


# ---------------------------------------------------------------------------
# 10. 模块常量哨兵（契约冻结）
# ---------------------------------------------------------------------------
def test_module_constants_frozen():
    assert STATS_QUEUE_MAXSIZE == 1000
    assert STATS_FLUSH_BATCH_SIZE == 50


# ---------------------------------------------------------------------------
# 11. P2 修复：shutdown 不得在后台 flusher 在途批次 mid-pipeline 时误判 drain 干净
# ---------------------------------------------------------------------------
class _BlockingPipeline:
    """execute() 阻塞在 asyncio.Event 上的假 pipeline——模拟 mid-pipeline 窗口。"""

    def __init__(self, gate: "asyncio.Event") -> None:
        self._gate = gate
        self.hincrby_calls: list[tuple] = []
        self.hset_calls: list[tuple] = []
        self.delete_calls: list[tuple] = []
        self.executed = 0
        self.execute_entered = None  # asyncio.Event，进入 execute 后 set

    def hincrby(self, key, field, amount=1):
        self.hincrby_calls.append((key, field, amount))
        return self

    def hset(self, key, field=None, value=None, mapping=None):
        self.hset_calls.append((key, field, value, mapping))
        return self

    def delete(self, *keys):
        self.delete_calls.append(keys)
        return self

    async def execute(self):
        if self.execute_entered is not None:
            self.execute_entered.set()
        await self._gate.wait()  # 阻塞直到测试 set(gate)
        self.executed += 1
        return []


class _BlockingRedis:
    def __init__(self, gate: "asyncio.Event") -> None:
        self.pipe = _BlockingPipeline(gate)
        self.hgetall = AsyncMock(return_value={})

    def pipeline(self, transaction: bool = True):
        return self.pipe


async def test_shutdown_waits_for_inflight_batch_no_loss():
    """竞态：后台 flusher 出队一批并 block 在 _apply_batch 的 pipeline.execute 里，
    此时并发调 shutdown()——shutdown 必须**不得**在批次在途时提前返回；放行 gate 后
    该批落盘、无数据丢失。"""
    import asyncio

    register_extension_tool("mcp_tool_a", "mcp", "server-a")
    gate = asyncio.Event()
    fake = _BlockingRedis(gate)
    stats = RedisExtensionStats(fake)
    fake.pipe.execute_entered = asyncio.Event()

    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    assert stats._queue.qsize() == 1

    # 后台起一轮 _flush_once：会 drain 出该批、在途计数 +1、block 在 execute。
    flush_task = asyncio.create_task(stats._flush_once())
    # 等 flusher 真正进入 pipeline.execute（此刻队列已空但批次在途）。
    await asyncio.wait_for(fake.pipe.execute_entered.wait(), timeout=1.0)
    assert stats._queue.empty()
    assert stats._flush_inflight == 1

    # 并发跑 shutdown：给足够大的 drain 窗口（默认 2s > 我们很快就会放行 gate）。
    shutdown_task = asyncio.create_task(stats.shutdown())

    # 让事件循环转几圈——在 gate 未放行前，shutdown 绝不能完成（批次仍在途）。
    for _ in range(5):
        await asyncio.sleep(0)
    assert not shutdown_task.done(), "shutdown 在批次仍 mid-pipeline 时提前返回——丢批风险"

    # 放行 pipeline：批次落盘完成、在途计数归零，shutdown 随即收尾。
    gate.set()
    await asyncio.wait_for(flush_task, timeout=1.0)
    await asyncio.wait_for(shutdown_task, timeout=1.0)

    assert fake.pipe.executed == 1  # 那批确实落盘（未随 cancel 丢失）
    assert stats._flush_inflight == 0
    key = build_stats_key("mcp", "server-a")
    hincrby_by_field = {(k, f): amt for (k, f, amt) in fake.pipe.hincrby_calls}
    assert hincrby_by_field[(key, "call_count")] == 1


async def test_shutdown_bounded_exit_when_pipeline_never_unblocks(monkeypatch, caplog):
    """兜底：若在途 pipeline 永不放行，shutdown 仍在死线内返回（丢批可接受 + warn）。"""
    import asyncio
    import logging

    import app.infrastructure.external.runtime_stats.redis_extension_stats as mod

    register_extension_tool("mcp_tool_a", "mcp", "server-a")
    gate = asyncio.Event()  # 永不 set
    fake = _BlockingRedis(gate)
    stats = RedisExtensionStats(fake)
    fake.pipe.execute_entered = asyncio.Event()

    # 把死线窗口压到很小（> 轮询间隔即可，仍是真时钟，无长 sleep）。
    monkeypatch.setattr(mod, "STATS_SHUTDOWN_DRAIN_SECONDS", 0.05)

    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    flush_task = asyncio.create_task(stats._flush_once())
    await asyncio.wait_for(fake.pipe.execute_entered.wait(), timeout=1.0)
    assert stats._flush_inflight == 1

    caplog.set_level(logging.WARNING)
    # shutdown 必须在死线内返回（不因永不放行的在途批次无界挂起）。
    await asyncio.wait_for(stats.shutdown(), timeout=1.0)
    assert any(
        "shutdown drain 超" in r.message for r in caplog.records
    ), "超时丢批应记 warn"

    # 收尾：放行 gate 让后台 flush_task 干净退出，避免悬挂 task 告警。
    gate.set()
    await asyncio.wait_for(flush_task, timeout=1.0)


# ---------------------------------------------------------------------------
# 12. PR3-R2#1：两批并发——shutdown 自身 drain 不得清掉后台 flusher 的在途标记
# ---------------------------------------------------------------------------
class _FirstBlockingRedis:
    """第一次 pipeline() 返回 gate 阻塞的 pipeline（批 1，后台 flusher 持有），
    后续 pipeline() 返回立即完成的 _FakePipeline（批 2，shutdown 自身 drain 用）。"""

    def __init__(self, gate: "asyncio.Event") -> None:
        self.blocking_pipe = _BlockingPipeline(gate)
        self.extra_pipes: list[_FakePipeline] = []
        self._returned_first = False
        self.hgetall = AsyncMock(return_value={})

    def pipeline(self, transaction: bool = True):
        if not self._returned_first:
            self._returned_first = True
            return self.blocking_pipe
        pipe = _FakePipeline()
        self.extra_pipes.append(pipe)
        return pipe


async def test_shutdown_concurrent_two_batches_counter_not_clobbered():
    """PR3-R2#1 精确竞态：后台 flusher block 在批 1 的 execute 里、队列还压着批 2；
    shutdown 并发启动、自身 drain 批 2 完成——其 finally 只归还**自己**的在途计数。
    共享布尔实现会在此刻被清成 False → 队列空 + 标记 False → shutdown 提前返回 →
    main.py cancel 后台 flusher → 批 1 丢失且无死线 warn。counter 实现下 shutdown
    必须等批 1 落盘（计数归零）才返回；放行 gate 后两批全部落盘、零丢失。"""
    import asyncio

    register_extension_tool("mcp_tool_a", "mcp", "server-a")
    gate = asyncio.Event()
    fake = _FirstBlockingRedis(gate)
    stats = RedisExtensionStats(fake)
    fake.blocking_pipe.execute_entered = asyncio.Event()

    # 批 1 入队 → 后台 flusher drain 出并 block 在 execute（在途计数 = 1）。
    stats.record("mcp_tool_a", success=True, latency_ms=1.0)
    flush_task = asyncio.create_task(stats._flush_once())
    await asyncio.wait_for(fake.blocking_pipe.execute_entered.wait(), timeout=1.0)
    assert stats._queue.empty()
    assert stats._flush_inflight == 1

    # 批 2 在后台 block 期间入队（尚未 shutdown，record 仍收）。
    stats.record("mcp_tool_a", success=False, latency_ms=1.0)
    assert stats._queue.qsize() == 1

    # 并发 shutdown：队列非空 → 自身 _flush_once drain 批 2（立即完成的 pipe2）。
    shutdown_task = asyncio.create_task(stats.shutdown())

    # 给 shutdown 时间处理完批 2 并进入在途轮询（轮询间隔 10ms；30ms 真时钟 < 0.1s）。
    await asyncio.sleep(0.03)
    assert len(fake.extra_pipes) == 1
    assert fake.extra_pipes[0].executed == 1  # 批 2 已由 shutdown 自身落盘
    # 关键断言：后台批 1 仍在途（计数 1 未被 shutdown 的 finally 清掉），shutdown 未返回。
    assert stats._flush_inflight == 1
    assert not shutdown_task.done(), (
        "shutdown 在后台批 1 仍 mid-pipeline 时提前返回——共享布尔被并发 flush 清掉（PR3-R2#1）"
    )

    # 放行批 1 → 计数归零 → shutdown 收尾；两批都落盘、零丢失。
    gate.set()
    await asyncio.wait_for(flush_task, timeout=1.0)
    await asyncio.wait_for(shutdown_task, timeout=1.0)

    assert stats._flush_inflight == 0
    assert fake.blocking_pipe.executed == 1
    key = build_stats_key("mcp", "server-a")
    total_call_count = sum(
        amt
        for pipe in [fake.blocking_pipe, *fake.extra_pipes]
        for (k, f, amt) in pipe.hincrby_calls
        if k == key and f == "call_count"
    )
    assert total_call_count == 2  # 批 1（成功）+ 批 2（失败）合计 2 次调用，无丢失
