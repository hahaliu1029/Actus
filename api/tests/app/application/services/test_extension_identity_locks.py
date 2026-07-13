"""T16 unit test — per-identity 进程锁注册表（D1a §3.6）。

单测锁 IdentityLockRegistry 契约：排序获取、body 异常后 finally 释放、
同 identity 串行化、get_identity_locks 进程单例。
"""
from __future__ import annotations

import asyncio

import pytest

from app.application.services.extension_identity_locks import (
    IdentityLockRegistry,
    get_identity_locks,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_acquire_all_releases_after_normal_exit() -> None:
    reg = IdentityLockRegistry()
    async with reg.acquire_all([("skill", "s1")]):
        pass
    # 正常退出后锁应已释放：再次获取不应阻塞（若未释放此处会挂）
    async with asyncio.timeout(1.0):
        async with reg.acquire_all([("skill", "s1")]):
            pass


async def test_lock_released_after_body_exception() -> None:
    reg = IdentityLockRegistry()
    with pytest.raises(ValueError):
        async with reg.acquire_all([("skill", "s1")]):
            raise ValueError("boom")
    # finally 必须释放已获取锁——否则下面这次获取会永久挂
    async with asyncio.timeout(1.0):
        async with reg.acquire_all([("skill", "s1")]):
            pass


async def test_same_identity_serialized() -> None:
    reg = IdentityLockRegistry()
    order: list[str] = []

    async def worker(tag: str) -> None:
        async with reg.acquire_all([("skill", "s1")]):
            order.append(f"enter-{tag}")
            await asyncio.sleep(0.01)
            order.append(f"exit-{tag}")

    await asyncio.gather(worker("a"), worker("b"))
    # 串行化：一个协程完整进出后另一个才进入（不交错）
    assert order in (
        ["enter-a", "exit-a", "enter-b", "exit-b"],
        ["enter-b", "exit-b", "enter-a", "exit-a"],
    )


async def test_distinct_identities_do_not_block() -> None:
    reg = IdentityLockRegistry()
    async with reg.acquire_all([("skill", "s1")]):
        # 不同 identity 的锁互不影响——不应阻塞
        async with asyncio.timeout(1.0):
            async with reg.acquire_all([("mcp", "m1")]):
                pass


async def test_acquire_all_sorted_and_deduped() -> None:
    reg = IdentityLockRegistry()
    # 重复 + 乱序 identity 集合：不应死锁（内部去重 + 排序获取）
    async with asyncio.timeout(1.0):
        async with reg.acquire_all(
            [("skill", "b"), ("skill", "a"), ("skill", "b")]
        ):
            pass


def test_get_identity_locks_is_process_singleton() -> None:
    first = get_identity_locks()
    second = get_identity_locks()
    assert first is second
    assert isinstance(first, IdentityLockRegistry)
