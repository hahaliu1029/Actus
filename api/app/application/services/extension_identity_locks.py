"""D1a §3.6 per-identity 进程内锁注册表（standalone 安装/卸载全程持锁）。

单一进程单例（``get_identity_locks``）——mcp/a2a/skill/saga 四路共享同一注册表，
保证同一 ``(kind, ext_id)`` 的 standalone 预检→落行不与并发写交错（T19/T22 消费）。

契约（避免 T16→T19 依赖倒挂，T19 只 import 消费）：
- 惰性建锁：每个 identity 首次触碰时按需创建 ``asyncio.Lock``。
- 排序获取：``acquire_all`` 按排序顺序获取多把锁——防止交叉获取死锁。
- try/finally 释放已获取子集：中途取消/异常只释放**已成功获取**的锁，
  不误碰未持有的锁。
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

Identity = tuple[str, str]  # (kind, ext_id)


class IdentityLockRegistry:
    """惰性建锁的 per-identity ``asyncio.Lock`` 注册表。"""

    def __init__(self) -> None:
        self._locks: dict[Identity, asyncio.Lock] = {}
        # 注册表字典写守卫——防两个协程同时为同一 identity 建两把锁
        self._registry_guard = asyncio.Lock()

    async def _get_lock(self, identity: Identity) -> asyncio.Lock:
        async with self._registry_guard:
            lock = self._locks.get(identity)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[identity] = lock
            return lock

    @asynccontextmanager
    async def acquire_all(
        self, identities: Sequence[Identity]
    ) -> AsyncIterator[None]:
        """按排序顺序获取一组 identity 锁；退出时逆序释放已获取子集。"""
        ordered = sorted(set(identities))
        acquired: list[asyncio.Lock] = []
        try:
            for identity in ordered:
                lock = await self._get_lock(identity)
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()


_process_registry: IdentityLockRegistry | None = None
_process_registry_guard = threading.Lock()


def get_identity_locks() -> IdentityLockRegistry:
    """进程单例工厂——同步调用点（``self._identity_locks or get_identity_locks()``）。

    ``asyncio.Lock`` 在 3.10+ 构造时不绑定事件循环（首次 await 惰性绑定），
    因此在无运行事件循环时创建单例是安全的。``threading.Lock`` 兜底跨线程
    首次创建竞态（进程内仅创建一次）。
    """
    global _process_registry
    if _process_registry is None:
        with _process_registry_guard:
            if _process_registry is None:
                _process_registry = IdentityLockRegistry()
    return _process_registry
