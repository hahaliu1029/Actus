"""SPM PR-1b Task 10 — LazyParentSandboxPort unit tests (r18 R17-Q1 / r21 R21-U6).

Covers: deferred provision (pure chat = zero ``get()``), first-parent-I/O single
provision + delegation, per-generation adapter cache (factory NOT re-called on a
same-generation re-entry), generation-change rebuild, and the DI invariant that
the wrapped adapter comes from the INJECTED factory (never a direct
``ParentSandboxAdapter`` construction).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.application.services.lazy_parent_sandbox_port import LazyParentSandboxPort

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeAccessor:
    """``peek()`` is None (``on_demand`` not-yet-provisioned); ``get()`` hands back
    handles in order and counts calls (a single handle repeats forever)."""

    def __init__(self, *handles: object) -> None:
        self._handles = list(handles) or [object()]
        self.get_calls = 0

    async def get(self) -> object:
        self.get_calls += 1
        idx = min(self.get_calls - 1, len(self._handles) - 1)
        return self._handles[idx]

    def peek(self) -> None:
        return None

    async def release_owned(self) -> None:  # protocol completeness
        pass


class _SpyFactory:
    """Records every handle it is invoked with; returns a fresh (or fixed) adapter."""

    def __init__(self, adapter: object | None = None) -> None:
        self.calls: list[object] = []
        self._adapter = adapter

    def __call__(self, handle: object) -> object:
        self.calls.append(handle)
        return self._adapter if self._adapter is not None else _make_adapter()


def _make_adapter() -> AsyncMock:
    """A fake ParentSandboxPort adapter with the 8 async methods stubbed."""
    a = AsyncMock()
    a.compute_digest = AsyncMock(return_value="deadbeef")
    a.exists = AsyncMock(return_value=True)
    a.read_file = AsyncMock(return_value=b"data")
    return a


class TestLazyProvisionDeferral:
    async def test_pure_chat_never_provisions(self) -> None:
        """Constructing + never touching parent I/O = ZERO ``accessor.get()``."""
        acc = _FakeAccessor(object())
        spy = _SpyFactory()
        LazyParentSandboxPort(acc, spy)
        assert acc.get_calls == 0
        assert spy.calls == []

    async def test_first_parent_io_provisions_once_and_delegates(self) -> None:
        acc = _FakeAccessor(object())
        adapter = _make_adapter()
        spy = _SpyFactory(adapter)
        port = LazyParentSandboxPort(acc, spy)

        result = await port.compute_digest("/home/ubuntu/x")

        assert result == "deadbeef"
        assert acc.get_calls == 1
        assert spy.calls == [acc._handles[0]]  # factory got the provisioned handle
        adapter.compute_digest.assert_awaited_once_with("/home/ubuntu/x")


class TestPerGenerationCache:
    async def test_same_generation_reuses_adapter(self) -> None:
        """Two calls, one generation → factory called ONCE (cache hit), get() twice."""
        handle = object()
        acc = _FakeAccessor(handle)  # repeats the SAME handle
        spy = _SpyFactory(_make_adapter())
        port = LazyParentSandboxPort(acc, spy)

        await port.compute_digest("/a")
        await port.exists("/b")

        assert acc.get_calls == 2       # provision-if-needed on each method
        assert spy.calls == [handle]    # but the adapter is built ONCE (cache hit)

    async def test_generation_change_rebuilds_adapter(self) -> None:
        """``get()`` returns a NEW handle (teardown + re-provision) → factory re-called."""
        h1, h2 = object(), object()
        acc = _FakeAccessor(h1, h2)
        spy = _SpyFactory()
        port = LazyParentSandboxPort(acc, spy)

        await port.compute_digest("/a")  # generation 1 (h1)
        await port.exists("/b")          # generation 2 (h2)

        assert spy.calls == [h1, h2]


class TestDIInvariant:
    async def test_adapter_comes_from_injected_factory(self) -> None:
        """r21/R21-U6 — the wrapped adapter is the INJECTED factory's output,
        never a direct ``ParentSandboxAdapter`` construction."""
        acc = _FakeAccessor(object())
        sentinel = _make_adapter()
        port = LazyParentSandboxPort(acc, _SpyFactory(sentinel))

        await port.read_file("/z")

        sentinel.read_file.assert_awaited_once_with("/z")


class TestAllMethodsDelegate:
    async def test_every_port_method_delegates(self) -> None:
        acc = _FakeAccessor(object())
        adapter = _make_adapter()
        port = LazyParentSandboxPort(acc, _SpyFactory(adapter))

        await port.compute_digest("/a")
        await port.exists("/b")
        await port.check_path("/c")
        await port.read_file("/d")
        await port.atomic_write_file("/e", b"x")
        await port.delete_file("/f")
        await port.kill_all_shell_sessions()
        await port.snapshot_workspace(
            "/root", max_paths=1, max_files=1, max_total_bytes=1, max_seconds=1.0
        )

        adapter.compute_digest.assert_awaited_once_with("/a")
        adapter.exists.assert_awaited_once_with("/b")
        adapter.check_path.assert_awaited_once_with("/c")
        adapter.read_file.assert_awaited_once_with("/d")
        adapter.atomic_write_file.assert_awaited_once_with("/e", b"x")
        adapter.delete_file.assert_awaited_once_with("/f")
        adapter.kill_all_shell_sessions.assert_awaited_once_with()
        adapter.snapshot_workspace.assert_awaited_once_with(
            "/root", max_paths=1, max_files=1, max_total_bytes=1, max_seconds=1.0
        )
