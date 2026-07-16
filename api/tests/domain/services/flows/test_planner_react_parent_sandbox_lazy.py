"""SPM PR-1b Task 10 — flow ``_make_parent_sandbox_port`` discriminator (r18 R17-Q1).

The coordinator ``parent_sandbox`` wiring forks on the sandbox accessor:

- Eager (``always`` / child): ``peek()`` non-None → direct ``factory(handle)`` wrap
  (byte-identical to pre-SPM; covered by ``test_planner_react_coord_config`` +
  ``test_coordinator_parent_sandbox_adapter_injection``).
- ``on_demand``: ``peek()`` is None → a ``LazyParentSandboxPort`` that provisions
  on the FIRST coordinator parent I/O. A pure chat (``_build_config`` only) must
  NOT provision. ``spec §4`` permits ``on_demand × coordinator``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.lazy_parent_sandbox_port import LazyParentSandboxPort
from tests.domain.services.flows.test_planner_react_coord_config import (
    _build_flow_with_real_coord_deps,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _OnDemandAccessor:
    """``on_demand`` accessor: ``peek()`` None until first ``get()`` provisions."""

    def __init__(self, handle: object) -> None:
        self._handle = handle
        self.get_calls = 0

    async def get(self) -> object:
        self.get_calls += 1
        return self._handle

    def peek(self) -> None:
        return None

    async def release_owned(self) -> None:
        pass


def test_eager_coordinator_keeps_direct_factory_wrap() -> None:
    """Sanity: the default Eager helper still takes the direct-wrap branch (peek
    non-None) — the ``parent_sandbox`` is the factory output, NOT a Lazy port."""
    flow, sentinels = _build_flow_with_real_coord_deps()
    cfg = flow._build_config()
    parent = cfg["configurable"]["parent_sandbox"]
    assert not isinstance(parent, LazyParentSandboxPort)
    sentinels["parent_sandbox_adapter_factory"].assert_called_once_with(
        flow._sandbox_accessor.peek()
    )


def test_on_demand_coordinator_pure_chat_zero_provision() -> None:
    """on_demand + coordinator, ``_build_config`` only (pure chat) → LazyPort +
    ZERO provision (no ``get()``, factory NOT invoked at build time)."""
    flow, sentinels = _build_flow_with_real_coord_deps()
    acc = _OnDemandAccessor(MagicMock(name="on_demand_handle"))
    flow._sandbox_accessor = acc
    factory = sentinels["parent_sandbox_adapter_factory"]
    factory.reset_mock()

    cfg = flow._build_config()
    parent = cfg["configurable"]["parent_sandbox"]

    assert isinstance(parent, LazyParentSandboxPort)
    assert acc.get_calls == 0          # deferred — nothing provisioned yet
    factory.assert_not_called()        # factory NOT consumed at _build_config time


async def test_on_demand_first_parent_io_provisions_once_via_injected_factory() -> None:
    """First coordinator parent I/O → single provision + wrap via the INJECTED
    factory + delegate to the real adapter (r21/R21-U6 DI invariant)."""
    flow, sentinels = _build_flow_with_real_coord_deps()
    handle = MagicMock(name="on_demand_handle")
    acc = _OnDemandAccessor(handle)
    flow._sandbox_accessor = acc
    factory = sentinels["parent_sandbox_adapter_factory"]
    adapter = AsyncMock()
    factory.side_effect = lambda h: adapter
    factory.reset_mock()

    cfg = flow._build_config()
    parent = cfg["configurable"]["parent_sandbox"]

    await parent.exists("/home/ubuntu/x")   # FIRST parent I/O
    await parent.compute_digest("/home/ubuntu/y")  # same generation → cache hit

    assert acc.get_calls == 2               # provision-if-needed per call
    factory.assert_called_once_with(handle)  # but wrapped ONCE (per-generation cache)
    adapter.exists.assert_awaited_once_with("/home/ubuntu/x")
    adapter.compute_digest.assert_awaited_once_with("/home/ubuntu/y")
