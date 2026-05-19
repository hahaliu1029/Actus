"""Unit regression for `_drain_subagent_cleanup` (Codex R4 P2#2).

Pure asyncio test — does not need pg/redis. Models the failure mode
Codex R3+R4 found in `ConnectionLease.release()`: cancel landing on
the second cleanup leaks Redis state. The helper's `create_task` +
loop-until-done pattern is what makes the cleanup deterministic.

Kept out of `tests/integration/test_subagent_research_endpoint.py` so
the cancel-storm contract can be exercised without integration infra
and without the integration autouse fixture chain.
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.anyio


async def test_drain_subagent_cleanup_completes_both_under_outer_cancel():
    """Both cleanup tasks must run to completion even if outer is cancelled
    mid-drain. Helper re-raises cancellation only AFTER both tasks finish.
    """
    from app.interfaces.endpoints.session_routes import (
        _drain_subagent_cleanup,
    )

    agen_closed = asyncio.Event()
    lease_released = asyncio.Event()

    class FakeAgen:
        async def aclose(self) -> None:
            await asyncio.sleep(0.05)
            agen_closed.set()

    class FakeLease:
        async def release(self) -> None:
            await asyncio.sleep(0.05)
            lease_released.set()

    cancelled_observed = False

    async def runner() -> None:
        nonlocal cancelled_observed
        try:
            await _drain_subagent_cleanup(FakeAgen(), FakeLease())
        except asyncio.CancelledError:
            cancelled_observed = True

    outer = asyncio.create_task(runner())
    # Let the helper start both cleanup tasks, then cancel the outer.
    await asyncio.sleep(0.01)
    outer.cancel()
    await asyncio.gather(outer, return_exceptions=True)

    assert agen_closed.is_set(), (
        "agen.aclose() did not complete to its set() — drain helper "
        "did not preserve the cleanup task across cancel"
    )
    assert lease_released.is_set(), (
        "lease.release() did not complete to its set() — drain helper "
        "did not preserve the cleanup task across cancel; the Redis "
        "zrem at rate_limit.py:124 would have been skipped"
    )
    assert cancelled_observed, (
        "outer cancel was not re-raised after cleanup — cooperative "
        "cancellation contract broken"
    )


async def test_drain_subagent_cleanup_no_cancel_returns_normally():
    """Sanity: no cancel, no exception — helper returns normally and
    both cleanups complete.
    """
    from app.interfaces.endpoints.session_routes import (
        _drain_subagent_cleanup,
    )

    log: list[str] = []

    class FakeAgen:
        async def aclose(self) -> None:
            log.append("agen.aclose")

    class FakeLease:
        async def release(self) -> None:
            log.append("lease.release")

    await _drain_subagent_cleanup(FakeAgen(), FakeLease())

    assert log == ["agen.aclose", "lease.release"], (
        f"cleanup order or count wrong: {log}"
    )


async def test_drain_subagent_cleanup_logs_and_continues_on_exception():
    """If `agen.aclose()` raises an arbitrary exception, helper logs and
    proceeds to `lease.release()` — lease must still be released.
    """
    from app.interfaces.endpoints.session_routes import (
        _drain_subagent_cleanup,
    )

    lease_released = asyncio.Event()

    class FakeAgen:
        async def aclose(self) -> None:
            raise RuntimeError("agen exploded")

    class FakeLease:
        async def release(self) -> None:
            lease_released.set()

    # Should NOT propagate the RuntimeError; should release lease.
    await _drain_subagent_cleanup(FakeAgen(), FakeLease())
    assert lease_released.is_set(), (
        "lease.release() must still run after agen.aclose() raises"
    )
