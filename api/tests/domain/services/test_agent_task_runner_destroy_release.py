"""SPM PR-1b Task 10 — runner ``destroy()`` category-D release chain (r4/r7 F3).

Frozen contract: ``destroy()`` closes the browser FIRST
(``browser_accessor.aclose()``), then releases the accessor-owned sandbox handle
(``sandbox_accessor.release_owned()``) — each best-effort (its own try/except so a
failure never skips the next step). PR-1b implements the Eager side only; the
OnDemand ``release_held_handle`` half is Task 14.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _bare_runner_with(sandbox_accessor, browser_accessor) -> AgentTaskRunner:
    """A runner shell that only wires what ``destroy()`` touches (no full ctor)."""
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._sandbox_accessor = sandbox_accessor
    runner._browser_accessor = browser_accessor
    runner._cleanup_tools = AsyncMock()
    runner._flow = AsyncMock()
    return runner


class TestDestroyReleaseChain:
    async def test_destroy_closes_browser_then_releases_sandbox(self) -> None:
        order: list[str] = []
        bacc = AsyncMock()
        bacc.aclose = AsyncMock(side_effect=lambda: order.append("browser"))
        sacc = AsyncMock()
        sacc.release_owned = AsyncMock(side_effect=lambda: order.append("sandbox"))
        runner = _bare_runner_with(sacc, bacc)

        await runner.destroy()

        bacc.aclose.assert_awaited_once()
        sacc.release_owned.assert_awaited_once()
        assert order == ["browser", "sandbox"], "browser MUST close before sandbox release"

    async def test_destroy_aclose_failure_still_releases_sandbox(self) -> None:
        """best-effort: aclose() raising must NOT skip release_owned() nor propagate."""
        bacc = AsyncMock()
        bacc.aclose = AsyncMock(side_effect=RuntimeError("browser boom"))
        sacc = AsyncMock()
        runner = _bare_runner_with(sacc, bacc)

        await runner.destroy()  # must NOT raise

        sacc.release_owned.assert_awaited_once()

    async def test_destroy_release_failure_is_swallowed(self) -> None:
        bacc = AsyncMock()
        sacc = AsyncMock()
        sacc.release_owned = AsyncMock(side_effect=RuntimeError("release boom"))
        runner = _bare_runner_with(sacc, bacc)

        await runner.destroy()  # must NOT raise

        bacc.aclose.assert_awaited_once()


class TestDestroyEagerSemantics:
    async def test_destroy_releases_wrapped_handle_once_and_is_idempotent(self) -> None:
        """Eager integration: destroy() → wrapped handle.release() once; a second
        destroy() is a no-op (release_owned clears the handle after the first)."""
        handle = MagicMock()
        handle.release = MagicMock()
        browser = AsyncMock()  # aclose() awaitable
        runner = _bare_runner_with(
            EagerSandboxAccessor(handle), EagerBrowserAccessor(browser)
        )

        await runner.destroy()
        await runner.destroy()  # idempotent — no second release

        handle.release.assert_called_once()
        browser.aclose.assert_awaited()

    async def test_destroy_no_provision_double_zero_call_does_not_crash(self) -> None:
        """Never-provisioned Eager wrappers over ``None``-free mocks: destroy is safe."""
        handle = MagicMock(release=MagicMock())
        browser = AsyncMock()
        runner = _bare_runner_with(
            EagerSandboxAccessor(handle), EagerBrowserAccessor(browser)
        )
        # release_owned() before any get() still works (Eager holds the handle).
        await runner.destroy()
        handle.release.assert_called_once()
