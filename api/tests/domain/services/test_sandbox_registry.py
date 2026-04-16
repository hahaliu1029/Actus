"""Unit tests for SandboxRegistry.

Covers §11.1: register/get_generation/acquire_handle/release_handle basics,
cancel_and_drain completes within timeout.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.external.sandbox.sandbox_registry import SandboxRegistry

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sbx-1") -> None:
        self._id = sandbox_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def cdp_url(self) -> str:
        return "http://test:9222"

    @property
    def shell_ws_url(self) -> str:
        return "ws://test:8080"

    @property
    def vnc_url(self) -> str:
        return "ws://test:5901"

    async def destroy(self) -> bool:
        return True


# ── register / remove / lookup ──


def test_register_and_lookup() -> None:
    reg = SandboxRegistry()
    sbx = FakeSandbox()
    reg.register("sess-1", sbx, generation=1)

    assert reg.get_sandbox("sess-1") is sbx
    assert reg.get_generation("sess-1") == 1


def test_remove_clears_all_state() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)
    reg.remove("sess-1")

    assert reg.get_sandbox("sess-1") is None
    assert reg.get_generation("sess-1") is None


def test_update_generation() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)
    reg.update_generation("sess-1", 5)
    assert reg.get_generation("sess-1") == 5


# ── acquire / release handle ──


def test_acquire_handle_creates_handle() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)

    handle = reg.acquire_handle("sess-1")
    assert handle.id == "sbx-1"
    assert handle.generation == 1
    assert len(reg._open_handles["sess-1"]) == 1


def test_release_handle_removes_from_set() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)
    handle = reg.acquire_handle("sess-1")

    reg.release_handle("sess-1", handle)
    assert len(reg._open_handles["sess-1"]) == 0


def test_release_handle_idempotent() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)
    handle = reg.acquire_handle("sess-1")

    reg.release_handle("sess-1", handle)
    reg.release_handle("sess-1", handle)  # no error
    assert len(reg._open_handles["sess-1"]) == 0


def test_release_handle_unknown_session_no_error() -> None:
    reg = SandboxRegistry()
    reg.release_handle("nonexistent", MagicMock())  # no KeyError


# ── inflight task tracking ──


def test_enroll_and_discharge_inflight() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)
    fake_task = MagicMock()

    reg.enroll_inflight("sess-1", fake_task)
    assert fake_task in reg._inflight_tasks["sess-1"]

    reg.discharge_inflight("sess-1", fake_task)
    assert fake_task not in reg._inflight_tasks["sess-1"]


def test_discharge_sets_drain_event_when_empty() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)
    fake_task = MagicMock()

    reg.enroll_inflight("sess-1", fake_task)
    assert not reg._drain_events["sess-1"].is_set()

    reg.discharge_inflight("sess-1", fake_task)
    assert reg._drain_events["sess-1"].is_set()


# ── cancel_and_drain ──


async def test_cancel_and_drain_with_no_inflight() -> None:
    """Empty inflight set → drain completes immediately."""
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)

    await reg.cancel_and_drain("sess-1", timeout=1.0)  # should not raise


async def test_cancel_and_drain_cancels_tasks() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)

    cancelled = False

    async def slow_work():
        nonlocal cancelled
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled = True
            raise

    task = asyncio.create_task(slow_work())
    reg.enroll_inflight("sess-1", task)
    await asyncio.sleep(0)  # yield to let slow_work start running

    await reg.cancel_and_drain("sess-1", timeout=5.0)
    assert cancelled


async def test_cancel_and_drain_closes_ws_holders() -> None:
    reg = SandboxRegistry()
    reg.register("sess-1", FakeSandbox(), generation=1)

    holder = MagicMock()
    holder.close = AsyncMock()
    reg.register_ws_holder("sess-1", holder)

    await reg.cancel_and_drain("sess-1", timeout=5.0)
    holder.close.assert_awaited_once()


# ── destroy_infra ──


async def test_destroy_infra_calls_sandbox_destroy() -> None:
    reg = SandboxRegistry()
    sbx = FakeSandbox()
    sbx.destroy = AsyncMock(return_value=True)
    reg.register("sess-1", sbx, generation=1)

    await reg.destroy_infra("sess-1")
    sbx.destroy.assert_awaited_once()


async def test_destroy_infra_no_sandbox_noop() -> None:
    reg = SandboxRegistry()
    await reg.destroy_infra("nonexistent")  # no error
