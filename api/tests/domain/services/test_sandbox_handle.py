"""Unit tests for SandboxHandleImpl.

Covers §11.1: generation check, release idempotent, context manager,
CancelledError discharge (eng review #6).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


from app.domain.errors.sandbox_lifecycle import SandboxPoisonedError
from app.infrastructure.external.sandbox.sandbox_handle import (
    SANDBOX_FORWARDED_METHODS,
    SandboxHandleImpl,
)
from app.infrastructure.external.sandbox.sandbox_registry import SandboxRegistry


class FakeSandbox:
    def __init__(self) -> None:
        self._id = "sbx-test"

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

    async def read_file(self, filepath: str, **kwargs):
        return MagicMock(success=True, data={"content": "hello"})

    async def exec_command(self, *args, **kwargs):
        return MagicMock(success=True)

    async def renew_timeout_lease(self, minutes=None) -> None:
        self.renewed_timeout_minutes = minutes

    async def destroy(self) -> bool:
        return True


def _make_handle(
    generation: int = 1,
    registry: SandboxRegistry | None = None,
) -> tuple[SandboxHandleImpl, SandboxRegistry]:
    if registry is None:
        registry = SandboxRegistry()
    sandbox = FakeSandbox()
    registry.register("sess-1", sandbox, generation=generation)
    handle = registry.acquire_handle("sess-1")
    return handle, registry


# ── Generation check ──


async def test_stale_generation_raises() -> None:
    handle, registry = _make_handle(generation=1)
    registry.update_generation("sess-1", 2)
    with pytest.raises(SandboxPoisonedError) as exc_info:
        await handle.read_file("/test")
    assert exc_info.value.expected_generation == 1
    assert exc_info.value.actual_generation == 2


async def test_matching_generation_succeeds() -> None:
    handle, _ = _make_handle(generation=1)
    result = await handle.read_file("/test")
    assert result.success


async def test_timeout_lease_renewal_is_generation_checked_and_forwarded() -> None:
    handle, _ = _make_handle(generation=1)

    await handle.renew_timeout_lease(7)

    assert "renew_timeout_lease" in SANDBOX_FORWARDED_METHODS
    assert handle._sandbox.renewed_timeout_minutes == 7


# ── Release ──


def test_release_removes_from_registry() -> None:
    handle, registry = _make_handle()
    assert len(registry._open_handles.get("sess-1", set())) == 1
    handle.release()
    assert len(registry._open_handles.get("sess-1", set())) == 0


def test_release_idempotent() -> None:
    handle, registry = _make_handle()
    handle.release()
    handle.release()
    assert len(registry._open_handles.get("sess-1", set())) == 0


async def test_concurrent_release_no_error() -> None:
    handle, registry = _make_handle()

    async def release_task():
        handle.release()

    await asyncio.gather(release_task(), release_task())
    assert len(registry._open_handles.get("sess-1", set())) == 0


# ── Context manager ──


async def test_context_manager_normal_exit_releases() -> None:
    handle, registry = _make_handle()
    async with handle:
        assert len(registry._open_handles.get("sess-1", set())) == 1
    assert len(registry._open_handles.get("sess-1", set())) == 0


async def test_context_manager_exception_exit_releases() -> None:
    handle, registry = _make_handle()
    with pytest.raises(ValueError):
        async with handle:
            raise ValueError("boom")
    assert len(registry._open_handles.get("sess-1", set())) == 0


# ── CancelledError discharge (eng review #6) ──


async def test_cancelled_error_discharges_inflight() -> None:
    registry = SandboxRegistry()
    sandbox = FakeSandbox()
    sandbox.read_file = AsyncMock(side_effect=asyncio.CancelledError)
    registry.register("sess-1", sandbox, generation=1)
    handle = registry.acquire_handle("sess-1")

    with pytest.raises(asyncio.CancelledError):
        await handle.read_file("/test")

    inflight = registry._inflight_tasks.get("sess-1", set())
    assert len(inflight) == 0


# ── Properties ──


def test_properties_cached_at_acquire() -> None:
    handle, _ = _make_handle()
    assert handle.id == "sbx-test"
    assert "test" in handle.cdp_url
    assert "test" in handle.shell_ws_url
    assert "test" in handle.vnc_url
    assert handle.generation == 1


# ── Forwarded methods ──


def test_forwarded_methods_exact_set() -> None:
    """Pin the exact forwarded method set, not a bare count.

    Asserting the full set documents WHICH Sandbox Protocol methods are
    generation-checked forwarded and fails with a meaningful diff when a
    method is added to / removed from ``SANDBOX_FORWARDED_METHODS``. Keep
    this list in sync with the source definition in ``sandbox_handle.py``.
    """
    expected = {
        "exec_command",
        "read_shell_output",
        "wait_process",
        "write_shell_input",
        "resize_shell_session",
        "kill_process",
        "write_file",
        "read_file",
        "check_file_exists",
        "delete_file",
        "list_files",
        "replace_in_file",
        "search_in_file",
        "find_files",
        "upload_file",
        "download_file",
        "ensure_sandbox",
        "renew_timeout_lease",
        "get_browser",
        "snapshot_workspace",
        "kill_all_shell_sessions",
    }
    assert SANDBOX_FORWARDED_METHODS == expected


def test_unknown_attribute_raises() -> None:
    handle, _ = _make_handle()
    with pytest.raises(AttributeError):
        handle.nonexistent_method  # noqa: B018
