"""Unit tests for the C2 leaked-sandbox startup reaper (spec §3.5 / tests 16–18).

Pure / fake-driven — no DB, no docker. Mirrors test_child_terminal_reconciler.
The real SQL query is exercised in tests/integration/test_coordinator_cancel_queries.py.
"""
from __future__ import annotations

import asyncio

import pytest

from app.application.services.sandbox_terminal_reaper import (
    SandboxReapStats,
    sweep_terminal_coordinator_active_sandboxes,
)
from app.domain.errors.sandbox_lifecycle import (
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
)
from app.domain.models.session import DestroyReason
from app.domain.repositories.session_repository import ChildLineageRow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeReaperRepo:
    def __init__(self, children, raises=False):
        self._children = children
        self._raises = raises

    async def find_terminal_coordinator_children_with_active_sandbox(self):
        if self._raises:
            raise RuntimeError("query down")
        return list(self._children)


class _FakeLifecycle:
    """try_register_from_binding returns register_map[id] (default True);
    destroy raises destroy_exc[id] if present."""

    def __init__(self, *, register_map=None, destroy_exc=None):
        self._register_map = register_map or {}
        self._destroy_exc = destroy_exc or {}
        self.register_calls: list[str] = []
        self.destroy_calls: list[tuple[str, DestroyReason]] = []

    async def try_register_from_binding(self, session_id):
        self.register_calls.append(session_id)
        return self._register_map.get(session_id, True)

    async def destroy(self, session_id, reason):
        self.destroy_calls.append((session_id, reason))
        exc = self._destroy_exc.get(session_id)
        if exc is not None:
            raise exc


def _children(n):
    return [ChildLineageRow(f"c{i}", "run-1", f"wu-{i}") for i in range(n)]


# 16 — all register True + destroy OK -> destroyed=N, reason is TERMINAL_CHILD_REAPER
async def test_sweep_all_destroyed():
    repo = _FakeReaperRepo(_children(3))
    svc = _FakeLifecycle()
    stats = await sweep_terminal_coordinator_active_sandboxes(
        session_repo=repo, lifecycle_service=svc
    )
    assert isinstance(stats, SandboxReapStats)
    assert (stats.scanned, stats.destroyed, stats.already_gone, stats.errored) == (3, 3, 0, 0)
    # The reaper consults the try_register guard for EVERY child BEFORE destroy
    # (mutation: skip the guard → register_calls empty → this assertion fails;
    # the gone-container skip path is further proven by test 17). R3 P3#2 — this
    # is the assertion that actually pins "reaper calls the guard", not test 14
    # (which only exercises the helper + destroy, never the reaper).
    assert svc.register_calls == ["c0", "c1", "c2"]
    assert svc.destroy_calls == [
        ("c0", DestroyReason.TERMINAL_CHILD_REAPER),
        ("c1", DestroyReason.TERMINAL_CHILD_REAPER),
        ("c2", DestroyReason.TERMINAL_CHILD_REAPER),
    ]


# 17 — mixed: True+OK / False(gone, skipped) / True+AlreadyDestroyed -> destroyed=1, already_gone=2
async def test_sweep_mixed_register_and_terminal_success():
    children = _children(3)  # c0, c1, c2
    svc = _FakeLifecycle(
        register_map={"c1": False},  # container gone -> skip destroy
        destroy_exc={"c2": SandboxAlreadyDestroyed("c2")},  # terminal-success
    )
    repo = _FakeReaperRepo(children)
    stats = await sweep_terminal_coordinator_active_sandboxes(
        session_repo=repo, lifecycle_service=svc
    )
    assert (stats.scanned, stats.destroyed, stats.already_gone, stats.errored) == (3, 1, 2, 0)
    # c1's destroy was skipped (register False); c0 + c2 attempted destroy.
    assert [sid for sid, _ in svc.destroy_calls] == ["c0", "c2"]


# 17b — SandboxBindingMissing is also terminal-success
async def test_sweep_binding_missing_is_already_gone():
    svc = _FakeLifecycle(destroy_exc={"c0": SandboxBindingMissing("c0")})
    stats = await sweep_terminal_coordinator_active_sandboxes(
        session_repo=_FakeReaperRepo(_children(1)), lifecycle_service=svc
    )
    assert (stats.destroyed, stats.already_gone, stats.errored) == (0, 1, 0)


# 18a — generic destroy error isolated; sweep continues
async def test_sweep_generic_error_isolated():
    svc = _FakeLifecycle(destroy_exc={"c1": RuntimeError("docker hiccup")})
    stats = await sweep_terminal_coordinator_active_sandboxes(
        session_repo=_FakeReaperRepo(_children(3)), lifecycle_service=svc
    )
    assert (stats.scanned, stats.destroyed, stats.errored) == (3, 2, 1)


# 18b — CancelledError from destroy propagates (not swallowed)
async def test_sweep_cancelled_propagates():
    svc = _FakeLifecycle(destroy_exc={"c0": asyncio.CancelledError()})
    with pytest.raises(asyncio.CancelledError):
        await sweep_terminal_coordinator_active_sandboxes(
            session_repo=_FakeReaperRepo(_children(1)), lifecycle_service=svc
        )


# 18c — empty list -> all zeros, no try_register / destroy
async def test_sweep_empty_noop():
    svc = _FakeLifecycle()
    stats = await sweep_terminal_coordinator_active_sandboxes(
        session_repo=_FakeReaperRepo([]), lifecycle_service=svc
    )
    assert (stats.scanned, stats.destroyed, stats.already_gone, stats.errored) == (0, 0, 0, 0)
    assert svc.register_calls == []
    assert svc.destroy_calls == []


# 18d — query failure propagates to the caller's OUTER best-effort try (main.py)
async def test_sweep_query_failure_propagates():
    with pytest.raises(RuntimeError):
        await sweep_terminal_coordinator_active_sandboxes(
            session_repo=_FakeReaperRepo([], raises=True), lifecycle_service=_FakeLifecycle()
        )
