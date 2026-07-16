"""PR-1: SessionService.create_session_with_parent — child session creation
for the Phase 1 minimal subagent feature.

Contract:
- Stores parent_session_id on the new Session entity
- Persists via uow.session.save exactly once
- Default title is "新对话" (parity with create_session)
- Does NOT trigger fs_reconciler walk (parent already walked the user dir);
  verified by spying the _spawn_fs_reconciler_walk hook directly to avoid
  fire-and-forget false-positives
- Serial calls (3 in a row) each persist a child — documents the
  serial-call discipline expected from SubagentResearchService (PR-4);
  concurrent-safety verification lives at the PR-4 caller boundary
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.session_service import SessionService
from app.domain.models.session import Session

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_uow_and_factory(
    *,
    parent_user_id: str = "u-1",
    parent_session_id: str = "parent-1",
    descendants_count: int = 0,
):
    """Returns (uow_mock, factory) so tests can both inject and introspect.

    C1a (PR-2): create_session_with_parent now performs
    ``lock_session_for_spawn`` + owner check + ``count_descendants`` inside
    the UoW. The mock must satisfy those async calls; defaults give a valid
    root parent with no descendants so the existing PR-1 assertions still
    hold.
    """
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.session = MagicMock()
    uow.session.save = AsyncMock()
    uow.session.lock_session_for_spawn = AsyncMock(
        return_value=Session(
            id=parent_session_id, user_id=parent_user_id, worker_type="root"
        )
    )
    uow.session.count_descendants = AsyncMock(return_value=descendants_count)
    return uow, lambda: uow


class TestCreateSessionWithParent:
    async def test_sets_parent_session_id(self) -> None:
        """create_session_with_parent stores parent_session_id on the child Session."""
        uow, factory = _make_uow_and_factory()
        service = SessionService(uow_factory=factory)

        child = await service.create_session_with_parent(
            user_id="u-1",
            parent_session_id="parent-1",
            # T12: preset is required for every child created via this method
            # (codex R1 P1 fix). Existing PR-1 contract preserved by passing
            # the canonical subagent_research preset.
            tool_filter_preset="subagent_research",
        )

        assert child.user_id == "u-1"
        assert child.parent_session_id == "parent-1"
        assert child.title == "新对话"

        uow.session.save.assert_awaited_once()
        saved = uow.session.save.call_args.args[0]
        assert saved.parent_session_id == "parent-1"
        assert saved.user_id == "u-1"

    async def test_does_not_trigger_fs_reconciler_walk(self) -> None:
        """Child sessions skip fs_reconciler walk (parent already walked the user dir).

        Spies the ``_spawn_fs_reconciler_walk`` hook directly — the parent
        ``create_session`` schedules the walk via ``asyncio.create_task`` inside
        this helper, so asserting on the hook itself (rather than on the
        downstream ``walk_user_directory`` awaitable) avoids a false-positive
        where a fire-and-forget task hasn't yet run when the test asserts.
        """
        uow, factory = _make_uow_and_factory()
        reconciler = MagicMock()
        reconciler.walk_user_directory = AsyncMock()

        service = SessionService(uow_factory=factory, fs_reconciler=reconciler)
        spawn_spy = MagicMock()
        service._spawn_fs_reconciler_walk = spawn_spy  # type: ignore[method-assign]

        await service.create_session_with_parent(
            user_id="u-1",
            parent_session_id="parent-1",
            tool_filter_preset="subagent_research",  # T12: required for children
        )

        spawn_spy.assert_not_called()
        reconciler.walk_user_directory.assert_not_called()

    async def test_serial_create_persists_each_child(self) -> None:
        """Sequential calls (3x) each persist a child Session.

        Documents the serial-call discipline expected from SubagentResearchService
        (PR-4): SessionService._uow is a single instance per service, so callers
        must invoke create_session_with_parent sequentially per prompt. This test
        only verifies the per-call persistence outcome; real concurrent-safety
        verification lives at the caller boundary in PR-4.
        """
        uow, factory = _make_uow_and_factory()
        service = SessionService(uow_factory=factory)

        children = []
        for _ in range(3):
            child = await service.create_session_with_parent(
                user_id="u-1",
                parent_session_id="parent-1",
                tool_filter_preset="subagent_research",  # T12: required
            )
            children.append(child)

        assert len(children) == 3
        assert all(c.parent_session_id == "parent-1" for c in children)
        assert uow.session.save.await_count == 3


# ── SPM PR-1c Task 17: vnc / takeover trigger three-classification (§5.2d) ──

import asyncio  # noqa: E402

from app.domain.errors.sandbox_lifecycle import SessionUnboundError  # noqa: E402


class _RecMetrics:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, str]] = []  # (mode, trigger, outcome)

    def record_provision(
        self, *, mode, trigger, outcome, latency_seconds=None, latency=None
    ) -> None:
        self.records.append((mode, trigger, outcome))

    def last(self, *, trigger: str):
        for mode, trg, outcome in reversed(self.records):
            if trg == trigger:
                return outcome
        return None


class _TakeoverHandle:
    vnc_url = "ws://sbx:5901"

    async def read_shell_output(self, *, session_id, console=False):
        return MagicMock(success=True, data={"output": "", "console": ""})


class _VncLifecycle:
    """acquire → UNBOUND (fresh) → bind_new; bind_new/resume raise on demand."""

    def __init__(self) -> None:
        self.bind_exc: BaseException | None = None
        self.handle = _TakeoverHandle()

    async def acquire(self, session_id):
        raise SessionUnboundError(session_id)

    async def bind_new(self, session_id, *, user_id=None):
        if self.bind_exc is not None:
            raise self.bind_exc
        return self.handle

    async def resume(self, session_id):
        return self.handle


def _vnc_service(lifecycle, metrics):
    session = Session(id="s-1", user_id="u-1")
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.session = MagicMock()
    uow.session.get_by_id = AsyncMock(return_value=session)
    return SessionService(
        uow_factory=lambda: uow,
        sandbox_lifecycle_service=lifecycle,
        sandbox_provision_metrics=metrics,
    )


class TestVncTakeoverTriggerClassification:
    async def test_vnc_records_ok(self) -> None:
        metrics = _RecMetrics()
        svc = _vnc_service(_VncLifecycle(), metrics)
        url = await svc.get_vnc_url("s-1", "u-1")
        assert url == "ws://sbx:5901"
        assert metrics.last(trigger="vnc") == "ok"

    async def test_vnc_records_failed(self) -> None:
        metrics = _RecMetrics()
        lifecycle = _VncLifecycle()
        lifecycle.bind_exc = RuntimeError("boom")
        svc = _vnc_service(lifecycle, metrics)
        with pytest.raises(RuntimeError):
            await svc.get_vnc_url("s-1", "u-1")
        assert metrics.last(trigger="vnc") == "failed"

    async def test_vnc_records_cancelled(self) -> None:
        metrics = _RecMetrics()
        lifecycle = _VncLifecycle()
        lifecycle.bind_exc = asyncio.CancelledError()
        svc = _vnc_service(lifecycle, metrics)
        with pytest.raises(asyncio.CancelledError):
            await svc.get_vnc_url("s-1", "u-1")
        assert metrics.last(trigger="vnc") == "cancelled"

    async def test_takeover_records_failed(self) -> None:
        metrics = _RecMetrics()
        lifecycle = _VncLifecycle()
        lifecycle.bind_exc = RuntimeError("boom")
        svc = _vnc_service(lifecycle, metrics)
        with pytest.raises(RuntimeError):
            await svc.ensure_takeover_shell_session("s-1", "tk-1", "u-1")
        assert metrics.last(trigger="takeover") == "failed"

    async def test_takeover_records_cancelled(self) -> None:
        metrics = _RecMetrics()
        lifecycle = _VncLifecycle()
        lifecycle.bind_exc = asyncio.CancelledError()
        svc = _vnc_service(lifecycle, metrics)
        with pytest.raises(asyncio.CancelledError):
            await svc.ensure_takeover_shell_session("s-1", "tk-1", "u-1")
        assert metrics.last(trigger="takeover") == "cancelled"
