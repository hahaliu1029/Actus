"""PR-5B regression: SessionService.create_session must trigger FsReconciler
lazy walk when ``fs_reconciler`` is injected.

Bug scenarios this prevents:
- No lazy walk means orphan files on disk never get quarantined and orphan
  DB rows never get rebuilt — three-view consistency silently rots until an
  operator manually runs the CLI.
- A crashing walk must not take the session creation down with it — ``create_session``
  has already persisted the session row; reconciler is advisory.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.session_service import SessionService

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_uow_factory():
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.session = MagicMock()
    uow.session.save = AsyncMock()
    return lambda: uow


class TestCreateSessionFiresWalk:
    async def test_walk_called_with_user_id(self) -> None:
        reconciler = MagicMock()
        reconciler.walk_user_directory = AsyncMock(return_value={"walked": True})

        service = SessionService(
            uow_factory=_make_uow_factory(),
            fs_reconciler=reconciler,
        )
        await service.create_session("user-abc")
        # fire-and-forget: yield to loop so the task runs
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        reconciler.walk_user_directory.assert_awaited_once_with("user-abc")

    async def test_walk_exception_does_not_break_session_creation(self) -> None:
        reconciler = MagicMock()
        reconciler.walk_user_directory = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        service = SessionService(
            uow_factory=_make_uow_factory(),
            fs_reconciler=reconciler,
        )
        session = await service.create_session("user-boom")
        # 强制让 fire-and-forget 任务跑完，不然测试 teardown 时 asyncio 会告警
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert session is not None
        assert session.user_id == "user-boom"

    async def test_no_reconciler_no_hook_call(self) -> None:
        """回归：未注入 reconciler 时 SessionService 仍然 work（legacy 路径）。"""
        service = SessionService(uow_factory=_make_uow_factory())
        session = await service.create_session("user-nohook")
        assert session is not None
