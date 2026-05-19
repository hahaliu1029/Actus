"""PR-1: SessionService.create_session_with_parent — child session creation
for the Phase 1 minimal subagent feature.

Contract:
- Stores sample_session_id on the new Session entity
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

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_uow_and_factory():
    """Returns (uow_mock, factory) so tests can both inject and introspect."""
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.session = MagicMock()
    uow.session.save = AsyncMock()
    return uow, lambda: uow


class TestCreateSessionWithParent:
    async def test_sets_sample_session_id(self) -> None:
        """create_session_with_parent stores sample_session_id on the child Session."""
        uow, factory = _make_uow_and_factory()
        service = SessionService(uow_factory=factory)

        child = await service.create_session_with_parent(
            user_id="u-1",
            sample_session_id="parent-1",
            # T12: preset is required for every child created via this method
            # (codex R1 P1 fix). Existing PR-1 contract preserved by passing
            # the canonical subagent_research preset.
            tool_filter_preset="subagent_research",
        )

        assert child.user_id == "u-1"
        assert child.sample_session_id == "parent-1"
        assert child.title == "新对话"

        uow.session.save.assert_awaited_once()
        saved = uow.session.save.call_args.args[0]
        assert saved.sample_session_id == "parent-1"
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
            sample_session_id="parent-1",
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
                sample_session_id="parent-1",
                tool_filter_preset="subagent_research",  # T12: required
            )
            children.append(child)

        assert len(children) == 3
        assert all(c.sample_session_id == "parent-1" for c in children)
        assert uow.session.save.await_count == 3
