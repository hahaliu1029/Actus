"""T12 / Phase 1 PR-X — SessionService.create_session_with_parent persists
``tool_filter_preset`` so ``AgentService._create_task`` can re-derive the
in-memory allowlist after a pod restart.

Companion to ``test_session_service.py`` (PR-1 contract) — that file pinned
``sample_session_id`` plumbing; this file pins the T12 ``tool_filter_preset``
extension. Kept separate so the T12 boundary stays grep-traceable.
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
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.session = MagicMock()
    uow.session.save = AsyncMock()
    return uow, lambda: uow


class TestCreateSessionWithParentToolFilterPreset:
    async def test_missing_preset_raises_value_error(self) -> None:
        """Codex R1 P1 fix: omitting the kwarg MUST raise — every child
        session created via this method needs a persisted restriction.

        This closes the bypass where a future caller could persist a child
        row with ``tool_filter_preset=NULL`` and have it reconstructed
        unrestricted on resume. The DB CHECK constraint
        ``ck_sessions_child_must_have_preset`` is the last-line defense;
        this ValueError is the friendly app-level diagnostic.
        """
        uow, factory = _make_uow_and_factory()
        service = SessionService(uow_factory=factory)

        with pytest.raises(ValueError, match="tool_filter_preset"):
            await service.create_session_with_parent(
                user_id="u-1", sample_session_id="parent-1",
            )

        # Critical: nothing should have hit the UoW save path on a rejected
        # call. Otherwise a partial-failure could leave a half-persisted row.
        uow.session.save.assert_not_called()

    async def test_unknown_preset_raises_value_error(self) -> None:
        """Unknown preset rejected at the app layer before reaching DB.

        Provides a friendlier error than the DB CHECK violation (which is
        the same defense but surfaces as IntegrityError after a roundtrip).
        Pins the app-level fail-closed contract.
        """
        uow, factory = _make_uow_and_factory()
        service = SessionService(uow_factory=factory)

        with pytest.raises(ValueError, match="unknown tool_filter_preset"):
            await service.create_session_with_parent(
                user_id="u-1",
                sample_session_id="parent-1",
                tool_filter_preset="not_a_real_preset",
            )
        uow.session.save.assert_not_called()

    async def test_preset_subagent_research_persisted(self) -> None:
        """SubagentResearchService passes 'subagent_research'; it must round-trip."""
        uow, factory = _make_uow_and_factory()
        service = SessionService(uow_factory=factory)

        child = await service.create_session_with_parent(
            user_id="u-1",
            sample_session_id="parent-1",
            tool_filter_preset="subagent_research",
        )
        assert child.tool_filter_preset == "subagent_research"
        assert child.sample_session_id == "parent-1"

        saved = uow.session.save.call_args.args[0]
        assert saved.tool_filter_preset == "subagent_research"

    async def test_preset_and_parent_id_independently_settable(self) -> None:
        """Validates the kwargs are positional-safe and independent.

        Future presets may add more keys to TOOL_FILTER_PRESETS; pin that
        the signature can carry any valid registry key alongside the
        parent linkage.
        """
        uow, factory = _make_uow_and_factory()
        service = SessionService(uow_factory=factory)

        child = await service.create_session_with_parent(
            user_id="u-1",
            sample_session_id="parent-x",
            tool_filter_preset="subagent_research",
        )
        assert child.sample_session_id == "parent-x"
        assert child.tool_filter_preset == "subagent_research"
