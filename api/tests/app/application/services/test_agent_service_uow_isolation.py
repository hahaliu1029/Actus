"""Tests for UoW per-call isolation, snapshot capture, and shutdown in AgentService."""
import asyncio
import re
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestUoWSourceInspection:
    """Auxiliary: static check that self._uow shared instance is gone."""

    def test_no_bare_self_uow_in_source(self):
        from app.application.services.agent_service import AgentService
        source = inspect.getsource(AgentService)
        bare = re.findall(r'self\._uow(?!_factory)', source)
        assert len(bare) == 0, f"Found {len(bare)} bare self._uow references"


class TestUoWRuntimeIsolation:
    """Behavioral: concurrent calls get independent UoW instances."""

    @pytest.mark.anyio
    async def test_concurrent_calls_get_distinct_uow(self):
        from app.application.services.agent_service import AgentService, _ConfigSnapshot

        captured_uows = []

        class FakeUoW:
            def __init__(self):
                self.session = MagicMock()
                self.session.get_by_id = AsyncMock(return_value=None)

            async def __aenter__(self):
                captured_uows.append(self)
                return self

            async def __aexit__(self, *args):
                pass

        svc = AgentService.__new__(AgentService)
        svc._uow_factory = FakeUoW
        svc._config_snapshot = MagicMock(spec=_ConfigSnapshot)
        svc._settings = MagicMock()
        svc._background_tasks = set()
        svc._pending_timeout_tasks = {}
        svc._takeover_timeout_tasks = {}

        # Call _get_accessible_session twice concurrently — both will fail with NotFoundError
        # because FakeUoW returns None from get_by_id, but we care about UoW isolation
        with pytest.raises(Exception):
            await asyncio.gather(
                svc._get_accessible_session("s1", "u1", False),
                svc._get_accessible_session("s2", "u1", False),
            )

        assert len(captured_uows) == 2
        assert captured_uows[0] is not captured_uows[1]


class TestShutdownCancelsRealTasks:
    """Behavioral: shutdown() cancels tasks and calls _task_cls.destroy()."""

    @pytest.mark.anyio
    async def test_shutdown_cancels_background_tasks(self):
        from app.application.services.agent_service import AgentService

        svc = AgentService.__new__(AgentService)
        svc._background_tasks = set()
        svc._pending_timeout_tasks = {}
        svc._takeover_timeout_tasks = {}
        svc._confirmation_sweep_task = None
        svc._task_cls = MagicMock()
        svc._task_cls.destroy = AsyncMock()

        async def never_finish():
            await asyncio.sleep(3600)

        task1 = asyncio.create_task(never_finish())
        task2 = asyncio.create_task(never_finish())
        svc._background_tasks.add(task1)
        svc._background_tasks.add(task2)

        await svc.shutdown()
        await asyncio.sleep(0)

        assert task1.cancelled() or task1.cancelling() > 0
        assert task2.cancelled() or task2.cancelling() > 0
        assert len(svc._background_tasks) == 0
        svc._task_cls.destroy.assert_awaited_once()
