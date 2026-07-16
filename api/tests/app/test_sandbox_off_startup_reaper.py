"""SPM PR-3 Task 28 — off startup does NOT schedule the Docker-dependent
terminal sandbox reaper.

The C2 leaked-sandbox startup reaper (``sweep_terminal_coordinator_active_sandboxes``)
destroys leaked per-child containers via ``lifecycle.destroy()`` → ``docker rm``.
Under off startup there is no Docker plane, so ``main.py`` must skip scheduling
it. We drive the extracted ``_run_terminal_sandbox_reaper_if_enabled`` helper
directly (the lifespan gate is ``docker_dependent_enabled=(mode != "off")``) and
spy on the sweep symbol.

G3 anti-fake guard: the ``enabled=True`` control asserts the sweep IS called —
if the spy target were mis-patched, the off "zero" would be a false green; the
control catches it (and also catches a swallowed pre-sweep exception).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.main import _run_terminal_sandbox_reaper_if_enabled

pytestmark = pytest.mark.anyio

_SWEEP_SYMBOL = (
    "app.application.services.sandbox_terminal_reaper."
    "sweep_terminal_coordinator_active_sandboxes"
)
_REPO_SYMBOL = (
    "app.infrastructure.repositories.db_session_repository.DBSessionRepository"
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSessionFactoryCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *a):
        return False


def _fake_postgres() -> MagicMock:
    pg = MagicMock()
    pg.session_factory = lambda: _FakeSessionFactoryCtx()
    return pg


def _fake_app_with_lifecycle() -> MagicMock:
    app = MagicMock()
    app.state.sandbox_lifecycle_service = MagicMock()  # non-None → reaper eligible
    return app


def _sweep_spy(calls: list):
    async def _spy(**kwargs):
        calls.append(kwargs)
        return MagicMock(destroyed=0, errored=0, scanned=0, already_gone=0)

    return _spy


async def test_off_startup_does_not_schedule_terminal_reaper() -> None:
    """docker_dependent_enabled=False (off) → the Docker-dependent sweep is
    never invoked (reaper not scheduled)."""
    calls: list = []
    with patch(_SWEEP_SYMBOL, _sweep_spy(calls)), patch(_REPO_SYMBOL, MagicMock()):
        await _run_terminal_sandbox_reaper_if_enabled(
            _fake_app_with_lifecycle(),
            _fake_postgres(),
            docker_dependent_enabled=False,
        )
    assert calls == []  # off → terminal sandbox reaper NOT scheduled


async def test_enabled_startup_schedules_terminal_reaper() -> None:
    """Control / G3 anti-fake: docker_dependent_enabled=True DOES call the sweep
    exactly once — proves the spy target is the real symbol (off 'zero' is real,
    not a mis-patch) and no pre-sweep exception is silently swallowed."""
    calls: list = []
    with patch(_SWEEP_SYMBOL, _sweep_spy(calls)), patch(_REPO_SYMBOL, MagicMock()):
        await _run_terminal_sandbox_reaper_if_enabled(
            _fake_app_with_lifecycle(),
            _fake_postgres(),
            docker_dependent_enabled=True,
        )
    assert len(calls) == 1  # non-off → terminal sandbox reaper scheduled
