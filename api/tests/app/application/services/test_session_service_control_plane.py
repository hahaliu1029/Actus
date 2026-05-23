"""C3 PR-4.5 — SessionService picks subagent_control_plane from feature flag.

Spec §11.2 — when ``mailbox_supervisor_enabled`` is true the new child's
``subagent_control_plane`` is ``'mailbox'``; otherwise ``'legacy'``. The
flag is read on every ``create_session_with_parent`` call so the rollback
runbook (§11.6) can flip behavior without a service restart.

Reuses the ``_FakeRepo`` / ``_FakeUoW`` pattern from
``test_session_service_c1a.py`` to keep the test pure (no DB / no FastAPI).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.application.services.session_service import SessionService
from app.domain.models.session import Session


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeRepo:
    def __init__(self, *, parent: Session | None, descendants_count: int = 0) -> None:
        self._parent = parent
        self._count = descendants_count
        self.saved: Session | None = None

    async def lock_session_for_spawn(self, parent_id: str, *, user_id: str):
        del parent_id
        if self._parent is None:
            return None
        if self._parent.user_id != user_id:
            return None
        return self._parent

    async def count_descendants(self, ancestor_id: str, *, user_id: str, cap: int) -> int:
        del ancestor_id, user_id, cap
        return self._count

    async def save(self, session: Session) -> None:
        self.saved = session


class _FakeUoW:
    def __init__(self, repo: _FakeRepo) -> None:
        self.session = repo

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, *args: Any) -> None:
        del args
        return None


def _uow_factory(repo: _FakeRepo):
    return lambda: _FakeUoW(repo)


def _settings(flag: bool) -> Any:
    # Lightweight stand-in for ``core.config.Settings``. SessionService only
    # touches ``mailbox_supervisor_enabled`` on it, so a SimpleNamespace
    # exposes exactly the contract being relied on.
    return SimpleNamespace(mailbox_supervisor_enabled=flag)


@pytest.mark.anyio
async def test_subagent_child_uses_mailbox_when_flag_enabled() -> None:
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo), settings=_settings(True))

    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )

    assert child.worker_type == "subagent"
    assert child.subagent_control_plane == "mailbox"


@pytest.mark.anyio
async def test_subagent_child_uses_legacy_when_flag_disabled() -> None:
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo), settings=_settings(False))

    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )

    assert child.subagent_control_plane == "legacy"


@pytest.mark.anyio
async def test_root_session_unaffected_by_flag() -> None:
    """Root sessions (``create_session``) never carry control_plane.

    Spec §11.2 narrows ``subagent_control_plane`` to subagent rows only.
    The DB CHECK constraint ``ck_sessions_subagent_control_plane_root_null``
    enforces this; the helper relies on it returning ``None`` for roots.
    """
    repo = _FakeRepo(parent=None)
    svc = SessionService(uow_factory=_uow_factory(repo), settings=_settings(True))

    root = await svc.create_session(user_id="u1")
    assert root.worker_type == "root"
    assert root.subagent_control_plane is None


@pytest.mark.anyio
async def test_mailbox_flag_reader_picks_up_live_env_changes() -> None:
    """codex r9 [R9-1, HIGH CONTRACT] — production wires a live env
    var reader instead of a cached Settings instance, so the rollback
    runbook §11.6 toggle takes effect without ``cache_clear()`` or a
    pod restart. This test pins that contract: a mutable
    ``flag_reader`` callable, called per request, drives the choice.
    """
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)

    flag = {"on": False}
    svc = SessionService(
        uow_factory=_uow_factory(repo),
        mailbox_flag_reader=lambda: flag["on"],
    )

    child_a = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )
    assert child_a.subagent_control_plane == "legacy"

    # Flip the flag — no restart, no cache_clear, no service rebuild.
    flag["on"] = True
    child_b = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )
    assert child_b.subagent_control_plane == "mailbox"


@pytest.mark.anyio
async def test_settings_stub_wins_over_flag_reader() -> None:
    """codex r9 [R9-1] — explicit Settings injection is the
    deterministic test override; it must take precedence over the
    flag_reader callable so test fixtures can pin behavior."""
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)

    svc = SessionService(
        uow_factory=_uow_factory(repo),
        settings=_settings(False),
        mailbox_flag_reader=lambda: True,  # would say mailbox but settings wins
    )
    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )
    assert child.subagent_control_plane == "legacy"


@pytest.mark.anyio
async def test_flag_flip_takes_effect_on_next_create() -> None:
    """codex r1 [R1-6, MEDIUM CONTRACT] — when a NEW SessionService is
    built with a different Settings stub, the new control_plane choice
    takes effect immediately. Rollback per §11.6 requires either a
    service restart (which gives a fresh ``get_settings()`` cache) OR
    ``get_settings.cache_clear()`` from an admin path; tests pin
    behavior via explicit Settings injection."""
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)

    # First service: flag on
    svc_on = SessionService(uow_factory=_uow_factory(repo), settings=_settings(True))
    child_a = await svc_on.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )
    assert child_a.subagent_control_plane == "mailbox"

    # Second service: flag off (simulates ops flipping
    # ``MAILBOX_SUPERVISOR_ENABLED`` and a fresh request hitting the new
    # Settings instance — the DI factory rebuilds SessionService per request)
    svc_off = SessionService(uow_factory=_uow_factory(repo), settings=_settings(False))
    child_b = await svc_off.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )
    assert child_b.subagent_control_plane == "legacy"
