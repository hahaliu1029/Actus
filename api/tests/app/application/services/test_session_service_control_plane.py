"""C3 PR-6 (spec §11.7) — SessionService unconditionally writes
``subagent_control_plane='mailbox'`` for new subagent children.

The PR-4.5 era ``mailbox_supervisor_enabled`` runtime feature flag was
retired in PR-6: the §11.6 rollback runbook is decommissioned and the
``c3pr6_retire_legacy_ctrl_plane`` alembic migration upgrades any
historic ``legacy`` rows to ``mailbox``. The constructor still accepts
the ``settings`` and ``mailbox_flag_reader`` kwargs (back-compat) but
the values are not consulted; this test pins both the new contract
(always-mailbox) and the back-compat surface (passing ``settings`` /
``mailbox_flag_reader`` with falsey values does NOT downgrade to
``legacy``).

Reuses the ``_FakeRepo`` / ``_FakeUoW`` pattern from
``test_session_service_c1a.py`` to keep the test pure (no DB / no
FastAPI).
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


def _settings_stub(flag: bool) -> Any:
    """Lightweight stand-in for ``core.config.Settings``.

    Post-PR-6 the value is ignored by ``create_session_with_parent``;
    we still build a stub so the back-compat tests can confirm the
    ctor accepts the kwarg without error.
    """
    return SimpleNamespace(mailbox_supervisor_enabled=flag)


@pytest.mark.anyio
async def test_subagent_child_always_uses_mailbox_plane() -> None:
    """C3 PR-6 contract — every new subagent gets
    ``subagent_control_plane='mailbox'`` regardless of how the service
    was constructed."""
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))

    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )

    assert child.worker_type == "subagent"
    assert child.subagent_control_plane == "mailbox"


@pytest.mark.anyio
async def test_subagent_child_uses_mailbox_even_when_settings_flag_false() -> None:
    """Back-compat surface check — passing a ``settings`` stub whose
    ``mailbox_supervisor_enabled=False`` does NOT downgrade the new
    child to ``legacy``. The §11.6 rollback runbook is decommissioned;
    the flag is inert."""
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(
        uow_factory=_uow_factory(repo),
        settings=_settings_stub(False),
    )

    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )

    assert child.subagent_control_plane == "mailbox"


@pytest.mark.anyio
async def test_subagent_child_uses_mailbox_even_when_flag_reader_returns_false() -> None:
    """Back-compat surface check — a ``mailbox_flag_reader`` callable
    returning ``False`` is now ignored; the legacy downgrade path is
    removed."""
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(
        uow_factory=_uow_factory(repo),
        mailbox_flag_reader=lambda: False,
    )

    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )

    assert child.subagent_control_plane == "mailbox"


@pytest.mark.anyio
async def test_root_session_carries_null_control_plane() -> None:
    """Root sessions (``create_session``) never carry a control plane.

    Spec §11.2 narrows ``subagent_control_plane`` to subagent rows in
    practice — ``SessionService.create_session()`` never sets the
    column, so root rows land with ``None``. The CHECK constraint
    ``ck_sessions_subagent_control_plane_valid`` permits
    ``NULL | 'legacy' | 'mailbox'`` for any row, so the NULL-for-roots
    invariant is enforced by ``SessionService``, not by the DB. PR-6
    does not change this behavior.
    """
    repo = _FakeRepo(parent=None)
    svc = SessionService(uow_factory=_uow_factory(repo))

    root = await svc.create_session(user_id="u1")
    assert root.worker_type == "root"
    assert root.subagent_control_plane is None
