import asyncio

import pytest
from app.application.errors.exceptions import ForbiddenError
from app.application.services.session_service import SessionService
from app.domain.models.session import Session


class FakeSessionRepo:
    def __init__(self, session: Session | None, all_sessions: list[Session] | None = None):
        self._session = session
        self._all_sessions = all_sessions or ([] if session is None else [session])

    async def get_by_id(self, session_id: str):
        if not self._session:
            return None
        return self._session if self._session.id == session_id else None

    async def get_all(self):
        return self._all_sessions

    async def get_all_by_user(self, user_id: str):
        return [session for session in self._all_sessions if session.user_id == user_id]

    async def save(self, session: Session):
        self._session = session
        return session


class FakeUnitOfWork:
    def __init__(self, session: Session | None, all_sessions: list[Session] | None = None):
        self.session = FakeSessionRepo(session=session, all_sessions=all_sessions)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sb-1") -> None:
        self._sandbox_id = sandbox_id
        self.ensure_called = False

    @property
    def id(self) -> str:
        return self._sandbox_id

    @property
    def vnc_url(self) -> str:
        return "ws://127.0.0.1:5901"

    async def ensure_sandbox(self) -> None:
        self.ensure_called = True

    @classmethod
    async def get(cls, sandbox_id: str):
        return None

    @classmethod
    async def create(cls):
        return cls()


def make_uow_factory(session: Session | None, all_sessions: list[Session] | None = None):
    def factory() -> FakeUnitOfWork:
        return FakeUnitOfWork(session=session, all_sessions=all_sessions)

    return factory


def test_get_session_rejects_non_owner() -> None:
    session = Session(id="s1", title="demo", user_id="owner")
    service = SessionService(
        uow_factory=make_uow_factory(session=session),
        sandbox_cls=FakeSandbox,
    )

    with pytest.raises(ForbiddenError):
        asyncio.run(service.get_session("s1", user_id="visitor", is_admin=False))


def test_get_session_allows_admin_cross_user() -> None:
    session = Session(id="s1", title="demo", user_id="owner")
    service = SessionService(
        uow_factory=make_uow_factory(session=session),
        sandbox_cls=FakeSandbox,
    )

    result = asyncio.run(service.get_session("s1", user_id="admin", is_admin=True))
    assert result.id == "s1"


def test_get_all_sessions_admin_can_get_all() -> None:
    sessions = [
        Session(id="s1", title="a", user_id="u1"),
        Session(id="s2", title="b", user_id="u2"),
    ]
    service = SessionService(
        uow_factory=make_uow_factory(session=sessions[0], all_sessions=sessions),
        sandbox_cls=FakeSandbox,
    )

    result = asyncio.run(service.get_all_sessions(user_id="admin", is_admin=True))
    assert len(result) == 2


def test_get_vnc_url_auto_creates_sandbox_when_missing() -> None:
    session = Session(id="s1", title="demo", user_id="owner", sandbox_id=None)
    service = SessionService(
        uow_factory=make_uow_factory(session=session),
        sandbox_cls=FakeSandbox,
    )

    vnc_url = asyncio.run(service.get_vnc_url("s1", user_id="owner", is_admin=False))
    assert vnc_url == "ws://127.0.0.1:5901"
