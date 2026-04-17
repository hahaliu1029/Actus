"""SessionService 访问控制 + vnc_url 单元测试.

Sandbox 路径已随 commit 5b9199f 迁到 SandboxLifecycleService；构造器不再接受
`sandbox_cls`。本文件跟随 `test_session_service_sandbox_resume.py` 的 fake
lifecycle 模式重新表达。
"""
import asyncio

import pytest
from app.application.errors.exceptions import ForbiddenError
from app.application.services.session_service import SessionService
from app.domain.errors.sandbox_lifecycle import SessionUnboundError
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


class FakeSandboxHandle:
    """最小 SandboxHandle replica — 只暴露 get_vnc_url 测试用到的 vnc_url."""

    @property
    def vnc_url(self) -> str:
        return "ws://127.0.0.1:5901"


class FakeLifecycle:
    """FakeLifecycle 复用 test_session_service_sandbox_resume.py pattern.

    acquire → SessionUnboundError, bind_new → handle. 这条路径精确对应
    `SessionService.get_vnc_url` 的"UNBOUND → bind_new"逻辑.
    """

    def __init__(self, handle: FakeSandboxHandle) -> None:
        self.handle = handle
        self.acquire_calls: list[str] = []
        self.bind_new_calls: list[str] = []

    async def acquire(self, session_id: str) -> FakeSandboxHandle:
        self.acquire_calls.append(session_id)
        raise SessionUnboundError(session_id)

    async def bind_new(self, session_id: str) -> FakeSandboxHandle:
        self.bind_new_calls.append(session_id)
        return self.handle


def make_uow_factory(session: Session | None, all_sessions: list[Session] | None = None):
    def factory() -> FakeUnitOfWork:
        return FakeUnitOfWork(session=session, all_sessions=all_sessions)

    return factory


def test_get_session_rejects_non_owner() -> None:
    session = Session(id="s1", title="demo", user_id="owner")
    service = SessionService(uow_factory=make_uow_factory(session=session))

    with pytest.raises(ForbiddenError):
        asyncio.run(service.get_session("s1", user_id="visitor", is_admin=False))


def test_get_session_allows_admin_cross_user() -> None:
    session = Session(id="s1", title="demo", user_id="owner")
    service = SessionService(uow_factory=make_uow_factory(session=session))

    result = asyncio.run(service.get_session("s1", user_id="admin", is_admin=True))
    assert result.id == "s1"


def test_get_all_sessions_admin_can_get_all() -> None:
    sessions = [
        Session(id="s1", title="a", user_id="u1"),
        Session(id="s2", title="b", user_id="u2"),
    ]
    service = SessionService(
        uow_factory=make_uow_factory(session=sessions[0], all_sessions=sessions),
    )

    result = asyncio.run(service.get_all_sessions(user_id="admin", is_admin=True))
    assert len(result) == 2


def test_get_vnc_url_auto_creates_sandbox_when_missing() -> None:
    """UNBOUND 会话调 get_vnc_url 时, lifecycle.acquire 抛 SessionUnboundError,
    SessionService 走 bind_new 路径返回新沙箱的 vnc_url."""
    session = Session(id="s1", title="demo", user_id="owner", sandbox_id=None)
    handle = FakeSandboxHandle()
    lifecycle = FakeLifecycle(handle)
    service = SessionService(
        uow_factory=make_uow_factory(session=session),
        sandbox_lifecycle_service=lifecycle,
    )

    vnc_url = asyncio.run(service.get_vnc_url("s1", user_id="owner", is_admin=False))
    assert vnc_url == "ws://127.0.0.1:5901"
    assert lifecycle.acquire_calls == ["s1"]
    assert lifecycle.bind_new_calls == ["s1"]
