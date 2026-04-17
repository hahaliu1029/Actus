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
        # 捕获 bind_new 收到的 user_id kwarg，用于回归 admin 越权 bug：
        # SessionService 必须**不**把 requester user_id 透传给 bind_new，
        # 应交由 bind_new 内部回退到 session.user_id。
        self.bind_new_user_ids: list[str | None] = []

    async def acquire(self, session_id: str) -> FakeSandboxHandle:
        self.acquire_calls.append(session_id)
        raise SessionUnboundError(session_id)

    async def bind_new(
        self, session_id: str, *, user_id: str | None = None
    ) -> FakeSandboxHandle:
        self.bind_new_calls.append(session_id)
        self.bind_new_user_ids.append(user_id)
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


# ── admin 越权 regression（M1 memory 系统配套修复） ─────────────────────────
# 场景：管理员通过 _get_accessible_session 拿到**他人** session 的访问权。
# 此时 SessionService 若把 requester 的 user_id 透传给 bind_new，bind_new 会
# 把管理员自己的 memory 目录挂进 session owner 的 sandbox 里，构成租户越权。
# 修复后 SessionService 必须不传 user_id，交由 bind_new 内部回退到 session.user_id。


def test_get_vnc_url_admin_does_not_leak_requester_user_id_into_bind_new() -> None:
    """admin 打开他人 session 的 VNC 时，bind_new 不应收到 admin 的 user_id。"""
    session = Session(id="s1", title="demo", user_id="session-owner", sandbox_id=None)
    handle = FakeSandboxHandle()
    lifecycle = FakeLifecycle(handle)
    service = SessionService(
        uow_factory=make_uow_factory(session=session),
        sandbox_lifecycle_service=lifecycle,
    )

    # admin 请求他人 session → 获得访问权
    asyncio.run(service.get_vnc_url("s1", user_id="admin-impersonator", is_admin=True))

    assert lifecycle.bind_new_calls == ["s1"]
    # 关键断言：SessionService 没有把 admin 的 user_id 透传给 bind_new。
    # bind_new 会走内部回退把 session.user_id("session-owner") 用作 mount user。
    assert lifecycle.bind_new_user_ids == [None], (
        "SessionService 不应把 requester user_id 透传给 bind_new；"
        f"实际收到 {lifecycle.bind_new_user_ids!r}"
    )


class _FakeResult:
    def __init__(self, success: bool) -> None:
        self.success = success
        self.message = "ok"


class _FakeTakeoverHandle:
    """ensure_takeover_shell_session 需要 handle 提供 read_shell_output."""

    @property
    def vnc_url(self) -> str:
        return "ws://127.0.0.1:5901"

    async def read_shell_output(self, *, session_id: str, console: bool):  # noqa: D401
        # success=True 直接跳过 exec_command 分支，让断言聚焦在 bind_new 调用
        return _FakeResult(success=True)


def test_ensure_takeover_admin_does_not_leak_requester_user_id_into_bind_new() -> None:
    """admin 接管他人 session 的 shell 时同样不应把 admin user_id 透传 bind_new。"""
    session = Session(id="s1", title="demo", user_id="session-owner", sandbox_id=None)
    handle = _FakeTakeoverHandle()
    lifecycle = FakeLifecycle(handle)  # handle 类型不影响本测试，只断言 bind_new
    service = SessionService(
        uow_factory=make_uow_factory(session=session),
        sandbox_lifecycle_service=lifecycle,
    )

    asyncio.run(
        service.ensure_takeover_shell_session(
            "s1",
            takeover_id="t1",
            user_id="admin-impersonator",
            is_admin=True,
        )
    )

    assert lifecycle.bind_new_calls == ["s1"]
    assert lifecycle.bind_new_user_ids == [None], (
        "ensure_takeover_shell_session 不应把 requester user_id 透传给 bind_new；"
        f"实际收到 {lifecycle.bind_new_user_ids!r}"
    )
