import asyncio

from app.application.services.session_service import SessionService
from app.domain.models.session import Session


class _FakeSessionRepo:
    def __init__(self, session: Session | None) -> None:
        self._session = session
        self.deleted_ids: list[str] = []

    async def get_by_id(self, session_id: str):
        if not self._session:
            return None
        return self._session if self._session.id == session_id else None

    async def delete_by_id(self, session_id: str) -> None:
        self.deleted_ids.append(session_id)
        if self._session and self._session.id == session_id:
            self._session = None


class _FakeUnitOfWork:
    def __init__(self, repo: _FakeSessionRepo) -> None:
        self.session = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class _FakeSandbox:
    registry: dict[str, "_FakeSandbox"] = {}

    def __init__(self, sandbox_id: str) -> None:
        self._sandbox_id = sandbox_id
        self.destroy_called = False

    @property
    def id(self) -> str:
        return self._sandbox_id

    @property
    def cdp_url(self) -> str:
        return "http://127.0.0.1:9222"

    @property
    def vnc_url(self) -> str:
        return "ws://127.0.0.1:5901"

    async def destroy(self) -> bool:
        self.destroy_called = True
        return True

    @classmethod
    async def get(cls, sandbox_id: str):
        return cls.registry.get(sandbox_id)


class _FakeTask:
    def __init__(self) -> None:
        self.cancel_called = False
        self.cancel_reason: str | None = None

    def cancel(self, reason: str = "stop") -> bool:
        self.cancel_called = True
        self.cancel_reason = reason
        return True


class _FakeTaskCls:
    registry: dict[str, _FakeTask] = {}

    @classmethod
    def get(cls, task_id: str):
        return cls.registry.get(task_id)


def _make_uow_factory(repo: _FakeSessionRepo):
    def factory() -> _FakeUnitOfWork:
        return _FakeUnitOfWork(repo=repo)

    return factory


def test_delete_session_cleans_related_task_and_sandbox() -> None:
    _FakeSandbox.registry.clear()
    _FakeTaskCls.registry.clear()

    session = Session(
        id="s-delete-1",
        title="demo",
        user_id="owner",
        sandbox_id="sb-1",
        task_id="task-1",
    )
    repo = _FakeSessionRepo(session=session)
    sandbox = _FakeSandbox("sb-1")
    task = _FakeTask()
    _FakeSandbox.registry["sb-1"] = sandbox
    _FakeTaskCls.registry["task-1"] = task

    service = SessionService(
        uow_factory=_make_uow_factory(repo),
        sandbox_cls=_FakeSandbox,
        task_cls=_FakeTaskCls,
    )

    asyncio.run(service.delete_session("s-delete-1", user_id="owner", is_admin=False))

    assert task.cancel_called is True
    assert task.cancel_reason == "session_delete"
    assert sandbox.destroy_called is True
    assert repo.deleted_ids == ["s-delete-1"]


def test_delete_session_skips_sandbox_destroy_when_shared_sandbox(monkeypatch) -> None:
    _FakeSandbox.registry.clear()
    _FakeTaskCls.registry.clear()

    session = Session(
        id="s-delete-2",
        title="demo",
        user_id="owner",
        sandbox_id="sb-shared",
        task_id=None,
    )
    repo = _FakeSessionRepo(session=session)
    sandbox = _FakeSandbox("sb-shared")
    _FakeSandbox.registry["sb-shared"] = sandbox

    class _Settings:
        sandbox_address = "shared-sandbox.example.com"

    monkeypatch.setattr(
        "app.application.services.session_service.get_settings",
        lambda: _Settings(),
    )

    service = SessionService(
        uow_factory=_make_uow_factory(repo),
        sandbox_cls=_FakeSandbox,
        task_cls=_FakeTaskCls,
    )

    asyncio.run(service.delete_session("s-delete-2", user_id="owner", is_admin=False))

    assert sandbox.destroy_called is False
    assert repo.deleted_ids == ["s-delete-2"]
