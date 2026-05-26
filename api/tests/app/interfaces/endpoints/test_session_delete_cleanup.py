"""SessionService.delete_session 清理行为单元测试.

commit 5b9199f 之后, sandbox destroy 的职责迁到 SandboxLifecycleService, 由
SessionService 通过 `self._lifecycle.destroy(session_id, DestroyReason.SESSION_DELETE)`
委托. 本测试验证两条 SessionService 层面的契约:

1. 当 lifecycle 注入时, delete_session 会发出 destroy 调用并传正确的
   DestroyReason.
2. 当 lifecycle 未注入 (shared-sandbox DI / 单测场景) 时, delete_session
   不试图 destroy, 避免空指针.

取代了 pre-refactor 依赖 `sandbox_cls` + `get_settings().sandbox_address`
的旧路径. 共享沙箱模式下 "skip destroy" 的决策现在在 DI wiring 层完成
(不注入 lifecycle), 由测试 2 间接覆盖.
"""
import asyncio

from app.application.services.session_service import SessionService
from app.domain.models.session import DestroyReason, Session


class _FakeSessionRepo:
    def __init__(
        self,
        session: Session | None,
        *,
        descendants: list[Session] | None = None,
    ) -> None:
        self._session = session
        self._sessions = {
            item.id: item
            for item in ([session] if session else []) + (descendants or [])
        }
        self._descendants = descendants or []
        self.deleted_ids: list[str] = []

    async def get_by_id(self, session_id: str):
        return self._sessions.get(session_id)

    async def find_descendants(
        self,
        ancestor_id: str,
        *,
        user_id: str,
        max_depth: int,
        limit: int,
    ):
        del max_depth, limit
        return [
            session
            for session in self._descendants
            if session.parent_session_id == ancestor_id and session.user_id == user_id
        ]

    async def delete_by_id(self, session_id: str) -> None:
        self.deleted_ids.append(session_id)
        self._sessions.pop(session_id, None)


class _FakeUnitOfWork:
    def __init__(self, repo: _FakeSessionRepo) -> None:
        self.session = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class _FakeLifecycle:
    """跟踪 destroy 调用. 模拟 SandboxLifecycleService 最小接口."""

    def __init__(self) -> None:
        self.destroy_calls: list[tuple[str, DestroyReason]] = []

    async def destroy(self, session_id: str, reason: DestroyReason) -> None:
        self.destroy_calls.append((session_id, reason))


class _FakeSupervisor:
    def __init__(self) -> None:
        self.cleanup_calls: list[dict[str, str]] = []

    async def cleanup_background_slot(self, **kwargs: str) -> None:
        self.cleanup_calls.append(kwargs)


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
    _FakeTaskCls.registry.clear()

    session = Session(
        id="s-delete-1",
        title="demo",
        user_id="owner",
        sandbox_id="sb-1",
        task_id="task-1",
    )
    repo = _FakeSessionRepo(session=session)
    task = _FakeTask()
    _FakeTaskCls.registry["task-1"] = task
    lifecycle = _FakeLifecycle()

    service = SessionService(
        uow_factory=_make_uow_factory(repo),
        task_cls=_FakeTaskCls,
        sandbox_lifecycle_service=lifecycle,
    )

    asyncio.run(service.delete_session("s-delete-1", user_id="owner", is_admin=False))

    assert task.cancel_called is True
    assert task.cancel_reason == "session_delete"
    assert lifecycle.destroy_calls == [("s-delete-1", DestroyReason.SESSION_DELETE)]
    assert repo.deleted_ids == ["s-delete-1"]


def test_delete_session_skips_sandbox_destroy_when_lifecycle_absent() -> None:
    """当 lifecycle 未注入 (e.g. shared-sandbox 模式下 DI 决定不给 SessionService
    挂 lifecycle), delete_session 必须跳过 destroy 路径, 不抛 AttributeError."""
    _FakeTaskCls.registry.clear()

    session = Session(
        id="s-delete-2",
        title="demo",
        user_id="owner",
        sandbox_id="sb-shared",
        task_id=None,
    )
    repo = _FakeSessionRepo(session=session)

    service = SessionService(
        uow_factory=_make_uow_factory(repo),
        task_cls=_FakeTaskCls,
        sandbox_lifecycle_service=None,  # DI 决定不注入: shared / 单测场景
    )

    asyncio.run(service.delete_session("s-delete-2", user_id="owner", is_admin=False))

    assert repo.deleted_ids == ["s-delete-2"]


def test_delete_session_releases_suspended_background_quota() -> None:
    _FakeTaskCls.registry.clear()

    session = Session(
        id="s-delete-bg",
        title="demo",
        user_id="owner",
        task_id=None,
        execution_mode="background",
        execution_phase="suspended",
        was_background=True,
    )
    repo = _FakeSessionRepo(session=session)
    supervisor = _FakeSupervisor()

    service = SessionService(
        uow_factory=_make_uow_factory(repo),
        task_cls=_FakeTaskCls,
        sandbox_lifecycle_service=None,
        execution_supervisor=supervisor,
    )

    asyncio.run(service.delete_session("s-delete-bg", user_id="admin", is_admin=True))

    assert supervisor.cleanup_calls == [
        {
            "session_id": "s-delete-bg",
            "user_id": "owner",
            "reason": "session_delete",
        }
    ]
    assert repo.deleted_ids == ["s-delete-bg"]


def test_delete_root_session_deletes_subagent_children_first() -> None:
    _FakeTaskCls.registry.clear()

    root = Session(
        id="root-1",
        title="root",
        user_id="owner",
        worker_type="root",
    )
    child = Session(
        id="child-1",
        title="child",
        user_id="owner",
        parent_session_id="root-1",
        worker_type="subagent",
        tool_filter_preset="subagent_research",
    )
    repo = _FakeSessionRepo(session=root, descendants=[child])

    service = SessionService(
        uow_factory=_make_uow_factory(repo),
        task_cls=_FakeTaskCls,
        sandbox_lifecycle_service=None,
    )

    asyncio.run(service.delete_session("root-1", user_id="owner", is_admin=False))

    assert repo.deleted_ids == ["child-1", "root-1"]
