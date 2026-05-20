"""C1a tests: SessionService.create_session_with_parent owner check + cap + worker_type."""
from __future__ import annotations

import pytest

from app.application.errors.exceptions import NotFoundError
from app.application.services.session_service import SessionService
from app.domain.models.session import Session
from app.domain.services.subagent_limits import (
    MAX_DESCENDANTS_PER_ROOT,
    SpawnCapExceeded,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeRepo:
    def __init__(self, *, parent: Session | None, descendants_count: int = 0):
        self._parent = parent
        self._count = descendants_count
        self.saved: Session | None = None

    async def lock_session_for_spawn(self, parent_id: str, *, user_id: str):
        del parent_id
        if self._parent is None:
            return None
        # Mirror the SQL `WHERE user_id == :user_id` filter so cross-tenant
        # calls collapse to None — identical to the real repository.
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

    async def __aexit__(self, *args):
        del args
        return None


def _uow_factory(repo: _FakeRepo):
    return lambda: _FakeUoW(repo)


@pytest.mark.anyio
async def test_creates_subagent_child_with_explicit_worker_type():
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))

    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )

    assert child.worker_type == "subagent"
    assert child.parent_session_id == "p"
    assert repo.saved is child


@pytest.mark.anyio
async def test_raises_not_found_when_parent_missing():
    repo = _FakeRepo(parent=None)
    svc = SessionService(uow_factory=_uow_factory(repo))
    with pytest.raises(NotFoundError):
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="nope",
            tool_filter_preset="subagent_research",
        )


@pytest.mark.anyio
async def test_raises_not_found_when_parent_belongs_to_other_user():
    """ID enumeration defense - foreign parent must look identical to missing."""
    parent = Session(id="p", user_id="OTHER", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))
    with pytest.raises(NotFoundError):
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="p",
            tool_filter_preset="subagent_research",
        )


@pytest.mark.anyio
async def test_raises_descendants_cap():
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent, descendants_count=MAX_DESCENDANTS_PER_ROOT)
    svc = SessionService(uow_factory=_uow_factory(repo))
    with pytest.raises(SpawnCapExceeded) as exc_info:
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="p",
            tool_filter_preset="subagent_research",
        )
    assert exc_info.value.kind == "descendants"


@pytest.mark.anyio
async def test_raises_depth_cap_when_parent_is_already_subagent():
    parent = Session(
        id="p", user_id="u1", worker_type="subagent", parent_session_id="root"
    )
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))
    with pytest.raises(SpawnCapExceeded) as exc_info:
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="p",
            tool_filter_preset="subagent_research",
        )
    assert exc_info.value.kind == "depth"


@pytest.mark.anyio
async def test_unknown_tool_filter_preset_raises():
    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))
    with pytest.raises(ValueError, match="unknown tool_filter_preset"):
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="p",
            tool_filter_preset="not_a_real_preset",
        )


@pytest.mark.anyio
async def test_custom_descendants_cap_is_used():
    """Codex P1 regression: SubagentLimitsConfig.max_descendants_per_root must drive the gate.

    Inject a custom limits instance (max=3) and seed 3 descendants; the next spawn
    must raise SpawnCapExceeded with cap=3 (NOT the module default of 10).
    """
    from core.config import SubagentLimitsConfig

    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent, descendants_count=3)
    svc = SessionService(
        uow_factory=_uow_factory(repo),
        subagent_limits=SubagentLimitsConfig(
            max_subagent_depth=1, max_descendants_per_root=3
        ),
    )
    with pytest.raises(SpawnCapExceeded) as exc_info:
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="p",
            tool_filter_preset="subagent_research",
        )
    assert exc_info.value.kind == "descendants"
    assert exc_info.value.cap == 3
    assert exc_info.value.current == 3


@pytest.mark.anyio
async def test_unsupported_max_subagent_depth_raises_not_implemented():
    """Codex P1 regression: max_subagent_depth > 1 must fail loudly (Phase 1 only supports 1)."""
    from core.config import SubagentLimitsConfig

    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(
        uow_factory=_uow_factory(repo),
        subagent_limits=SubagentLimitsConfig(
            max_subagent_depth=2, max_descendants_per_root=10
        ),
    )
    with pytest.raises(NotImplementedError, match="max_subagent_depth > 1"):
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="p",
            tool_filter_preset="subagent_research",
        )
