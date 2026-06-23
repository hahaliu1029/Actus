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
    """INV-B1: at default config=1, a valid depth-1 subagent parent → reject
    with the byte-identical legacy tuple SpawnCapExceeded('depth', 2, 1)."""
    parent = Session(
        id="p",
        user_id="u1",
        worker_type="subagent",
        parent_session_id="root",
        depth=1,
        root_session_id="root",
    )
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))  # default config=1
    with pytest.raises(SpawnCapExceeded) as exc_info:
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="p",
            tool_filter_preset="subagent_research",
        )
    assert exc_info.value.kind == "depth"
    assert exc_info.value.current == 2
    assert exc_info.value.cap == 1


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
async def test_max_subagent_depth_2_accepts_child_from_root():
    """S3 PR-2: with the ceiling lifted to 2, a root parent still yields a
    valid depth-1 child (the only prod-reachable shape; INV-B1/INV-0/INV-B2)."""
    from core.config import SubagentLimitsConfig

    parent = Session(id="p", user_id="u1", worker_type="root")
    repo = _FakeRepo(parent=parent)
    svc = SessionService(
        uow_factory=_uow_factory(repo),
        subagent_limits=SubagentLimitsConfig(
            max_subagent_depth=2, max_descendants_per_root=10
        ),
    )
    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )
    assert child.depth == 1
    assert child.root_session_id == "p"


@pytest.mark.anyio
async def test_depth_2_grandchild_accepted_at_ceiling_2():
    """S3 PR-2: direct construction of a valid depth-1 subagent parent; at
    config=2 a grandchild (depth 2) is accepted, chaining root to the true root.
    Dormant in prod (INV-B2) — reachable only from this unit test."""
    from core.config import SubagentLimitsConfig

    parent = Session(
        id="c1",
        user_id="u1",
        worker_type="subagent",
        parent_session_id="root",
        depth=1,
        root_session_id="root",
    )
    repo = _FakeRepo(parent=parent)
    svc = SessionService(
        uow_factory=_uow_factory(repo),
        subagent_limits=SubagentLimitsConfig(
            max_subagent_depth=2, max_descendants_per_root=10
        ),
    )
    grandchild = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="c1",
        tool_filter_preset="subagent_research",
    )
    assert grandchild.depth == 2
    assert grandchild.root_session_id == "root"  # true root, not the immediate parent


@pytest.mark.anyio
async def test_depth_3_rejected_with_current_3_not_legacy_literal_2():
    """S3 PR-2: a depth-2 parent → child_depth=3 > ceiling=2 → reject with
    current=3 (the legacy gate hardcoded the literal 2 regardless of depth)."""
    from core.config import SubagentLimitsConfig

    parent = Session(
        id="c2",
        user_id="u1",
        worker_type="subagent",
        parent_session_id="c1",
        depth=2,
        root_session_id="root",
    )
    repo = _FakeRepo(parent=parent)
    svc = SessionService(
        uow_factory=_uow_factory(repo),
        subagent_limits=SubagentLimitsConfig(
            max_subagent_depth=2, max_descendants_per_root=10
        ),
    )
    with pytest.raises(SpawnCapExceeded) as exc_info:
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="c2",
            tool_filter_preset="subagent_research",
        )
    assert exc_info.value.kind == "depth"
    assert exc_info.value.current == 3
    assert exc_info.value.cap == 2


@pytest.mark.anyio
async def test_corrupt_lineage_parent_set_but_depth_zero_fails_closed():
    """S3 PR-2 §4.2: a row with parent_session_id set but depth=0 violates
    INV-A1. The gate fails closed (kind='depth') rather than spawning from it."""
    parent = Session(
        id="bad",
        user_id="u1",
        worker_type="subagent",
        parent_session_id="root",
        depth=0,  # corrupt: a subagent must be depth ≥ 1
    )
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))  # default config=1
    with pytest.raises(SpawnCapExceeded) as exc_info:
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="bad",
            tool_filter_preset="subagent_research",
        )
    assert exc_info.value.kind == "depth"


@pytest.mark.anyio
async def test_corrupt_lineage_root_with_positive_depth_fails_closed():
    """S3 PR-2 §4.2: the mirror corruption — no parent but depth>0 — also fails
    closed before the depth math."""
    parent = Session(
        id="bad2",
        user_id="u1",
        worker_type="root",
        parent_session_id=None,
        depth=2,  # corrupt: a root must be depth 0
    )
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))
    with pytest.raises(SpawnCapExceeded) as exc_info:
        await svc.create_session_with_parent(
            user_id="u1",
            parent_session_id="bad2",
            tool_filter_preset="subagent_research",
        )
    assert exc_info.value.kind == "depth"


@pytest.mark.anyio
async def test_child_lineage_set_from_root_parent():
    """S3 PR-1 / INV-A1: a child spawned from a root carries depth=1 and
    root_session_id = parent.id (parent's effective root)."""
    parent = Session(id="p", user_id="u1", worker_type="root")  # depth=0, root=None
    repo = _FakeRepo(parent=parent)
    svc = SessionService(uow_factory=_uow_factory(repo))
    child = await svc.create_session_with_parent(
        user_id="u1",
        parent_session_id="p",
        tool_filter_preset="subagent_research",
    )
    assert child.depth == 1
    assert child.root_session_id == "p"


@pytest.mark.anyio
async def test_create_session_root_lineage_defaults():
    """S3 PR-1: a root session created via create_session has depth=0 / root=None."""
    repo = _FakeRepo(parent=None)
    svc = SessionService(uow_factory=_uow_factory(repo))
    s = await svc.create_session("u1")
    assert s.depth == 0
    assert s.root_session_id is None
    assert repo.saved is s
