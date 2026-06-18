"""PR-9b-B INV-B4 — PatchApplier.apply(lineage=...) threads root+parent
session_id into CoordinatorApplyEvent. Per-child fields stay None
(apply event is group-level).

Deviation from the plan snippet: the plan used ``MagicMock()`` as the
plan + ``PatchApplier(emit_event=...)``. Neither drives ``apply()`` to the
real ``_finalize`` emit (the apply path iterates ``plan.files``, splits
``plan.coordinator_run_id`` on ``:``, reads ``plan.total_size_bytes``,
and ``__init__`` requires 4 keyword-only ctor args). We mirror the
working construction in ``test_patch_applier_emit.py`` instead: a real
single-entry ``PatchApplyPlan`` + mocked sandbox/minio/audit/redis that
reach the SUCCESS emit, plus a capturing ``emit_event`` hook.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.group_lineage import GroupLineageFields
from app.application.services.patch_applier import (
    ApplyStatus,
    PatchApplier,
)
from app.domain.external.parent_sandbox import SandboxPathCheck
from app.domain.models.event import CoordinatorApplyEvent
from app.domain.models.patch_apply_plan import PatchApplyPlan
from app.domain.models.patch_manifest import FilePatchEntry

pytestmark = pytest.mark.anyio

_SHA_A = "a" * 64
_NEW_DIGEST = "b" * 64
_NEW_CONTENT = b"new content"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _NoopAsyncCM:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: object) -> bool:
        return False


@pytest.fixture
def parent_sandbox() -> MagicMock:
    s = MagicMock()
    s.compute_digest = AsyncMock(side_effect=[_SHA_A, _NEW_DIGEST])
    s.exists = AsyncMock(return_value=True)
    s.check_path = AsyncMock(
        return_value=SandboxPathCheck(exists=True, kind="regular")
    )
    s.read_file = AsyncMock(return_value=b"old content")
    s.atomic_write_file = AsyncMock()
    return s


@pytest.fixture
def minio() -> MagicMock:
    m = MagicMock()
    m.get_bytes = AsyncMock(return_value=_NEW_CONTENT)
    return m


@pytest.fixture
def snapshot_store() -> MagicMock:
    store = MagicMock()
    store.save = AsyncMock()
    store.discard = AsyncMock()
    return store


@pytest.fixture
def audit_repo() -> MagicMock:
    r = MagicMock()
    r.insert_in_progress = AsyncMock(return_value=42)
    r.update_terminal = AsyncMock()
    return r


@pytest.fixture
def redis_mock() -> MagicMock:
    r = MagicMock()
    r.lock = MagicMock(return_value=_NoopAsyncCM())
    return r


def _build_applier(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
    emit_event: AsyncMock,
) -> PatchApplier:
    return PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=redis_mock,
        emit_event=emit_event,
    )


def _modify_plan(coordinator_run_id: str) -> PatchApplyPlan:
    return PatchApplyPlan(
        coordinator_run_id=coordinator_run_id,
        files=(
            FilePatchEntry(
                path="d/x.py",
                op="modify",
                base_digest=_SHA_A,
                new_digest=_NEW_DIGEST,
                content_ref="ref-r1",
                content_size=len(_NEW_CONTENT),
            ),
        ),
        total_size_bytes=len(_NEW_CONTENT),
        file_count=1,
        source_work_unit_ids=("wu1",),
    )


def _apply_event(emit_event: AsyncMock) -> CoordinatorApplyEvent:
    """Extract the single CoordinatorApplyEvent emitted (HealthEvent
    only fires on rollback_partial, which doesn't happen on SUCCESS)."""
    calls = [
        call
        for call in emit_event.await_args_list
        if isinstance(call.args[0], CoordinatorApplyEvent)
    ]
    assert len(calls) == 1
    return calls[0].args[0]


async def test_apply_with_lineage_populates_group_fields(
    parent_sandbox: MagicMock,
    minio: MagicMock,
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
) -> None:
    emit_event = AsyncMock()
    applier = _build_applier(
        snapshot_store, audit_repo, redis_mock, emit_event,
    )
    plan = _modify_plan("run-7")
    lineage = GroupLineageFields(
        root_session_id="root-7", parent_session_id="parent-7",
    )

    out = await applier.apply(
        plan,
        parent_sandbox=parent_sandbox,
        minio_client=minio,
        lineage=lineage,
    )
    assert out.status == ApplyStatus.SUCCESS

    ev = _apply_event(emit_event)
    assert ev.root_session_id == "root-7"
    assert ev.parent_session_id == "parent-7"
    assert ev.coordinator_run_id == "run-7"
    # Apply event is group-level: per-child fields stay None.
    assert ev.child_session_id is None
    assert ev.work_unit_id is None


async def test_apply_with_partial_lineage_only_root(
    parent_sandbox: MagicMock,
    minio: MagicMock,
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
) -> None:
    """Partial GroupLineageFields (only root_session_id) populates that
    field without clobbering parent_session_id. Locks that the two
    conditional branches in _finalize are independent."""
    emit_event = AsyncMock()
    applier = _build_applier(
        snapshot_store, audit_repo, redis_mock, emit_event,
    )

    out = await applier.apply(
        _modify_plan("run-9"),
        parent_sandbox=parent_sandbox,
        minio_client=minio,
        lineage=GroupLineageFields(root_session_id="root-9"),
    )
    assert out.status == ApplyStatus.SUCCESS

    ev = _apply_event(emit_event)
    assert ev.root_session_id == "root-9"
    assert ev.parent_session_id is None
    assert ev.child_session_id is None
    assert ev.work_unit_id is None


async def test_apply_without_lineage_preserves_legacy_behavior(
    parent_sandbox: MagicMock,
    minio: MagicMock,
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
) -> None:
    """When lineage=None (legacy callers), only coordinator_run_id is set;
    root/parent_session_id default to None."""
    emit_event = AsyncMock()
    applier = _build_applier(
        snapshot_store, audit_repo, redis_mock, emit_event,
    )
    plan = _modify_plan("run-8")

    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.SUCCESS

    ev = _apply_event(emit_event)
    assert ev.coordinator_run_id == "run-8"
    assert ev.root_session_id is None
    assert ev.parent_session_id is None
    assert ev.child_session_id is None
    assert ev.work_unit_id is None
