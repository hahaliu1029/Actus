"""C2 PR-8 §13 Task 8.4 — emit tests for patch_applier._finalize.

Asserts that the applier emits a CoordinatorApplyEvent through its
`_emit_event` async hook after the terminal audit row is written.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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


@pytest.fixture
def emit_event() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def applier(
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


def _modify_plan() -> PatchApplyPlan:
    return PatchApplyPlan(
        coordinator_run_id="r1",
        files=(
            FilePatchEntry(
                path="d/x.py", op="modify",
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


@pytest.mark.anyio
async def test_finalize_emits_coordinator_apply_event_on_success(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    emit_event: AsyncMock,
) -> None:
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.SUCCESS

    # Find the CoordinatorApplyEvent emit (HealthEvent fires only on
    # rollback_partial, which doesn't apply here).
    apply_ev_calls = [
        call for call in emit_event.await_args_list
        if isinstance(call.args[0], CoordinatorApplyEvent)
    ]
    assert len(apply_ev_calls) == 1
    ev = apply_ev_calls[0].args[0]
    assert ev.apply_status == "success"
    assert ev.file_count == 1
    assert ev.total_bytes == len(_NEW_CONTENT)
    assert ev.failed_at_path is None
    assert ev.rollback_status is None
    assert ev.coordinator_run_id == "r1"


@pytest.mark.anyio
async def test_finalize_emits_failed_apply_event_with_path(
    applier: PatchApplier,
    parent_sandbox: MagicMock,
    minio: MagicMock,
    emit_event: AsyncMock,
) -> None:
    """Digest-drift abort → CoordinatorApplyEvent with failed_at_path set."""
    parent_sandbox.compute_digest = AsyncMock(return_value="c" * 64)
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.DIGEST_DRIFT

    apply_ev_calls = [
        call for call in emit_event.await_args_list
        if isinstance(call.args[0], CoordinatorApplyEvent)
    ]
    assert len(apply_ev_calls) == 1
    ev = apply_ev_calls[0].args[0]
    assert ev.apply_status == "digest_drift"
    assert ev.failed_at_path == "d/x.py"
    assert ev.coordinator_run_id == "r1"


@pytest.mark.anyio
async def test_finalize_emit_failure_does_not_mask_outcome(
    snapshot_store: MagicMock,
    audit_repo: MagicMock,
    redis_mock: MagicMock,
    parent_sandbox: MagicMock,
    minio: MagicMock,
) -> None:
    """If the emit_event hook raises during CoordinatorApplyEvent, the
    apply outcome is still returned to the caller."""
    bad_emit = AsyncMock(side_effect=RuntimeError("emit broke"))
    applier = PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=redis_mock,
        emit_event=bad_emit,
    )
    plan = _modify_plan()
    out = await applier.apply(
        plan, parent_sandbox=parent_sandbox, minio_client=minio,
    )
    assert out.status == ApplyStatus.SUCCESS
