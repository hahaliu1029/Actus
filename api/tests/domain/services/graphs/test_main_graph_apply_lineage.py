"""[PR-9b-B Task B6] ``main_graph._run_parallel_backend`` MUST thread a
``GroupLineageFields`` into the ``PatchApplier.apply(lineage=...)`` call so the
group-level ``CoordinatorApplyEvent`` carries ``root_session_id`` /
``parent_session_id`` parity (closing the B5→B6 production wiring).

The lineage is derived from the graph state's already-computed locals:
  - ``parent_session_id = state["session_id"]``     (session_id → parent mapping)
  - ``root_session_id   = state["root_session_id"] or parent_session_id``
This test pins that the apply call site builds ``GroupLineageFields`` from those
exact locals (not a recompute / not ``state.get("parent_session_id")`` which
MainGraphState does not carry).

Harness mirrors
``test_parallel_execution_subgraph_emit.test_run_parallel_backend_constructs_applier_from_deps_with_emit``:
monkeypatch ``PatchApplier`` inside main_graph's import scope with a capturing
fake, drive the subgraph to SUCCESS, and inspect the kwargs apply() received.
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_run_parallel_backend_threads_group_lineage_into_apply() -> None:
    """[Task B6 / INV-B4 wiring] The per-run apply() call receives a
    ``GroupLineageFields`` whose ``root_session_id`` / ``parent_session_id``
    match the lineage derived from graph state."""
    from app.application.services.group_lineage import GroupLineageFields
    from app.application.services.patch_applier import (
        ApplyDiagnostics,
        ApplyOutcome,
        ApplyStatus,
    )
    from app.application.services.patch_applier_deps import PatchApplierDeps
    from app.domain.models.patch_apply_plan import GroupOutcome, PatchApplyPlan
    from app.domain.models.patch_manifest import FilePatchEntry
    from app.domain.services.graphs.main_graph import _run_parallel_backend

    sha = hashlib.sha256(b"x").hexdigest()
    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(FilePatchEntry(
            path="d/f.py", op="add", new_digest=sha,
            content_ref="r", content_size=1,
        ),),
        total_size_bytes=1, file_count=1,
        source_work_unit_ids=("wu1",),
    )

    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(return_value={
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": plan,
        "step_result_candidate": "reducer-text",
    })

    # Capture the kwargs apply() is called with so we can assert the lineage
    # threading explicitly.
    captured_apply_kwargs: dict = {}

    class _CapturingApplier:
        def __init__(self, **kwargs) -> None:
            pass

        async def apply(self, plan, *, parent_sandbox, minio_client,
                        cancel_event=None, lineage=None,
                        on_rollback=None) -> ApplyOutcome:
            captured_apply_kwargs["lineage"] = lineage
            captured_apply_kwargs["cancel_event"] = cancel_event
            captured_apply_kwargs["on_rollback"] = on_rollback
            return ApplyOutcome(
                status=ApplyStatus.SUCCESS,
                applied_files=(), failed_at=None, rollback_status=None,
                diagnostics=ApplyDiagnostics(duration_ms=1),
            )

    import app.domain.services.graphs.main_graph as mg

    orig_applier = mg.__dict__.get("PatchApplier", None)
    mg.PatchApplier = _CapturingApplier
    try:
        deps = PatchApplierDeps(
            snapshot_store=MagicMock(),
            audit_repo=MagicMock(),
            redis=MagicMock(),
        )
        event_queue: asyncio.Queue = asyncio.Queue()

        step = MagicMock()
        step.id = "step1"
        step.parallel_work_units = MagicMock()
        step.parallel_work_units.work_units = []

        # session_id → parent_session_id; explicit root_session_id distinct from
        # parent so the mapping is observable (not collapsed by the `or` fallback).
        state = {
            "session_id": "parent-test",
            "user_id": "u1",
            "root_session_id": "root-test",
        }
        cancel_event = asyncio.Event()
        config = {"configurable": {
            "parallel_execution_subgraph": subgraph,
            "patch_applier_deps": deps,
            "parent_sandbox": AsyncMock(),
            "artifact_storage": AsyncMock(),
            "event_queue": event_queue,
            "cancel_event": cancel_event,
        }}

        out = (await _run_parallel_backend(state, config, step)).summary
    finally:
        if orig_applier is None:
            mg.__dict__.pop("PatchApplier", None)
        else:
            mg.PatchApplier = orig_applier

    # apply() was reached and received a GroupLineageFields.
    assert "lineage" in captured_apply_kwargs, "apply() was not called"
    lineage = captured_apply_kwargs["lineage"]
    assert isinstance(lineage, GroupLineageFields)
    # Derived from state: session_id → parent, explicit root preserved.
    assert lineage.parent_session_id == "parent-test"
    assert lineage.root_session_id == "root-test"

    # Sanity: the pre-existing cancel_event wiring is preserved alongside lineage.
    assert captured_apply_kwargs["cancel_event"] is cancel_event
    assert captured_apply_kwargs["on_rollback"] is None

    # Apply succeeded → main_graph returns the success text (behavior unchanged).
    assert "应用成功" in out


async def test_run_parallel_backend_lineage_root_falls_back_to_parent() -> None:
    """When state has no ``root_session_id``, the lineage's root falls back to
    the parent (Phase 1 ``max_subagent_depth=1`` invariant: parent IS root)."""
    from app.application.services.group_lineage import GroupLineageFields
    from app.application.services.patch_applier import (
        ApplyDiagnostics,
        ApplyOutcome,
        ApplyStatus,
    )
    from app.application.services.patch_applier_deps import PatchApplierDeps
    from app.domain.models.patch_apply_plan import GroupOutcome, PatchApplyPlan
    from app.domain.models.patch_manifest import FilePatchEntry
    from app.domain.services.graphs.main_graph import _run_parallel_backend

    sha = hashlib.sha256(b"x").hexdigest()
    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(FilePatchEntry(
            path="d/f.py", op="add", new_digest=sha,
            content_ref="r", content_size=1,
        ),),
        total_size_bytes=1, file_count=1,
        source_work_unit_ids=("wu1",),
    )

    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(return_value={
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": plan,
        "step_result_candidate": "reducer-text",
    })

    captured_apply_kwargs: dict = {}

    class _CapturingApplier:
        def __init__(self, **kwargs) -> None:
            pass

        async def apply(self, plan, *, parent_sandbox, minio_client,
                        cancel_event=None, lineage=None,
                        on_rollback=None) -> ApplyOutcome:
            captured_apply_kwargs["lineage"] = lineage
            captured_apply_kwargs["on_rollback"] = on_rollback
            return ApplyOutcome(
                status=ApplyStatus.SUCCESS,
                applied_files=(), failed_at=None, rollback_status=None,
                diagnostics=ApplyDiagnostics(duration_ms=1),
            )

    import app.domain.services.graphs.main_graph as mg

    orig_applier = mg.__dict__.get("PatchApplier", None)
    mg.PatchApplier = _CapturingApplier
    try:
        deps = PatchApplierDeps(
            snapshot_store=MagicMock(),
            audit_repo=MagicMock(),
            redis=MagicMock(),
        )
        step = MagicMock()
        step.id = "step1"
        step.parallel_work_units = MagicMock()
        step.parallel_work_units.work_units = []

        # No root_session_id → root falls back to parent (== session_id).
        state = {"session_id": "solo-parent", "user_id": "u1"}
        config = {"configurable": {
            "parallel_execution_subgraph": subgraph,
            "patch_applier_deps": deps,
            "parent_sandbox": AsyncMock(),
            "artifact_storage": AsyncMock(),
            "event_queue": asyncio.Queue(),
        }}

        await _run_parallel_backend(state, config, step)
    finally:
        if orig_applier is None:
            mg.__dict__.pop("PatchApplier", None)
        else:
            mg.PatchApplier = orig_applier

    lineage = captured_apply_kwargs["lineage"]
    assert isinstance(lineage, GroupLineageFields)
    assert lineage.parent_session_id == "solo-parent"
    assert lineage.root_session_id == "solo-parent"
    assert captured_apply_kwargs["on_rollback"] is None
