import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.mailbox_envelope import ResultReadyOutcome
from app.domain.models.patch_manifest import FilePatchEntry, PatchManifest
from app.domain.models.work_unit import PathLease, WorkUnit
from app.domain.services.graphs.parallel_execution_subgraph import (
    _build_pre_results_from_terminal,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_SHA = hashlib.sha256(b"a").hexdigest()


def _manifest(wu_id: str) -> PatchManifest:
    return PatchManifest(
        patch_id=f"r1:{wu_id}:p",
        coordinator_run_id="r1",
        work_unit_id=wu_id,
        files=(
            FilePatchEntry(
                path="d/a.py",
                op="add",
                new_digest=_SHA,
                content_ref="ref-1",
                content_size=1,
            ),
        ),
    )


def _record(*, wu_id, payload) -> MagicMock:
    rec = MagicMock()
    rec.envelope_type = "RESULT_READY"
    rec.child_session_id = f"c_{wu_id}"
    rec.payload = payload
    return rec


def _write_wu(wu_id: str) -> WorkUnit:
    return WorkUnit(
        work_unit_id=wu_id,
        objective="o",
        phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="d/a.py", op="add")],
    )


def _explore_wu(wu_id: str) -> WorkUnit:
    return WorkUnit(
        work_unit_id=wu_id,
        objective="o",
        phase="exploration",
        allowed_tools=["file_read"],
    )


async def test_rehydrate_resolves_manifest_by_ref() -> None:
    pm = _manifest("wu1")
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(
        return_value=json.dumps(pm.model_dump(mode="json")).encode("utf-8")
    )
    terminal = {
        "wu1": _record(
            wu_id="wu1",
            payload={
                "outcome": "success",
                "patch_manifest_ref": "coordinator/r1/wu1/manifest/abc",
            },
        )
    }
    out = await _build_pre_results_from_terminal(
        terminal,
        artifact_storage=artifact,
        work_units_by_id={"wu1": _write_wu("wu1")},
    )
    assert len(out) == 1
    assert out[0].outcome == ResultReadyOutcome.SUCCESS
    assert out[0].patch_manifest is not None
    artifact.get_bytes.assert_awaited_once_with("coordinator/r1/wu1/manifest/abc")


async def test_rehydrate_inline_manifest_unchanged() -> None:
    pm = _manifest("wu1")
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    terminal = {
        "wu1": _record(
            wu_id="wu1",
            payload={
                "outcome": "success",
                "patch_manifest": pm.model_dump(mode="json"),
            },
        )
    }
    out = await _build_pre_results_from_terminal(
        terminal,
        artifact_storage=artifact,
        work_units_by_id={"wu1": _write_wu("wu1")},
    )
    assert out[0].outcome == ResultReadyOutcome.SUCCESS
    assert out[0].patch_manifest is not None
    artifact.get_bytes.assert_not_awaited()


async def test_rehydrate_demotes_write_success_no_manifest() -> None:
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    terminal = {
        "wu1": _record(wu_id="wu1", payload={"outcome": "success"})
    }
    out = await _build_pre_results_from_terminal(
        terminal,
        artifact_storage=artifact,
        work_units_by_id={"wu1": _write_wu("wu1")},
    )
    assert out[0].outcome == ResultReadyOutcome.FAILED
    assert out[0].patch_manifest is None


async def test_rehydrate_demotes_write_success_unresolvable_ref() -> None:
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(side_effect=KeyError("gone"))
    terminal = {
        "wu1": _record(
            wu_id="wu1",
            payload={
                "outcome": "success",
                "patch_manifest_ref": "coordinator/r1/wu1/manifest/gone",
            },
        )
    }
    out = await _build_pre_results_from_terminal(
        terminal,
        artifact_storage=artifact,
        work_units_by_id={"wu1": _write_wu("wu1")},
    )
    assert out[0].outcome == ResultReadyOutcome.FAILED
    assert out[0].patch_manifest is None


async def test_rehydrate_non_write_phase_demotion_guard_inert() -> None:
    # [S2 §3.2 C1 / §5 invariant 5] A non-write (here: exploration) work unit
    # yields manifest_required=False, so the rehydrate demotion guard is INERT
    # and leaves the outcome untouched. This does NOT claim an exploration
    # SUCCESS terminal is producible (an exploration child finalizes
    # NEEDS_AUTHORIZATION — coordinator_child_runner.py:562, locked by Task
    # 2.4a). It only pins that the demotion fires SOLELY on the write-phase
    # (manifest_required) row, so the rehydrate fail-close is scoped exactly to
    # the dropped/truncated write-phase manifest it exists to catch.
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    terminal = {
        "wu1": _record(wu_id="wu1", payload={"outcome": "success"})
    }
    out = await _build_pre_results_from_terminal(
        terminal,
        artifact_storage=artifact,
        work_units_by_id={"wu1": _explore_wu("wu1")},
    )
    # manifest_required=False (non-write phase) ⇒ no demotion ⇒ outcome as-persisted.
    assert out[0].outcome == ResultReadyOutcome.SUCCESS
    assert out[0].patch_manifest is None
