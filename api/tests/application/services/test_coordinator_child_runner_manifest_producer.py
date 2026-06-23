import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.mailbox_envelope import ResultReadyOutcome
from app.domain.models.patch_manifest import FilePatchEntry, PatchManifest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _entry(i: int) -> FilePatchEntry:
    import hashlib

    return FilePatchEntry(
        path=f"d/f{i}.py",
        op="add",
        new_digest=hashlib.sha256(f"{i}".encode()).hexdigest(),
        content_ref=f"ref-{i}",
        content_size=1,
    )


def _manifest(n_files: int) -> PatchManifest:
    return PatchManifest(
        patch_id="r1:wu1:p",
        coordinator_run_id="r1",
        work_unit_id="wu1",
        files=tuple(_entry(i) for i in range(n_files)),
    )


def _runner(artifact_storage) -> "CoordinatorChildRunner":
    # Build a runner with the collaborators the helper touches stubbed. The
    # build-only helper touches ONLY _artifact_storage (it does NOT publish —
    # the caller does), but we also stub _publish_result_ready so the
    # _finalize_success integration test can drive the full success path.
    # Other ctor deps are not exercised; pass MagicMock/None per the existing
    # test fixtures (see test_coordinator_child_seed_install.py:_wire_runner for
    # the canonical construction pattern — mirror it here).
    from app.application.services.coordinator_child_runner import (
        CoordinatorChildRunner,
    )

    runner = CoordinatorChildRunner.__new__(CoordinatorChildRunner)
    runner._artifact_storage = artifact_storage
    runner._publish_result_ready = AsyncMock()
    return runner


@pytest.mark.anyio
async def test_small_manifest_built_inline() -> None:
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock()
    runner = _runner(artifact)

    pm = _manifest(1)
    wu = MagicMock()
    wu.work_unit_id = "wu1"
    payload = await runner._build_manifest_payload_inline_or_ref(
        "r1", wu, pm, summary="completed wu1",
    )
    assert payload.outcome == ResultReadyOutcome.SUCCESS
    assert payload.patch_manifest is not None
    assert payload.patch_manifest_ref is None
    artifact.put_content_addressed_bytes.assert_not_awaited()
    # [defect-fix R2 P1] The helper is BUILD-ONLY — it must NOT publish.
    runner._publish_result_ready.assert_not_awaited()


@pytest.mark.anyio
async def test_oversized_manifest_built_by_ref() -> None:
    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(
        return_value="coordinator/r1/wu1/manifest/abc"
    )
    runner = _runner(artifact)

    # Force the by-ref branch deterministically regardless of the real ceiling:
    # patch the threshold down to 0 so any non-empty manifest serializes over it.
    import app.application.services.coordinator_child_runner as ccr_mod

    pm = _manifest(3)
    wu = MagicMock()
    wu.work_unit_id = "wu1"

    orig = ccr_mod._MAX_INLINE_MANIFEST_BYTES
    ccr_mod._MAX_INLINE_MANIFEST_BYTES = 0
    try:
        payload = await runner._build_manifest_payload_inline_or_ref(
            "r1", wu, pm, summary="completed wu1",
        )
    finally:
        ccr_mod._MAX_INLINE_MANIFEST_BYTES = orig

    assert payload.outcome == ResultReadyOutcome.SUCCESS
    assert payload.patch_manifest is None
    assert payload.patch_manifest_ref == "coordinator/r1/wu1/manifest/abc"
    # uploaded the canonical JSON under the manifest/ prefix.
    call = artifact.put_content_addressed_bytes.await_args
    assert call.kwargs["prefix"] == "coordinator/r1/wu1/manifest/"
    assert json.loads(call.kwargs["content"].decode("utf-8"))["work_unit_id"] == "wu1"
    # [defect-fix R2 P1] BUILD-ONLY — the helper must NOT publish.
    runner._publish_result_ready.assert_not_awaited()


def test_build_child_prompt_forwards_tree_lease_prefix() -> None:
    """[C2-full S2 PR-5] A shell-mode write unit carrying ``write_tree_lease``
    must have its ADD-only tree prefixes reach the child prompt — the codex-found
    gap was ``_build_child_prompt`` dropping ``wu.write_tree_lease`` so a tree-add
    unit was mislabeled '(none — read-only exploration)'."""
    from app.application.services.coordinator_child_runner import (
        CoordinatorChildRunner,
    )
    from app.domain.models.work_unit import TreeLease, WorkUnit

    wu = WorkUnit(
        work_unit_id="wu-tree",
        objective="generate code under the tree",
        phase="write",
        write_tree_lease=[TreeLease(prefix="workspace/gen", ops=frozenset({"add"}))],
        shell_mode=True,
    )
    runner = CoordinatorChildRunner.__new__(CoordinatorChildRunner)
    prompt = runner._build_child_prompt(wu, manifest=MagicMock())

    # The leased tree prefix reaches the prompt.
    assert "workspace/gen" in prompt
    # And the tree-only write unit is NOT mislabeled read-only.
    assert "_(none — read-only exploration)_" not in prompt


def test_build_child_prompt_no_tree_lease_unchanged_no_tree_block() -> None:
    """[flag-OFF safety] A non-shell-mode unit (``write_tree_lease == []``) ⇒
    empty ``allowed_trees`` ⇒ no ADD-only tree block in the rendered prompt."""
    from app.application.services.coordinator_child_runner import (
        CoordinatorChildRunner,
    )
    from app.domain.models.work_unit import PathLease, WorkUnit

    wu = WorkUnit(
        work_unit_id="wu-paths",
        objective="patch a file",
        phase="write",
        write_lease=[
            PathLease(path="workspace/a.py", op="add"),
        ],
    )
    runner = CoordinatorChildRunner.__new__(CoordinatorChildRunner)
    prompt = runner._build_child_prompt(wu, manifest=MagicMock())

    assert "workspace/a.py" in prompt
    assert "directory tree" not in prompt.lower()


@pytest.mark.anyio
async def test_finalize_success_upload_failure_publishes_failed_terminal() -> None:
    """[defect-fix R2 P1] The by-ref MinIO upload is a NEW failure source. It is
    invoked INSIDE _finalize_success's extraction try, so an upload failure hits
    the generic ``except`` → _finalize_failed → a FAILED terminal IS published
    (not an escape out of run_work_unit that would strand the parent waiter)."""
    import app.application.services.coordinator_child_runner as ccr_mod

    artifact = MagicMock()
    artifact.put_content_addressed_bytes = AsyncMock(
        side_effect=RuntimeError("minio down")
    )
    runner = _runner(artifact)
    # Stub extraction so _finalize_success builds a (non-empty) manifest, then
    # routes it through the build helper INSIDE the try.
    runner._extract_patch_files_from_history = AsyncMock(
        return_value=[_entry(0)]
    )

    wu = MagicMock()
    wu.work_unit_id = "wu1"

    # Force the by-ref branch so the (failing) upload is reached.
    orig = ccr_mod._MAX_INLINE_MANIFEST_BYTES
    ccr_mod._MAX_INLINE_MANIFEST_BYTES = 0
    try:
        out = await runner._finalize_success("r1", wu, "c1", done_event=None)
    finally:
        ccr_mod._MAX_INLINE_MANIFEST_BYTES = orig

    # The upload failure degraded to a terminal FAILED envelope — published,
    # not propagated out of the finally-only run_work_unit wrapper.
    assert out.outcome == ResultReadyOutcome.FAILED
    runner._publish_result_ready.assert_awaited_once()
    published_payload = runner._publish_result_ready.await_args.args[1]
    assert published_payload.outcome == ResultReadyOutcome.FAILED
