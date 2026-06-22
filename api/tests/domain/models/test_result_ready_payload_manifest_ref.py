import hashlib

import pytest

from app.domain.models.mailbox_envelope import (
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.patch_manifest import FilePatchEntry, PatchManifest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_SHA = hashlib.sha256(b"a").hexdigest()


def _manifest() -> PatchManifest:
    return PatchManifest(
        patch_id="r1:wu1:p",
        coordinator_run_id="r1",
        work_unit_id="wu1",
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


def test_manifest_ref_allowed_on_success() -> None:
    p = ResultReadyPayload(
        summary="ok",
        outcome=ResultReadyOutcome.SUCCESS,
        patch_manifest_ref="coordinator/r1/wu1/manifest/abc",
    )
    assert p.patch_manifest_ref == "coordinator/r1/wu1/manifest/abc"
    assert p.patch_manifest is None


def test_manifest_ref_defaults_none() -> None:
    p = ResultReadyPayload(summary="ok", outcome=ResultReadyOutcome.SUCCESS)
    assert p.patch_manifest_ref is None


def test_manifest_ref_forbidden_on_non_success() -> None:
    with pytest.raises(ValueError, match="patch_manifest_ref"):
        ResultReadyPayload(
            summary="fail",
            outcome=ResultReadyOutcome.FAILED,
            patch_manifest_ref="coordinator/r1/wu1/manifest/abc",
        )


def test_manifest_ref_mutually_exclusive_with_inline_manifest() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ResultReadyPayload(
            summary="ok",
            outcome=ResultReadyOutcome.SUCCESS,
            patch_manifest=_manifest(),
            patch_manifest_ref="coordinator/r1/wu1/manifest/abc",
        )


def test_inline_manifest_still_allowed_alone() -> None:
    p = ResultReadyPayload(
        summary="ok",
        outcome=ResultReadyOutcome.SUCCESS,
        patch_manifest=_manifest(),
    )
    assert p.patch_manifest is not None
    assert p.patch_manifest_ref is None
