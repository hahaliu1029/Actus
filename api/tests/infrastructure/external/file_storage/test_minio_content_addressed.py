"""C2 PR-4 Task 4.9 — MinioFileStorage.put_content_addressed_bytes tests.

Spec ref: §6.3 — coordinator uses content-addressed MinIO refs for SpawnManifest,
PathLease seed bytes, and rationale_ref artifacts. Same content → same key
→ dedup. The SHA-256 hex digest is the basename under the given prefix.
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.external.file_storage.minio_file_storage import (
    MinioFileStorage,
)


pytestmark = pytest.mark.anyio


def _mk_storage() -> tuple[MinioFileStorage, MagicMock]:
    minio_store = MagicMock()
    minio_store.upload_fileobj = AsyncMock()
    uow_factory = MagicMock(return_value=MagicMock())
    storage = MinioFileStorage(
        bucket="test-bucket", minio_store=minio_store, uow_factory=uow_factory,
    )
    return storage, minio_store


async def test_returns_key_under_prefix_with_sha256_basename() -> None:
    """Default behavior: key = ``{prefix}{sha256_hex}``. No filename override."""
    storage, _ = _mk_storage()
    content = b"hello world"
    expected_digest = hashlib.sha256(content).hexdigest()
    ref = await storage.put_content_addressed_bytes(
        prefix="coordinator/r1/wu1/seed/", content=content,
    )
    assert ref == f"coordinator/r1/wu1/seed/{expected_digest}"
    assert len(ref.rsplit("/", 1)[1]) == 64


async def test_filename_override_prefixed_with_short_digest() -> None:
    """[r1 P1#6 fix] When ``filename`` is supplied, the key STILL contains
    a content digest prefix (first 16 hex chars). Pure path-addressing
    (``{prefix}{filename}``) would break the API name's content-addressed
    contract — two divergent serializations of the same logical object
    would silently overwrite under the same key on re-dispatch (PR-7
    crash recovery scope)."""
    storage, _ = _mk_storage()
    content = b"{}"
    expected_short = hashlib.sha256(content).hexdigest()[:16]
    ref = await storage.put_content_addressed_bytes(
        prefix="coordinator/r1/wu1/", content=content, filename="manifest.json",
    )
    assert ref == f"coordinator/r1/wu1/{expected_short}-manifest.json"
    # 16-char digest prefix is enough for collision resistance for
    # coordinator-scoped artifacts (~10^9 unique manifests before a 50%
    # collision probability — well past any practical run count).
    assert len(expected_short) == 16


async def test_filename_with_different_content_produces_different_keys() -> None:
    """[r1 P1#6 corollary] Two different serializations of the same logical
    manifest land at different keys under the same prefix + filename.
    Without this, replay safety degrades to "caller-must-be-bit-exact"."""
    storage, _ = _mk_storage()
    ref1 = await storage.put_content_addressed_bytes(
        prefix="p/", content=b"v1", filename="manifest.json",
    )
    ref2 = await storage.put_content_addressed_bytes(
        prefix="p/", content=b"v2", filename="manifest.json",
    )
    assert ref1 != ref2
    assert ref1.endswith("-manifest.json")
    assert ref2.endswith("-manifest.json")


async def test_same_content_produces_same_key() -> None:
    """Dedup invariant: identical bytes → identical key under same prefix."""
    storage, _ = _mk_storage()
    content = b"deterministic content"
    ref1 = await storage.put_content_addressed_bytes(
        prefix="a/", content=content,
    )
    ref2 = await storage.put_content_addressed_bytes(
        prefix="a/", content=content,
    )
    assert ref1 == ref2


async def test_different_content_produces_different_keys() -> None:
    storage, _ = _mk_storage()
    ref1 = await storage.put_content_addressed_bytes(
        prefix="a/", content=b"foo",
    )
    ref2 = await storage.put_content_addressed_bytes(
        prefix="a/", content=b"bar",
    )
    assert ref1 != ref2


async def test_upload_call_uses_correct_bucket_and_length() -> None:
    storage, minio = _mk_storage()
    content = b"some bytes payload"
    await storage.put_content_addressed_bytes(
        prefix="p/", content=content,
    )
    minio.upload_fileobj.assert_awaited_once()
    kwargs = minio.upload_fileobj.await_args.kwargs
    assert kwargs["bucket_name"] == "test-bucket"
    assert kwargs["object_name"].startswith("p/")
    assert kwargs["length"] == len(content)


async def test_empty_content_allowed() -> None:
    """Defensive: empty bytes still produce a key (SHA-256 of empty string)."""
    storage, _ = _mk_storage()
    ref = await storage.put_content_addressed_bytes(
        prefix="empty/", content=b"",
    )
    empty_digest = hashlib.sha256(b"").hexdigest()
    assert ref == f"empty/{empty_digest}"


async def test_content_type_defaults_to_octet_stream() -> None:
    """No content_type override → octet-stream (generic binary).
    Content-addressed artifacts are opaque blobs; the consumer (reducer,
    rationale viewer) knows how to interpret them by context."""
    storage, minio = _mk_storage()
    await storage.put_content_addressed_bytes(prefix="p/", content=b"x")
    kwargs = minio.upload_fileobj.await_args.kwargs
    assert kwargs["content_type"] == "application/octet-stream"
