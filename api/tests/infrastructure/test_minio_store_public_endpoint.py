from __future__ import annotations

import io
from unittest.mock import MagicMock, call
from urllib.parse import urlsplit

import pytest
from minio import Minio

from app.infrastructure.storage import minio as minio_module
from core.config import Settings


pytestmark = pytest.mark.anyio


def _settings(
    *,
    internal_endpoint: str = "minio:9000",
    internal_secure: bool = False,
    public_endpoint: str | None = "localhost:19000",
    public_secure: bool | None = False,
) -> Settings:
    return Settings(
        _env_file=None,
        env="test",
        jwt_secret_key="unit-test-secret",
        minio_endpoint=internal_endpoint,
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_region="us-east-1",
        minio_secure=internal_secure,
        minio_public_endpoint=public_endpoint,
        minio_public_secure=public_secure,
    )


def _store(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> minio_module.MinioStore:
    monkeypatch.setattr(minio_module, "get_settings", lambda: settings)
    return minio_module.MinioStore()


async def test_presigned_url_uses_explicit_public_endpoint_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(monkeypatch, _settings())
    await store.init()

    monkeypatch.setattr(
        store.signing_client,
        "_execute",
        MagicMock(
            side_effect=AssertionError("presigning must not perform network I/O")
        ),
    )
    url = await store.presigned_get_url("artifacts", "runs/report.txt")

    parsed = urlsplit(url)
    assert parsed.scheme == "http"
    assert parsed.netloc == "localhost:19000"
    assert parsed.path == "/artifacts/runs/report.txt"
    assert store.public_endpoint == "localhost:19000"
    assert store.public_secure is False


async def test_presigned_url_honors_public_secure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(monkeypatch, _settings(public_secure=True))
    await store.init()

    url = await store.presigned_get_url("artifacts", "report.txt")

    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "localhost:19000"


async def test_unconfigured_public_endpoint_reuses_internal_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(
        monkeypatch,
        _settings(public_endpoint=None, public_secure=None),
    )

    await store.init()

    assert store.signing_client is store.client
    assert store.public_endpoint == "minio:9000"
    assert store.public_secure is False


async def test_explicit_matching_public_settings_reuse_internal_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(
        monkeypatch,
        _settings(
            public_endpoint="minio:9000",
            public_secure=False,
        ),
    )

    await store.init()

    assert store.signing_client is store.client


async def test_distinct_public_endpoint_uses_separate_signing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(monkeypatch, _settings())

    await store.init()

    assert isinstance(store.client, Minio)
    assert isinstance(store.signing_client, Minio)
    assert store.signing_client is not store.client


def test_signing_client_requires_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(monkeypatch, _settings())

    with pytest.raises(RuntimeError, match="MinIO.*init"):
        _ = store.signing_client


async def test_object_io_uses_internal_client_and_only_presign_uses_signer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(monkeypatch, _settings())
    internal = MagicMock(spec=Minio)
    signer = MagicMock(spec=Minio)
    internal.bucket_exists.return_value = True
    internal.put_object.return_value = MagicMock(etag="etag", version_id=None)
    response = MagicMock()
    response.read.return_value = b"payload"
    internal.get_object.return_value = response
    signer.presigned_get_object.return_value = (
        "http://localhost:19000/bucket/object"
    )
    store._client = internal
    store._signing_client = signer

    assert await store.bucket_exists("bucket") is True
    await store.ping("bucket")
    await store.upload_fileobj("bucket", "object", io.BytesIO(b"x"), 1)
    downloaded = await store.download_fileobj("bucket", "object")
    await store.delete_object("bucket", "object")
    await store.smoke_test("bucket")
    url = await store.presigned_get_url("bucket", "object")

    assert downloaded.read() == b"payload"
    assert url == "http://localhost:19000/bucket/object"
    assert [entry[0] for entry in internal.method_calls] == [
        "bucket_exists",
        "bucket_exists",
        "put_object",
        "get_object",
        "remove_object",
        "bucket_exists",
        "put_object",
        "get_object",
        "remove_object",
    ]
    assert signer.method_calls == [
        call.presigned_get_object(
            "bucket",
            "object",
            expires=minio_module.timedelta(seconds=3600),
        )
    ]


async def test_repeated_init_does_not_rebuild_and_shutdown_clears_both_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(monkeypatch, _settings())
    internal = MagicMock(spec=Minio)
    signer = MagicMock(spec=Minio)
    builder = MagicMock(side_effect=[internal, signer])
    monkeypatch.setattr(store, "_build_client", builder, raising=False)

    await store.init()
    await store.init()

    assert builder.call_count == 2
    assert store.client is internal
    assert store.signing_client is signer

    await store.shutdown()

    assert store._client is None
    assert store._signing_client is None


async def test_second_client_build_failure_keeps_init_atomic_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(monkeypatch, _settings())
    first_internal = MagicMock(spec=Minio)
    builder = MagicMock(side_effect=[first_internal, RuntimeError("signer failed")])
    monkeypatch.setattr(store, "_build_client", builder, raising=False)

    with pytest.raises(RuntimeError, match="signer failed"):
        await store.init()

    assert store._client is None
    assert store._signing_client is None

    retry_internal = MagicMock(spec=Minio)
    retry_signer = MagicMock(spec=Minio)
    builder.side_effect = [retry_internal, retry_signer]
    await store.init()

    assert store.client is retry_internal
    assert store.signing_client is retry_signer
