from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import unquote, urlparse

import pytest
from fastapi import UploadFile

from app.infrastructure.external.file_storage.minio_file_storage import (
    MinioFileStorage,
)


pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("public_secure", "expected_scheme"),
    [(False, "http"), (True, "https")],
)
@pytest.mark.parametrize(
    ("filename", "expected_key_suffix", "expected_encoded"),
    [
        ("a.png?download=1", ".png?download=1", "%3F"),
        ("a.png#preview", ".png#preview", "%23"),
        ("a.some ext", ".some ext", "%20"),
        ("a.png%raw", ".png%raw", "%25"),
    ],
)
async def test_upload_file_uses_public_endpoint_in_filepath(
    public_secure: bool,
    expected_scheme: str,
    filename: str,
    expected_key_suffix: str,
    expected_encoded: str,
) -> None:
    minio_store = SimpleNamespace(
        public_endpoint="localhost:19000",
        public_secure=public_secure,
        upload_fileobj=AsyncMock(),
    )
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.file.save = AsyncMock()
    storage = MinioFileStorage(
        bucket="manus",
        minio_store=minio_store,
        uow_factory=MagicMock(return_value=uow),
    )
    upload = UploadFile(filename=filename, file=io.BytesIO(b"hello"))

    result = await storage.upload_file(upload)

    parsed = urlparse(result.filepath)
    assert parsed.scheme == expected_scheme
    assert parsed.netloc == "localhost:19000"
    assert parsed.query == ""
    assert parsed.fragment == ""
    assert unquote(parsed.path) == f"/manus/{result.key}"
    assert expected_encoded in parsed.path
    assert result.key.endswith(expected_key_suffix)
    assert "minio:9000" not in result.filepath
    minio_store.upload_fileobj.assert_awaited_once()
    uow.file.save.assert_awaited_once_with(result)
