"""B12 P1: MinioFileViewImageResolver storage-first + allowlist SSRF 门。"""
import asyncio
from unittest.mock import AsyncMock


class _FakeStorage:
    bucket = "manus"

    def __init__(self, data: dict[str, bytes]):
        self._data = data

    async def get_bytes(self, ref: str) -> bytes | None:
        return self._data.get(ref)


def _make_resolver(storage, allow: str = "minio:9000"):
    from app.infrastructure.external.file_view.image_bytes_resolver import (
        MinioFileViewImageResolver,
    )
    return MinioFileViewImageResolver(storage, allow)


def test_storage_first_hit_returns_bytes() -> None:
    resolver = _make_resolver(_FakeStorage({"images/a.png": b"PNGDATA"}))
    out = asyncio.run(resolver.load_image_bytes(
        "https://minio:9000/manus/images/a.png?X-Amz-Sig=abc", max_bytes=1024))
    assert out == b"PNGDATA"


def test_public_endpoint_allowlist_uses_storage_first() -> None:
    resolver = _make_resolver(
        _FakeStorage({"images/a.png": b"PUBLIC-ENDPOINT-DATA"}),
        allow="localhost:19000",
    )
    out = asyncio.run(resolver.load_image_bytes(
        "http://localhost:19000/manus/images/a.png?X-Amz-Sig=abc",
        max_bytes=1024,
    ))
    assert out == b"PUBLIC-ENDPOINT-DATA"


def test_encoded_object_key_is_unquoted_for_storage_first() -> None:
    original_key = "images/a.png?name#space 100%.png"
    resolver = _make_resolver(
        _FakeStorage({original_key: b"ENCODED-PATH-DATA"}),
        allow="localhost:19000",
    )
    resolver._http_fallback = AsyncMock(
        side_effect=AssertionError("encoded key must resolve through storage-first"),
    )

    out = asyncio.run(resolver.load_image_bytes(
        "http://localhost:19000/manus/"
        "images/a.png%3Fname%23space%20100%25.png",
        max_bytes=1024,
    ))

    assert out == b"ENCODED-PATH-DATA"
    resolver._http_fallback.assert_not_awaited()


def test_storage_first_oversize_returns_none() -> None:
    resolver = _make_resolver(_FakeStorage({"images/a.png": b"X" * 100}))
    out = asyncio.run(resolver.load_image_bytes(
        "https://minio:9000/manus/images/a.png?sig=x", max_bytes=10))
    assert out is None


def test_key_parse_strips_bucket_prefix() -> None:
    resolver = _make_resolver(_FakeStorage({"deep/path/a.png": b"OK"}))
    out = asyncio.run(resolver.load_image_bytes(
        "https://minio:9000/manus/deep/path/a.png?sig=1", max_bytes=1024))
    assert out == b"OK"


def test_non_allowlist_host_refused() -> None:
    """host 非 allowlist → None（入口硬门，早于 storage-first / HTTP）。"""
    resolver = _make_resolver(_FakeStorage({}), allow="minio:9000")
    out = asyncio.run(resolver.load_image_bytes(
        "https://evil.example.com/manus/x.png", max_bytes=1024))
    assert out is None


def test_non_allowlist_host_with_valid_key_refused() -> None:
    """R1#P2-2: key 存在于 bucket，但 host 非 allowlist → 仍 None（host 门早于 storage-first，
    不让非 allowlist host + 命中 path 绕过 host 语义读到任意 object）。"""
    resolver = _make_resolver(_FakeStorage({"x.png": b"SECRET"}), allow="minio:9000")
    out = asyncio.run(resolver.load_image_bytes(
        "https://evil.example.com/manus/x.png", max_bytes=1024))
    assert out is None
