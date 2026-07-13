"""Run from ``api/`` as ``python -m scripts.verify_local_minio write|verify``."""

import argparse
import asyncio
import io
import json
import sys
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from app.infrastructure.storage.minio import MinioStore
from core.config import Settings, get_settings


BUCKET_NAME = "a2a-mcp"
OBJECT_NAME = "__local_minio_verify__/persistence.txt"
PAYLOAD = b"actus-local-minio-persistence-v1\n"

# Acceptance-only contract: the isolated Compose project runs these phases
# serially. ``write`` replaces this run's fixed payload before the container
# rebuild; this is intentionally not a concurrent/general-purpose utility.


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _validate_acceptance_settings(settings: Settings) -> None:
    checks = (
        (
            "minio_endpoint",
            settings.minio_endpoint,
            "localhost:19000",
            settings.minio_endpoint == "localhost:19000",
        ),
        (
            "effective_minio_public_endpoint",
            settings.effective_minio_public_endpoint,
            "localhost:19000",
            settings.effective_minio_public_endpoint == "localhost:19000",
        ),
        (
            "minio_secure",
            settings.minio_secure,
            False,
            settings.minio_secure is False,
        ),
        (
            "effective_minio_public_secure",
            settings.effective_minio_public_secure,
            False,
            settings.effective_minio_public_secure is False,
        ),
        (
            "minio_region",
            settings.minio_region,
            "us-east-1",
            settings.minio_region == "us-east-1",
        ),
        (
            "minio_bucket_name",
            settings.minio_bucket_name,
            BUCKET_NAME,
            settings.minio_bucket_name == BUCKET_NAME,
        ),
    )
    mismatches = [
        f"{field}: expected {expected!r}, got {actual!r}"
        for field, actual, expected, matches in checks
        if not matches
    ]
    if mismatches:
        raise RuntimeError(
            "refusing local MinIO acceptance with unsafe settings: "
            + "; ".join(mismatches)
        )


def _download_public(url: str) -> bytes:
    opener = build_opener(_NoRedirectHandler())
    with opener.open(url, timeout=10) as response:
        return response.read(len(PAYLOAD) + 1)


async def _assert_public_download(store: MinioStore) -> str:
    url = await store.presigned_get_url(BUCKET_NAME, OBJECT_NAME)
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError(f"invalid presigned URL port: {url}") from exc

    # Hard-coded host/port are the isolated acceptance endpoint contract.
    if (
        parsed.scheme != "http"
        or parsed.hostname != "localhost"
        or port != 19000
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError(f"unexpected presigned URL endpoint: {url}")

    public_bytes = await asyncio.to_thread(_download_public, url)
    if public_bytes != PAYLOAD:
        raise RuntimeError(
            f"public download mismatch: expected {len(PAYLOAD)} bytes, "
            f"got {len(public_bytes)}"
        )
    return url


async def _write(store: MinioStore) -> dict[str, object]:
    smoke = await store.smoke_test(BUCKET_NAME)
    if smoke.get("ok") is not True:
        raise RuntimeError(f"MinIO smoke test failed: {smoke}")

    await store.upload_fileobj(
        BUCKET_NAME,
        OBJECT_NAME,
        io.BytesIO(PAYLOAD),
        len(PAYLOAD),
        content_type="text/plain",
    )
    url = await _assert_public_download(store)
    return {"phase": "write", "smoke": smoke, "presigned_url": url}


async def _verify(store: MinioStore) -> dict[str, object]:
    downloaded = await store.download_fileobj(BUCKET_NAME, OBJECT_NAME)
    try:
        internal_bytes = downloaded.read()
    finally:
        downloaded.close()
    if internal_bytes != PAYLOAD:
        raise RuntimeError(
            f"internal download mismatch: expected {len(PAYLOAD)} bytes, "
            f"got {len(internal_bytes)}"
        )
    url = await _assert_public_download(store)
    await store.delete_object(BUCKET_NAME, OBJECT_NAME)
    return {"phase": "verify", "presigned_url": url, "deleted": True}


async def main(phase: str) -> None:
    settings = get_settings()
    _validate_acceptance_settings(settings)
    # MinioStore resolves the same lru-cached public Settings object internally.
    store = MinioStore()
    try:
        await store.init()
        result = await (_write(store) if phase == "write" else _verify(store))
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    finally:
        await store.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one serial phase of the isolated local MinIO acceptance."
    )
    parser.add_argument(
        "phase",
        choices=("write", "verify"),
        help=(
            "write: store the fixed persistence object before rebuild; "
            "verify: verify the persisted object and delete it (one-shot cleanup)"
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(args.phase))
