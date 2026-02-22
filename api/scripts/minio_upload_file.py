import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.infrastructure.storage.minio import get_minio
from core.config import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local file to MinIO.")
    parser.add_argument("path", help="Local file path")
    parser.add_argument("--bucket", default=None, help="Bucket name (default: MINIO_BUCKET_NAME)")
    parser.add_argument("--object", dest="object_name", default=None, help="Object name/key")
    parser.add_argument("--prefix", default="uploads", help="Object key prefix when --object not set")
    parser.add_argument("--expiry-seconds", type=int, default=3600, help="Presigned GET expiry")
    parser.add_argument("--no-presign", action="store_true", help="Do not print presigned GET url")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    settings = get_settings()
    bucket_name = args.bucket or settings.minio_bucket_name
    object_name = (args.object_name or "").strip().lstrip("/").replace("\\", "/")
    if not object_name:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix_clean = args.prefix.strip().strip("/").replace("\\", "/")
        object_name = (
            f"{prefix_clean}/{timestamp}-{uuid4().hex}-{path.name}"
            if prefix_clean
            else f"{timestamp}-{uuid4().hex}-{path.name}"
        )

    store = get_minio()
    await store.init()

    if not await store.bucket_exists(bucket_name):
        await store.shutdown()
        raise SystemExit(f"Bucket not exists: {bucket_name}")

    size = path.stat().st_size
    with path.open("rb") as f:
        result = await store.upload_fileobj(
            bucket_name=bucket_name,
            object_name=object_name,
            data=f,
            length=size,
            content_type="application/octet-stream",
        )

    if not args.no_presign:
        result["presigned_get_url"] = await store.presigned_get_url(
            bucket_name=bucket_name,
            object_name=object_name,
            expiry_seconds=args.expiry_seconds,
        )

    result["size"] = size
    result["local_path"] = str(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    await store.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
