import asyncio
import json

from app.infrastructure.storage.minio import get_minio
from core.config import get_settings


async def main() -> None:
    settings = get_settings()
    store = get_minio()
    await store.init()

    bucket_name = settings.minio_bucket_name
    result = {
        "ping": await store.ping(bucket_name=bucket_name),
        "smoke": await store.smoke_test(bucket_name=bucket_name),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    await store.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
