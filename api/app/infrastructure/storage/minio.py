import io
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial
from typing import Any, BinaryIO, Optional
from uuid import uuid4

import anyio
from core.config import Settings, get_settings
from minio import Minio

logger = logging.getLogger(__name__)


class MinioStore:
    """MinIO（S3兼容）对象存储"""

    def __init__(self):
        """构造函数：获取配置 + 初始化 client 占位"""
        self._settings: Settings = get_settings()
        self._client: Optional[Minio] = None

    async def init(self) -> None:
        """创建 MinIO 客户端（Minio SDK 为同步客户端，但初始化本身很轻）"""
        if self._client is not None:
            logger.warning("MinIO 对象存储已初始化，无需重复操作")
            return

        try:
            self._client = Minio(
                endpoint=self._settings.minio_endpoint,  # 例如: "s3.example.com"
                access_key=self._settings.minio_access_key,
                secret_key=self._settings.minio_secret_key,
                secure=self._settings.minio_secure,  # True/False
                region=getattr(self._settings, "minio_region", None),
            )
            logger.info("MinIO 对象存储初始化成功")
        except Exception as e:
            logger.error(f"MinIO 对象存储初始化失败: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """关闭 MinIO 客户端（SDK 无显式 close，释放引用即可）"""
        if self._client is not None:
            self._client = None
            logger.info("关闭 MinIO 对象存储成功")

        get_minio.cache_clear()

    @property
    def client(self) -> Minio:
        """只读属性：返回 MinIO 客户端"""
        if self._client is None:
            raise RuntimeError("MinIO 未初始化，请调用 init() 完成初始化")
        return self._client

    async def _run_sync(self, fn, /, *args, **kwargs):
        return await anyio.to_thread.run_sync(partial(fn, *args, **kwargs))

    async def bucket_exists(self, bucket_name: str) -> bool:
        client = self.client
        return await self._run_sync(client.bucket_exists, bucket_name)

    async def ping(self, bucket_name: str) -> dict[str, Any]:
        """连通性检查：验证 endpoint 可访问、bucket 可探测"""
        client = self.client
        bucket_exists = await self._run_sync(client.bucket_exists, bucket_name)
        return {
            "ok": bucket_exists,
            "reachable": True,
            "endpoint": self._settings.minio_endpoint,
            "secure": self._settings.minio_secure,
            "bucket": bucket_name,
            "bucket_exists": bucket_exists,
        }

    async def upload_fileobj(
        self,
        bucket_name: str,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        client = self.client
        result = await self._run_sync(
            client.put_object,
            bucket_name,
            object_name,
            data,
            length,
            content_type=content_type or "application/octet-stream",
        )
        return {
            "bucket": bucket_name,
            "object": object_name,
            "etag": getattr(result, "etag", None),
            "version_id": getattr(result, "version_id", None),
        }

    async def presigned_get_url(
        self, bucket_name: str, object_name: str, expiry_seconds: int = 3600
    ) -> str:
        client = self.client
        return await self._run_sync(
            client.presigned_get_object,
            bucket_name,
            object_name,
            expires=timedelta(seconds=expiry_seconds),
        )

    async def download_fileobj(
        self,
        bucket_name: str,
        object_name: str,
    ) -> BinaryIO:
        """从MinIO下载文件对象，返回文件流"""
        client = self.client

        def _download() -> BinaryIO:
            resp = client.get_object(bucket_name, object_name)
            # 将响应内容读取到BytesIO中，因为原始响应需要释放连接
            content = resp.read()
            resp.close()
            resp.release_conn()
            return io.BytesIO(content)

        return await anyio.to_thread.run_sync(_download)

    async def delete_object(
        self,
        bucket_name: str,
        object_name: str,
    ) -> None:
        """从MinIO删除指定对象"""
        client = self.client
        await self._run_sync(client.remove_object, bucket_name, object_name)

    async def smoke_test(self, bucket_name: str) -> dict[str, Any]:
        """读写自检：put/get/remove 一次性验证"""
        client = self.client

        bucket_exists = await self._run_sync(client.bucket_exists, bucket_name)
        if not bucket_exists:
            return {
                "ok": False,
                "endpoint": self._settings.minio_endpoint,
                "secure": self._settings.minio_secure,
                "bucket": bucket_name,
                "bucket_exists": False,
                "error": "bucket_not_exists",
            }

        object_name = (
            f"__healthcheck__/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            f"-{uuid4().hex}.txt"
        )
        payload = f"minio-smoke {datetime.now(timezone.utc).isoformat()}".encode(
            "utf-8"
        )

        uploaded = False
        try:
            await self._run_sync(
                client.put_object,
                bucket_name,
                object_name,
                io.BytesIO(payload),
                len(payload),
                content_type="text/plain",
            )
            uploaded = True

            def _download() -> bytes:
                resp = client.get_object(bucket_name, object_name)
                try:
                    return resp.read()
                finally:
                    resp.close()
                    resp.release_conn()

            downloaded = await anyio.to_thread.run_sync(_download)
            match = downloaded == payload
            return {
                "ok": match,
                "endpoint": self._settings.minio_endpoint,
                "secure": self._settings.minio_secure,
                "bucket": bucket_name,
                "bucket_exists": True,
                "object": object_name,
                "uploaded_bytes": len(payload),
                "downloaded_bytes": len(downloaded),
                "match": match,
            }
        finally:
            if uploaded:
                try:
                    await self._run_sync(client.remove_object, bucket_name, object_name)
                except Exception:
                    logger.exception("MinIO smoke test cleanup failed")


@lru_cache()
def get_minio() -> MinioStore:
    """lru_cache 单例：获取 MinIO 对象存储实例"""
    return MinioStore()
