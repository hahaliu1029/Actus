"""B12 P1: FileViewImageBytesResolver 实现——storage-first + allowlist HTTP fallback。

file_view 图片今日恒为自家 presigned URL（image.py:82 上传自签），故 storage-first
几乎总命中；HTTP fallback 仅防御，host 必须 == 配置的 MinIO endpoint（allowlist 优先
于盲拒 private IP——自托管 MinIO 本就在私网）。禁 redirect + 超时 + streaming 限读 +
Content-Type image/* 校验。
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class MinioFileViewImageResolver:
    def __init__(self, file_storage, allowlist_host: str) -> None:
        # file_storage: MinioFileStorage — 有 .bucket 属性 + async get_bytes(ref)
        self._file_storage = file_storage
        self._allowlist_host = allowlist_host  # settings.minio_endpoint, e.g. "minio:9000"

    async def load_image_bytes(self, display_url: str, *, max_bytes: int) -> bytes | None:
        # R1#P2-2: SSRF 硬门统一到入口——host 必须 == 配置 MinIO endpoint（storage-first
        # 与 HTTP fallback 都要过）。否则非 allowlist host + path 命中 bucket/key 时，
        # storage-first 会绕过 host 语义读到任意 object。presigned URL netloc 恒 = endpoint，
        # 正常流不受影响。
        if urlparse(display_url).netloc != self._allowlist_host:
            logger.warning(
                "[B12 P1] refusing non-allowlist image host %r (allow=%r)",
                urlparse(display_url).netloc, self._allowlist_host,
            )
            return None
        # 1. storage-first（无 HTTP，最安全）
        key = self._parse_object_key(display_url)
        if key is not None:
            try:
                raw = await self._file_storage.get_bytes(key)
                if raw is not None:
                    if len(raw) <= max_bytes:
                        return raw
                    logger.debug("[B12 P1] storage image %d > max_bytes=%d", len(raw), max_bytes)
                    return None
            except Exception as e:  # noqa: BLE001
                logger.debug("[B12 P1] storage-first get_bytes failed key=%s: %s", key, e)
        # 2. HTTP fallback（allowlist 硬门）
        return await self._http_fallback(display_url, max_bytes=max_bytes)

    def _parse_object_key(self, url: str) -> str | None:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return None
            path = parsed.path.lstrip("/")  # "{bucket}/{object...}"
            bucket = getattr(self._file_storage, "bucket", None)
            if bucket and path.startswith(f"{bucket}/"):
                return path[len(bucket) + 1:]
            return None
        except Exception:  # noqa: BLE001
            return None

    async def _http_fallback(self, url: str, *, max_bytes: int) -> bytes | None:
        parsed = urlparse(url)
        # SSRF 硬门：host 必须 == 配置的 MinIO endpoint（allowlist 优先于拒 private IP）
        if parsed.netloc != self._allowlist_host:
            logger.warning(
                "[B12 P1] refusing non-allowlist image host %r (allow=%r)",
                parsed.netloc, self._allowlist_host,
            )
            return None
        try:
            import httpx

            async with httpx.AsyncClient(follow_redirects=False, timeout=10.0) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        return None
                    ctype = resp.headers.get("content-type", "")
                    if not ctype.startswith("image/"):
                        logger.warning("[B12 P1] refusing non-image content-type %r", ctype)
                        return None
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > max_bytes:  # 读到 max_bytes+1 早停
                            logger.debug("[B12 P1] HTTP image exceeds max_bytes=%d", max_bytes)
                            return None
                    return bytes(buf)
        except Exception as e:  # noqa: BLE001
            logger.debug("[B12 P1] HTTP fallback failed for %s: %s", url[:64], e)
            return None
