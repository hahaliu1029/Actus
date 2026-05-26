import asyncio
import hashlib
import io as _io
import logging
import os.path
import uuid
from datetime import datetime
from typing import BinaryIO, Callable, Tuple

from app.application.errors.exceptions import BadRequestError
from app.domain.external.file_storage import FileStorage
from app.domain.models.file import File
from app.domain.repositories.uow import IUnitOfWork
from app.infrastructure.external.image_processor import (
    IMAGE_MAX_FILE_SIZE,
    MEDIA_TYPE_TO_EXT,
    ImageProcessError,
    ImageProcessor,
)

# from app.domain.repositories.file_repository import FileRepository
from app.infrastructure.storage.minio import MinioStore
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class MinioFileStorage(FileStorage):
    """基于MinIO的文件存储扩展"""

    def __init__(
        self,
        bucket: str,
        minio_store: MinioStore,
        uow_factory: Callable[[], IUnitOfWork],
    ) -> None:
        """构造函数，完成MinIO文件存储扩展初始化"""
        self.bucket = bucket
        self.minio_store = minio_store
        self._uow_factory = uow_factory
        self._uow = uow_factory()

    async def upload_file(self, upload_file: UploadFile) -> File:
        """根据传递的文件源将文件上传到MinIO，图片自动压缩。"""
        try:
            # 0. 早期拦截：图片 MIME + 已知大小 → 拒绝前不读入内存
            declared_size = upload_file.size  # from Content-Length, may be None
            if (
                declared_size is not None
                and declared_size > IMAGE_MAX_FILE_SIZE
                and (upload_file.content_type or "").startswith("image/")
            ):
                raise BadRequestError(
                    f"图片文件过大 ({declared_size // 1024 // 1024}MB)，"
                    f"限制为 {IMAGE_MAX_FILE_SIZE // 1024 // 1024}MB"
                )

            # 1. 读取原始字节 + 生成文件 ID
            file_id = str(uuid.uuid4())
            file_content = await upload_file.read()
            file_size = len(file_content)
            _, file_extension = os.path.splitext(upload_file.filename or "")
            file_extension = file_extension or ""
            filename = upload_file.filename or file_id
            content_type = upload_file.content_type or ""

            # 2. 图片检测 + 压缩
            image_meta = None
            is_image_but_failed = False
            # Detect if MIME claims image but magic bytes disagree (spoofed file)
            mime_claims_image = content_type.startswith("image/")

            if ImageProcessor.is_image(file_content, content_type):
                # Safety net: reject after read if pre-read check was skipped
                # (e.g. upload_file.size was None on chunked/sandbox paths)
                if file_size > ImageProcessor.MAX_FILE_SIZE:
                    raise BadRequestError(
                        f"图片文件过大 ({file_size // 1024 // 1024}MB)，"
                        f"限制为 {ImageProcessor.MAX_FILE_SIZE // 1024 // 1024}MB"
                    )
                else:
                    try:
                        result = await asyncio.to_thread(
                            ImageProcessor.process, file_content, file_size, content_type
                        )
                        file_content = result.buffer
                        file_size = len(result.buffer)
                        image_meta = result
                    except ImageProcessError:
                        logger.warning(
                            "Image compression failed for %s, uploading original",
                            filename,
                        )
                        is_image_but_failed = True
            elif mime_claims_image:
                # MIME says image but magic bytes disagree → spoofed
                logger.warning(
                    "File %s has image MIME (%s) but invalid magic bytes, "
                    "correcting MIME and marking as not multimodal eligible",
                    filename, content_type,
                )
                content_type = "application/octet-stream"
                is_image_but_failed = True

            # 3. 格式变更传播：MIME / 扩展名 / 文件名对齐
            if image_meta:
                content_type = image_meta.media_type
                new_ext = MEDIA_TYPE_TO_EXT.get(image_meta.media_type, "")
                if new_ext and new_ext != file_extension:
                    if file_extension:
                        filename = filename.removesuffix(file_extension) + new_ext
                    else:
                        filename = filename + new_ext
                    file_extension = new_ext

            # 4. 生成对象 key 并上传到 MinIO
            date_path = datetime.now().strftime("%Y/%m/%d")
            object_name = f"{date_path}/{file_id}{file_extension}"

            await self.minio_store.upload_fileobj(
                bucket_name=self.bucket,
                object_name=object_name,
                data=_io.BytesIO(file_content),
                length=file_size,
                content_type=content_type,
            )
            logger.info(f"文件上传成功: {filename} (ID: {file_id})")

            # 5. 构建文件访问路径
            settings = self.minio_store._settings
            protocol = "https" if settings.minio_secure else "http"
            filepath = (
                f"{protocol}://{settings.minio_endpoint}/{self.bucket}/{object_name}"
            )

            # 6. 确定多模态可用性
            if image_meta:
                multimodal_eligible = True
            elif is_image_but_failed:
                multimodal_eligible = False
            else:
                multimodal_eligible = None

            # 7. 构建 File 模型并持久化
            file = File(
                id=file_id,
                filename=filename,
                filepath=filepath,
                key=object_name,
                extension=file_extension,
                mime_type=content_type,
                size=file_size,
                width=image_meta.width if image_meta else None,
                height=image_meta.height if image_meta else None,
                original_width=image_meta.original_width if image_meta else None,
                original_height=image_meta.original_height if image_meta else None,
                multimodal_eligible=multimodal_eligible,
            )
            async with self._uow:
                await self._uow.file.save(file)

            return file
        except Exception as e:
            logger.error(f"上传文件[{upload_file.filename}]失败: {str(e)}")
            raise

    async def download_file(self, file_id: str) -> Tuple[BinaryIO, File]:
        """根据文件id查询数据并下载文件"""
        try:
            # 1.查询对应的文件记录是否存在
            async with self._uow:
                file = await self._uow.file.get_by_id(file_id)
            if not file:
                raise ValueError(f"该文件不存在, 文件id: {file_id}")

            # 2.使用MinioStore下载文件
            response = await self.minio_store.download_fileobj(
                bucket_name=self.bucket,
                object_name=file.key,
            )

            # 3.返回文件流+文件信息
            return response, file
        except Exception as e:
            logger.error(f"下载文件[{file_id}]失败: {str(e)}")
            raise

    async def delete_file(self, file_id: str) -> None:
        """根据文件id删除MinIO中的文件和数据库记录"""
        try:
            # 1.查询对应的文件记录是否存在
            async with self._uow:
                file = await self._uow.file.get_by_id(file_id)
            if not file:
                raise ValueError(f"该文件不存在, 文件id: {file_id}")

            # 2.从MinIO中删除文件
            await self.minio_store.delete_object(
                bucket_name=self.bucket,
                object_name=file.key,
            )

            # 3.从数据库中删除文件记录
            async with self._uow:
                await self._uow.file.delete(file_id)
            logger.info(f"文件删除成功: {file.filename} (ID: {file_id})")
        except Exception as e:
            logger.error(f"删除文件[{file_id}]失败: {str(e)}")
            raise

    async def put_content_addressed_bytes(
        self,
        *,
        prefix: str,
        content: bytes,
        filename: str | None = None,
    ) -> str:
        """[C2 PR-4 §6.3] Content-addressed bytes upload to MinIO.

        Used by the coordinator dispatch_node for:
        - SpawnManifest JSON  (filename='manifest.json' → digest-prefixed key)
        - PathLease seed content (no filename → SHA-256 hex basename)
        - rationale_ref artifacts (NeedsAuthorizationDetails.proposed_write_plan)

        Key shape (r1 P1#6 fix — always content-addressed):
        - ``filename is None`` → ``{prefix}{digest_hex}``
        - ``filename`` given   → ``{prefix}{digest_hex[:16]}-{filename}``

        Why include the digest even with filename: PR-7 crash recovery may
        re-dispatch a coordinator run; if the serialized bytes diverge
        (dict ordering drift, timestamp embedding) and the key were purely
        ``{prefix}{filename}``, the second write would silently overwrite
        the first. Including the digest prefix preserves the
        "content-addressed" contract the API name promises while keeping
        a human-readable suffix for ops/debugging.

        Returns the full object key. Same content + same prefix + same
        filename → same key → MinIO PUT is idempotent at the wire level.

        Note: This method does NOT touch the ``files`` table — content-addressed
        artifacts are coordinator-internal blobs, not user-facing File records.
        The ORM ``File`` model is only populated by ``upload_file`` for the
        user's chat-attached uploads.
        """
        digest = hashlib.sha256(content).hexdigest()
        if filename is None:
            key = f"{prefix}{digest}"
        else:
            key = f"{prefix}{digest[:16]}-{filename}"
        await self.minio_store.upload_fileobj(
            bucket_name=self.bucket,
            object_name=key,
            data=_io.BytesIO(content),
            length=len(content),
            content_type="application/octet-stream",
        )
        return key

    async def get_presigned_url(
        self, file: File, expiry_seconds: int = 86400
    ) -> str | None:
        """生成 MinIO 预签名 URL。"""
        try:
            if not file.key:
                return None
            return await self.minio_store.presigned_get_url(
                bucket_name=self.bucket,
                object_name=file.key,
                expiry_seconds=expiry_seconds,
            )
        except Exception as e:
            logger.debug("MinIO presigned URL 生成失败: %s", e)
            return None
