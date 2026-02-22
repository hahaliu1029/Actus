import logging
import os.path
import uuid
from datetime import datetime
from typing import BinaryIO, Callable, Tuple

from app.domain.external.file_storage import FileStorage
from app.domain.models.file import File
from app.domain.repositories.uow import IUnitOfWork

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
        """根据传递的文件源将文件上传到MinIO"""
        try:
            # 1.生成随机的uuid作为文件id并获取文件扩展名
            file_id = str(uuid.uuid4())
            _, file_extension = os.path.splitext(upload_file.filename)
            if not file_extension:
                file_extension = ""

            # 2.生成日期路径并拼接最终key
            date_path = datetime.now().strftime("%Y/%m/%d")
            object_name = f"{date_path}/{file_id}{file_extension}"

            # 3.获取文件大小
            file_content = await upload_file.read()
            file_size = len(file_content)

            # 4.重置文件指针并上传文件到MinIO
            await upload_file.seek(0)
            await self.minio_store.upload_fileobj(
                bucket_name=self.bucket,
                object_name=object_name,
                data=upload_file.file,
                length=file_size,
                content_type=upload_file.content_type,
            )
            logger.info(f"文件上传成功: {upload_file.filename} (ID: {file_id})")

            # 5.构建文件访问路径
            settings = self.minio_store._settings
            protocol = "https" if settings.minio_secure else "http"
            filepath = (
                f"{protocol}://{settings.minio_endpoint}/{self.bucket}/{object_name}"
            )

            # 6.构建file模型并将数据存储到数据库中
            file = File(
                id=file_id,
                filename=upload_file.filename,
                filepath=filepath,
                key=object_name,
                extension=file_extension,
                mime_type=upload_file.content_type or "",
                size=file_size,
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
