from typing import BinaryIO, Callable, Optional, Tuple

from app.application.errors.exceptions import ForbiddenError, NotFoundError
from app.domain.external.file_storage import FileStorage
from app.domain.models.file import File

# from app.domain.repositories.file_repository import FileRepository
from app.domain.repositories.uow import IUnitOfWork
from fastapi import UploadFile


class FileService:
    """Actus文件系统服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        file_storage: FileStorage,
    ) -> None:
        """构造函数，完成文件服务的初始化"""
        self.file_storage = file_storage
        self._uow_factory = uow_factory
        self._uow = uow_factory()

    async def upload_file(
        self, upload_file: UploadFile, user_id: Optional[str] = None
    ) -> File:
        """将传递的文件上传到腾讯云cos并记录上传数据"""
        file = await self.file_storage.upload_file(upload_file=upload_file)
        # 设置文件所属用户并持久化，避免后续附件鉴权读取到空 user_id
        if user_id and file.user_id != user_id:
            file.user_id = user_id
            async with self._uow:
                await self._uow.file.save(file)
        return file

    def _check_access(self, file: File, user_id: str, is_admin: bool = False) -> None:
        """检查用户是否有权限访问文件"""
        if is_admin:
            return
        if not file.user_id or file.user_id != user_id:
            raise ForbiddenError("无权访问此文件")

    async def get_file_info(
        self, file_id: str, user_id: str, is_admin: bool = False
    ) -> File:
        """根据传递的文件id获取文件信息"""
        async with self._uow:
            file = await self._uow.file.get_by_id(file_id)
        if not file:
            raise NotFoundError(f"该文件[{file_id}]不存在")
        self._check_access(file, user_id=user_id, is_admin=is_admin)
        return file

    async def download_file(
        self, file_id: str, user_id: str, is_admin: bool = False
    ) -> Tuple[BinaryIO, File]:
        """根据传递的文件id下载文件"""
        await self.get_file_info(file_id, user_id=user_id, is_admin=is_admin)
        return await self.file_storage.download_file(file_id)

    async def delete_file(self, file_id: str, user_id: str, is_admin: bool = False) -> None:
        """根据传递的文件id删除文件"""
        # 1.检查文件是否存在
        await self.get_file_info(file_id, user_id=user_id, is_admin=is_admin)

        # 2.调用存储层删除文件
        await self.file_storage.delete_file(file_id)
