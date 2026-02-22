from typing import Optional, Protocol

from app.domain.models.file import File


class FileRepository(Protocol):
    """文件模型数据仓库"""

    async def save(self, file: File) -> None:
        """新增或更新文件信息"""
        ...

    async def get_by_id(self, file_id: str) -> Optional[File]:
        """根据传递的文件id获取文件信息"""
        ...

    async def delete(self, file_id: str) -> None:
        """根据传递的文件id删除文件记录"""
        ...
