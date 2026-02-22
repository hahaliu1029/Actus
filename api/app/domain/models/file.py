import uuid
from typing import Optional

from pydantic import BaseModel, Field


class File(BaseModel):
    """文件信息Domain模型，用于记录Manus/Human上传or生成的文件"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # 文件id
    filename: str = ""  # 文件名字
    filepath: str = ""  # 文件路径
    key: str = ""  # minio中的路径
    extension: str = ""  # 扩展名
    mime_type: str = ""  # mime-type类型
    size: int = 0  # 文件大小，单位为字节
    user_id: Optional[str] = None  # 文件所属用户ID
