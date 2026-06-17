from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class FileReadResult(BaseModel):
    """文件读取结果"""

    filepath: str = Field(..., description="读取文件的路径(回显调用方原始入参)")
    content: str = Field(..., description="读取的文件内容")


class FileWriteResult(BaseModel):
    """文件写入结果"""

    filepath: str = Field(..., description="写入文件的路径(回显调用方原始入参)")
    bytes_written: Optional[int] = Field(
        default=None, description="写入文件内容的字节数"
    )


class FileReplaceResult(BaseModel):
    """文件内容替换结果模型"""

    filepath: str = Field(..., description="替换内容的文件路径(回显调用方原始入参)")
    replaced_count: int = Field(default=0, description="替换内容的次数")


class FileSearchResult(BaseModel):
    """文件搜索结果"""

    filepath: str = Field(..., description="搜索内容的文件路径(回显调用方原始入参)")
    matches: List[str] = Field(default_factory=list, description="匹配内容列表")
    line_numbers: List[int] = Field(default_factory=list, description="匹配的行号列表")


class FileFindResult(BaseModel):
    """文件查找结果"""

    dir_path: str = Field(..., description="搜索的目录路径(回显调用方原始入参)")
    files: List[str] = Field(default_factory=list, description="检索到的文件列表")


class FileUploadResult(BaseModel):
    """文件上传结果"""

    filepath: str = Field(..., description="上传文件的路径(回显调用方原始入参)")
    file_size: int = Field(default=0, description="上传文件的大小, 单位为字节")
    success: bool = Field(..., description="是否上传成功")


FileKind = Literal[
    "missing", "regular", "directory", "symlink",
    "fifo", "socket", "block", "char", "other",
]


class FileCheckResult(BaseModel):
    """文件检查是否存在结果"""

    filepath: str = Field(..., description="检查文件的路径(回显调用方原始入参)")
    exists: bool = Field(..., description="文件是否存在")
    kind: FileKind = Field(
        default="other",
        description="inode 类型（os.lstat 分类，不跟随符号链接）；旧客户端缺省为 other",
    )


class FileDeleteResult(BaseModel):
    """文件删除结果模型"""

    filepath: str = Field(..., description="删除文件的路径(回显调用方原始入参)")
    deleted: bool = Field(..., description="文件是否删除成功")
