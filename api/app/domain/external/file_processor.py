from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.domain.models.tool_result import DocumentPreview


# Shared constant: max image blocks per file_view call.
# Used by react_graph (tool_node truncation) and video processor (self-limiting).
MAX_FILE_VIEW_IMAGES = 10


@dataclass(frozen=True)
class FileProcessResult:
    """文件处理结果。text 进入 ToolMessage，image_blocks 注入 HumanMessage。"""
    text: str
    image_blocks: tuple[dict, ...] = ()
    document_blocks: tuple[dict, ...] = ()
    # B12 P2: processor 探测的原始 media_type（简单版；storage 压缩后 mime 漂移
    # 可接受，见 spec §4）。file_view 按 flag 决定是否挂到 MultimodalPayload。
    media_type: str | None = None
    # B12 P5: 结构化文档预览（PdfFileProcessor 填；其他 processor 恒 None）。
    document_preview: "DocumentPreview | None" = None


class FileProcessor(Protocol):
    """文件处理器协议。每种文件类型一个实现。"""

    async def process(
        self,
        sandbox_path: str,
        filename: str,
        mime_type: str,
        supports_vision: bool,
        supports_pdf_input: bool = False,
    ) -> FileProcessResult: ...


class FileProcessorLookup(Protocol):
    """文件处理器查找协议。domain 层使用此协议。"""

    def get_processor(self, mime_type: str) -> FileProcessor | None: ...
