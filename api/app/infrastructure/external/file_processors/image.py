from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, Awaitable, Callable

from app.domain.external.file_processor import FileProcessResult
from app.domain.external.sandbox import SandboxHandle
from app.infrastructure.external.llm.message_sanitizer import MAX_IMAGE_BYTES

logger = logging.getLogger(__name__)

FileUploader = Callable[[bytes, str], Awaitable[str | None]]

_MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB


class ImageFileProcessor:
    """图片文件处理器。

    supports_vision=True: 上传 MinIO → presigned URL → image_url block
    supports_vision=False + vision_model: 调视觉模型描述 → 纯文本
    supports_vision=False 无 vision_model: 仅返回元数据
    """

    def __init__(
        self,
        sandbox: SandboxHandle,
        file_uploader: FileUploader,
        vision_model: Any | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._file_uploader = file_uploader
        self._vision_model = vision_model

    async def process(
        self,
        sandbox_path: str,
        filename: str,
        mime_type: str,
        supports_vision: bool,
        supports_pdf_input: bool = False,
    ) -> FileProcessResult:
        file_io = await self._sandbox.download_file(sandbox_path)
        file_bytes = file_io.read() if hasattr(file_io, "read") else file_io

        if len(file_bytes) > _MAX_IMAGE_SIZE:
            return FileProcessResult(
                text=f"[Image: {filename}, {len(file_bytes)} bytes — too large to process]"
            )

        try:
            w, h = await asyncio.to_thread(self._get_dimensions, file_bytes)
            text = f"[Image: {filename}, {w}x{h}]"
        except Exception as e:
            logger.warning("Image dimension detection failed for %s: %s", filename, e)
            return FileProcessResult(
                text=f"[Image: {filename} — invalid or corrupted image file]"
            )

        # Upload only when a URL is actually needed (vision path or fallback).
        # Protected so uploader failure doesn't break metadata-only path.
        url: str | None = None
        if supports_vision or self._vision_model:
            try:
                url = await self._file_uploader(file_bytes, filename)
            except Exception as e:
                logger.warning("Image upload failed: %s", e)

        if supports_vision:
            # Prefer presigned URL; fall back to inline data URL for small images
            if url:
                image_url = url
            elif len(file_bytes) <= MAX_IMAGE_BYTES:
                b64 = base64.b64encode(file_bytes).decode()
                image_url = f"data:{mime_type};base64,{b64}"
            else:
                # Too large for data URL and upload failed → metadata only
                return FileProcessResult(text=text)
            return FileProcessResult(
                text=text,
                image_blocks=(
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
                ),
                media_type=mime_type,
            )
        elif self._vision_model:
            desc = await self._describe_with_vision(file_bytes, mime_type, url)
            return FileProcessResult(text=f"{text}\n{desc}")
        else:
            return FileProcessResult(text=text)

    @staticmethod
    def _get_dimensions(file_bytes: bytes) -> tuple[int, int]:
        from PIL import Image

        img = Image.open(io.BytesIO(file_bytes))
        return img.size

    async def _describe_with_vision(
        self, file_bytes: bytes, mime_type: str, presigned_url: str | None = None
    ) -> str:
        from langchain_core.messages import HumanMessage

        # Prefer presigned URL over inline data URL to avoid adapter sanitizer
        # stripping oversized base64. Uses shared limit (imported at module level).
        if presigned_url:
            image_url = presigned_url
        elif len(file_bytes) <= MAX_IMAGE_BYTES:
            # Small enough for inline data URL (won't be stripped by sanitizer)
            b64 = base64.b64encode(file_bytes).decode()
            image_url = f"data:{mime_type};base64,{b64}"
        else:
            # Too large for data URL and no presigned URL available — can't describe
            logger.warning(
                "Image too large for inline data URL (%d bytes) and upload failed; "
                "skipping vision description",
                len(file_bytes),
            )
            return "[Vision analysis unavailable: image too large and upload failed]"

        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this image concisely. Focus on key visual elements, text, layout, and purpose."},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
        ])
        try:
            response = await asyncio.wait_for(
                self._vision_model.ainvoke([msg]),
                timeout=200.0,
            )
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:
            logger.warning("Vision fallback failed: %s", e)
            return "[Vision analysis unavailable]"
