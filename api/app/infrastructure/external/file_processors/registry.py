from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.domain.external.file_processor import FileProcessor
from app.domain.external.sandbox import SandboxHandle
from app.infrastructure.external.file_processors.image import ImageFileProcessor

logger = logging.getLogger(__name__)

FileUploader = Callable[[bytes, str], Awaitable[str | None]]


class FileProcessorRegistry:
    """MIME 前缀 → Processor 映射。实现 domain FileProcessorLookup Protocol。"""

    def __init__(
        self,
        sandbox: SandboxHandle,
        file_uploader: FileUploader,
        vision_model: Any | None = None,
        audio_config: Any | None = None,
        video_config: Any | None = None,
    ) -> None:
        self._processors: list[tuple[str, FileProcessor]] = []

        self._processors.append((
            "image/",
            ImageFileProcessor(
                sandbox=sandbox,
                file_uploader=file_uploader,
                vision_model=vision_model,
            ),
        ))

        from app.infrastructure.external.file_processors.pdf import PdfFileProcessor
        self._processors.append((
            "application/pdf",
            PdfFileProcessor(sandbox=sandbox, file_uploader=file_uploader),
        ))

        self._audio_processor: AudioFileProcessor | None = None
        if audio_config and getattr(audio_config, "provider", "disabled") != "disabled":
            from app.infrastructure.external.file_processors.audio import AudioFileProcessor
            audio_proc = AudioFileProcessor(sandbox=sandbox, config=audio_config)
            self._processors.append(("audio/", audio_proc))
            self._audio_processor = audio_proc

        if video_config:
            from app.infrastructure.external.file_processors.video import VideoFileProcessor
            self._processors.append((
                "video/",
                VideoFileProcessor(
                    sandbox=sandbox,
                    file_uploader=file_uploader,
                    audio_processor=self._audio_processor,
                    video_config=video_config,
                    vision_model=vision_model,
                ),
            ))

    # MIME types that match a prefix but have no working processor
    _EXCLUDED_MIMES = frozenset({"image/svg+xml"})

    def get_processor(self, mime_type: str) -> FileProcessor | None:
        if mime_type in self._EXCLUDED_MIMES:
            return None
        for prefix, processor in self._processors:
            if mime_type.startswith(prefix) or mime_type == prefix:
                return processor
        return None
