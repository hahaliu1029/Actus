from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable

from app.domain.external.file_processor import FileProcessor, FileProcessResult
from app.domain.external.sandbox import SandboxHandle
from app.infrastructure.external.file_processors.image import ImageFileProcessor

logger = logging.getLogger(__name__)

FileUploader = Callable[[bytes, str], Awaitable[str | None]]


class FileProcessorRegistry:
    """MIME 前缀 → Processor 映射。实现 domain FileProcessorLookup Protocol。"""

    # B12 P3: session-scoped result cache（flag gate 在 file_view 编排侧）
    _CACHE_CAP = 200
    _PRESIGNED_TTL = 86400   # mirror minio_file_storage presigned expiry_seconds
    _CACHE_MARGIN = 300      # 近过期当 miss（PDF render/extract timeout ~60/120s）

    def __init__(
        self,
        sandbox: SandboxHandle,
        file_uploader: FileUploader,
        vision_model: Any | None = None,
        audio_config: Any | None = None,
        video_config: Any | None = None,
        *,
        pdf_page_parallel_enabled: bool = False,
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
            PdfFileProcessor(
                sandbox=sandbox,
                file_uploader=file_uploader,
                page_parallel_enabled=pdf_page_parallel_enabled,
            ),
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

        # B12 P3: session-scoped LRU（key = 11 元组，见 Task 3.2 + PR-3 R1#P2 ctime_ns；value = (result, expires_at)）
        self._result_cache: "OrderedDict[tuple, tuple[FileProcessResult, float]]" = OrderedDict()

    # MIME types that match a prefix but have no working processor
    _EXCLUDED_MIMES = frozenset({"image/svg+xml"})

    def get_processor(self, mime_type: str) -> FileProcessor | None:
        if mime_type in self._EXCLUDED_MIMES:
            return None
        for prefix, processor in self._processors:
            if mime_type.startswith(prefix) or mime_type == prefix:
                return processor
        return None

    def cache_get(self, key: tuple) -> "FileProcessResult | None":
        import time as _time
        entry = self._result_cache.get(key)
        if entry is None:
            return None
        result, expires_at = entry
        if _time.time() >= expires_at:
            self._result_cache.pop(key, None)  # 近过期/已过期 → miss（presigned 可能死）
            return None
        self._result_cache.move_to_end(key)
        return result

    def cache_put(self, key: tuple, result: "FileProcessResult", process_start_time: float) -> None:
        expires_at = process_start_time + self._PRESIGNED_TTL - self._CACHE_MARGIN
        self._result_cache[key] = (result, expires_at)
        self._result_cache.move_to_end(key)
        while len(self._result_cache) > self._CACHE_CAP:
            self._result_cache.popitem(last=False)
