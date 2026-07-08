import asyncio
import io

import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image

from app.domain.external.file_processor import FileProcessResult
from app.infrastructure.external.file_processors.image import ImageFileProcessor


def _make_png_bytes(width: int = 100, height: int = 200) -> bytes:
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_sandbox(png_bytes: bytes | None = None):
    mock = AsyncMock()
    mock.download_file = AsyncMock(return_value=io.BytesIO(png_bytes or _make_png_bytes()))
    return mock


def _make_uploader(url: str = "https://minio.example.com/presigned/test.png"):
    return AsyncMock(return_value=url)


class TestImageFileProcessor:
    def test_vision_mode_returns_image_block(self):
        proc = ImageFileProcessor(sandbox=_make_sandbox(), file_uploader=_make_uploader())
        result = asyncio.run(
            proc.process("/tmp/test.png", "test.png", "image/png", supports_vision=True)
        )
        assert isinstance(result, FileProcessResult)
        assert "100x200" in result.text
        assert len(result.image_blocks) == 1
        assert result.image_blocks[0]["type"] == "image_url"

    def test_non_vision_no_fallback_returns_metadata_only(self):
        proc = ImageFileProcessor(sandbox=_make_sandbox(), file_uploader=_make_uploader())
        result = asyncio.run(
            proc.process("/tmp/test.png", "test.png", "image/png", supports_vision=False)
        )
        assert result.image_blocks == ()
        assert "100x200" in result.text

    def test_non_vision_with_fallback_calls_vision_model(self):
        from langchain_core.messages import AIMessage as LcAIMessage

        vision_model = AsyncMock()
        vision_model.ainvoke = AsyncMock(return_value=LcAIMessage(content="A red square image"))

        proc = ImageFileProcessor(
            sandbox=_make_sandbox(), file_uploader=_make_uploader(), vision_model=vision_model
        )
        result = asyncio.run(
            proc.process("/tmp/test.png", "test.png", "image/png", supports_vision=False)
        )
        assert result.image_blocks == ()
        assert "A red square image" in result.text
        assert "100x200" in result.text
        vision_model.ainvoke.assert_awaited_once()

    def test_oversized_file_returns_error_text(self):
        big_bytes = b"\x00" * (21 * 1024 * 1024)
        proc = ImageFileProcessor(sandbox=_make_sandbox(big_bytes), file_uploader=AsyncMock())
        result = asyncio.run(
            proc.process("/tmp/big.png", "big.png", "image/png", supports_vision=True)
        )
        assert "too large" in result.text
        assert result.image_blocks == ()

    def test_upload_failure_small_image_falls_back_to_data_url(self):
        """Small image + upload returns None → data URL fallback in vision path."""
        proc = ImageFileProcessor(sandbox=_make_sandbox(), file_uploader=AsyncMock(return_value=None))
        result = asyncio.run(
            proc.process("/tmp/test.png", "test.png", "image/png", supports_vision=True)
        )
        # Small test image → should fall back to inline data URL
        assert len(result.image_blocks) == 1
        assert result.image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "100x200" in result.text

    def test_upload_exception_non_vision_returns_metadata(self):
        """Uploader raising an exception must not crash the metadata-only path."""
        proc = ImageFileProcessor(
            sandbox=_make_sandbox(),
            file_uploader=AsyncMock(side_effect=ConnectionError("MinIO down")),
        )
        result = asyncio.run(
            proc.process("/tmp/test.png", "test.png", "image/png", supports_vision=False)
        )
        assert "100x200" in result.text
        assert result.image_blocks == ()

    def test_upload_exception_vision_falls_back_to_data_url(self):
        """Uploader exception + small image → data URL fallback in vision path."""
        proc = ImageFileProcessor(
            sandbox=_make_sandbox(),
            file_uploader=AsyncMock(side_effect=ConnectionError("MinIO down")),
        )
        result = asyncio.run(
            proc.process("/tmp/test.png", "test.png", "image/png", supports_vision=True)
        )
        # Small test image → should fall back to inline data URL despite upload error
        assert len(result.image_blocks) == 1
        assert result.image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "100x200" in result.text

    def test_vision_mode_sets_media_type(self):
        proc = ImageFileProcessor(sandbox=_make_sandbox(), file_uploader=_make_uploader())
        result = asyncio.run(
            proc.process("/tmp/test.png", "test.png", "image/png", supports_vision=True)
        )
        assert result.media_type == "image/png"
