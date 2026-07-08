"""Tests for PdfFileProcessor — dual-path: native PDF block or pymupdf4llm extraction."""
from __future__ import annotations

import asyncio
import io
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.external.file_processor import FileProcessResult
from app.domain.models.tool_result import ToolResult
from app.infrastructure.external.file_processors.pdf import (
    PdfFileProcessor,
    _is_complex_page,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PDF_HEADER = b"%PDF-1.4 fake pdf content"
_INVALID_PDF_HEADER = b"NOTAPDF content"


def _make_exec_result(returncode: int = 0, output: str = "") -> ToolResult:
    """Build a ToolResult that looks like exec_command output from the sandbox."""
    return ToolResult(
        success=(returncode == 0),
        message="",
        data={"returncode": returncode, "output": output},
    )


def _make_sandbox(
    file_bytes: bytes = _VALID_PDF_HEADER,
    exec_results: list[ToolResult] | None = None,
    download_extras: dict[str, bytes] | None = None,
) -> AsyncMock:
    """Create a mock Sandbox.

    Args:
        file_bytes: bytes returned by the first download_file call (the PDF).
        exec_results: successive ToolResult values for exec_command (cycled if short).
        download_extras: mapping of filepath → bytes for subsequent download_file calls.
    """
    mock = AsyncMock()

    # download_file: first call returns the PDF, subsequent calls use download_extras.
    extra = download_extras or {}
    call_count = [0]

    async def _download(path: str) -> io.BytesIO:
        if call_count[0] == 0:
            call_count[0] += 1
            return io.BytesIO(file_bytes)
        # For rendered page images etc.
        return io.BytesIO(extra.get(path, b"\xff\xd8\xff"))  # minimal JPEG header

    mock.download_file = AsyncMock(side_effect=_download)

    # exec_command: return results in sequence; last one repeated if list exhausted.
    exec_queue = list(exec_results or [_make_exec_result(0, "5")])
    exec_idx = [0]

    async def _exec(session_id: str, exec_dir: str, command: str) -> ToolResult:
        idx = min(exec_idx[0], len(exec_queue) - 1)
        exec_idx[0] += 1
        return exec_queue[idx]

    mock.exec_command = AsyncMock(side_effect=_exec)
    mock.write_file = AsyncMock(return_value=_make_exec_result(0, ""))
    return mock


def _make_uploader(url: str = "https://storage.example.com/pdf_page.jpg") -> AsyncMock:
    return AsyncMock(return_value=url)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _is_complex_page unit tests
# ---------------------------------------------------------------------------

class TestIsComplexPage:
    def test_text_only_page_is_not_complex(self):
        md = "# Introduction\n\nThis is a regular text page with some content."
        assert _is_complex_page(md) is False

    def test_page_with_many_images_is_complex(self):
        # Three or more image references → complex
        md = "![img1](a.png) ![img2](b.png) ![img3](c.png)\nsome text"
        assert _is_complex_page(md) is True

    def test_short_page_with_single_image_is_complex(self):
        # Short page (< 200 chars) that has an image reference → complex
        md = "![chart](chart.png)"
        assert _is_complex_page(md) is True

    def test_short_page_with_table_is_complex(self):
        # Short page with markdown table separator → complex
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert _is_complex_page(md) is True

    def test_code_block_images_not_counted(self):
        # Image refs inside fenced code blocks should not count
        md = "```\n![img1](a.png) ![img2](b.png) ![img3](c.png)\n```\n"
        assert _is_complex_page(md) is False

    def test_long_page_with_two_images_not_complex(self):
        # More than 200 chars + only 2 image refs (threshold is >=3)
        text = "a" * 300
        md = f"{text}\n![img1](a.png) ![img2](b.png)"
        assert _is_complex_page(md) is False


# ---------------------------------------------------------------------------
# Native path tests (supports_pdf_input=True)
# ---------------------------------------------------------------------------

class TestNativePath:
    def test_valid_pdf_returns_document_block(self):
        """Native path: valid PDF with known page count returns document_blocks."""
        sandbox = _make_sandbox(
            file_bytes=_VALID_PDF_HEADER,
            exec_results=[_make_exec_result(0, "5")],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/test.pdf", "test.pdf", "application/pdf",
                         supports_vision=True, supports_pdf_input=True)
        )
        assert isinstance(result, FileProcessResult)
        assert "test.pdf" in result.text
        assert "5 pages" in result.text
        assert len(result.document_blocks) == 1
        block = result.document_blocks[0]
        assert block["type"] == "file"
        assert block["file"]["filename"] == "test.pdf"
        assert block["file"]["file_data"].startswith("data:application/pdf;base64,")

    def test_invalid_magic_bytes_returns_error_text(self):
        """Native path: file that doesn't start with %PDF- is rejected."""
        sandbox = _make_sandbox(file_bytes=_INVALID_PDF_HEADER)
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/notpdf.pdf", "notpdf.pdf", "application/pdf",
                         supports_vision=True, supports_pdf_input=True)
        )
        assert result.document_blocks == ()
        assert "invalid header" in result.text

    def test_oversized_pdf_falls_back_to_extraction(self):
        """Native path fails for >50MB → falls back to extraction path."""
        import json
        big_bytes = b"%PDF-" + b"x" * (50 * 1024 * 1024 + 1)
        extraction_output = json.dumps([{"metadata": {"page": 0}, "text": "# Fallback page"}])
        sandbox = _make_sandbox(
            file_bytes=big_bytes,
            exec_results=[
                # extraction: write_file result (cycled) then extraction result
                _make_exec_result(0, extraction_output),
            ],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/big.pdf", "big.pdf", "application/pdf",
                         supports_vision=True, supports_pdf_input=True)
        )
        # Should fall back to extraction, not hard fail
        assert result.document_blocks == ()
        assert "Fallback page" in result.text

    def test_page_count_failure_falls_back_to_extraction(self):
        """Native path: page count fails → falls back to extraction path."""
        import json
        extraction_output = json.dumps([{"metadata": {"page": 0}, "text": "# Extracted text"}])
        sandbox = _make_sandbox(
            file_bytes=_VALID_PDF_HEADER,
            exec_results=[
                _make_exec_result(1, "error: fitz not found"),  # page count fails
                _make_exec_result(0, extraction_output),         # extraction succeeds
            ],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/test.pdf", "test.pdf", "application/pdf",
                         supports_vision=True, supports_pdf_input=True)
        )
        assert result.document_blocks == ()
        assert "Extracted text" in result.text

    def test_too_many_pages_falls_back_to_extraction(self):
        """Native path: >100 pages → falls back to extraction path."""
        import json
        extraction_output = json.dumps([{"metadata": {"page": 0}, "text": "# Page 1 of many"}])
        sandbox = _make_sandbox(
            file_bytes=_VALID_PDF_HEADER,
            exec_results=[
                _make_exec_result(0, "150"),                     # 150 pages
                _make_exec_result(0, extraction_output),         # extraction succeeds
            ],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/large.pdf", "large.pdf", "application/pdf",
                         supports_vision=True, supports_pdf_input=True)
        )
        assert result.document_blocks == ()
        assert "Page 1 of many" in result.text


# ---------------------------------------------------------------------------
# Extraction path tests (supports_pdf_input=False)
# ---------------------------------------------------------------------------

class TestExtractionPath:
    def _make_page_output(self, pages: list[dict]) -> str:
        """Build JSON output as the extraction script would produce it."""
        return json.dumps(pages, ensure_ascii=False)

    def test_valid_pdf_extraction_returns_text(self):
        """Extraction path: valid PDF produces text with page content."""
        pages = [
            {"text": "# Chapter One\n\nSome content here."},
            {"text": "# Chapter Two\n\nMore content."},
        ]
        exec_result = _make_exec_result(0, self._make_page_output(pages))
        # First exec_command is for extraction; last cleanup exec_command is rm -rf
        sandbox = _make_sandbox(
            file_bytes=_VALID_PDF_HEADER,
            exec_results=[exec_result, _make_exec_result(0, "")],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/test.pdf", "test.pdf", "application/pdf",
                         supports_vision=False, supports_pdf_input=False)
        )
        assert isinstance(result, FileProcessResult)
        assert "test.pdf" in result.text
        assert "Chapter One" in result.text
        assert "Chapter Two" in result.text
        assert result.document_blocks == ()

    def test_empty_document_returns_error_text(self):
        """Extraction path: empty page list produces 'empty document' message."""
        exec_result = _make_exec_result(0, self._make_page_output([]))
        sandbox = _make_sandbox(
            file_bytes=_VALID_PDF_HEADER,
            exec_results=[exec_result, _make_exec_result(0, "")],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/empty.pdf", "empty.pdf", "application/pdf",
                         supports_vision=False, supports_pdf_input=False)
        )
        assert "empty document" in result.text

    def test_extraction_failure_returns_error_text(self):
        """Extraction path: non-zero returncode from sandbox produces error message."""
        exec_result = _make_exec_result(1, "ModuleNotFoundError: pymupdf4llm")
        sandbox = _make_sandbox(
            file_bytes=_VALID_PDF_HEADER,
            exec_results=[exec_result, _make_exec_result(0, "")],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/test.pdf", "test.pdf", "application/pdf",
                         supports_vision=False, supports_pdf_input=False)
        )
        assert "extraction failed" in result.text

    def test_invalid_magic_bytes_in_extraction_path(self):
        """Extraction path also rejects non-PDF files based on magic bytes."""
        sandbox = _make_sandbox(file_bytes=_INVALID_PDF_HEADER)
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/notpdf.pdf", "notpdf.pdf", "application/pdf",
                         supports_vision=False, supports_pdf_input=False)
        )
        assert "invalid header" in result.text
        assert result.image_blocks == ()

    def test_complex_pages_rendered_as_images_with_vision(self):
        """Extraction path: complex pages produce image_blocks when supports_vision=True."""
        # Three image refs on page 0 → complex
        complex_md = "![img1](a.png) ![img2](b.png) ![img3](c.png)\nsome text"
        pages = [
            {"text": complex_md},
            {"text": "# Normal page\n\nRegular text content."},
        ]
        extract_result = _make_exec_result(0, self._make_page_output(pages))
        render_result = _make_exec_result(0, "")
        cleanup_result = _make_exec_result(0, "")
        sandbox = _make_sandbox(
            file_bytes=_VALID_PDF_HEADER,
            exec_results=[extract_result, render_result, cleanup_result],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/test.pdf", "test.pdf", "application/pdf",
                         supports_vision=True, supports_pdf_input=False)
        )
        assert "rendered as image" in result.text
        assert len(result.image_blocks) == 1
        assert result.image_blocks[0]["type"] == "image_url"

    def test_complex_pages_not_rendered_without_vision(self):
        """Extraction path: complex pages are included as text when supports_vision=False."""
        complex_md = "![img1](a.png) ![img2](b.png) ![img3](c.png)\nsome text"
        pages = [{"text": complex_md}]
        extract_result = _make_exec_result(0, self._make_page_output(pages))
        cleanup_result = _make_exec_result(0, "")
        sandbox = _make_sandbox(
            file_bytes=_VALID_PDF_HEADER,
            exec_results=[extract_result, cleanup_result],
        )
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/test.pdf", "test.pdf", "application/pdf",
                         supports_vision=False, supports_pdf_input=False)
        )
        # No image blocks since vision not supported
        assert result.image_blocks == ()
        # The page text should be included directly
        assert "--- Page 1 ---" in result.text

    def test_oversized_pdf_rejected_in_extraction_path(self):
        """Extraction path: PDF exceeding 100 MB is rejected before running sandbox."""
        big_bytes = b"%PDF-" + b"x" * (100 * 1024 * 1024 + 1)
        sandbox = _make_sandbox(file_bytes=big_bytes)
        proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
        result = _run(
            proc.process("/tmp/huge.pdf", "huge.pdf", "application/pdf",
                         supports_vision=False, supports_pdf_input=False)
        )
        assert "too large to extract" in result.text
        # exec_command should not have been called for extraction (only possibly cleanup)
        sandbox.write_file.assert_not_called()


def test_native_path_sets_media_type():
    sandbox = _make_sandbox(
        file_bytes=_VALID_PDF_HEADER,
        exec_results=[_make_exec_result(0, "3")],  # page count = 3
    )
    proc = PdfFileProcessor(sandbox=sandbox, file_uploader=_make_uploader())
    result = _run(proc.process("/tmp/r.pdf", "r.pdf", "application/pdf",
                               supports_vision=True, supports_pdf_input=True))
    assert result.media_type == "application/pdf"
    assert result.document_blocks  # native path produced a file block
