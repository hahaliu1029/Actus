from unittest.mock import AsyncMock

from app.infrastructure.external.file_processors.registry import FileProcessorRegistry


def _make_registry(**kwargs):
    return FileProcessorRegistry(
        sandbox=AsyncMock(),
        file_uploader=AsyncMock(),
        **kwargs,
    )


class TestFileProcessorRegistry:
    def test_image_png_returns_image_processor(self):
        proc = _make_registry().get_processor("image/png")
        assert proc is not None
        assert type(proc).__name__ == "ImageFileProcessor"

    def test_image_jpeg_returns_image_processor(self):
        proc = _make_registry().get_processor("image/jpeg")
        assert proc is not None

    def test_unknown_mime_returns_none(self):
        assert _make_registry().get_processor("text/plain") is None

    def test_svg_excluded_from_image_processor(self):
        """SVG matches image/ prefix but Pillow can't parse it — must return None."""
        assert _make_registry().get_processor("image/svg+xml") is None

    def test_audio_disabled_by_default(self):
        assert _make_registry().get_processor("audio/mpeg") is None

    def test_pdf_depends_on_module_availability(self):
        """PDF processor availability depends on whether the module exists."""
        proc = _make_registry().get_processor("application/pdf")
        # May be None (ImportError) or PdfFileProcessor — just ensure no crash
        if proc is not None:
            assert type(proc).__name__ == "PdfFileProcessor"


def test_registry_threads_pdf_parallel_flag_to_processor() -> None:
    from unittest.mock import AsyncMock
    from app.infrastructure.external.file_processors.registry import FileProcessorRegistry
    from app.infrastructure.external.file_processors.pdf import PdfFileProcessor

    reg = FileProcessorRegistry(
        sandbox=AsyncMock(), file_uploader=AsyncMock(), pdf_page_parallel_enabled=True,
    )
    pdf_proc = reg.get_processor("application/pdf")
    assert isinstance(pdf_proc, PdfFileProcessor)
    assert pdf_proc._page_parallel_enabled is True


def test_registry_pdf_parallel_flag_defaults_off() -> None:
    from unittest.mock import AsyncMock
    from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

    reg = FileProcessorRegistry(sandbox=AsyncMock(), file_uploader=AsyncMock())
    assert reg.get_processor("application/pdf")._page_parallel_enabled is False
