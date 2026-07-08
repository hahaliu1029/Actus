"""B12 P1: _translate_outcome materialize (file_view LLM-facing image reshape)."""
import asyncio

from app.domain.models.tool_result import (
    ImageUrlBlock,
    ImageUrlPayload,
    MultimodalPayload,
    Passthrough,
)
from app.domain.services.graphs.react_graph import _translate_outcome
from app.domain.services.tools.tool_source_resolver import ToolSource


class _Profile:
    provider_id = "kimi_k2"
    accepts_image_url = False
    accepts_image_base64 = True
    image_max_bytes = 5 * 1024 * 1024


class _OpenAIProfile:
    provider_id = "openai"
    accepts_image_url = True
    accepts_image_base64 = True
    image_max_bytes = 5 * 1024 * 1024


class _Resolver:
    def __init__(self, data): self._data = data
    async def load_image_bytes(self, display_url, *, max_bytes): return self._data


def _passthrough(url):
    return Passthrough(
        content="[file_view: file_view — 1 image(s) loaded]",
        data=MultimodalPayload(blocks=[ImageUrlBlock(image_url=ImageUrlPayload(url=url))]),
    )


def _run(outcome, **kw):
    tc = {"id": "c1", "name": "file_view", "args": {}, "type": "tool_call"}
    ts = ToolSource(source="native", category="file", canonical_name="file_view")
    return asyncio.run(_translate_outcome(
        outcome, tc, ts, None,  # session_ctx unused (del'd)
        tool_result_max_chars=8000, guide_injector=None, **kw,
    ))


def _image_urls(deferred):
    human = deferred[0]
    return [b["image_url"]["url"] for b in human.content if b.get("type") == "image_url"]


def test_materialize_off_keeps_http_url() -> None:
    _msg, deferred, _events = _run(_passthrough("https://minio/a.png"))
    assert _image_urls(deferred) == ["https://minio/a.png"]


def test_materialize_on_url_forbidden_becomes_base64() -> None:
    _msg, deferred, _events = _run(
        _passthrough("https://minio/a.png"),
        materialize_enabled=True,
        image_transport_profile=_Profile(),
        image_bytes_resolver=_Resolver(b"PNGBYTES"),
    )
    urls = _image_urls(deferred)
    assert urls and urls[0].startswith("data:image/png;base64,")


def test_materialize_on_url_ok_profile_keeps_url() -> None:
    _msg, deferred, _events = _run(
        _passthrough("https://minio/a.png"),
        materialize_enabled=True,
        image_transport_profile=_OpenAIProfile(),
        image_bytes_resolver=_Resolver(None),
    )
    assert _image_urls(deferred) == ["https://minio/a.png"]


def test_materialize_on_url_forbidden_fetch_fails_placeholder() -> None:
    _msg, deferred, _events = _run(
        _passthrough("https://minio/a.png"),
        materialize_enabled=True,
        image_transport_profile=_Profile(),
        image_bytes_resolver=_Resolver(None),  # fetch returns None
    )
    human = deferred[0]
    assert _image_urls(deferred) == []  # no image_url block
    assert any("[image unavailable:" in b.get("text", "") for b in human.content)
    # header re-counts image blocks → 0
    assert any("0 image(s) loaded" in b.get("text", "") for b in human.content if b.get("type") == "text")


def test_materialize_on_uses_payload_media_type() -> None:
    """R1#P2-1: base64 data URL 用 payload.media_type（非硬编码 image/png）。"""
    outcome = Passthrough(
        content="[file_view: file_view — 1 image(s) loaded]",
        data=MultimodalPayload(
            blocks=[ImageUrlBlock(image_url=ImageUrlPayload(url="https://minio/a.jpg"))],
            media_type="image/jpeg",
        ),
    )
    _msg, deferred, _events = _run(
        outcome,
        materialize_enabled=True,
        image_transport_profile=_Profile(),
        image_bytes_resolver=_Resolver(b"JPEGBYTES"),
    )
    urls = _image_urls(deferred)
    assert urls and urls[0].startswith("data:image/jpeg;base64,")


def test_materialize_pdf_page_uses_image_mime_not_document_mime() -> None:
    """R5#P2: extraction PDF 的 payload media_type='application/pdf' **不能**套到 image_url
    block（页图是 JPEG）——否则产 data:application/pdf 被 sanitizer strip。有
    document_preview.thumbnail → 用其 image/jpeg；绝不产 data:application/pdf。"""
    from app.domain.models.tool_result import DocumentPreview, DocumentThumbnail

    outcome = Passthrough(
        content="[PDF: scan.pdf, 1 pages]",
        data=MultimodalPayload(
            blocks=[ImageUrlBlock(image_url=ImageUrlPayload(url="https://minio/scan_p0.jpg"))],
            media_type="application/pdf",
            document_preview=DocumentPreview(
                filename="scan.pdf", media_type="application/pdf", page_count=1,
                thumbnail=DocumentThumbnail(
                    url="https://minio/scan_p0.jpg", media_type="image/jpeg", page=0)),
        ),
    )
    _msg, deferred, _events = _run(
        outcome,
        materialize_enabled=True,
        image_transport_profile=_Profile(),
        image_bytes_resolver=_Resolver(b"JPEGBYTES"),
    )
    urls = _image_urls(deferred)
    assert urls and urls[0].startswith("data:image/jpeg;base64,")
    assert "application/pdf" not in urls[0]


def test_build_react_graph_accepts_image_bytes_resolver() -> None:
    """smoke: build_react_graph 接受新 image_bytes_resolver 参数并返回图。"""
    from unittest.mock import MagicMock
    from app.domain.services.graphs.react_graph import build_react_graph

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.profile = _Profile()
    graph = build_react_graph(
        llm=llm, tools=[], image_bytes_resolver=_Resolver(b"x"),
    )
    assert graph is not None
