"""B12: DocumentPreview/DocumentThumbnail models + MultimodalPayload
optional metadata None-omitting serializer."""
import pytest
from pydantic import ValidationError

from app.domain.models.tool_result import DocumentPreview, DocumentThumbnail


def test_document_thumbnail_construct_and_dump() -> None:
    t = DocumentThumbnail(url="https://minio/p0.jpg", media_type="image/jpeg", page=0)
    assert t.model_dump() == {
        "url": "https://minio/p0.jpg",
        "media_type": "image/jpeg",
        "page": 0,
    }


def test_document_preview_native_no_thumbnail() -> None:
    p = DocumentPreview(filename="report.pdf", media_type="application/pdf", page_count=12)
    assert p.thumbnail is None
    assert p.model_dump() == {
        "filename": "report.pdf",
        "media_type": "application/pdf",
        "page_count": 12,
        "thumbnail": None,
    }


def test_document_preview_with_thumbnail() -> None:
    p = DocumentPreview(
        filename="deck.pdf",
        media_type="application/pdf",
        page_count=3,
        thumbnail=DocumentThumbnail(url="https://minio/p0.jpg", media_type="image/jpeg", page=0),
    )
    dumped = p.model_dump()
    assert dumped["thumbnail"]["page"] == 0
    assert dumped["filename"] == "deck.pdf"


def test_document_preview_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        DocumentPreview(filename="x", media_type="application/pdf", bogus=1)  # type: ignore[call-arg]


from app.domain.models.tool_result import (  # noqa: E402
    ImageUrlBlock,
    ImageUrlPayload,
    MultimodalPayload,
)


def test_multimodal_payload_omits_none_b12_fields() -> None:
    """INV-B12-1: media_type/document_preview 为 None 时不出现在 dump
    （否则破坏 R2 round-trip 结构门 test_r2_golden_matrix.py:114）。"""
    payload = MultimodalPayload(blocks=[
        ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,x")),
    ])
    dumped = payload.model_dump(mode="json", by_alias=True)
    assert "media_type" not in dumped
    assert "document_preview" not in dumped
    assert dumped == {
        "blocks": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x", "detail": "auto"}},
        ]
    }


def test_multimodal_payload_includes_media_type_when_set() -> None:
    payload = MultimodalPayload(
        blocks=[ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,x"))],
        media_type="image/png",
    )
    dumped = payload.model_dump(mode="json", by_alias=True)
    assert dumped["media_type"] == "image/png"
    assert "document_preview" not in dumped  # still None → omitted


def test_multimodal_payload_includes_document_preview_when_set() -> None:
    payload = MultimodalPayload(
        blocks=[],
        document_preview=DocumentPreview(
            filename="r.pdf", media_type="application/pdf", page_count=2
        ),
    )
    dumped = payload.model_dump(mode="json", by_alias=True)
    assert dumped["document_preview"]["filename"] == "r.pdf"
    assert "media_type" not in dumped
