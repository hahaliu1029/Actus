from app.interfaces.schemas.conversation_compaction import RecoveredMessage
from app.interfaces.schemas.recovered_message_sanitizer import (
    _MAX_RECURSION_DEPTH,
    _has_oversized_inline_string,
    sanitize_recovered_message,
)


def test_string_content_truncated_when_over_1mb():
    huge = "x" * 2_000_000
    msg = RecoveredMessage(type="human", content=huge)
    out = sanitize_recovered_message(msg)
    assert out.content.startswith("[content omitted")


def test_multimodal_image_url_oversized_blob_replaced():
    blocks = [
        {"type": "text", "text": "ok"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 2_000_000}},
    ]
    msg = RecoveredMessage(type="human", content=blocks)
    out = sanitize_recovered_message(msg)
    assert out.content[0] == {"type": "text", "text": "ok"}
    assert "omitted" in out.content[1]["text"]


def test_small_content_passthrough():
    msg = RecoveredMessage(type="ai", content="short")
    assert sanitize_recovered_message(msg).content == "short"


def test_deeply_nested_oversized_blob_replaced():
    """[CXR2-P3-7] Recursive scan finds oversized strings deeper than 1 level."""
    blocks = [
        {"type": "text", "text": "ok"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + "A" * 2_000_000,
                "detail": "auto",
            },
        },
        {
            "type": "tool_result",
            "result": {"nested": {"payload": "B" * 2_000_000}},
        },
    ]
    msg = RecoveredMessage(type="human", content=blocks)
    out = sanitize_recovered_message(msg)
    assert out.content[0] == {"type": "text", "text": "ok"}
    assert "omitted" in out.content[1]["text"]
    assert "omitted" in out.content[2]["text"]


def test_string_content_truncated_when_utf8_bytes_exceed_1mb():
    """Non-ASCII content > 1 MB UTF-8 bytes is truncated even if char count is smaller.

    Each Chinese character encodes to 3 UTF-8 bytes.
    350_000 chars × 3 bytes = 1_050_000 bytes > 1_000_000 (limit).
    Without the UTF-8 fix the old char-count check (350_000 < 1_000_000) would pass
    the string through unchanged — this test catches that regression.
    """
    # 350_000 × 3 bytes = 1_050_000 bytes — just over the 1 MB limit.
    huge_cn = "中" * 350_000
    msg = RecoveredMessage(type="human", content=huge_cn)
    out = sanitize_recovered_message(msg)
    assert out.content.startswith("[content omitted")


def test_recursion_depth_cap_returns_oversized():
    """Exceed _MAX_RECURSION_DEPTH in _has_oversized_inline_string → returns True (fail-closed)."""
    # Build a dict nested _MAX_RECURSION_DEPTH + 2 levels deep with a small value at the leaf.
    def _nest(depth: int):
        if depth == 0:
            return {"leaf": "small"}
        return {"child": _nest(depth - 1)}

    deep = _nest(_MAX_RECURSION_DEPTH + 2)
    # The function should return True (treat as oversized) rather than raising RecursionError.
    assert _has_oversized_inline_string(deep) is True
