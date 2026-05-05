"""Strip multimodal blobs >1MB from RecoveredMessage payloads before HTTP response.

Spec § Section 3 Response 200 + Risk Register: original-content can carry
huge image / PDF base64 blobs that bloat the JSON response. Drop those.
"""
from __future__ import annotations

from typing import Any

from app.interfaces.schemas.conversation_compaction import RecoveredMessage

_MAX_INLINE_BYTES = 1_000_000  # 1 MB — measured in UTF-8 encoded bytes
_MAX_RECURSION_DEPTH = 32  # fail-closed on malformed deeply-nested content


def _has_oversized_inline_string(value: Any, depth: int = 0) -> bool:
    """[CXR2-P3-7] Recursively scan for any string > _MAX_INLINE_BYTES (UTF-8 encoded).

    Returns True (oversized) when recursion depth exceeds _MAX_RECURSION_DEPTH so that
    malformed deeply-nested content is dropped rather than causing a RecursionError.
    """
    if depth > _MAX_RECURSION_DEPTH:
        return True  # fail-closed: treat over-deep as oversized
    if isinstance(value, str):
        # Use UTF-8 byte length, not char length, to bound HTTP response bytes correctly.
        return len(value.encode("utf-8", errors="replace")) > _MAX_INLINE_BYTES
    if isinstance(value, dict):
        return any(_has_oversized_inline_string(v, depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(_has_oversized_inline_string(v, depth + 1) for v in value)
    return False


def _shrink_block(block: dict[str, Any]) -> dict[str, Any]:
    """If a multimodal block carries an oversized inline blob anywhere in its
    nested structure, replace with a stub. [CXR2-P3-7] Recursive scan handles
    LangChain shapes deeper than 1 level (e.g., image_url.url, source.data,
    or any other nested string)."""
    if _has_oversized_inline_string(block):
        return {"type": block.get("type", "unknown"), "text": "[content omitted: too large for response]"}
    return block


def sanitize_recovered_message(msg: RecoveredMessage) -> RecoveredMessage:
    if isinstance(msg.content, str):
        # Long string content (e.g., huge tool output) — also cap.
        # Use UTF-8 byte length to bound HTTP response bytes correctly.
        if len(msg.content.encode("utf-8", errors="replace")) > _MAX_INLINE_BYTES:
            return msg.model_copy(update={"content": "[content omitted: too large for response]"})
        return msg
    if isinstance(msg.content, list):
        new_content = [_shrink_block(b) if isinstance(b, dict) else b for b in msg.content]
        return msg.model_copy(update={"content": new_content})
    return msg


def sanitize_recovered_messages(msgs: list[RecoveredMessage]) -> list[RecoveredMessage]:
    return [sanitize_recovered_message(m) for m in msgs]
