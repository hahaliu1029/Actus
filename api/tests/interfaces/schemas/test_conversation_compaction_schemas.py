"""Schema shape lock: list / detail / original-content responses."""
import pytest

from app.interfaces.schemas.conversation_compaction import (
    ConversationCompactionListResponse,
    ConversationCompactionDetailResponse,
)


def test_list_response_kinds_strict_union():
    item = {
        "compaction_id": "a" * 16,
        "kinds": ["llm_summary"],
        "summary_preview": "p",
        "tokens_before_total": 100,
        "tokens_after_total": 50,
        "messages_removed_total": 5,
        "first_visible_event_id": None,
        "last_visible_event_id": None,
        "has_recoverable_original": False,
        "created_at": "2026-05-03T00:00:00Z",
    }
    resp = ConversationCompactionListResponse(items=[item])
    assert resp.items[0].kinds == ["llm_summary"]


def test_list_response_kinds_rejects_unknown_value():
    with pytest.raises(Exception):
        ConversationCompactionListResponse(items=[{
            "compaction_id": "a" * 16, "kinds": ["junk_kind"],
            "summary_preview": "", "tokens_before_total": 0,
            "tokens_after_total": 0, "messages_removed_total": 0,
            "first_visible_event_id": None, "last_visible_event_id": None,
            "has_recoverable_original": False, "created_at": "2026-05-03T00:00:00Z",
        }])


def test_detail_response_operations_strict_union():
    detail = ConversationCompactionDetailResponse(
        compaction_id="a" * 16, session_id="s",
        summary="x", summary_tokens=1,
        operations=[{"kind": "hard_truncate", "tokens_before": 10, "tokens_after": 5,
                     "messages_removed": 3, "messages_kept": 19}],
        parent_compaction_id=None,
        first_visible_event_id=None, last_visible_event_id=None,
        pre_compact_checkpoint_id=None,
        tokens_before_total=10, tokens_after_total=5, messages_removed_total=3,
        created_at="2026-05-03T00:00:00Z",
    )
    assert detail.operations[0].kind == "hard_truncate"
