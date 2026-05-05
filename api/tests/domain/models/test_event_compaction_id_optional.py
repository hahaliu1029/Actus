"""Pre-B6 JSONB rows without compaction_id must still validate (legacy compat)."""
from app.domain.models.event import CompactionEvent


def test_compaction_event_compaction_id_optional_default_none():
    ev = CompactionEvent(
        level=2, tokens_before=100, tokens_after=50,
        messages_removed=5, usage_ratio_after=0.5,
    )
    assert ev.compaction_id is None
    assert ev.type == "compaction"


def test_compaction_event_accepts_compaction_id_when_provided():
    ev = CompactionEvent(
        level=2, tokens_before=100, tokens_after=50,
        messages_removed=5, usage_ratio_after=0.5,
        compaction_id="aabbccddeeff0011",
    )
    assert ev.compaction_id == "aabbccddeeff0011"


def test_legacy_jsonb_payload_without_compaction_id_validates():
    payload = {
        "type": "compaction",
        "level": 3,
        "tokens_before": 8000,
        "tokens_after": 2000,
        "messages_removed": 12,
        "usage_ratio_after": 0.25,
    }
    ev = CompactionEvent.model_validate(payload)
    assert ev.compaction_id is None
