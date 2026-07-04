"""B8 PR-1: recall DTO 形状契约（P-1）。"""
import dataclasses
from datetime import datetime, timezone

import pytest

from app.domain.models.memory_recall import (
    RecallCachePayload,
    RecalledMemory,
    RecalledMemoryItem,
    RecallQueryMaterial,
)


def _item(**overrides):
    defaults = dict(
        chunk_id="c1",
        category="fact",
        content="数据库是 PostgreSQL 17",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        score=0.42,
    )
    defaults.update(overrides)
    return RecalledMemoryItem(**defaults)


class TestRecallDtos:
    def test_material_entry_is_required(self):
        with pytest.raises(TypeError):
            RecallQueryMaterial(message="hi", original_request=None, session_title=None)

    def test_material_is_frozen(self):
        m = RecallQueryMaterial(
            message="hi", original_request=None, session_title=None, entry="graph",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.message = "other"

    def test_material_replace_fills_title(self):
        m = RecallQueryMaterial(
            message="hi", original_request=None, session_title=None, entry="detection",
        )
        m2 = dataclasses.replace(m, session_title="真实标题")
        assert m2.session_title == "真实标题"
        assert m2.entry == "detection"

    def test_item_category_none_allowed(self):
        assert _item(category=None).category is None

    def test_recalled_memory_shape(self):
        rm = RecalledMemory(items=(_item(),), query_hash="h", cache_hit=False, recall_id="rid")
        assert isinstance(rm.items, tuple)
        names = {f.name for f in dataclasses.fields(RecalledMemory)}
        assert names == {"items", "query_hash", "cache_hit", "recall_id"}

    def test_cache_payload_has_no_cache_hit_or_recall_id(self):
        names = {f.name for f in dataclasses.fields(RecallCachePayload)}
        assert names == {"items", "candidate_count"}
