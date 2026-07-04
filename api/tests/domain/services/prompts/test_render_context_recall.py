"""B8 PR-2: RenderContext.recalled_memory 字段 + build_render_context kwarg。"""
import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.domain.models.memory_recall import RecalledMemory, RecalledMemoryItem
from app.domain.services.prompts.render_context import build_render_context
from app.domain.services.prompts.section import RenderContext


def _recalled():
    item = RecalledMemoryItem(
        chunk_id="c1", category="fact", content="内容",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc), score=0.5,
    )
    return RecalledMemory(items=(item,), query_hash="qh", cache_hit=False, recall_id="rid")


_AGENT_CONFIG = SimpleNamespace(supports_vision=True, supports_pdf_input=False)


class TestRenderContextRecalledMemory:
    def test_field_defaults_to_none(self):
        ctx = RenderContext(lang="zh")
        assert ctx.recalled_memory is None

    def test_field_is_frozen(self):
        ctx = RenderContext(lang="zh")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.recalled_memory = _recalled()

    def test_build_render_context_default_none(self):
        ctx = build_render_context({"language": "zh"}, {"configurable": {}}, _AGENT_CONFIG)
        assert ctx.recalled_memory is None

    def test_build_render_context_kwarg_passthrough(self):
        recalled = _recalled()
        ctx = build_render_context(
            {"language": "zh"}, {"configurable": {}}, _AGENT_CONFIG,
            recalled_memory=recalled,
        )
        assert ctx.recalled_memory is recalled
