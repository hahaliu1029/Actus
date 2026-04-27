"""Tests for GradualCompactor.

Uses TokenEstimator(strategy="char") for predictable token counts:
  char strategy: len(text) // 3  (integer division)
  MESSAGE_OVERHEAD_TOKENS = 3 per message
"""
from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from unittest.mock import AsyncMock

from app.domain.services.graphs.compaction import (
    SUMMARY_END,
    SUMMARY_START,
    CompactionResult,
    GradualCompactor,
    extract_identifiers,
    extract_summary_section,
    inject_summary_section,
)
from app.domain.services.graphs.token_estimator import (
    MESSAGE_OVERHEAD_TOKENS,
    TokenEstimator,
)


class TestCompactionResult:
    def test_frozen_immutable(self):
        r = CompactionResult(
            messages=(),
            level_applied=0,
            tokens_before=100,
            tokens_after=100,
            summary_injected=False,
            messages_removed=0,
            usage_ratio_after=0.5,
        )
        with pytest.raises(AttributeError):
            r.level_applied = 1  # type: ignore[misc]

    def test_fields_stored(self):
        msgs = (SystemMessage(content="sys"),)
        r = CompactionResult(
            messages=msgs,
            level_applied=2,
            tokens_before=1000,
            tokens_after=600,
            summary_injected=True,
            messages_removed=5,
            usage_ratio_after=0.6,
        )
        assert r.level_applied == 2
        assert r.tokens_before == 1000
        assert r.messages_removed == 5
        assert r.usage_ratio_after == 0.6


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_compactor(**overrides) -> GradualCompactor:
    defaults = dict(
        token_estimator=TokenEstimator(strategy="char"),
        soft_trigger_ratio=0.85,
        hard_trigger_ratio=0.95,
        target_ratio=0.65,
        summary_max_chars=16_000,
        token_safety_factor=1.0,  # disable safety factor for predictable tests
    )
    defaults.update(overrides)
    return GradualCompactor(**defaults)


def _pad(n: int) -> str:
    """Return a string whose char-strategy token count is exactly n - MESSAGE_OVERHEAD_TOKENS.
    char strategy: tokens = len(text) // 3, plus MESSAGE_OVERHEAD_TOKENS per msg.
    So text length = (n - MESSAGE_OVERHEAD_TOKENS) * 3.
    """
    text_tokens = n - MESSAGE_OVERHEAD_TOKENS
    return "x" * (text_tokens * 3)


# ── TestHardCompact ───────────────────────────────────────────────────────────


class TestHardCompact:
    def test_keeps_sys_plus_last_19(self):
        c = _make_compactor()
        sys_msg = SystemMessage(content="system prompt")
        msgs = [sys_msg] + [HumanMessage(content=f"msg-{i}") for i in range(24)]
        assert len(msgs) == 25
        result = c._hard_compact(msgs, context_window=1000, tokens_before=950)
        assert len(result.messages) == 20
        assert result.messages[0].content.startswith("system prompt")
        assert result.messages[-1].content == "msg-23"
        assert result.messages_removed == 5

    def test_truncation_marker_injected(self):
        c = _make_compactor()
        sys_msg = SystemMessage(content="system prompt")
        msgs = [sys_msg] + [HumanMessage(content=f"m-{i}") for i in range(24)]
        result = c._hard_compact(msgs, context_window=1000, tokens_before=950)
        assert SUMMARY_START in result.messages[0].content
        assert "硬截断" in result.messages[0].content

    def test_preserves_existing_summary(self):
        summary_section = f"\n\n{SUMMARY_START}\n## 对话历史摘要（自动生成）\n\nold summary\n{SUMMARY_END}"
        sys_msg = SystemMessage(content="system prompt" + summary_section)
        c = _make_compactor()
        msgs = [sys_msg] + [HumanMessage(content=f"m-{i}") for i in range(24)]
        result = c._hard_compact(msgs, context_window=1000, tokens_before=950)
        content = result.messages[0].content
        assert "old summary" in content
        assert "硬截断" in content
        assert content.count(SUMMARY_START) == 1

    def test_small_list_no_truncation(self):
        c = _make_compactor()
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        result = c._hard_compact(msgs, context_window=100, tokens_before=95)
        assert len(result.messages) == 2
        assert result.messages_removed == 0
        assert result.level_applied == 3


# ── TestTryCompactRouting ─────────────────────────────────────────────────────


class TestTryCompactRouting:
    @pytest.mark.anyio
    async def test_below_soft_no_compact(self):
        c = _make_compactor()
        msgs = [SystemMessage(content=_pad(80))]
        result = await c.try_compact(msgs, context_window=100, summary_llm=None)
        assert result.level_applied == 0

    @pytest.mark.anyio
    async def test_above_hard_goes_to_level3(self):
        c = _make_compactor()
        sys_msg = SystemMessage(content=_pad(10))
        others = [HumanMessage(content=_pad(4)) for _ in range(24)]
        msgs = [sys_msg] + others
        result = await c.try_compact(msgs, context_window=100, summary_llm=None)
        assert result.level_applied == 3
        assert len(result.messages) == 20

    @pytest.mark.anyio
    async def test_between_soft_hard_no_llm_goes_to_level3(self):
        c = _make_compactor()
        sys_msg = SystemMessage(content=_pad(10))
        others = [HumanMessage(content=_pad(4)) for _ in range(20)]
        msgs = [sys_msg] + others
        result = await c.try_compact(msgs, context_window=100, summary_llm=None)
        assert result.level_applied == 3

    @pytest.mark.anyio
    async def test_safety_factor_applied(self):
        c = _make_compactor(token_safety_factor=1.2)
        msgs = [SystemMessage(content=_pad(75))]
        result = await c.try_compact(msgs, context_window=100, summary_llm=None)
        assert result.level_applied == 3


# ── TestIdentifierExtraction ──────────────────────────────────────────────────


class TestIdentifierExtraction:
    def test_extracts_file_paths(self):
        text = "Modified /app/domain/models/foo.py and /app/core/config.py"
        ids = extract_identifiers(text)
        assert "/app/domain/models/foo.py" in ids
        assert "/app/core/config.py" in ids

    def test_extracts_urls(self):
        text = "See https://example.com/api/v1/users for docs"
        ids = extract_identifiers(text)
        assert "https://example.com/api/v1/users" in ids

    def test_extracts_uuids(self):
        text = "Task ID: 550e8400-e29b-41d4-a716-446655440000"
        ids = extract_identifiers(text)
        assert "550e8400-e29b-41d4-a716-446655440000" in ids

    def test_deduplicates(self):
        text = "/app/foo.py and /app/foo.py again"
        ids = extract_identifiers(text)
        assert ids.count("/app/foo.py") == 1

    def test_ignores_short_paths(self):
        text = "use /n for newline"
        ids = extract_identifiers(text)
        assert len(ids) == 0


# ── TestSummarySection ────────────────────────────────────────────────────────


class TestSummarySection:
    def test_inject_into_clean_content(self):
        result = inject_summary_section("original prompt", "summary text", 5, 3000)
        assert "original prompt" in result
        assert SUMMARY_START in result
        assert "summary text" in result

    def test_replace_existing_section(self):
        old = f"prompt\n\n{SUMMARY_START}\nold\n{SUMMARY_END}"
        result = inject_summary_section(old, "new summary", 3, 2000)
        assert "old" not in result
        assert "new summary" in result
        assert result.count(SUMMARY_START) == 1

    def test_extract_existing_summary(self):
        content = f"prompt\n\n{SUMMARY_START}\n## 摘要\n\nsome summary\n{SUMMARY_END}"
        extracted = extract_summary_section(content)
        assert "some summary" in extracted

    def test_extract_no_summary(self):
        assert extract_summary_section("plain content") == ""


# ── TestSoftCompact ───────────────────────────────────────────────────────────


def _make_mock_llm(response: str = "Summary of conversation.") -> AsyncMock:
    mock = AsyncMock()
    mock.ainvoke.return_value = AIMessage(content=response)
    return mock


class TestSoftCompact:
    @pytest.mark.anyio
    async def test_summarizes_oldest_messages(self):
        # Call _soft_compact directly to verify level-2 behaviour independent of post-verify.
        # Post-verify (in try_compact) can legitimately escalate to level 3 when the
        # injected summary block itself adds tokens that keep usage above the threshold.
        c = _make_compactor()
        sys_msg = SystemMessage(content=_pad(10))
        humans = [HumanMessage(content=_pad(4)) for _ in range(20)]
        msgs = [sys_msg] + humans
        mock_llm = _make_mock_llm("Short summary.")
        tokens_before = c._estimator.estimate_messages(msgs)
        result = await c._soft_compact(
            messages=msgs,
            context_window=100,
            summary_llm=mock_llm,
            tokens_before=tokens_before,
        )
        assert result.level_applied == 2
        assert result.summary_injected is True
        assert result.messages_removed > 0
        assert SUMMARY_START in result.messages[0].content

    @pytest.mark.anyio
    async def test_llm_exception_falls_to_level3(self):
        c = _make_compactor()
        sys_msg = SystemMessage(content=_pad(10))
        msgs = [sys_msg] + [HumanMessage(content=_pad(4)) for _ in range(24)]
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("API timeout")
        result = await c.try_compact(msgs, context_window=100, summary_llm=mock_llm)
        assert result.level_applied == 3

    @pytest.mark.anyio
    async def test_summary_output_hard_truncated(self):
        c = _make_compactor(summary_max_chars=20)
        sys_msg = SystemMessage(content=_pad(10))
        msgs = [sys_msg] + [HumanMessage(content=_pad(4)) for _ in range(20)]
        mock_llm = _make_mock_llm("A" * 100)
        result = await c.try_compact(msgs, context_window=100, summary_llm=mock_llm)
        if result.level_applied == 2:
            section = extract_summary_section(result.messages[0].content)
            assert len(section) < 200

    @pytest.mark.anyio
    async def test_cumulative_summary_merges_old(self):
        c = _make_compactor()
        existing = f"prompt\n\n{SUMMARY_START}\n## 对话历史摘要（自动生成）\n\nold summary\n{SUMMARY_END}"
        sys_msg = SystemMessage(content=existing)
        msgs = [sys_msg] + [HumanMessage(content=_pad(4)) for _ in range(20)]
        mock_llm = _make_mock_llm("Merged summary.")
        result = await c.try_compact(msgs, context_window=100, summary_llm=mock_llm)
        if result.level_applied == 2:
            call_args = mock_llm.ainvoke.call_args
            prompt_text = str(call_args)
            assert "old summary" in prompt_text

    @pytest.mark.anyio
    async def test_post_verify_fallback_to_level3(self):
        c = _make_compactor(summary_max_chars=50000)
        sys_msg = SystemMessage(content=_pad(10))
        msgs = [sys_msg] + [HumanMessage(content=_pad(4)) for _ in range(24)]
        mock_llm = _make_mock_llm("x" * 30000)
        result = await c.try_compact(msgs, context_window=100, summary_llm=mock_llm)
        assert result.level_applied == 3


@pytest.mark.anyio
async def test_try_compact_threads_config_to_summary_llm_ainvoke():
    """Audit Round 11 P1 #1 (Round 12 P1 #1 size-fix) regression-lock for B2.

    B2's `_build_on_context_overflow_callback` (Task 2.6 Step 2) sends a
    `config=` payload to `compactor.try_compact(...)` so the Level-2
    summary LLM call lands in the B4 cost ledger. That depends on
    `GradualCompactor` threading `config` all the way through
    `_soft_compact` (`compaction.py:411-417`) into
    `summary_llm.ainvoke(..., config=config)`.

    Without this test, a future refactor that drops the `**ainvoke_kwargs`
    splat in `_soft_compact` would silently break B2's cost-ledger path.

    Routing math (must enter Level 2):
        soft=0.85, hard=0.95, safety_factor=1.0 (via _make_compactor() defaults)
        sys=_pad(10) + 20×_pad(4) = 10 + 80 = 90 tokens
        ratio = 90 / 100 = 0.9  ∈ (0.85, 0.95)  → ENTERS _soft_compact

    Audit Round 13 P1 #1: do NOT pin `result.level_applied == 2`. After
    `_soft_compact`, `try_compact` may post-verify-escalate to Level 3 if
    the post-summary ratio is still above target. The capture happens
    INSIDE ainvoke (Level-2 path) before any escalation — that's the
    authoritative routing proof.
    """
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    captured: dict = {}

    class _CapturingSummaryLLM(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "capture-stub"

        def _generate(self, *a, **kw):
            raise NotImplementedError

        async def ainvoke(self, prompt, **kwargs):
            captured["config"] = kwargs.get("config")
            return AIMessage(
                content='{"summary":"ok","key_facts":[],"open_questions":[]}',
            )

        def with_structured_output(self, schema, **kwargs):
            return self

    sentinel = {
        "callbacks": [object()],
        "metadata": {"langgraph_node": "context_compaction"},
    }

    c = _make_compactor()
    sys_msg = SystemMessage(content=_pad(10))
    others = [HumanMessage(content=_pad(4)) for _ in range(20)]
    msgs = [sys_msg] + others

    await c.try_compact(
        messages=msgs,
        context_window=100,
        summary_llm=_CapturingSummaryLLM(),
        config=sentinel,
    )

    assert captured.get("config") is sentinel, (
        "GradualCompactor must forward `config` to summary_llm.ainvoke. "
        "B2 emergency-compact cost-ledger relies on this contract. "
        "If captured is empty, _soft_compact was never entered — check "
        "soft/hard trigger ratios and re-tune _pad sizing so "
        "usage_ratio ∈ (soft_trigger_ratio, hard_trigger_ratio)."
    )
