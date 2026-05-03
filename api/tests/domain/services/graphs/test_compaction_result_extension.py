"""Verify CompactionResult carries summary_text + operations + compaction_id,
and that escalation produces a 2-entry operations array per [R2-P2-8]."""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.services.graphs.compaction import (
    CompactionResult,
    GradualCompactor,
)
from app.domain.services.graphs.token_estimator import TokenEstimator


def test_compaction_result_has_new_fields():
    r = CompactionResult(
        messages=(),
        level_applied=2,
        tokens_before=100,
        tokens_after=50,
        summary_injected=True,
        messages_removed=5,
        usage_ratio_after=0.5,
        summary_text="hello",
        operations=[{"kind": "llm_summary"}],
        compaction_id="aabbccddeeff0011",
    )
    assert r.summary_text == "hello"
    assert r.operations == [{"kind": "llm_summary"}]
    assert r.compaction_id == "aabbccddeeff0011"


def test_compaction_result_defaults_for_legacy_callers():
    """Existing callers that don't supply new fields should still work."""
    r = CompactionResult(
        messages=(),
        level_applied=0,
        tokens_before=10,
        tokens_after=10,
        summary_injected=False,
        messages_removed=0,
        usage_ratio_after=0.1,
    )
    assert r.summary_text is None
    assert r.operations == []
    assert r.compaction_id is None


def test_hard_compact_returns_single_op_entry():
    estimator = TokenEstimator(strategy="char")
    compactor = GradualCompactor(estimator)
    msgs = [SystemMessage(content="sys")] + [HumanMessage(content=f"m{i}") for i in range(30)]
    result = compactor._hard_compact(messages=msgs, context_window=100, tokens_before=200)
    assert result.level_applied == 3
    assert len(result.operations) == 1
    assert result.operations[0]["kind"] == "hard_truncate"
    assert "messages_kept" in result.operations[0]
    assert "tokens_before" in result.operations[0]
    assert "tokens_after" in result.operations[0]


def test_hard_compact_with_prior_operations_rolls_up_totals():
    """[CXR1-P1-1] When prior_operations is supplied, totals roll up:
    - tokens_before = first prior op's tokens_before
    - tokens_after = current hard step's tokens_after
    - messages_removed = sum across the chain (summarized + removed)
    """
    estimator = TokenEstimator(strategy="char")
    compactor = GradualCompactor(estimator)
    msgs = [SystemMessage(content="sys")] + [HumanMessage(content=f"m{i}") for i in range(30)]
    prior = [{
        "kind": "llm_summary",
        "tokens_before": 10_000,
        "tokens_after": 4_000,
        "messages_summarized": 25,
        "summary_chars": 500,
        "identifiers_preserved_count": 3,
    }]
    result = compactor._hard_compact(
        messages=msgs,
        context_window=8_000,
        tokens_before=4_000,         # = prior soft step's tokens_after
        prior_operations=prior,
    )
    # Operations array preserves the chain
    kinds = [op["kind"] for op in result.operations]
    assert kinds == ["llm_summary", "hard_truncate"]
    # Rollup totals match spec § Operations Array Schema
    assert result.tokens_before == 10_000          # first op's tokens_before
    assert result.tokens_after == result.operations[-1]["tokens_after"]  # last op's tokens_after
    assert result.messages_removed == 25 + result.operations[-1]["messages_removed"]


# This test uses anyio convention. The plan template originally had @pytest.mark.asyncio
# but project convention is anyio (pytest-asyncio NOT installed).
@pytest.mark.anyio
async def test_escalation_produces_two_entry_operations():
    """End-to-end: Level 2 fires but post-verify fails → 2-entry [llm_summary, hard_truncate].

    Sizing rationale (TokenEstimator(strategy="char") empirically yields ~0.386 token/char
    for these messages):
      input estimate ≈ 19_324 tokens (50 × 1000-char HumanMessages + sys)
      window = 21_000 → input/window ≈ 0.92 (between 0.85 soft and 0.95 hard → Level 2 fires
                                              first, NOT direct jump to Level 3)
      fake summary = 100_000 chars ≈ 38_600 tokens (>> target_ratio + 0.03 = 0.53 × window
                                                    → post-verify fails → escalation to Level 3)
      summary_max_chars = 200_000 prevents the compactor's own truncation from shrinking
                          the summary below the post-verify threshold.
    """
    estimator = TokenEstimator(strategy="char")

    class _FakeLLM:
        async def ainvoke(self, prompt, **kw):
            from langchain_core.messages import AIMessage
            return AIMessage(content="x" * 100_000)  # >> target_ratio × window

    compactor = GradualCompactor(
        estimator,
        soft_trigger_ratio=0.85,
        hard_trigger_ratio=0.95,
        target_ratio=0.5,
        summary_max_chars=200_000,  # let the huge fake summary survive truncation
    )
    msgs = [SystemMessage(content="sys")] + [HumanMessage(content="x" * 1000) for _ in range(50)]
    result = await compactor.try_compact(
        messages=msgs,
        context_window=21_000,
        summary_llm=_FakeLLM(),
    )
    # Unconditional contract assertions — escalation MUST fire with this config
    assert result.level_applied == 3
    assert len(result.operations) == 2
    kinds = [op["kind"] for op in result.operations]
    assert kinds == ["llm_summary", "hard_truncate"]
    # Rolled-up totals match the spec § Operations Array Schema
    assert result.tokens_before == result.operations[0]["tokens_before"]
    assert result.tokens_after == result.operations[1]["tokens_after"]
    # P1.1 fix: summary_text must survive escalation so the persisted record carries
    # the LLM's summary even after Level 3 hard truncation overwrote messages.
    assert result.summary_text is not None
    assert len(result.summary_text) > 0
