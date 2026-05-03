"""Helper: builds ConversationCompaction from CompactionResult, computes id, calls repo."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.application.services.compaction_recorder import record_compaction
from app.domain.services.graphs.compaction import CompactionResult

pytestmark = pytest.mark.anyio


async def test_record_compaction_builds_record_and_calls_repo():
    uow = MagicMock()
    uow.compaction = MagicMock()

    async def _create_or_get(rec):
        return rec

    uow.compaction.create_or_get = AsyncMock(side_effect=_create_or_get)

    result = CompactionResult(
        messages=(),
        level_applied=2,
        tokens_before=1000,
        tokens_after=500,
        summary_injected=True,
        messages_removed=10,
        usage_ratio_after=0.5,
        summary_text="summary",
        operations=[{"kind": "llm_summary", "tokens_before": 1000, "tokens_after": 500}],
    )

    cid = await record_compaction(
        uow=uow,
        session_id="sess_1",
        result=result,
        messages_input_hash="aabbccddeeff0011",
    )

    assert isinstance(cid, str)
    assert len(cid) == 16
    uow.compaction.create_or_get.assert_awaited_once()
    rec = uow.compaction.create_or_get.await_args.args[0]
    assert rec.session_id == "sess_1"
    assert rec.summary == "summary"
    assert rec.tokens_before_total == 1000
    assert rec.tokens_after_total == 500
    assert rec.messages_removed_total == 10
    assert rec.operations == result.operations
    assert rec.first_visible_event_id is None
    assert rec.last_visible_event_id is None


async def test_record_compaction_truncates_summary_to_16k():
    uow = MagicMock()
    uow.compaction = MagicMock()
    uow.compaction.create_or_get = AsyncMock(side_effect=lambda r: r)

    huge = "x" * 30_000
    result = CompactionResult(
        messages=(),
        level_applied=2,
        tokens_before=1, tokens_after=1, summary_injected=True,
        messages_removed=0, usage_ratio_after=0.1,
        summary_text=huge,
        operations=[{"kind": "llm_summary"}],
    )
    await record_compaction(uow=uow, session_id="s", result=result, messages_input_hash="h" * 16)
    rec = uow.compaction.create_or_get.await_args.args[0]
    assert len(rec.summary) == 16_000


async def test_record_compaction_raises_on_empty_operations():
    uow = MagicMock()
    result = CompactionResult(
        messages=(), level_applied=2, tokens_before=1, tokens_after=1,
        summary_injected=False, messages_removed=0, usage_ratio_after=0.1,
        operations=[],
    )
    with pytest.raises(ValueError, match="operations must be non-empty"):
        await record_compaction(uow=uow, session_id="s", result=result, messages_input_hash="h" * 16)


async def test_record_compaction_id_retry_idempotent():
    """[R4-P1-1 + R4-P2-5] Same input → same compaction_id."""
    uow = MagicMock()
    uow.compaction = MagicMock()
    uow.compaction.create_or_get = AsyncMock(side_effect=lambda r: r)

    result = CompactionResult(
        messages=(), level_applied=2, tokens_before=1000, tokens_after=500,
        summary_injected=True, messages_removed=10, usage_ratio_after=0.5,
        summary_text="same summary", operations=[{"kind": "llm_summary"}],
    )

    cid1 = await record_compaction(uow=uow, session_id="s1", result=result, messages_input_hash="aabbccddeeff0011")
    cid2 = await record_compaction(uow=uow, session_id="s1", result=result, messages_input_hash="aabbccddeeff0011")
    assert cid1 == cid2


async def test_record_compaction_id_distinct_compactions_get_distinct_ids():
    """[R4-P1-1 + R4-P2-5] Different input hash → different compaction_id."""
    uow = MagicMock()
    uow.compaction = MagicMock()
    uow.compaction.create_or_get = AsyncMock(side_effect=lambda r: r)

    result = CompactionResult(
        messages=(), level_applied=2, tokens_before=1000, tokens_after=500,
        summary_injected=True, messages_removed=10, usage_ratio_after=0.5,
        summary_text="x", operations=[{"kind": "llm_summary"}],
    )

    cid_a = await record_compaction(uow=uow, session_id="s1", result=result, messages_input_hash="aaaaaaaaaaaaaaaa")
    cid_b = await record_compaction(uow=uow, session_id="s1", result=result, messages_input_hash="bbbbbbbbbbbbbbbb")
    assert cid_a != cid_b


async def test_record_compaction_persists_escalation_rollups():
    """[CXR1-P1-1] Escalation result fed to helper → *_total columns equal the rollup, not the hard step alone."""
    uow = MagicMock()
    uow.compaction = MagicMock()
    uow.compaction.create_or_get = AsyncMock(side_effect=lambda r: r)

    # Synthesize an escalation result whose CompactionResult totals are already rolled up
    # (Task 8 _hard_compact + prior_operations contract guarantees this)
    escalated = CompactionResult(
        messages=(), level_applied=3,
        tokens_before=10_000,    # = first soft op's tokens_before
        tokens_after=2_000,      # = hard op's tokens_after
        summary_injected=True,
        messages_removed=30,     # = soft 25 summarized + hard 5 removed
        usage_ratio_after=0.25,
        summary_text="combined",
        operations=[
            {"kind": "llm_summary", "tokens_before": 10_000, "tokens_after": 4_000, "messages_summarized": 25},
            {"kind": "hard_truncate", "tokens_before": 4_000, "tokens_after": 2_000, "messages_removed": 5, "messages_kept": 19},
        ],
    )

    await record_compaction(uow=uow, session_id="s1", result=escalated, messages_input_hash="h" * 16)
    rec = uow.compaction.create_or_get.await_args.args[0]
    assert rec.tokens_before_total == 10_000
    assert rec.tokens_after_total == 2_000
    assert rec.messages_removed_total == 30
    assert [op["kind"] for op in rec.operations] == ["llm_summary", "hard_truncate"]
