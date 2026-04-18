"""M2-PR3: integration tests — memory sections under global budget pressure.

Verifies design doc §637 invariant: when memory sections collectively
exceed the global ``SystemPromptBudget``, PromptAssembler drops them in
priority-DESC order:

1. ``memory_fact_index`` (priority 5) drops FIRST
2. ``memory_user_profile`` (priority 7) drops SECOND
3. ``memory_rules`` (priority 8) is PROTECTED by
   ``critical_priority_min=8`` and survives under pressure

These are assembler-level assertions — individual section budgets
(``Section.max_tokens``) only apply INSIDE each section's render.
Cross-section budget pressure is handled by the assembler's priority-DESC
drop loop in ``assembler.py:140-151``.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.domain.models.memory_chunk import MemoryChunk
from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts.assembler import PromptAssembler
from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.bundles.zh import ZH_BUNDLE
from app.domain.services.prompts.memory_snapshot import MemorySnapshot
from app.domain.services.prompts.section import PromptMode, RenderContext


# ---- Fixtures ---------------------------------------------------------- #


def _chunk(
    content: str,
    *,
    category: str | None = None,
    pinned: bool = False,
    chunk_id: str | None = None,
) -> MemoryChunk:
    now = datetime.now(timezone.utc)
    return MemoryChunk(
        id=chunk_id or str(uuid.uuid4()),
        user_id="u1",
        content=content,
        content_hash="h",
        source="session_flush",
        metadata={},
        created_at=now,
        updated_at=now,
        session_id=None,
        embedding=None,
        category=category,
        auto_promoted_at=None,
        fs_synced=True,
        pinned=pinned,
    )


def _make_full_snapshot(
    *, bullet_chars: int = 200
) -> MemorySnapshot:
    """Build a snapshot where each category is populated with bulky chunks
    so that all three memory sections produce non-trivial output.

    ``bullet_chars`` controls per-chunk content size (CJK). Default 200
    means each bullet ~300 tokens; multiplied by snapshot caps
    (10 user + 20 rule + 50 fact) this produces a prompt fragment well
    past any sane budget."""
    big_content = "长" * bullet_chars
    return MemorySnapshot(
        user_chunks=tuple(
            _chunk(f"{big_content}-user-{i}", category="user")
            for i in range(10)
        ),
        rule_chunks=tuple(
            _chunk(f"{big_content}-rule-{i}", category="rule")
            for i in range(20)
        ),
        fact_chunks=tuple(
            _chunk(
                f"{big_content}-fact-{i}",
                category="fact",
                chunk_id=f"f{i}",
            )
            for i in range(50)
        ),
    )


def _make_assembler(*, max_tokens: int) -> PromptAssembler:
    return PromptAssembler(
        budget=SystemPromptBudget(max_tokens=max_tokens, critical_priority_min=8),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )


def _assemble(
    *, snapshot: MemorySnapshot, max_tokens: int
) -> tuple[str, list[str], list[str]]:
    ctx = RenderContext(lang="zh", memory_snapshot=snapshot)
    result = _make_assembler(max_tokens=max_tokens).assemble(
        ZH_BUNDLE.executor, ctx, PromptMode.FULL
    )
    return result.text, result.sections_included, result.sections_dropped


# ---- Tests ------------------------------------------------------------- #


class TestMemorySectionsPriorityDropOrder:
    def test_generous_budget_includes_all_memory_sections(self) -> None:
        """Baseline: with ample budget, all three memory sections emit."""
        snapshot = _make_full_snapshot(bullet_chars=20)
        _, included, dropped = _assemble(snapshot=snapshot, max_tokens=100_000)
        assert "memory_rules" in included
        assert "memory_user_profile" in included
        assert "memory_fact_index" in included
        for mid in ("memory_rules", "memory_user_profile", "memory_fact_index"):
            assert mid not in dropped

    def test_fact_index_drops_first_under_moderate_pressure(self) -> None:
        """With a budget that fits rules + user profile but not the fact
        index, fact_index (priority 5) must be the sole memory section
        dropped. Rules (priority 8) and user profile (priority 7) survive.

        Size calibration on the default bundle with this snapshot:
        - identity: ~302 tokens
        - behavior_core: ~1892 tokens
        - output_format: ~233 tokens
        - memory_rules: ~2468 tokens (internal cap 2500)
        - memory_user_profile: ~1288 tokens (under internal cap 1500)
        - memory_fact_index: ~1022 tokens (about at internal cap 1000)

        Sum of critical (prio >= 8) = 4895. Adding user (1288) = 6183.
        Adding fact (1022) = 7205. A 6500-token budget fits everything
        except fact (6183 < 6500 < 7205) — fact is the lone drop.
        """
        snapshot = _make_full_snapshot(bullet_chars=200)
        _, included, dropped = _assemble(snapshot=snapshot, max_tokens=6500)
        assert "memory_rules" in included
        assert "memory_user_profile" in included
        assert "memory_fact_index" in dropped

    def test_user_profile_drops_before_rules_under_heavy_pressure(self) -> None:
        """Tighter budget: fact_index AND user profile drop; rules survive
        because priority=8 >= critical_priority_min=8."""
        snapshot = _make_full_snapshot(bullet_chars=200)
        # With a 4000-token budget, rules alone ~2500 + fixed ~500 =
        # 3000. Adding user profile's 1500 would push to 4500 > budget.
        # So user profile drops. fact_index already dropped.
        _, included, dropped = _assemble(snapshot=snapshot, max_tokens=4000)
        assert "memory_rules" in included, (
            f"memory_rules (priority 8) should be protected; "
            f"got dropped={dropped}"
        )
        assert "memory_user_profile" in dropped
        assert "memory_fact_index" in dropped

    def test_rules_survive_even_at_critical_priority_pressure(self) -> None:
        """critical_priority_min=8 makes rules survive even when the budget
        alone could drop it. This is the hard-constraint guarantee: the
        agent must see project rules, full stop."""
        snapshot = _make_full_snapshot(bullet_chars=200)
        # Very tight budget — so tight that every non-critical section
        # should be dropped. Rules (prio 8) and identity/behavior_core
        # (prio 10) must still emit.
        _, included, dropped = _assemble(snapshot=snapshot, max_tokens=3500)
        assert "identity" in included
        assert "behavior_core" in included
        assert "memory_rules" in included
        assert "memory_user_profile" in dropped
        assert "memory_fact_index" in dropped


class TestAssembledOutputOrdering:
    """Declaration-order test: regardless of which sections drop under
    budget, the kept sections appear in registry-declaration order (NOT
    priority order). See assembler.py step 3 comment."""

    def test_memory_sections_render_in_declaration_order_when_kept(self) -> None:
        snapshot = _make_full_snapshot(bullet_chars=20)
        text, included, _ = _assemble(snapshot=snapshot, max_tokens=100_000)
        # Registry declaration order in ZH bundle:
        #   ... tools_guide_dynamic, memory_rules, memory_user_profile,
        #   skill_context, conversation_summaries, memory_fact_index, ...
        rules_idx = text.index("## 项目规则")
        profile_idx = text.index("## 用户画像")
        fact_idx = text.index("## 事实索引")
        assert rules_idx < profile_idx < fact_idx

    def test_no_memory_section_emitted_without_snapshot(self) -> None:
        """Sanity: with ``memory_snapshot=None``, none of the 3 memory
        sections emit text. This is what non-memory call sites observe
        and ensures M2-PR3 doesn't regress the legacy executor prompt."""
        ctx = RenderContext(lang="zh", memory_snapshot=None)
        result = _make_assembler(max_tokens=100_000).assemble(
            ZH_BUNDLE.executor, ctx, PromptMode.FULL
        )
        for mid in ("memory_rules", "memory_user_profile", "memory_fact_index"):
            assert mid not in result.sections_included
            assert mid not in result.sections_dropped
