"""Tests for memory_ranker pure functions."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.domain.models.memory_chunk import MemoryChunk

from tests.conftest import TEST_USER_ID_FIXED


def _make_chunk(
    embedding: tuple[float, ...] | None = None,
    **overrides,
) -> MemoryChunk:
    defaults = dict(
        id="chunk-1",
        user_id=TEST_USER_ID_FIXED,
        content="test content",
        content_hash="hash1",
        source="session_flush",
        metadata={},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        embedding=embedding,
    )
    defaults.update(overrides)
    return MemoryChunk(**defaults)


def _norm(v: list[float]) -> tuple[float, ...]:
    """L2 normalize a vector, return as tuple."""
    a = np.array(v, dtype=np.float64)
    return tuple((a / np.linalg.norm(a)).tolist())


class TestComputeRelevance:
    def test_ordering_by_similarity(self) -> None:
        """Higher cosine similarity chunks should rank first."""
        from app.domain.services.memory_ranker import compute_relevance

        query = [1.0, 0.0, 0.0]
        close = _make_chunk(id="close", embedding=_norm([0.9, 0.1, 0.0]))
        far = _make_chunk(id="far", embedding=_norm([0.1, 0.9, 0.0]))

        result = compute_relevance([far, close], query)

        assert len(result) == 2
        assert result[0][0].id == "close"
        assert result[1][0].id == "far"
        assert result[0][1] > result[1][1]

    def test_skips_none_embedding(self) -> None:
        """Chunks with embedding=None should be excluded."""
        from app.domain.services.memory_ranker import compute_relevance

        query = [1.0, 0.0]
        valid = _make_chunk(id="valid", embedding=_norm([1.0, 0.0]))
        no_emb = _make_chunk(id="no_emb", embedding=None)

        result = compute_relevance([valid, no_emb], query)

        assert len(result) == 1
        assert result[0][0].id == "valid"

    def test_zero_vector_gets_zero_similarity(self) -> None:
        """A zero embedding should get similarity 0.0, not crash."""
        from app.domain.services.memory_ranker import compute_relevance

        query = [1.0, 0.0]
        zero = _make_chunk(id="zero", embedding=(0.0, 0.0))

        result = compute_relevance([zero], query)

        assert len(result) == 1
        assert result[0][1] == 0.0


class TestApplyTemporalDecay:
    def test_basic_half_life(self) -> None:
        """30 days old chunk with half_life=30 → score ≈ relevance * 0.5."""
        from app.domain.services.memory_ranker import apply_temporal_decay

        now = datetime(2026, 2, 1, tzinfo=timezone.utc)
        chunk = _make_chunk(
            created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            embedding=_norm([1.0, 0.0]),
        )
        scored = [(chunk, 0.8)]

        result = apply_temporal_decay(scored, half_life_days=30, now=now)

        assert len(result) == 1
        assert result[0][1] == pytest.approx(0.8 * 0.5, abs=0.01)

    def test_evergreen_immune(self) -> None:
        """Chunks with metadata evergreen=True should not decay."""
        from app.domain.services.memory_ranker import apply_temporal_decay

        now = datetime(2026, 4, 1, tzinfo=timezone.utc)
        old = _make_chunk(
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            metadata={"evergreen": True},
            embedding=_norm([1.0, 0.0]),
        )
        scored = [(old, 0.9)]

        result = apply_temporal_decay(scored, half_life_days=30, now=now)

        assert result[0][1] == 0.9  # unchanged

    def test_zero_age_no_decay(self) -> None:
        """Just-created chunk should have score ≈ original relevance."""
        from app.domain.services.memory_ranker import apply_temporal_decay

        now = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        chunk = _make_chunk(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            embedding=_norm([1.0, 0.0]),
        )
        scored = [(chunk, 0.95)]

        result = apply_temporal_decay(scored, half_life_days=30, now=now)

        assert result[0][1] == pytest.approx(0.95, abs=0.001)

    def test_result_sorted_by_decayed_score(self) -> None:
        """After decay, results should be re-sorted by final score."""
        from app.domain.services.memory_ranker import apply_temporal_decay

        now = datetime(2026, 4, 1, tzinfo=timezone.utc)
        old_high = _make_chunk(
            id="old_high",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            embedding=_norm([1.0, 0.0]),
        )
        new_low = _make_chunk(
            id="new_low",
            created_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
            embedding=_norm([0.5, 0.5]),
        )
        scored = [(old_high, 0.95), (new_low, 0.5)]

        result = apply_temporal_decay(scored, half_life_days=30, now=now)

        # new_low should rank higher: 0.5 * ~1.0 > 0.95 * ~0.0 (450+ days old)
        assert result[0][0].id == "new_low"


class TestApplyMmr:
    def test_selects_diverse(self) -> None:
        """Given 2 similar + 1 different, MMR should select the different one."""
        from app.domain.services.memory_ranker import apply_mmr

        similar_a = _make_chunk(id="sim_a", embedding=_norm([1.0, 0.0, 0.0]))
        similar_b = _make_chunk(id="sim_b", embedding=_norm([0.99, 0.01, 0.0]))
        different = _make_chunk(id="diff", embedding=_norm([0.0, 1.0, 0.0]))

        scored = [(similar_a, 0.95), (similar_b, 0.90), (different, 0.70)]

        result = apply_mmr(scored, lambda_=0.5, top_k=2)

        ids = [c.id for c in result]
        assert ids[0] == "sim_a"
        assert "diff" in ids

    def test_lambda_1_pure_relevance(self) -> None:
        """lambda=1.0 → pure relevance ordering, no diversity penalty."""
        from app.domain.services.memory_ranker import apply_mmr

        a = _make_chunk(id="a", embedding=_norm([1.0, 0.0]))
        b = _make_chunk(id="b", embedding=_norm([0.99, 0.01]))
        c = _make_chunk(id="c", embedding=_norm([0.0, 1.0]))

        scored = [(a, 0.9), (b, 0.8), (c, 0.7)]

        result = apply_mmr(scored, lambda_=1.0, top_k=3)

        assert [r.id for r in result] == ["a", "b", "c"]

    def test_respects_top_k(self) -> None:
        """Should return at most top_k results."""
        from app.domain.services.memory_ranker import apply_mmr

        chunks = [
            _make_chunk(id=f"c{i}", embedding=_norm([float(i), 1.0]))
            for i in range(5)
        ]
        scored = [(c, 0.9 - i * 0.1) for i, c in enumerate(chunks)]

        result = apply_mmr(scored, lambda_=0.7, top_k=2)

        assert len(result) == 2

    def test_fewer_candidates_than_top_k(self) -> None:
        """If fewer candidates than top_k, return all."""
        from app.domain.services.memory_ranker import apply_mmr

        a = _make_chunk(id="a", embedding=_norm([1.0, 0.0]))
        scored = [(a, 0.9)]

        result = apply_mmr(scored, lambda_=0.7, top_k=5)

        assert len(result) == 1
        assert result[0].id == "a"


class TestRankMemoryResults:
    def test_integration_relevance_decay_mmr(self) -> None:
        """End-to-end: relevance + decay + MMR combined."""
        from app.domain.services.memory_ranker import rank_memory_results

        now = datetime(2026, 4, 1, tzinfo=timezone.utc)
        query = [1.0, 0.0, 0.0]

        recent = _make_chunk(
            id="recent",
            created_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
            embedding=_norm([0.9, 0.1, 0.0]),
        )
        old = _make_chunk(
            id="old",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            embedding=_norm([0.95, 0.05, 0.0]),
        )
        diverse = _make_chunk(
            id="diverse",
            created_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
            embedding=_norm([0.0, 0.0, 1.0]),
        )

        result = rank_memory_results(
            [old, recent, diverse],
            query_embedding=query,
            half_life_days=30,
            mmr_lambda=0.5,
            top_k=2,
            now=now,
        )

        assert len(result) == 2
        ids = [c.id for c in result]
        assert ids[0] == "recent"

    def test_empty_input(self) -> None:
        from app.domain.services.memory_ranker import rank_memory_results

        result = rank_memory_results([], [1.0, 0.0], 30, 0.7, 5)
        assert result == []

    def test_single_chunk(self) -> None:
        from app.domain.services.memory_ranker import rank_memory_results

        chunk = _make_chunk(embedding=_norm([1.0, 0.0]))
        result = rank_memory_results([chunk], [1.0, 0.0], 30, 0.7, 5)
        assert len(result) == 1
        assert result[0].id == chunk.id
