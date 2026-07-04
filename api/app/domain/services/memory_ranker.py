"""Pure-function memory ranking: relevance scoring, temporal decay, MMR.

No I/O, no state. All functions operate on in-memory data.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.domain.models.memory_chunk import MemoryChunk


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with explicit L2 normalization.

    Does NOT assume inputs are pre-normalized — the write path
    (memory_flush_service._embed_batch) stores raw provider vectors.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a / norm_a, b / norm_b))


def compute_relevance(
    chunks: list[MemoryChunk],
    query_embedding: list[float],
) -> list[tuple[MemoryChunk, float]]:
    """Compute cosine similarity between each chunk and the query.

    Chunks with embedding=None are skipped.
    Returns (chunk, similarity) pairs sorted by similarity descending.
    """
    query_vec = np.array(query_embedding, dtype=np.float64)
    scored: list[tuple[MemoryChunk, float]] = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        chunk_vec = np.array(chunk.embedding, dtype=np.float64)
        sim = _cosine_similarity(query_vec, chunk_vec)
        scored.append((chunk, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def apply_temporal_decay(
    scored_chunks: list[tuple[MemoryChunk, float]],
    half_life_days: int,
    now: datetime | None = None,
) -> list[tuple[MemoryChunk, float]]:
    """Multiply each score by a time-decay factor.

    Formula: final_score = relevance * 0.5 ** (days / half_life_days)
    Chunks with metadata["evergreen"] == True are immune to decay.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    result: list[tuple[MemoryChunk, float]] = []
    for chunk, relevance in scored_chunks:
        if chunk.metadata.get("evergreen", False):
            result.append((chunk, relevance))
            continue
        days = (now - chunk.created_at).total_seconds() / 86400.0
        decay = 0.5 ** (days / half_life_days)
        result.append((chunk, relevance * decay))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def apply_mmr(
    scored_chunks: list[tuple[MemoryChunk, float]],
    lambda_: float,
    top_k: int,
) -> list[MemoryChunk]:
    """MMR diversity reranking using cosine similarity on embeddings.

    mmr_score = λ * norm_relevance - (1-λ) * max_cosine_sim_to_selected

    Requires all chunks to have non-None embedding.
    Chunks with None embedding are skipped.
    lambda_=1.0 degenerates to pure relevance ordering.
    """
    if not scored_chunks:
        return []

    # Filter out None embeddings (defensive)
    valid = [
        (chunk, score)
        for chunk, score in scored_chunks
        if chunk.embedding is not None
    ]
    if not valid:
        return []

    # Precompute normalized embedding vectors
    emb_map: dict[str, np.ndarray] = {}
    for chunk, _ in valid:
        vec = np.array(chunk.embedding, dtype=np.float64)
        norm = np.linalg.norm(vec)
        emb_map[chunk.id] = vec / norm if norm > 0 else vec

    # Normalize relevance scores to [0, 1]
    scores = [s for _, s in valid]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score
    if score_range > 0:
        norm_scores = {
            chunk.id: (score - min_score) / score_range
            for chunk, score in valid
        }
    else:
        norm_scores = {chunk.id: 1.0 for chunk, _ in valid}

    # Original scores for tie-breaking
    orig_scores = {chunk.id: score for chunk, score in valid}

    candidates = {chunk.id: chunk for chunk, _ in valid}
    selected: list[MemoryChunk] = []
    selected_embs: list[np.ndarray] = []

    for _ in range(min(top_k, len(valid))):
        best_id: str | None = None
        best_mmr = float("-inf")
        best_orig = float("-inf")

        for cid in candidates:
            relevance = norm_scores[cid]
            if selected_embs:
                max_sim = max(
                    float(np.dot(emb_map[cid], sel_emb))
                    for sel_emb in selected_embs
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_ * relevance - (1.0 - lambda_) * max_sim

            if (mmr_score > best_mmr) or (
                mmr_score == best_mmr and orig_scores[cid] > best_orig
            ):
                best_mmr = mmr_score
                best_orig = orig_scores[cid]
                best_id = cid

        if best_id is None:
            break

        selected.append(candidates.pop(best_id))
        selected_embs.append(emb_map[best_id])

    return selected


def rank_memory_results_with_scores(
    chunks: list[MemoryChunk],
    query_embedding: list[float],
    half_life_days: int,
    mmr_lambda: float,
    top_k: int,
    now: datetime | None = None,
) -> list[tuple[MemoryChunk, float]]:
    """B8: composite pipeline 的保分变体（单一实现双出口）。

    与 ``rank_memory_results`` 完全同管线（relevance → temporal decay →
    MMR → top_k）；附带的 score = **decayed relevance**（relevance ×
    time decay）。MMR 只决定选择与顺序，不改报告的分值。id→score 回填
    安全：``chunk.id`` 是 UUID 主键，无碰撞。
    """
    if not chunks:
        return []
    scored = compute_relevance(chunks, query_embedding)
    if not scored:
        return []
    decayed = apply_temporal_decay(scored, half_life_days, now=now)
    selected = apply_mmr(decayed, mmr_lambda, top_k)
    score_by_id = {chunk.id: score for chunk, score in decayed}
    return [(chunk, score_by_id[chunk.id]) for chunk in selected]


def rank_memory_results(
    chunks: list[MemoryChunk],
    query_embedding: list[float],
    half_life_days: int,
    mmr_lambda: float,
    top_k: int,
    now: datetime | None = None,
) -> list[MemoryChunk]:
    """Composite entry point: relevance → temporal decay → MMR → top_k.

    B8 起是 ``rank_memory_results_with_scores`` 的 thin wrapper（单一
    实现两个出口，行为字节等价——parity 测试锁定；spec R1#1+R3）。
    """
    return [
        chunk
        for chunk, _ in rank_memory_results_with_scores(
            chunks, query_embedding, half_life_days, mmr_lambda, top_k, now=now,
        )
    ]
