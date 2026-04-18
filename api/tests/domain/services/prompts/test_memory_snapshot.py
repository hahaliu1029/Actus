"""M2-PR3: unit tests for MemorySnapshot and build_memory_snapshot.

Isolated from infrastructure: uses a ``FakeMemoryChunkRepository`` that
records per-call arguments so tests can assert the build_memory_snapshot
contract (three category-specific calls, correct caps, user overfetch)
without a live database.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.domain.models.memory_chunk import MemoryChunk
from app.domain.services.prompts.memory_snapshot import (
    FACT_CHUNK_CAP,
    MemorySnapshot,
    RULE_CHUNK_CAP,
    USER_CHUNK_CAP,
    build_memory_snapshot,
)

pytestmark = pytest.mark.anyio


# ---- Fixtures ---------------------------------------------------------- #


def _chunk(
    content: str,
    *,
    category: str | None = None,
    pinned: bool = False,
    updated_at: datetime | None = None,
    chunk_id: str | None = None,
) -> MemoryChunk:
    now = updated_at or datetime.now(timezone.utc)
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


@dataclass
class _Call:
    user_id: str
    category: str | None
    pinned: bool | None
    limit: int
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeRepo:
    """Records every ``list_by_user`` invocation and applies the
    ``category`` + ``pinned`` filters on the stored rows so tests can
    verify two-phase pinned/unpinned fetch behavior.

    ``store`` is keyed by category; each stored list is expected to be
    in the repo's canonical order (``updated_at DESC``) because
    ``build_memory_snapshot`` trusts that ordering from the real repo.
    """

    store: dict[str, list[MemoryChunk]] = field(default_factory=dict)
    calls: list[_Call] = field(default_factory=list)

    async def list_by_user(
        self,
        user_id: str,
        *,
        category: str | None = None,
        pinned: bool | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[MemoryChunk]:
        self.calls.append(
            _Call(
                user_id=user_id,
                category=category,
                pinned=pinned,
                limit=limit,
                kwargs=kwargs,
            )
        )
        rows = self.store.get(category or "__all__", [])
        if pinned is not None:
            rows = [r for r in rows if r.pinned is pinned]
        return list(rows[:limit])


# ---- MemorySnapshot data-class basics ---------------------------------- #


class TestMemorySnapshotBasics:
    def test_default_is_empty(self) -> None:
        s = MemorySnapshot()
        assert s.user_chunks == ()
        assert s.rule_chunks == ()
        assert s.fact_chunks == ()
        assert s.is_empty() is True

    def test_empty_classmethod_returns_empty_snapshot(self) -> None:
        assert MemorySnapshot.empty().is_empty() is True

    def test_is_empty_false_when_any_bucket_non_empty(self) -> None:
        rule = _chunk("r1", category="rule")
        s = MemorySnapshot(rule_chunks=(rule,))
        assert s.is_empty() is False

    def test_is_frozen(self) -> None:
        """Snapshot must be immutable — sections rely on this to treat
        tuples as safe to share across the render loop."""
        s = MemorySnapshot()
        with pytest.raises(Exception):  # FrozenInstanceError / AttributeError
            s.user_chunks = ()  # type: ignore[misc]

    def test_tuples_are_actually_tuple_type(self) -> None:
        """A list default would leak mutability even behind a frozen
        wrapper. Enforce ``tuple`` at the type level."""
        s = MemorySnapshot(
            user_chunks=(_chunk("u"),),
            rule_chunks=(_chunk("r"),),
            fact_chunks=(_chunk("f"),),
        )
        assert isinstance(s.user_chunks, tuple)
        assert isinstance(s.rule_chunks, tuple)
        assert isinstance(s.fact_chunks, tuple)


# ---- build_memory_snapshot: per-category repo call contract ----------- #


class TestBuildMemorySnapshotCalls:
    async def test_pinned_user_fetched_first(self) -> None:
        """Post-codex-review: user bucket is two-phase. First call MUST
        be the pinned fetch so an old pinned chunk can't be pre-empted
        by a recency-windowed single fetch."""
        repo = FakeRepo(store={"user": [], "rule": [], "fact": []})
        await build_memory_snapshot(repo, "u1")
        assert repo.calls[0].category == "user"
        assert repo.calls[0].pinned is True
        assert repo.calls[0].limit == USER_CHUNK_CAP

    async def test_unpinned_user_fetch_limit_shrinks_by_pinned_count(
        self,
    ) -> None:
        """When N pinned rows already fill N of the USER_CHUNK_CAP slots,
        the unpinned fetch should only ask for ``USER_CHUNK_CAP - N``
        more rows. Prevents over-fetching and keeps the merged result
        bounded at ``USER_CHUNK_CAP``."""
        now = datetime.now(timezone.utc)
        pinned = [
            _chunk(f"p{i}", category="user", pinned=True, updated_at=now)
            for i in range(3)
        ]
        repo = FakeRepo(store={"user": pinned, "rule": [], "fact": []})
        await build_memory_snapshot(repo, "u1")
        # calls[0] is pinned fetch; calls[1] is unpinned fetch (if any)
        pinned_call = next(
            c for c in repo.calls if c.category == "user" and c.pinned is True
        )
        unpinned_call = next(
            c for c in repo.calls if c.category == "user" and c.pinned is False
        )
        assert pinned_call.limit == USER_CHUNK_CAP
        assert unpinned_call.limit == USER_CHUNK_CAP - 3

    async def test_unpinned_fetch_skipped_when_pinned_fills_cap(self) -> None:
        """If pinned already fills USER_CHUNK_CAP, we must skip the
        unpinned fetch entirely (it would produce nothing usable)."""
        now = datetime.now(timezone.utc)
        pinned = [
            _chunk(f"p{i}", category="user", pinned=True, updated_at=now)
            for i in range(USER_CHUNK_CAP)
        ]
        repo = FakeRepo(store={"user": pinned, "rule": [], "fact": []})
        await build_memory_snapshot(repo, "u1")
        unpinned_calls = [
            c for c in repo.calls if c.category == "user" and c.pinned is False
        ]
        assert unpinned_calls == []

    async def test_rule_and_fact_use_plain_caps(self) -> None:
        repo = FakeRepo(store={"user": [], "rule": [], "fact": []})
        await build_memory_snapshot(repo, "u1")
        rule_call = next(c for c in repo.calls if c.category == "rule")
        fact_call = next(c for c in repo.calls if c.category == "fact")
        assert rule_call.limit == RULE_CHUNK_CAP
        assert fact_call.limit == FACT_CHUNK_CAP
        # Rule and fact queries do NOT filter by pinned (DB CHECK keeps
        # non-user rows unpinned anyway; an extra filter would just be noise).
        assert rule_call.pinned is None
        assert fact_call.pinned is None

    async def test_user_id_propagates_to_all_calls(self) -> None:
        repo = FakeRepo(store={"user": [], "rule": [], "fact": []})
        await build_memory_snapshot(repo, "user-abc")
        assert all(c.user_id == "user-abc" for c in repo.calls)


# ---- build_memory_snapshot: user pinned sort ---------------------------- #


class TestBuildMemorySnapshotPinnedSort:
    async def test_pinned_block_comes_before_unpinned_in_result(self) -> None:
        """Pinned chunks must surface at the front of ``user_chunks``.
        Single old pinned + single new unpinned; the pinned row sits
        first in the final snapshot regardless of recency."""
        now = datetime.now(timezone.utc)
        old_pinned = _chunk(
            "pinned-old",
            category="user",
            pinned=True,
            updated_at=now - timedelta(days=30),
        )
        new_unpinned = _chunk(
            "unpinned-new",
            category="user",
            pinned=False,
            updated_at=now,
        )
        # FakeRepo applies the pinned filter per call — store can be in
        # any order; both sub-fetches will pull the right rows.
        repo = FakeRepo(
            store={"user": [new_unpinned, old_pinned], "rule": [], "fact": []}
        )
        s = await build_memory_snapshot(repo, "u1")
        assert s.user_chunks[0].content == "pinned-old"
        assert s.user_chunks[1].content == "unpinned-new"

    async def test_old_pinned_survives_when_unpinned_saturate_recency_window(
        self,
    ) -> None:
        """P2 regression (codex post-PR-3 review): if we had used a single
        ``list_by_user`` fetch with a recency window, an old pinned chunk
        sitting beyond the window would be silently dropped. The
        two-phase fetch guarantees pinned is retrieved independently of
        recency.

        Construct a worst case: 1 VERY OLD pinned + 40 newer unpinned.
        A ``limit=10`` single fetch would see only the 10 most-recent
        (all unpinned) and miss the pinned entirely. Two-phase fetch
        returns the pinned in phase 1 regardless of age.
        """
        now = datetime.now(timezone.utc)
        ancient_pinned = _chunk(
            "ancient-pin",
            category="user",
            pinned=True,
            updated_at=now - timedelta(days=365),
        )
        recent_unpinned = [
            _chunk(
                f"U{i}",
                category="user",
                pinned=False,
                updated_at=now - timedelta(minutes=i),
            )
            for i in range(40)
        ]
        # Store order: recent unpinned first (updated_at DESC), then
        # ancient pinned at the tail — the exact layout that would
        # starve a single-fetch design.
        repo = FakeRepo(
            store={
                "user": recent_unpinned + [ancient_pinned],
                "rule": [],
                "fact": [],
            }
        )
        s = await build_memory_snapshot(repo, "u1")
        assert s.user_chunks[0].content == "ancient-pin", (
            "ancient pinned chunk must surface despite being ~365 days "
            "older than the 10 most-recent unpinned — this is the codex P2 "
            "regression."
        )
        # Cap holds
        assert len(s.user_chunks) == USER_CHUNK_CAP

    async def test_pinned_and_unpinned_counts_respect_cap(self) -> None:
        """If pinned fills fewer than USER_CHUNK_CAP slots, unpinned
        completes the bucket exactly up to the cap."""
        now = datetime.now(timezone.utc)
        pinned = [
            _chunk(
                f"P{i}",
                category="user",
                pinned=True,
                updated_at=now - timedelta(days=i),
            )
            for i in range(3)
        ]
        unpinned = [
            _chunk(
                f"U{i}",
                category="user",
                pinned=False,
                updated_at=now - timedelta(minutes=i),
            )
            for i in range(15)
        ]
        repo = FakeRepo(
            store={"user": unpinned + pinned, "rule": [], "fact": []}
        )
        s = await build_memory_snapshot(repo, "u1")
        assert len(s.user_chunks) == USER_CHUNK_CAP
        pinned_in_result = [c for c in s.user_chunks if c.pinned]
        unpinned_in_result = [c for c in s.user_chunks if not c.pinned]
        assert len(pinned_in_result) == 3
        assert len(unpinned_in_result) == USER_CHUNK_CAP - 3
        # Pinned block stays at the head
        assert all(c.pinned for c in s.user_chunks[:3])
        assert not any(c.pinned for c in s.user_chunks[3:])

    async def test_truncates_user_to_user_chunk_cap_when_all_unpinned(
        self,
    ) -> None:
        """No pinned → first phase returns []; second phase's limit is
        USER_CHUNK_CAP, result is exactly that."""
        now = datetime.now(timezone.utc)
        many = [
            _chunk(
                f"u{i}",
                category="user",
                pinned=False,
                updated_at=now - timedelta(minutes=i),
            )
            for i in range(USER_CHUNK_CAP * 3)
        ]
        repo = FakeRepo(store={"user": many, "rule": [], "fact": []})
        s = await build_memory_snapshot(repo, "u1")
        assert len(s.user_chunks) == USER_CHUNK_CAP
        # Most-recent unpinned wins
        assert [c.content for c in s.user_chunks] == [
            f"u{i}" for i in range(USER_CHUNK_CAP)
        ]

    async def test_pinned_count_exceeding_cap_truncates_within_pinned(
        self,
    ) -> None:
        """User accidentally pins more than USER_CHUNK_CAP entries.
        The repo limit on the pinned phase enforces the cap — only
        USER_CHUNK_CAP pinned rows come back, unpinned phase skipped."""
        now = datetime.now(timezone.utc)
        overflow_pinned = [
            _chunk(
                f"P{i}",
                category="user",
                pinned=True,
                updated_at=now - timedelta(minutes=i),  # newer → first
            )
            for i in range(USER_CHUNK_CAP + 5)
        ]
        repo = FakeRepo(
            store={"user": overflow_pinned, "rule": [], "fact": []}
        )
        s = await build_memory_snapshot(repo, "u1")
        assert len(s.user_chunks) == USER_CHUNK_CAP
        assert all(c.pinned for c in s.user_chunks)


# ---- build_memory_snapshot: rule/fact straight passthrough ------------- #


class TestBuildMemorySnapshotPassthrough:
    async def test_rule_order_preserved_from_repo(self) -> None:
        """Rule bucket does not re-sort — the repo's ``updated_at DESC``
        is canonical. Verify by passing out-of-order and checking the
        snapshot preserves repo's return order."""
        # Repo-emulated order (already updated_at DESC — we don't verify
        # that here, we just check the snapshot doesn't re-shuffle).
        rules = [_chunk(f"r{i}", category="rule") for i in range(3)]
        repo = FakeRepo(store={"user": [], "rule": rules, "fact": []})
        s = await build_memory_snapshot(repo, "u1")
        assert [c.content for c in s.rule_chunks] == ["r0", "r1", "r2"]

    async def test_fact_order_preserved_from_repo(self) -> None:
        facts = [_chunk(f"f{i}", category="fact") for i in range(3)]
        repo = FakeRepo(store={"user": [], "rule": [], "fact": facts})
        s = await build_memory_snapshot(repo, "u1")
        assert [c.content for c in s.fact_chunks] == ["f0", "f1", "f2"]

    async def test_empty_repo_produces_empty_snapshot(self) -> None:
        repo = FakeRepo(store={"user": [], "rule": [], "fact": []})
        s = await build_memory_snapshot(repo, "u1")
        assert s.is_empty()
