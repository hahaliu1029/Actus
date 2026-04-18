"""M2-PR3: MemorySnapshot bundles memory chunks by category for prompt injection.

Built upstream (next PR will wire this into ``build_render_context`` /
``agent_task_runner._build_runtime_system_context``) and attached to
``RenderContext.memory_snapshot``. The three memory sections
(``memory_rules``, ``memory_user_profile``, ``memory_fact_index``) render
from this snapshot at prompt-assembly time.

**Why a pre-built snapshot**: sections are pure sync functions of
``RenderContext``. Fetching from ``MemoryChunkRepository`` is async and
uses an infrastructure dependency; doing that at render time would break
both properties. The snapshot encapsulates the async fetch so sections
stay pure and the registry startup-validation loop (which renders every
section against ``_FIXTURE_CTX``) stays sync.

Sorting rules baked in at construction so sections don't duplicate them:
- ``user_chunks``: pinned block first (by ``updated_at DESC`` internally),
  then unpinned block (``updated_at DESC``). Implemented via two repo
  calls with ``pinned`` filter — see ``build_memory_snapshot`` docstring
  for why a single ``list_by_user`` with client-side re-sort cannot
  guarantee "pinned always surfaces" (post-PR-3 codex review fix).
- ``rule_chunks``: updated_at DESC (matches repo default)
- ``fact_chunks``: updated_at DESC (matches repo default)

**Category chunk caps** (hard count caps, applied BEFORE section
internal token truncation, per design doc §596-598):
- user: 10
- rule: 20
- fact: 50

Caps bound snapshot size regardless of how many memories the user has —
a user with 5000 fact chunks still only gets the 50 most recent into
prompt, and ``memory_fact_index`` further truncates to fit its
1000-token budget. Cap + section budget together give a predictable
upper bound on prompt contribution per category.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.models.memory_chunk import MemoryChunk
    from app.domain.repositories.memory_chunk_repository import (
        MemoryChunkRepository,
    )


USER_CHUNK_CAP = 10
RULE_CHUNK_CAP = 20
FACT_CHUNK_CAP = 50


@dataclass(frozen=True)
class MemorySnapshot:
    """Per-render snapshot of a user's memories, bucketed by category.

    All tuples are empty by default — sections render nothing when their
    bucket is empty, which matches the "snapshot is absent" fallback.
    Uses ``tuple`` for immutability so a buggy section cannot mutate
    shared state across the render loop (``RenderContext`` is frozen).
    """

    user_chunks: tuple["MemoryChunk", ...] = ()
    rule_chunks: tuple["MemoryChunk", ...] = ()
    fact_chunks: tuple["MemoryChunk", ...] = ()

    @classmethod
    def empty(cls) -> "MemorySnapshot":
        """Sentinel empty snapshot. Used by ``_FIXTURE_CTX`` and tests."""
        return cls()

    def is_empty(self) -> bool:
        return not (self.user_chunks or self.rule_chunks or self.fact_chunks)


async def build_memory_snapshot(
    repo: "MemoryChunkRepository",
    user_id: str,
) -> MemorySnapshot:
    """Fetch category-bucketed chunks for the user and assemble a snapshot.

    **User bucket — two-phase fetch for the "pinned always surfaces"
    invariant.** A single ``list_by_user`` with only ``category='user'``
    orders by ``updated_at DESC`` and caps at 10; a pinned chunk older
    than the 10 most-recent user-category rows would be silently
    missed. With a per-user daily quota of 500 writes, the window of
    "most recent 10" can span just hours — clearly small enough to
    exclude an old pinned memory in production. We fix it by fetching
    ALL pinned user chunks first (bounded by USER_CHUNK_CAP and backed
    by the ``ix_memory_chunks_user_pinned`` partial index, so this is
    independent of recency), then fill the remaining slots with the
    top-N unpinned user rows. Both sub-queries stay under their own
    cap, so the total result is always <= USER_CHUNK_CAP.

    Rule and fact buckets are plain single-category fetches — no
    pinned semantics there (DB CHECK: ``pinned=true`` requires
    ``category='user'``), and the repo's ``updated_at DESC`` is the
    canonical order.
    """
    pinned_user = await repo.list_by_user(
        user_id, category="user", pinned=True, limit=USER_CHUNK_CAP,
    )
    remaining = max(0, USER_CHUNK_CAP - len(pinned_user))
    unpinned_user: list = []
    if remaining > 0:
        unpinned_user = await repo.list_by_user(
            user_id, category="user", pinned=False, limit=remaining,
        )

    rule_raw = await repo.list_by_user(
        user_id, category="rule", limit=RULE_CHUNK_CAP,
    )
    fact_raw = await repo.list_by_user(
        user_id, category="fact", limit=FACT_CHUNK_CAP,
    )

    return MemorySnapshot(
        user_chunks=tuple(pinned_user) + tuple(unpinned_user),
        rule_chunks=tuple(rule_raw),
        fact_chunks=tuple(fact_raw),
    )
