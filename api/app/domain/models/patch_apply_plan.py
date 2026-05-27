"""C2 v1 PatchApplyPlan + GroupOutcome (spec §9.2).

Wire-stable contract between the reducer and the applier:

- ``GroupOutcome`` — closed StrEnum mapped 1:1 from ``ResultReadyOutcome``
  plus three reducer-derived values (``CONFLICT`` / ``INCOMPLETE`` /
  ``MIXED``) that have no single-worker analogue.
- ``resolve_priority`` — deterministic multi-outcome resolver. The ladder
  is wire-load-bearing: ``test_patch_apply_plan.py::test_full_ladder_order``
  pins every step. Any reorder requires a spec change.
- ``PatchApplyPlan`` — frozen dataclass the applier consumes once.

Pure domain model: no DB, no SQLAlchemy, no FastAPI imports. The dataclass
is intentionally frozen so the reducer → applier handoff cannot smuggle
in-flight mutations across nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.domain.models.patch_manifest import FilePatchEntry


class GroupOutcome(StrEnum):
    """Single-value summary the reducer hands to the orchestrator + SSE.

    Five of these values mirror ``ResultReadyOutcome`` (per-worker
    terminal state); the remaining three (``CONFLICT`` / ``INCOMPLETE`` /
    ``MIXED``) are reducer-only and have no single-worker source.
    """

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    NEEDS_AUTHORIZATION = "needs_authorization"
    CONFLICT = "conflict"
    INCOMPLETE = "incomplete"
    MIXED = "mixed"


# [spec §9.2] Priority ladder (multi-outcome resolution).
#
# Reading the ladder: when multiple workers reach different outcomes the
# orchestrator must collapse them to one. Higher priority wins because the
# operator's next action diverges by outcome:
# - NEEDS_AUTHORIZATION first: the human gate blocks ANY apply, regardless
#   of other workers' progress. Losing this signal would silently drop a
#   pending permission decision.
# - CONFLICT next: even if every worker succeeded individually, overlapping
#   writes invalidate the apply plan — the orchestrator must route to
#   retry / split, never to apply.
# - FAILED > TIMED_OUT > CANCELLED: a hard failure is louder than a
#   deadline miss, which is louder than an explicit cancel. The operator
#   wants the most-actionable cause surfaced first.
# - INCOMPLETE only fires when expected workers never returned at all; it
#   ranks above SUCCESS because partial returns aren't safe to apply, but
#   below every per-worker terminal state we did receive.
_PRIORITY: tuple[GroupOutcome, ...] = (
    GroupOutcome.NEEDS_AUTHORIZATION,
    GroupOutcome.CONFLICT,
    GroupOutcome.FAILED,
    GroupOutcome.TIMED_OUT,
    GroupOutcome.CANCELLED,
    GroupOutcome.INCOMPLETE,
    GroupOutcome.SUCCESS,
)
# ``MIXED`` is intentionally absent from ``_PRIORITY``. It is a reducer
# diagnostic label that surfaces only when the seven-step algorithm
# explicitly elects it (e.g. degraded multi-cause routing in PR-6) — it
# never participates in normal priority resolution and must not be
# elected by the empty-set fallback below.


def resolve_priority(outcomes: set[GroupOutcome]) -> GroupOutcome:
    """Return the highest-priority outcome present in ``outcomes``.

    Empty input falls back to ``SUCCESS`` — degenerate-but-safe: the
    upstream completeness check (§9.3 step 1) has already rejected the
    only realistic empty case (no workers returned), so reaching here
    with an empty set means the caller resolved a vacuous outcome group
    (e.g. exploration-only step with no per-worker results aggregated).
    """
    for p in _PRIORITY:
        if p in outcomes:
            return p
    return GroupOutcome.SUCCESS


@dataclass(frozen=True)
class PatchApplyPlan:
    """Reducer → applier handoff.

    ``files`` is the deduplicated, path-sorted entry list the reducer
    builds in step 6. ``source_work_unit_ids`` lets the audit table trace
    each plan back to its contributing workers.

    Frozen so the in-flight value cannot be mutated between the reducer
    return and the applier consume — the orchestrator passes this around
    as part of the subgraph final_state, and a slip there would corrupt
    the audit trail.
    """

    coordinator_run_id: str
    files: tuple[FilePatchEntry, ...]
    total_size_bytes: int
    file_count: int
    source_work_unit_ids: tuple[str, ...]
