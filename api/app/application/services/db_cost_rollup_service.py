"""[PR-9b-B Task B3] DbCostRollupService — pull-cost authority + push-only hook.

Concrete implementation of :class:`CostRollupService` Protocol (see
``cost_rollup_service.py``). Replaces PR-9b-A's
:class:`MetricHookCostRollupService` stub: this class implements BOTH the
``aggregate(...)`` pull path (authoritative for
``CoordinatorReduceEvent.cost_total``) and the ``rollup_to_parent(...)``
push path (non-authoritative metric hook preserved for parity).

INV-B3 contract (plan §B3):

  * ``aggregate(...)`` queries the durable ``cost_records`` ledger, JOINing
    ``sessions`` on ``session_id`` to validate caller-supplied
    ``child_session_ids`` belong to the named ``coordinator_run_id``.
    Mismatched-run rows are EXCLUDED from the SUM **and** surface via
    ``missing_children`` so the reducer can build the
    ``diagnostics_summary='cost_unavailable: <ids>'`` carrier per spec.
  * Per-child zero-rows (ledger empty OR ``coordinator_run_id`` mismatch
    OR the child_session_id is not present in ``sessions``) are detected
    by application-side set difference between requested IDs and IDs that
    contributed at least one row.
  * INV-B1 hard-lock: the SQL SUMs ``total_usd`` (NOT ``cost_usd``); this
    matches the ORM column at ``cost_record_orm.py:112``.
  * **Deliberate ``tool_call_count=0``**: ``cost_records`` has no
    ``tool_call_count`` column (cost_record_orm.py:46-128). ``aggregate(...)``
    surfaces 0 on the wire as a *known* regression vs. A3's push path —
    the reducer's diagnostics path does not consume tool_call_count from
    the pull source. Tracked as a B4 follow-up: add a ledger column or
    wire ``COUNT(*)`` as a proxy. Do NOT treat the 0 as a bug.
  * **Explicit ``float(total_usd)`` at the boundary**: avoids leaning on
    Pydantic v2's implicit Decimal → float coercion (would re-break on a
    Pydantic v3+ behavior change).

The ``rollup_to_parent`` half preserves the existing 4-kwarg signature
(``parent_session_id`` / ``cost`` / ``source`` / ``idempotency_key``) and
the never-raise + idempotent semantics A3 shipped — supervisor PEL retry
must NOT double-count, and a metric sink failure must NOT raise into the
mailbox handler. A3's :class:`MetricHookCostRollupService` is deleted in
the same change; its INV-A9 idempotency coverage is now in
``test_db_cost_rollup_service::test_rollup_to_parent_idempotent``.

Clean Architecture note: lives in ``application/`` because cost rollup is
an orchestration concern owned by the coordinator dispatch path. SQLAlchemy
import is allowed in application; the query is built once as the class
attribute ``AGGREGATE_SQL`` so the SQL-shape unit tests can introspect it
without touching Postgres.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.services.cost_rollup_service import AggregateResult
from app.domain.models.mailbox_envelope import CostAggregate

logger = logging.getLogger(__name__)

# Hard cap on the push-path idempotency cache (rollup_to_parent). The cache only
# needs to dedup RESULT_READY redeliveries within a supervisor PEL-retry window
# (seconds-to-minutes), so 50k distinct envelope_ids is far more than any retry
# storm can produce. A redelivery older than 50k distinct envelopes is
# astronomically unlikely, and even if one double-fired, rollup_to_parent is an
# explicitly non-load-bearing metric hook (see its docstring) so a rare metric
# double-count is acceptable — never a correctness break. Module-level (not a
# ctor param) so the ctor signature stays stable; tests monkeypatch this for
# small-cap eviction coverage.
_MAX_SEEN_KEYS: int = 50_000


class _MetricSink(Protocol):
    def record(
        self,
        *,
        parent_session_id: str,
        cost: Mapping[str, Any],
        source: str,
    ) -> None: ...


class _NoopMetricSink:
    """Inert metric sink used when DI does not provide one.

    Mirrors the ``_NoopMetricSink`` defined inline in
    ``service_dependencies.build_coordinator_runtime_deps``; keeps the
    push-only hook codepath identical to A3 without forcing every caller
    to construct a sink they don't yet have a destination for.
    """

    def record(self, **kwargs: Any) -> None:  # noqa: ANN003
        return None


class DbCostRollupService:
    """Pull-cost authority + idempotent push-only metric hook.

    Constructed once at composition-root time (lifespan startup) and shared
    by all coordinator runs. The ``async_session_factory`` is the same
    factory the rest of the coordinator runtime uses
    (``get_postgres().session_factory``); per-call short sessions are opened
    inside ``aggregate(...)`` so we never hold a long-lived connection.

    A3 → B3 supersession:
      * ``MetricHookCostRollupService`` (A3) implemented only
        ``rollup_to_parent`` and explicitly carried a one-PR scope.
      * ``DbCostRollupService`` (B3) implements BOTH ``aggregate`` and
        ``rollup_to_parent`` — the supervisor's ResultReadyHandler push
        and the reducer's pull both bind to this single concrete instance.
    """

    # SQL is captured as a class attribute so the unit SQL-shape tests
    # (``test_db_cost_rollup_service_sql.py``) can introspect the literal
    # without instantiating the service or touching Postgres. The query
    # JOINs cost_records to sessions on session_id, filters by
    # sessions.coordinator_run_id AND cost_records.session_id ∈ :ids, and
    # GROUPs BY session_id so every contributing child gets its own row
    # (application-side set difference surfaces zero-row children).
    AGGREGATE_SQL: str = (
        "SELECT cr.session_id AS session_id, "
        "       SUM(cr.input_tokens) AS sum_input_tokens, "
        "       SUM(cr.output_tokens) AS sum_output_tokens, "
        "       SUM(cr.total_usd) AS sum_total_usd, "
        "       COUNT(*) AS row_count "
        "FROM cost_records cr "
        "JOIN sessions s ON s.id = cr.session_id "
        "WHERE s.coordinator_run_id = :coordinator_run_id "
        "  AND cr.session_id IN :child_session_ids "
        "GROUP BY cr.session_id"
    )

    def __init__(
        self,
        *,
        async_session_factory: async_sessionmaker[AsyncSession],
        metric_sink: _MetricSink | None = None,
    ) -> None:
        self._session_factory = async_session_factory
        self._metric_sink: _MetricSink = metric_sink or _NoopMetricSink()
        # Bounded-LRU idempotency cache for the push path. Capped at
        # ``_MAX_SEEN_KEYS`` (default 50k) and evicting oldest-first via
        # ``OrderedDict.popitem(last=False)`` so a long-running singleton does
        # NOT leak memory linearly with completed-subtask count. PEL retry storms
        # still dedup correctly because the key is the envelope_id (not a
        # timestamp) and the cap dwarfs any realistic in-flight redelivery window.
        self._seen_keys: "OrderedDict[str, None]" = OrderedDict()

    # ── Pull authority ──────────────────────────────────────────────────────
    async def aggregate(
        self,
        *,
        coordinator_run_id: str,
        child_session_ids: list[str],
    ) -> AggregateResult:
        """Pull authoritative cost for the named coordinator run.

        Returns an :class:`AggregateResult` whose ``cost`` field is a
        :class:`CostAggregate` summed over the matched ledger rows and whose
        ``missing_children`` field lists IDs that contributed ZERO rows
        (ledger empty OR sessions.coordinator_run_id mismatch).

        Empty ``child_session_ids`` short-circuits to a zero CostAggregate
        with empty missing_children — the reducer treats "no children"
        and "all children unavailable" as distinct upstream states.
        """
        if not child_session_ids:
            return AggregateResult(
                cost=CostAggregate(
                    total_input_tokens=0,
                    total_output_tokens=0,
                    total_usd=Decimal("0"),
                    tool_call_count=0,
                ),
                missing_children=(),
            )

        stmt = text(self.AGGREGATE_SQL).bindparams(
            bindparam("child_session_ids", expanding=True),
        )

        async with self._session_factory() as session:
            result = await session.execute(
                stmt,
                {
                    "coordinator_run_id": coordinator_run_id,
                    "child_session_ids": list(child_session_ids),
                },
            )
            rows = result.mappings().all()

        # Application-side reduction: collapse per-child rows into the
        # CostAggregate shape the supervisor's CoordinatorReduceEvent expects.
        # Track which IDs contributed at least one row so the set difference
        # against the caller-supplied list produces missing_children.
        # [B3 KNOWN LIMITATION — cost_status NOT filtered; deliberate post-B
        # deferral] Any ledger row makes a child "present", INCLUDING a
        # ``cost_status='unknown'`` degraded marker
        # (CostCallbackHandler.write_session_degraded_marker writes a zero-cost
        # UNKNOWN/``node_name='persist_degraded'`` row when a child's terminal
        # cost flush fails). So a child whose ONLY ledger row is a degraded
        # marker is counted present-with-zero rather than surfaced in
        # ``missing_children``. This faithfully implements the plan's INV-B3
        # contract ("zero ROWS → missing") — a degraded child HAS a row — but it
        # under-reports a flush-failed child's true (unknown) cost on the
        # coordinator reduce event. A correct fix is out of B3 scope: it needs
        # cost_status-aware aggregation PLUS a ``degraded_children`` carrier on
        # AggregateResult + reducer handling (the legacy GET /cost path flags
        # UNKNOWN separately via cost_aggregation_service). Deferred to a
        # post-B follow-up; flag-gated OFF in production so there is no live
        # impact until the C2 coordinator flag is flipped on.
        present_ids: set[str] = set()
        total_input = 0
        total_output = 0
        total_usd = Decimal("0")
        for row in rows:
            present_ids.add(row["session_id"])
            total_input += int(row["sum_input_tokens"] or 0)
            total_output += int(row["sum_output_tokens"] or 0)
            row_usd = row["sum_total_usd"]
            if row_usd is not None:
                total_usd += Decimal(row_usd) if not isinstance(row_usd, Decimal) else row_usd

        # Preserve caller-supplied ordering for missing_children so the
        # reducer's diagnostics_summary string is deterministic; using a set
        # difference would scramble order across Python runs.
        missing = tuple(
            sid for sid in child_session_ids if sid not in present_ids
        )
        if missing:
            logger.warning(
                "[B3 INV-B3] cost aggregate found mismatched/empty children "
                "for coordinator_run_id=%s: missing=%s present=%s",
                coordinator_run_id, list(missing), sorted(present_ids),
            )

        return AggregateResult(
            cost=CostAggregate(
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                # Explicit float coercion — CostAggregate.total_usd is typed
                # float (mailbox_envelope.py:167); guard against Pydantic v3+
                # behavior change. Without this we'd lean on Pydantic v2's
                # implicit Decimal → float coercion, tying the wire boundary
                # to Pydantic internals.
                total_usd=float(total_usd),
                # TODO(post-B-tasks): ``cost_records`` ledger has no
                # ``tool_call_count`` column today (cost_record_orm.py:46-128).
                # B3 surfaces 0 here; this is a deliberate regression vs. A3's
                # push path which carried caller counts. The reducer's
                # diagnostics path does not consume tool_call_count from the
                # pull source, but any downstream consumer reading
                # CoordinatorReduceEvent.cost_total.tool_call_count will see 0
                # until either (a) the ledger schema gains a tool_call_count
                # column or (b) COUNT(*) is wired here as a proxy. File
                # tracked separately as a B4 follow-up.
                tool_call_count=0,
            ),
            missing_children=missing,
        )

    # ── Push hook (A3 parity) ───────────────────────────────────────────────
    async def rollup_to_parent(
        self,
        *,
        parent_session_id: str,
        cost: Mapping[str, Any],
        source: str,
        idempotency_key: str,
    ) -> None:
        """Push-only metric hook fired by the supervisor's ResultReadyHandler.

        Inherits A3's contract verbatim:
          * NEVER raises business exceptions — sink failures are WARN-logged
            and swallowed so the mailbox handler stays on the happy path.
          * Idempotent on ``idempotency_key`` — PEL retry must not
            double-count the same RESULT_READY envelope.

        Does NOT write to ``cost_records`` — the ledger is populated by the
        per-step LLM-call infrastructure (B4 M0). This method is a passive
        metric hook only; the authoritative cost source is ``aggregate(...)``.
        """
        if idempotency_key in self._seen_keys:
            # [PR-9b-B codex F4 — LOW] True-LRU refresh on a dedup HIT: move the
            # re-seen key to the tail so eviction is recency-based (matches the
            # "bounded-LRU" docstring + the eviction test), not FIFO-by-first-
            # insertion. Without this a frequently-redelivered key could still
            # be evicted as "oldest" while staler keys survive.
            self._seen_keys.move_to_end(idempotency_key)
            return
        self._seen_keys[idempotency_key] = None
        # Evict oldest-first to keep the cache bounded. ``> `` (not ``>=``)
        # leaves the cache holding exactly _MAX_SEEN_KEYS entries at steady
        # state; reads the module global each call so tests can shrink the cap.
        while len(self._seen_keys) > _MAX_SEEN_KEYS:
            self._seen_keys.popitem(last=False)
        try:
            self._metric_sink.record(
                parent_session_id=parent_session_id,
                cost=cost,
                source=source,
            )
        except Exception as exc:  # noqa: BLE001 — metric hook is never load-bearing
            logger.warning(
                "[B3] metric hook cost rollup failed: parent=%s exc=%s",
                parent_session_id, exc,
            )
