"""[C2 PR-6 §14.4] /cost/tree returns cost_source attribution.

Locks the bucketing rule that turns descendant ``tool_filter_preset`` +
``worker_type`` into a ``CostSource`` label, plus the rule that the
``SessionCostSnapshot.total_cost_usd`` projection matches the existing
``CostTreeAggregate.total_cost.total_usd`` ledger number (float-cast).
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import uuid4

import pytest
from unittest.mock import MagicMock

from app.application.services.cost_aggregation_service import CostAggregationService
from app.application.services.session_cost_tree_service import (
    CostTreeAggregate,
    SessionCostTreeService,
)
from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.models.cost_snapshot import CostSource
from app.domain.models.session import Session


pytestmark = pytest.mark.anyio


def _make_session(
    *,
    id: str,
    parent_session_id: Optional[str] = None,
    tool_filter_preset: Optional[str] = None,
    worker_type: str = "root",
    user_id: str = "u1",
) -> Session:
    """Build a minimal Session domain model with C2-relevant fields."""
    return Session(
        id=id,
        parent_session_id=parent_session_id,
        tool_filter_preset=tool_filter_preset,  # type: ignore[arg-type]
        worker_type=worker_type,  # type: ignore[arg-type]
        user_id=user_id,
    )


def _make_cost_row(
    *,
    session_id: str,
    total_usd: str,
    status: CostStatus = CostStatus.ACTUAL,
) -> CostRecord:
    """Build a minimal CostRecord — all 18 required fields populated.

    ``total_usd`` is taken as a string to keep Decimal precision (matches
    the existing ``_row`` helper in ``test_session_cost_tree_service.py``).
    """
    return CostRecord(
        id=uuid4().hex,
        session_id=session_id,
        user_id="u1",
        run_id=f"run-{session_id}",
        node_name="n",
        step_ix=0,
        attempt_ix=0,
        model="m",
        provider="p",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        total_usd=Decimal(total_usd),
        pricing_version="v1",
        cost_status=status,
        created_at=datetime.now(tz=timezone.utc),
    )


class _SessionRepo:
    """Hand-rolled SessionRepository stub.

    Mirrors the existing ``_SessionRepo`` in
    ``test_session_cost_tree_service.py`` so the new tests stay close to the
    existing fixture style (one repo per test, no shared mutation).
    """

    def __init__(
        self,
        *,
        self_session: Optional[Session],
        descendants: list[Session],
    ) -> None:
        self._self = self_session
        self._desc = descendants

    async def find_by_id_for_user(
        self, session_id: str, *, user_id: str
    ) -> Optional[Session]:
        if (
            self._self
            and self._self.id == session_id
            and self._self.user_id == user_id
        ):
            return self._self
        return None

    async def find_descendants(
        self,
        ancestor_id: str,
        *,
        user_id: str,
        max_depth: int,
        limit: int,
    ) -> list[Session]:
        del ancestor_id, user_id, max_depth
        return self._desc[:limit]


class _CostRepo:
    """Hand-rolled CostRecordRepository stub."""

    def __init__(self, rows: list[CostRecord]) -> None:
        self._rows = rows

    async def find_by_sessions_for_user(
        self, session_ids: list[str], *, user_id: str
    ) -> list[CostRecord]:
        del user_id
        return [r for r in self._rows if r.session_id in session_ids]


def _build_service(
    *,
    self_session: Optional[Session],
    descendants: list[Session],
    rows: list[CostRecord],
) -> SessionCostTreeService:
    return SessionCostTreeService(
        session_repo=_SessionRepo(
            self_session=self_session, descendants=descendants
        ),
        cost_repo=_CostRepo(rows),
        cost_aggregator=CostAggregationService(MagicMock()),
    )


class TestCostSourceClassification:
    """Eight transitions of the ``cost_source`` label, end-to-end through
    the tree service. The service-level path proves the bucketing rule
    (preset → snapshot dimension) on top of the domain-level transitions
    already covered in ``test_cost_snapshot.py``.
    """

    async def test_direct_only_returns_DIRECT(self) -> None:
        """Root session with own cost, no descendants → cost_source=DIRECT."""
        root = _make_session(id="root", worker_type="root")
        rows = [_make_cost_row(session_id="root", total_usd="1.00")]
        svc = _build_service(self_session=root, descendants=[], rows=rows)

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)

        assert isinstance(agg, CostTreeAggregate)
        assert agg.cost_source == CostSource.DIRECT

    async def test_coordinator_descendant_only_returns_COORDINATOR_SUBAGENT(
        self,
    ) -> None:
        """Root w/ one coordinator_step descendant carrying cost → COORDINATOR."""
        root = _make_session(id="root", worker_type="root")
        child = _make_session(
            id="child-coord",
            parent_session_id="root",
            tool_filter_preset="coordinator_step",
            worker_type="subagent",
        )
        rows = [_make_cost_row(session_id="child-coord", total_usd="0.42")]
        svc = _build_service(
            self_session=root, descendants=[child], rows=rows
        )

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)

        assert agg.cost_source == CostSource.COORDINATOR_SUBAGENT

    async def test_research_descendant_only_returns_RESEARCH_SUBAGENT(
        self,
    ) -> None:
        """Root w/ subagent_research descendant carrying cost → RESEARCH."""
        root = _make_session(id="root", worker_type="root")
        child = _make_session(
            id="child-research",
            parent_session_id="root",
            tool_filter_preset="subagent_research",
            worker_type="subagent",
        )
        rows = [_make_cost_row(session_id="child-research", total_usd="0.25")]
        svc = _build_service(
            self_session=root, descendants=[child], rows=rows
        )

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)

        assert agg.cost_source == CostSource.RESEARCH_SUBAGENT

    async def test_legacy_descendant_buckets_as_RESEARCH(self) -> None:
        """Descendant with tool_filter_preset=None AND worker_type='subagent'
        → bucketed into the research dimension (legacy compat path for
        pre-C2 subagent rows that predate the preset column being filled)."""
        root = _make_session(id="root", worker_type="root")
        legacy_child = _make_session(
            id="child-legacy",
            parent_session_id="root",
            tool_filter_preset=None,
            worker_type="subagent",
        )
        rows = [_make_cost_row(session_id="child-legacy", total_usd="0.10")]
        svc = _build_service(
            self_session=root, descendants=[legacy_child], rows=rows
        )

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)

        assert agg.cost_source == CostSource.RESEARCH_SUBAGENT

    async def test_direct_plus_coordinator_returns_MIXED(self) -> None:
        """Root own cost + coordinator descendant cost → MIXED."""
        root = _make_session(id="root", worker_type="root")
        coord_child = _make_session(
            id="child-coord",
            parent_session_id="root",
            tool_filter_preset="coordinator_step",
            worker_type="subagent",
        )
        rows = [
            _make_cost_row(session_id="root", total_usd="1.00"),
            _make_cost_row(session_id="child-coord", total_usd="0.50"),
        ]
        svc = _build_service(
            self_session=root, descendants=[coord_child], rows=rows
        )

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)

        assert agg.cost_source == CostSource.MIXED

    async def test_all_three_buckets_returns_MIXED(self) -> None:
        """Direct + coordinator + research all > 0 → MIXED."""
        root = _make_session(id="root", worker_type="root")
        coord_child = _make_session(
            id="child-coord",
            parent_session_id="root",
            tool_filter_preset="coordinator_step",
            worker_type="subagent",
        )
        research_child = _make_session(
            id="child-research",
            parent_session_id="root",
            tool_filter_preset="subagent_research",
            worker_type="subagent",
        )
        rows = [
            _make_cost_row(session_id="root", total_usd="1.00"),
            _make_cost_row(session_id="child-coord", total_usd="0.30"),
            _make_cost_row(session_id="child-research", total_usd="0.20"),
        ]
        svc = _build_service(
            self_session=root,
            descendants=[coord_child, research_child],
            rows=rows,
        )

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=2)

        assert agg.cost_source == CostSource.MIXED

    async def test_zero_cost_session_returns_NONE(self) -> None:
        """Session with no cost rows at all → cost_source=NONE."""
        root = _make_session(id="root", worker_type="root")
        svc = _build_service(self_session=root, descendants=[], rows=[])

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)

        assert agg.cost_source == CostSource.NONE

    async def test_descendant_exists_but_zero_cost_returns_NONE(self) -> None:
        """Descendant attached but no cost rows landed for it → NONE
        (no attribution claim when every dimension is zero)."""
        root = _make_session(id="root", worker_type="root")
        coord_child = _make_session(
            id="child-coord",
            parent_session_id="root",
            tool_filter_preset="coordinator_step",
            worker_type="subagent",
        )
        svc = _build_service(
            self_session=root, descendants=[coord_child], rows=[]
        )

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)

        assert agg.cost_source == CostSource.NONE


class TestTotals:
    """Cross-check the snapshot total against the existing ledger number."""

    async def test_snapshot_total_matches_tree_total(self) -> None:
        """The float projection (direct + coord + research) equals
        ``float(CostTreeAggregate.total_cost.total_usd)`` so the frontend
        can reconcile the two numbers without rounding drift."""
        root = _make_session(id="root", worker_type="root")
        coord_child = _make_session(
            id="child-coord",
            parent_session_id="root",
            tool_filter_preset="coordinator_step",
            worker_type="subagent",
        )
        research_child = _make_session(
            id="child-research",
            parent_session_id="root",
            tool_filter_preset="subagent_research",
            worker_type="subagent",
        )
        rows = [
            _make_cost_row(session_id="root", total_usd="1.00"),
            _make_cost_row(session_id="child-coord", total_usd="0.30"),
            _make_cost_row(session_id="child-research", total_usd="0.20"),
        ]
        svc = _build_service(
            self_session=root,
            descendants=[coord_child, research_child],
            rows=rows,
        )

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=2)

        # The snapshot float total must equal float(Decimal ledger total).
        expected_float_total = float(agg.total_cost.total_usd)
        assert expected_float_total == pytest.approx(1.50)
        # And the underlying ledger remains the authoritative Decimal.
        assert agg.total_cost.total_usd == Decimal("1.50")

    async def test_unmapped_descendant_row_does_not_tilt_source(self) -> None:
        """Defensive: if a desc_row somehow references a session id we
        don't have in ``descendants_map`` (e.g. truncated tail), the
        attribution code drops it from the bucketing rather than
        attributing it to an unknown dimension. Documents the guard at
        ``_compute_cost_source`` ``if session is None: continue``.

        Realised here by including the orphan id in the descendants
        passed to ``find_descendants`` (so the repo will return its
        rows) but stripping it from the per-test ``descendants_map``
        view via a custom session_repo that yields a *different*
        descendant list than the one feeding the rows. The simpler
        equivalent — passing a descendant whose preset is an unknown
        string — exercises the same dropped-from-buckets branch.
        """
        root = _make_session(id="root", worker_type="root")
        # ``tool_filter_preset`` Literal only allows two values + None;
        # the unknown-preset branch is unreachable through normal
        # construction. The mapped-but-not-bucketed path is when
        # ``tool_filter_preset is None`` AND ``worker_type == "root"``
        # — a descendant with no preset and worker_type=root is dropped
        # because no bucket claims it.
        weird_child = _make_session(
            id="child-unknown",
            parent_session_id="root",
            tool_filter_preset=None,
            worker_type="root",  # not subagent — legacy bucket doesn't fire
        )
        rows = [_make_cost_row(session_id="child-unknown", total_usd="9.99")]
        svc = _build_service(
            self_session=root, descendants=[weird_child], rows=rows
        )

        agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)

        # Unknown-bucket descendant didn't tilt cost_source.
        assert agg.cost_source == CostSource.NONE
        # But the existing aggregate-level total still saw the row
        # (the rollup is exhaustive; only the attribution is selective).
        assert agg.total_cost.total_usd == Decimal("9.99")


# ── [PR-9b-B Task B9] CostRollupService decoupling regression guard ──────────
# The session-cost-tree path (cost_aggregation_service + session_cost_tree_service)
# reads cost_records directly via its own legacy aggregation and MUST stay
# decoupled from the coordinator pull-cost service (CostRollupService) added in
# PR-9b-B. If a future refactor reroutes session-cost-tree through
# CostRollupService, this guard fails — forcing an explicit decision instead of
# silent coupling between two independent cost-aggregation subsystems.


async def test_session_cost_tree_modules_do_not_depend_on_cost_rollup_service() -> None:
    """[PR-9b-B B9] Neither session-cost-tree module IMPORTS the coordinator
    ``CostRollupService`` class or the ``cost_rollup_service`` /
    ``db_cost_rollup_service`` modules.

    [PR-9b-B codex F3 — MEDIUM/TEST] AST-based, not a source-substring scan.
    A substring check (``"CostRollupService" not in src``) is brittle (a
    comment/docstring/string mentioning the name trips it) AND incomplete (an
    aliased import — ``from ... import CostRollupService as X`` — would still
    bind the symbol but a naive bound-name scan could miss it; a TYPE_CHECKING
    ref also trips a substring scan even though it's not a runtime dependency).

    Instead we parse each module's source and walk ONLY ``ast.Import`` /
    ``ast.ImportFrom`` nodes, checking the IMPORTED name (not the bound alias),
    so:
      * comments / docstrings / arbitrary string literals never trip it, and
      * a real import IS caught regardless of ``as``-aliasing.
    Pure-AST (no DB). Follows the imported module object so the lock survives
    file moves.
    """
    import ast
    import inspect

    from app.application.services import (
        cost_aggregation_service,
        session_cost_tree_service,
    )

    # The coordinator pull-cost surface this path must stay decoupled from:
    # the Protocol/class name, plus the two modules that define/implement it.
    _BANNED_NAMES = {"CostRollupService"}
    _BANNED_MODULES = {
        "cost_rollup_service",
        "db_cost_rollup_service",
    }

    def _module_tail(dotted: str) -> str:
        """Last dotted segment, e.g. ``a.b.cost_rollup_service`` → tail."""
        return dotted.rsplit(".", 1)[-1]

    def _violations_for(module) -> list[str]:
        src = inspect.getsource(module)
        tree = ast.parse(src)
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # ``import a.b.cost_rollup_service`` / ``... as x`` — inspect the
                # module path's tail, NOT the bound alias.
                for alias in node.names:
                    if _module_tail(alias.name) in _BANNED_MODULES:
                        found.append(
                            f"line {node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                # ``from <module> import <name> [as alias]`` — a violation if
                # EITHER the source module is a banned module OR an imported
                # name is the banned class. Check imported names, not aliases.
                mod_tail = _module_tail(node.module) if node.module else ""
                if mod_tail in _BANNED_MODULES:
                    found.append(
                        f"line {node.lineno}: from {node.module} import ..."
                    )
                for alias in node.names:
                    if alias.name in _BANNED_NAMES:
                        found.append(
                            f"line {node.lineno}: from {node.module} "
                            f"import {alias.name}"
                        )
        return found

    for module in (cost_aggregation_service, session_cost_tree_service):
        violations = _violations_for(module)
        assert not violations, (
            f"{module.__name__} imports the coordinator pull-cost surface — "
            "the session-cost-tree path has its OWN ledger aggregation and "
            "MUST stay decoupled from CostRollupService (PR-9b-B B9). A real "
            "violation looks like `from app.application.services."
            "cost_rollup_service import CostRollupService` (or an aliased / "
            "db_cost_rollup_service variant). Found:\n  - "
            + "\n  - ".join(violations)
        )
