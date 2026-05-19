"""T12 / Phase 1 PR-X — F8 pod-restart resilience for tool_filter.

Pins the precedence contract of ``AgentService._resolve_effective_tool_filter``:

1. Explicit caller ``tool_filter`` always wins — including the empty
   ``frozenset()`` "deny-all" sentinel that fresh chat would never receive
   but a test or a future caller might supply.
2. If caller passes ``None`` and ``session.tool_filter_preset`` is set, the
   preset's allowlist is restored.
3. ``None`` caller + no preset → ``None`` (back-compat for pre-T12 sessions).
4. Unknown preset → ``ValueError`` (fail-closed; mirrors the DB CHECK
   constraint's last-line defense).

The helper is the single source of truth consumed by every ``_create_task``
reconstruction path (fresh chat, resume, FINISHING, orphan sweep, preflight
rebuild); testing it directly avoids spinning up the full sandbox / browser
/ runner stack just to assert a 3-line precedence rule.
"""

from __future__ import annotations

import pytest

from app.application.services.agent_service import AgentService
from app.domain.models.session import Session
from app.domain.services.tool_filter_presets import (
    SUBAGENT_RESEARCH_ALLOWED_TOOLS,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _session(preset: str | None) -> Session:
    return Session(
        id="sess-t12",
        user_id="user-1",
        tool_filter_preset=preset,  # type: ignore[arg-type]
    )


class TestResolveEffectiveToolFilter:
    """Precedence: explicit caller wins; preset fills the gap; None on miss."""

    def test_caller_none_preset_none_returns_none(self) -> None:
        """Back-compat: pre-T12 sessions have no preset → no restriction."""
        result = AgentService._resolve_effective_tool_filter(
            _session(preset=None), tool_filter=None,
        )
        assert result is None

    def test_caller_none_preset_subagent_research_restores_allowlist(self) -> None:
        """F8 restore: resume path passes None; preset must reinstate the allowlist.

        This is the critical security assertion — without it a pod restart
        silently widens a child session's tool surface back to the parent's
        full registry.
        """
        result = AgentService._resolve_effective_tool_filter(
            _session(preset="subagent_research"), tool_filter=None,
        )
        assert result is SUBAGENT_RESEARCH_ALLOWED_TOOLS

    def test_caller_explicit_filter_wins_over_preset(self) -> None:
        """Caller-supplied tool_filter takes precedence — fresh-chat path."""
        explicit = frozenset({"search_web"})
        result = AgentService._resolve_effective_tool_filter(
            _session(preset="subagent_research"), tool_filter=explicit,
        )
        assert result is explicit

    def test_caller_explicit_empty_frozenset_wins_over_preset(self) -> None:
        """``frozenset()`` is "deny-all"; preset must NOT widen it.

        Pins the falsy/None boundary: ``not frozenset()`` is True in Python
        but the contract is "explicit value wins" — only literal ``None``
        triggers preset restore.
        """
        deny_all: frozenset[str] = frozenset()
        result = AgentService._resolve_effective_tool_filter(
            _session(preset="subagent_research"), tool_filter=deny_all,
        )
        assert result is deny_all

    def test_caller_none_preset_unknown_raises_value_error(self) -> None:
        """Fail-closed on unknown preset (DB CHECK should prevent this from
        ever reaching the resolver, but the resolver still must not silently
        fall back to no restriction).

        Build session via ``model_construct`` to bypass Pydantic's Literal
        validation — simulates a row written before a preset was rolled
        back or via a direct SQL insert that bypassed the CHECK constraint.
        """
        sess = Session.model_construct(
            id="sess-bad",
            user_id="user-1",
            tool_filter_preset="not_a_real_preset",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError):
            AgentService._resolve_effective_tool_filter(sess, tool_filter=None)

    def test_caller_none_session_without_attribute_returns_none(self) -> None:
        """Defensive: a session model that somehow lacks the attribute (e.g.
        a stub in unrelated tests) must not crash — ``getattr`` default
        kicks in and we fall through to ``None``."""
        class _StubSession:
            id = "stub"

        result = AgentService._resolve_effective_tool_filter(
            _StubSession(),  # type: ignore[arg-type]
            tool_filter=None,
        )
        assert result is None

    def test_empty_string_preset_raises_not_silently_drops(self) -> None:
        """Codex R1 P2 fix: empty string MUST flow through to resolve_preset
        (which raises ValueError) rather than being coerced to "no
        restriction" by truthiness.

        Pins the ``is None`` predicate vs the previous ``not preset_name``
        which would silently widen permissions on a malformed row (e.g.
        post-DB-corruption or a future schema change that didn't update the
        Literal type). Tests the operative semantic: only literal ``None``
        means "no preset"; any other falsy value is data inconsistency and
        must surface.
        """
        sess = Session.model_construct(
            id="sess-empty",
            user_id="user-1",
            tool_filter_preset="",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError):
            AgentService._resolve_effective_tool_filter(sess, tool_filter=None)


class TestOrmRoundTripThroughResolver:
    """Codex R1 P2 fix: lock in the full pod-restart reconstruction path.

    Exercises ``Session → SessionModel.from_domain → to_domain → resolver``
    end-to-end so a future mapper / Pydantic / column-rename regression
    breaks here rather than silently widening a restored child's tool
    surface. Uses the bi-directional mapper rather than constructing
    SessionModel from scratch — the ORM relies on DB server_defaults for
    many columns that aren't set on raw-construct, so a from_domain seed
    is the only honest way to hydrate a SessionModel in-memory without
    actually round-tripping through Postgres.
    """

    def _hydrate_orm(self, seed: Session):
        """Build an ORM row the way Postgres would after INSERT-SELECT.

        ``SessionModel.from_domain`` deliberately excludes ``updated_at`` /
        ``created_at`` (the columns carry ``server_default
        CURRENT_TIMESTAMP(0)`` so the DB fills them on flush). When we
        construct in-memory we have to fill them ourselves; without a real
        DB roundtrip those attrs are ``None`` and the subsequent
        ``Session.model_validate(orm, from_attributes=True)`` fails the
        datetime check.
        """
        from datetime import datetime
        from app.infrastructure.models.session import SessionModel

        orm = SessionModel.from_domain(seed)
        now = datetime.now()
        orm.updated_at = now
        orm.created_at = now
        return orm

    def test_orm_session_with_preset_restores_allowlist(self) -> None:
        seed = Session(
            id="sess-orm-t12",
            user_id="user-1",
            title="reloaded child",
            sample_session_id="parent-1",
            tool_filter_preset="subagent_research",
        )
        orm = self._hydrate_orm(seed)

        # Forward mapping pinned the column.
        assert orm.tool_filter_preset == "subagent_research"
        assert orm.sample_session_id == "parent-1"

        # ``to_domain`` uses ``Session.model_validate(self,
        # from_attributes=True)`` so any attribute-name drift between ORM
        # and domain blows up here.
        reloaded = orm.to_domain()
        assert reloaded.tool_filter_preset == "subagent_research"
        assert reloaded.sample_session_id == "parent-1"

        # And the resolver re-derives the canonical allowlist — the actual
        # F8 fix observable end-to-end on a "post-restart" reload.
        restored = AgentService._resolve_effective_tool_filter(
            reloaded, tool_filter=None,
        )
        assert restored is SUBAGENT_RESEARCH_ALLOWED_TOOLS

    def test_orm_session_without_preset_restores_no_filter(self) -> None:
        """Pre-T12 sessions (NULL preset) still flow through to None.

        Back-compat invariant — a regular parent session (NULL preset)
        reloaded from DB must not be coerced into the subagent allowlist.
        """
        seed = Session(
            id="sess-orm-parent",
            user_id="user-1",
            title="reloaded parent",
            sample_session_id=None,
            tool_filter_preset=None,
        )
        orm = self._hydrate_orm(seed)
        reloaded = orm.to_domain()

        assert reloaded.tool_filter_preset is None
        restored = AgentService._resolve_effective_tool_filter(
            reloaded, tool_filter=None,
        )
        assert restored is None
