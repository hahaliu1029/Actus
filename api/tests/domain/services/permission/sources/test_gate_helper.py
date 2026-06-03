"""PE-1 §2.3 — gate_helper: is_pe_enabled_for_source + PE_SUPPORTED_SOURCES.

Round 2 P1#2 additions: ``is_pe_eligible_tool_source`` considers BOTH
source and category, so skill creator / skill guide tools (``source="skill"``
but ``category in {"skill creator", "skill guide"}``) bypass PE → legacy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.services.permission.sources.gate_helper import (
    PE_SUPPORTED_SOURCES,
    is_pe_eligible_tool_source,
    is_pe_enabled_for_source,
)


@dataclass
class _StubTC:
    """Mimics ToolConfirmationConfig surface used by the gate.

    PE-4c: per-source flags deleted; only the master ``enabled`` switch remains.
    """
    enabled: bool = True


@dataclass(frozen=True)
class _StubToolSource:
    """Minimal duck-typed ToolSource for gate tests.

    Avoids importing the real Pydantic ``ToolSource`` (which validates
    ``source`` against a Literal) so we can exercise edge cases like
    ``source=None`` or unsupported sources.
    """
    source: Any = None
    category: Any = None
    canonical_name: str = "stub"


class TestRegisteredSetMembership:
    def test_pe_3_registers_native_skill_mcp_a2a(self):
        assert PE_SUPPORTED_SOURCES == frozenset({"native", "skill", "mcp", "a2a"})

    def test_mcp_in_supported_set(self):
        assert "mcp" in PE_SUPPORTED_SOURCES

    def test_a2a_in_supported_set(self):
        assert "a2a" in PE_SUPPORTED_SOURCES


class TestGateHelperBehavior:
    def test_native_supported_and_flag_on_returns_true(self):
        tc = _StubTC()
        assert is_pe_enabled_for_source("native", tc) is True

    def test_skill_supported_and_flag_on_returns_true(self):
        tc = _StubTC()
        assert is_pe_enabled_for_source("skill", tc) is True

    def test_mcp_supported_and_flag_on_returns_true(self):
        """mcp is PE-supported (PE-2); master on → True (PE-4c: per-source flag gone)."""
        tc = _StubTC()
        assert is_pe_enabled_for_source("mcp", tc) is True

    def test_a2a_supported_and_flag_on_returns_true(self):
        """a2a is PE-supported (PE-3); master on → True (PE-4c: per-source flag gone)."""
        tc = _StubTC()
        assert is_pe_enabled_for_source("a2a", tc) is True

    def test_unknown_source_returns_false(self):
        tc = _StubTC()
        assert is_pe_enabled_for_source("mystery", tc) is False
        assert is_pe_enabled_for_source("", tc) is False

    def test_master_disabled_overrides_all_sources(self):
        tc = _StubTC(enabled=False)
        assert is_pe_enabled_for_source("native", tc) is False
        assert is_pe_enabled_for_source("skill", tc) is False

    def test_all_registered_sources_enabled_when_master_on(self):
        """PE-4c: with the per-source flags gone, every registered source is
        PE-enabled iff the master switch is on."""
        tc = _StubTC()
        for src in PE_SUPPORTED_SOURCES:
            assert is_pe_enabled_for_source(src, tc) is True


class TestIsPeEligibleToolSource:
    """Round 2 P1#2: ``is_pe_eligible_tool_source`` is the source+category-aware
    funnel used by ``react_graph._pe_dispatch`` (pre-loop + per-call defensive)
    and ``agent_service._batch_has_non_pe_eligible_pending`` /
    ``preflight_resume_tool_confirmation``.
    """

    def test_native_known_tool_with_flag_on_is_eligible(self):
        tc = _StubTC()
        ts = _StubToolSource(source="native", category="file")
        assert is_pe_eligible_tool_source(ts, tc) is True

    def test_skill_dynamic_wrapper_is_eligible(self):
        """source='skill' + category='skill' is the dynamic SkillTool path."""
        tc = _StubTC()
        ts = _StubToolSource(source="skill", category="skill")
        assert is_pe_eligible_tool_source(ts, tc) is True

    def test_skill_creator_is_not_eligible(self):
        """brainstorm_skill / generate_skill / install_skill all map to
        source='skill' + category='skill creator' — must fall back to legacy."""
        tc = _StubTC()
        ts = _StubToolSource(source="skill", category="skill creator")
        assert is_pe_eligible_tool_source(ts, tc) is False

    def test_skill_guide_is_not_eligible(self):
        """get_skill_guide maps to source='skill' + category='skill guide'
        — must fall back to legacy."""
        tc = _StubTC()
        ts = _StubToolSource(source="skill", category="skill guide")
        assert is_pe_eligible_tool_source(ts, tc) is False

    def test_mcp_real_tool_is_eligible(self):
        """MCP is now in PE_SUPPORTED_SOURCES (PE-2); a real mcp tool
        (category='mcp') is PE-eligible → True."""
        tc = _StubTC()
        ts = _StubToolSource(source="mcp", category="mcp")
        assert is_pe_eligible_tool_source(ts, tc) is True

    def test_a2a_tool_is_eligible(self):
        """A2A is in PE_SUPPORTED_SOURCES (PE-3); category='a2a' → eligible."""
        tc = _StubTC()
        ts = _StubToolSource(source="a2a", category="a2a")
        assert is_pe_eligible_tool_source(ts, tc) is True

    def test_none_tool_source_returns_false(self):
        """Unknown tool / ToolSourceUnknownError → caller passes None → False."""
        tc = _StubTC()
        assert is_pe_eligible_tool_source(None, tc) is False

    def test_tool_source_with_source_none_returns_false(self):
        """Defensive: a ToolSource object whose source attr is None still
        returns False (no crash on getattr chain)."""
        tc = _StubTC()
        ts = _StubToolSource(source=None, category=None)
        assert is_pe_eligible_tool_source(ts, tc) is False

    def test_master_disabled_overrides_eligible_tool(self):
        """When the master switch is off, even a perfectly eligible tool
        falls back to legacy."""
        tc = _StubTC(enabled=False)
        ts = _StubToolSource(source="native", category="file")
        assert is_pe_eligible_tool_source(ts, tc) is False
        ts_skill = _StubToolSource(source="skill", category="skill")
        assert is_pe_eligible_tool_source(ts_skill, tc) is False

    def test_skill_creator_blocked_even_when_skill_flag_on(self):
        """The category gate is independent of the master switch: creator/guide
        tools must always bypass PE even when confirmation is on, because
        ``build_skill_call_metadata`` cannot resolve them."""
        tc = _StubTC()
        ts_creator = _StubToolSource(source="skill", category="skill creator")
        ts_guide = _StubToolSource(source="skill", category="skill guide")
        assert is_pe_eligible_tool_source(ts_creator, tc) is False
        assert is_pe_eligible_tool_source(ts_guide, tc) is False


class TestIsPeEligibleToolSourceWithRealToolSource:
    """Smoke test: the helper works with the real Pydantic ``ToolSource``
    model (frozen, extra='forbid'), not just our dataclass stub."""

    def test_real_native_file_tool_is_eligible(self):
        from app.domain.services.tools.tool_source_resolver import ToolSource

        tc = _StubTC()
        ts = ToolSource(
            source="native",
            category="file",
            canonical_name="file_write",
        )
        assert is_pe_eligible_tool_source(ts, tc) is True

    def test_real_brainstorm_skill_is_not_eligible(self):
        """Round 2 P1#2: brainstorm_skill is registered as
        source='skill', category='skill creator' per
        _CANONICAL_TOOL_IDENTITIES."""
        from app.domain.services.tools.tool_source_resolver import (
            resolve_tool_source,
        )

        tc = _StubTC()
        ts = resolve_tool_source("brainstorm_skill")
        assert ts.source == "skill"
        assert ts.category == "skill creator"
        assert is_pe_eligible_tool_source(ts, tc) is False

    def test_real_get_skill_guide_is_not_eligible(self):
        from app.domain.services.tools.tool_source_resolver import (
            resolve_tool_source,
        )

        tc = _StubTC()
        ts = resolve_tool_source("get_skill_guide")
        assert ts.source == "skill"
        assert ts.category == "skill guide"
        assert is_pe_eligible_tool_source(ts, tc) is False

    def test_real_dynamic_skill_prefix_is_eligible(self):
        """Heuristic resolution for skill_* maps to source='skill',
        category='skill'."""
        from app.domain.services.tools.tool_source_resolver import (
            resolve_tool_source,
        )

        tc = _StubTC()
        ts = resolve_tool_source("skill_some_dynamic")
        assert ts.source == "skill"
        assert ts.category == "skill"
        assert is_pe_eligible_tool_source(ts, tc) is True

    def test_real_mcp_discovery_tool_is_not_eligible(self):
        """list_mcp_tools / get_mcp_tool are source='mcp' — PE-2 adds support."""
        from app.domain.services.tools.tool_source_resolver import (
            resolve_tool_source,
        )

        tc = _StubTC()
        ts = resolve_tool_source("list_mcp_tools")
        assert ts.source == "mcp"
        assert is_pe_eligible_tool_source(ts, tc) is False
