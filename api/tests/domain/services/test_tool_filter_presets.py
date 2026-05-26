"""Unit tests for the tool_filter preset registry.

T12 / Phase 1 PR-X: ``Session.tool_filter_preset`` carries a persisted name
that ``AgentService._create_task`` resolves back to an in-memory allowlist
via ``resolve_preset``. This file pins:

1. The known preset ``subagent_research`` resolves to the documented
   read-only allowlist (so a future drift on the constant doesn't silently
   widen permissions on every restored child session).
2. ``None`` passthrough — caller passing ``None`` means "no restriction".
3. Unknown preset → ``ValueError`` (fail-closed; the DB CHECK constraint
   blocks unknown values at insert time, so resolve-time ValueError surfaces
   the rare-but-possible code/data drift instead of degrading silently).
"""

from __future__ import annotations

import pytest

from app.domain.services.tool_filter_presets import (
    SUBAGENT_RESEARCH_ALLOWED_TOOLS,
    TOOL_FILTER_PRESETS,
    resolve_preset,
)


class TestSubagentResearchAllowlist:
    def test_allowlist_is_frozenset(self) -> None:
        assert isinstance(SUBAGENT_RESEARCH_ALLOWED_TOOLS, frozenset)

    def test_allowlist_contents_pinned(self) -> None:
        """Pin the exact allowlist so a future drift surfaces as a test diff.

        These are the 10 read-only / read-output tools the subagent research
        spec greenlights; any change must be a deliberate spec update.
        """
        assert SUBAGENT_RESEARCH_ALLOWED_TOOLS == frozenset({
            "search_web",
            "file_read",
            "file_list",
            "file_view",
            "list_mcp_tools",
            "get_mcp_tool",
            "get_skill_guide",
            "memory_search",
            "memory_get",
            "shell_read_output",
        })

    def test_allowlist_excludes_write_surface(self) -> None:
        """Sanity guard: write-side tools must not appear in the read-only preset."""
        forbidden = {
            "shell_execute",
            "file_write",
            "file_str_replace",
            "memory_save",
            "browser_navigate",
            "browser_click",
            "skill_install",
            "delegate_to_subagent",
        }
        leaked = SUBAGENT_RESEARCH_ALLOWED_TOOLS & forbidden
        assert leaked == set(), f"forbidden tools leaked into allowlist: {leaked}"


class TestResolvePreset:
    def test_none_passes_through(self) -> None:
        assert resolve_preset(None) is None

    def test_known_preset_returns_allowlist(self) -> None:
        result = resolve_preset("subagent_research")
        assert result is SUBAGENT_RESEARCH_ALLOWED_TOOLS

    def test_unknown_preset_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc:
            resolve_preset("not_a_real_preset")
        msg = str(exc.value)
        assert "not_a_real_preset" in msg
        assert "subagent_research" in msg

    def test_empty_string_treated_as_unknown(self) -> None:
        """An empty string is not None — it's an unknown preset name.

        Pins the boundary between truthiness and explicit None: only literal
        None means "no preset". Anything else gets looked up.
        """
        with pytest.raises(ValueError):
            resolve_preset("")


class TestToolFilterPresetsRegistry:
    def test_registry_contains_only_known_keys(self) -> None:
        """Pin the registry's keyset; new presets need their own test update."""
        assert set(TOOL_FILTER_PRESETS) == {"subagent_research", "coordinator_step"}

    def test_subagent_research_key_points_to_canonical_allowlist(self) -> None:
        assert TOOL_FILTER_PRESETS["subagent_research"] is (
            SUBAGENT_RESEARCH_ALLOWED_TOOLS
        )
