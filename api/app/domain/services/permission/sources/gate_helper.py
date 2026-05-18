"""Source-uniform gate helper for PE routing.

Replaces 10 native-only callsites (spec §2.5 table) with a single
two-level gate: (1) source registered in PE_SUPPORTED_SOURCES, (2)
source-specific flag enabled.

PE-2 will add "mcp" to the supported set in the same PR that ships
MCPSource. PE-3 adds "a2a".

Spec §2.3; Round 1 P0#3.
"""

from __future__ import annotations

from typing import Any

# Single source of truth for "which sources flow through PE at this rev?"
# PE-2 PR appends "mcp"; PE-3 PR appends "a2a". CI invariants
# (validate_pe_source_registry at DI time) ensure DI registers every
# entry in this set.
PE_SUPPORTED_SOURCES_AFTER_PE_1: frozenset[str] = frozenset({"native", "skill"})

# Mapping from source string to the corresponding ToolConfirmationConfig
# flag attribute. Centralized to keep is_pe_enabled_for_source O(1)
# without per-call dict allocation.
_SOURCE_FLAG_ATTR: dict[str, str] = {
    "native": "permission_engine_native_enabled",
    "skill": "permission_engine_skill_enabled",
    # PE-2 PR: "mcp": "permission_engine_mcp_enabled",
    # PE-3 PR: "a2a": "permission_engine_a2a_enabled",
}


def is_pe_enabled_for_source(source: str, config: Any) -> bool:
    """Two-level gate.

    Returns False (caller routes to legacy path) when:
      - master switch off (``config.enabled is False``)
      - source not in ``PE_SUPPORTED_SOURCES_AFTER_PE_1`` (e.g., mcp/a2a in PE-1)
      - unknown source string (typo / future source not yet planned)
      - source-specific flag off

    Returns True only when both gates pass.

    The two-level design lets us ship per-source feature flags WHILE keeping
    a registered-set guard so an enabled flag for an unimplemented source
    cannot accidentally route calls into PE.
    """
    if not getattr(config, "enabled", True):
        return False
    if source not in PE_SUPPORTED_SOURCES_AFTER_PE_1:
        return False
    flag_attr = _SOURCE_FLAG_ATTR.get(source)
    if flag_attr is None:
        # supported set entry missing from flag map = code bug; fail-closed.
        return False
    return bool(getattr(config, flag_attr, False))


def is_pe_eligible_tool_source(tool_source: Any, config: Any) -> bool:
    """PE-1 §2.5 + Round 2 P1#2: PE-eligibility considers BOTH source and category.

    ``source="skill"`` covers three different tool categories per
    ``resolve_tool_source``:
      - ``category="skill"`` — dynamic SkillTool wrappers (in
        ``SkillTool._tool_bindings``) — PE-eligible via SkillSource.
      - ``category="skill creator"`` — ``brainstorm_skill``,
        ``generate_skill``, ``install_skill`` — NOT in ``_tool_bindings``,
        ``build_skill_call_metadata`` cannot resolve them; falls back to
        legacy.
      - ``category="skill guide"`` — ``get_skill_guide`` — same as above.

    Allowing source-only gating would route the creator / guide tools into
    PE, where ``_pe_dispatch`` would then emit
    ``AllowError(code="skill_metadata_unresolvable")``. PE-2 / PE-3 may
    register separate source adapters for those categories, but for PE-1
    they must bypass PE → legacy.

    Returns False (caller routes to legacy path) when:
      - ``tool_source`` is None (caller already gave up on resolution; this
        also gives the unknown-source case a single funnel so HTTP preflight,
        graph dispatch, and the batch guard all agree)
      - source not in ``PE_SUPPORTED_SOURCES_AFTER_PE_1``
      - source-specific flag off (or master switch off, via
        ``is_pe_enabled_for_source``)
      - ``source="skill"`` but ``category != "skill"`` (creator / guide)

    Returns True only when every gate passes.
    """
    if tool_source is None:
        return False
    source = getattr(tool_source, "source", None)
    if source is None:
        return False
    if not is_pe_enabled_for_source(source, config):
        return False
    # PE-1 §2.5 Round 2 P1#2: skill creator/guide are NOT PE-eligible.
    if source == "skill":
        category = getattr(tool_source, "category", None)
        if category != "skill":
            return False
    return True
