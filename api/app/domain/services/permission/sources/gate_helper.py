"""Source-uniform gate helper for PE routing.

Replaces 10 native-only callsites (spec §2.5 table) with a single
two-level gate: (1) source registered in PE_SUPPORTED_SOURCES, (2)
[PE-4c] per-source flags retired — master switch only.

Current supported sources: native, skill, mcp, a2a.

Spec §2.3; Round 1 P0#3.
"""

from __future__ import annotations

from typing import Any

# Single source of truth for "which sources flow through PE at this rev?"
# PE-3 PR appends "a2a". CI invariants (validate_pe_source_registry at DI
# time) ensure DI registers every entry in this set.
PE_SUPPORTED_SOURCES: frozenset[str] = frozenset({"native", "skill", "mcp", "a2a"})

def is_pe_enabled_for_source(source: str, config: Any) -> bool:
    """Master-switch + registered-source gate (PE-4c: per-source flags retired).

    Returns False (caller routes to legacy path) when:
      - master switch off (``config.enabled is False``)
      - source not in ``PE_SUPPORTED_SOURCES`` (e.g., a not-yet-supported
        future source, or a typo)

    Returns True when both gates pass. The per-source kill-switch flags
    (``permission_engine_{native,skill,mcp,a2a}_enabled``) were deleted in
    PE-4c — every registered source flows through PE whenever the master
    ``enabled`` switch is on. The ``PE_SUPPORTED_SOURCES`` membership guard
    remains so an unimplemented future source cannot accidentally route into
    PE before its Source adapter is registered at DI.
    """
    if not getattr(config, "enabled", True):
        return False
    return source in PE_SUPPORTED_SOURCES


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
    ``AllowError(code="skill_metadata_unresolvable")``. PE-2 registers
    McpSource for ``category="mcp"`` real remote tools (the mcp-discovery
    meta-tools are carved out below); PE-3 added A2aSource for ``source="a2a"``
    (single ``category="a2a"``, no discovery split). The skill creator / guide
    categories still bypass PE → legacy.

    Returns False (caller routes to legacy path) when:
      - ``tool_source`` is None (caller already gave up on resolution; this
        also gives the unknown-source case a single funnel so HTTP preflight,
        graph dispatch, and the batch guard all agree)
      - source not in ``PE_SUPPORTED_SOURCES``
      - master switch off (PE-4c retired the per-source flags; the only
        remaining toggle is the master ``enabled`` switch, via
        ``is_pe_enabled_for_source``)
      - ``source="skill"`` but ``category != "skill"`` (creator / guide)
      - ``source="mcp"`` but ``category != "mcp"`` (discovery meta-tools)

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
    # PE-2 §5: mcp discovery meta-tools (list_mcp_tools / get_mcp_tool) are
    # capability-discovery actions, NOT external permissioned actions — they
    # extend the next step's bindable tool set rather than producing an external
    # side effect, so they are NOT PE-eligible. Only category=="mcp" real remote
    # tools route through PE → McpSource. (get_mcp_tool mutates the activated set;
    # that is not a permissioned side effect — content-injection scanning of the
    # fetched description is a separate follow-up, spec §12.)
    if source == "mcp":
        category = getattr(tool_source, "category", None)
        if category != "mcp":
            return False
    return True
