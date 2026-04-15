"""R1 CS1 ToolSource resolver — identity contract for Actus tool sources.

See docs/superpowers/specs/2026-04-14-r1-toolsource-resolver-design.md (v3.5)
for full design rationale.

This module owns:
- ToolSource Pydantic model (frozen)
- KNOWN_CATEGORIES (12 canonical categories)
- _CANONICAL_TOOL_IDENTITIES (37 static identities, seeded at module import)
- _REGISTRY (process-wide name -> ToolSource map)
- annotate_and_register_tool_source helper (factory single entry point)
- resolve_tool_source / resolve_tool_source_from_tool (public query API)
- ToolSourceUnknownError / RegistryConflictError exceptions

Does NOT export any clear/reset helper — test-only reset lives in
api/tests/_tool_source_testing.py to keep it out of the prod import path.
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


# ---- Canonical category vocabulary ------------------------------------- #

KNOWN_CATEGORIES: frozenset[str] = frozenset({
    "shell", "file", "browser", "message", "search", "memory",
    "skill", "skill creator", "skill guide",
    "mcp", "mcp discovery",
    "a2a",
})
"""The 12 canonical category values from tools_guide_dynamic.py _DISPLAY_ORDER.

Note the multi-word values use SPACE separators ("skill creator", not
"skill_creation"). R1 aligns agent_task_runner to this canonical form.
"""


# ---- Model ------------------------------------------------------------- #


class ToolSource(BaseModel):
    """Identity-only description of a tool.

    Identity contract: answers "what kind of tool is this name?".
    Does NOT answer "is this tool currently bound / available / invokable" —
    that is the availability concern handled by bound_tool_names set checks.

    Frozen via ConfigDict so instances cannot mutate after construction.
    Pydantic-based (not dataclass) so R4 can embed ToolSource directly into
    ToolEvent (BaseModel) with zero serialization friction.
    """

    model_config = ConfigDict(frozen=True)

    source: Literal["native", "mcp", "a2a", "skill"]
    category: str
    canonical_name: str

    @field_validator("category")
    @classmethod
    def _warn_unknown_category(cls, v: str) -> str:
        if v not in KNOWN_CATEGORIES:
            logger.warning(
                "Unknown category %r; expected one of %s",
                v, sorted(KNOWN_CATEGORIES),
            )
        return v


# ---- Exceptions -------------------------------------------------------- #


class ToolSourceUnknownError(Exception):
    """Raised when resolve_tool_source cannot identify a tool name.

    Fail-closed contract: no 'unknown' sentinel ToolSource, no native fallback.
    Raised when the name is neither in _REGISTRY nor matched by the migration
    heuristic (mcp_/skill_ prefix).
    """


# ---- Canonical static tool identities ---------------------------------- #

_CANONICAL_TOOL_IDENTITIES: dict[str, tuple[Literal["native", "mcp", "a2a", "skill"], str]] = {
    # native / browser (12 tools, from langchain_tools.py:287+)
    "browser_view": ("native", "browser"),
    "browser_navigate": ("native", "browser"),
    "browser_click": ("native", "browser"),
    "browser_input": ("native", "browser"),
    "browser_move_mouse": ("native", "browser"),
    "browser_press_key": ("native", "browser"),
    "browser_select_option": ("native", "browser"),
    "browser_scroll_up": ("native", "browser"),
    "browser_scroll_down": ("native", "browser"),
    "browser_console_exec": ("native", "browser"),
    "browser_console_view": ("native", "browser"),
    "browser_restart": ("native", "browser"),
    # native / shell (5 tools, from langchain_tools.py:159+)
    "shell_execute": ("native", "shell"),
    "shell_read_output": ("native", "shell"),
    "shell_wait_process": ("native", "shell"),
    "shell_write_input": ("native", "shell"),
    "shell_kill_process": ("native", "shell"),
    # native / file (7 tools: 6 from _make_file_tools + file_view from _make_file_view_tools)
    "file_read": ("native", "file"),
    "file_write": ("native", "file"),
    "file_str_replace": ("native", "file"),
    "file_find_in_content": ("native", "file"),
    "file_find_by_name": ("native", "file"),
    "file_list": ("native", "file"),
    "file_view": ("native", "file"),
    # native / message (2 tools, from langchain_tools.py:60+)
    "message_notify_user": ("native", "message"),
    "message_ask_user": ("native", "message"),
    # native / search (1 tool, from langchain_tools.py:387+)
    "search_web": ("native", "search"),
    # native / memory (2 tools, from memory_tools.py:26 create_memory_tools)
    "memory_search": ("native", "memory"),
    "memory_get": ("native", "memory"),
    # a2a (2 tools, hardcoded in langchain_a2a.py:16-48)
    "get_remote_agent_cards": ("a2a", "a2a"),
    "call_remote_agent": ("a2a", "a2a"),
    # mcp discovery (2 tools, hardcoded in langchain_mcp_discovery.py)
    "list_mcp_tools": ("mcp", "mcp discovery"),
    "get_mcp_tool": ("mcp", "mcp discovery"),
    # skill creator (3 tools, hardcoded in langchain_skill_tools.py:35+)
    "brainstorm_skill": ("skill", "skill creator"),
    "generate_skill": ("skill", "skill creator"),
    "install_skill": ("skill", "skill creator"),
    # skill guide (1 tool, hardcoded in langchain_skill_tools.py:147+)
    "get_skill_guide": ("skill", "skill guide"),
}
# Total: 29 native + 2 a2a + 2 mcp discovery + 3 skill creator + 1 skill guide = 37


# ---- Module-global registry -------------------------------------------- #

_REGISTRY: dict[str, ToolSource] = {}


def _bootstrap_registry() -> None:
    """Seed _REGISTRY with canonical static tool identities at module import.

    Runs BEFORE any factory call so resolve_tool_source() works for all
    well-known tool names immediately after `import tool_source_resolver`.

    This lets SectionRegistry.__post_init__ validate against
    _FIXTURE_CTX.bound_tool_names at AgentTaskRunner.__init__ time —
    preserving the existing startup fail-fast semantics.

    Factories still call annotate_and_register_tool_source at construction
    time. For names already in the bootstrap seed, that call is a same-value
    idempotent no-op (see Task 3). If a factory disagrees with the seed,
    RegistryConflictError is raised — fail-fast on drift.
    """
    for name, (source, category) in _CANONICAL_TOOL_IDENTITIES.items():
        _REGISTRY[name] = ToolSource(
            source=source, category=category, canonical_name=name,
        )


_bootstrap_registry()


# ---- Helper ------------------------------------------------------------ #


class RegistryConflictError(Exception):
    """Raised when a tool_name is already registered with a DIFFERENT ToolSource.

    Same name + same value is a no-op (idempotent). Same name + different value
    is a bug — either a factory typo or genuine naming collision between
    heterogeneous tool providers.
    """

    def __init__(self, tool_name: str, existing: ToolSource, new: ToolSource):
        self.tool_name = tool_name
        self.existing = existing
        self.new = new
        super().__init__(
            f"Tool {tool_name!r} already registered as {existing!r}; "
            f"refusing to overwrite with {new!r}"
        )


def annotate_and_register_tool_source(
    tool,  # langchain_core.tools.BaseTool — avoid top-level import to prevent cycles
    *,
    source: Literal["native", "mcp", "a2a", "skill"],
    category: str,
):
    """Single source of truth: annotate tool.metadata AND register in _REGISTRY.

    This is the ONLY function factories should use. Do NOT write
    `tool.metadata["_actus_source"]` or `_REGISTRY[name]` directly.

    Rules:
    - Same (name, source, category): idempotent no-op — safe across factory
      rebuilds and bootstrap overlap. Metadata still gets written because
      bootstrap only populates _REGISTRY, not tool.metadata on the live
      tool object.
    - Same name, different (source, category): raise RegistryConflictError
      BEFORE writing any metadata. The tool passed in must remain
      semantically unchanged — never leave the caller holding a tool whose
      metadata disagrees with _REGISTRY.

    Returns the tool for chaining.
    """
    ts = ToolSource(source=source, category=category, canonical_name=tool.name)

    # Conflict check FIRST — do not mutate tool.metadata on the conflict path.
    existing = _REGISTRY.get(tool.name)
    if existing is not None and existing != ts:
        raise RegistryConflictError(tool.name, existing, ts)

    # Conflict check passed. Either idempotent no-op (existing == ts) or
    # new registration (existing is None). Either way, metadata write is
    # safe and matches _REGISTRY.
    tool.metadata = (tool.metadata or {}) | {"_actus_source": ts}
    if existing is None:
        _REGISTRY[tool.name] = ts
    return tool


# ---- Heuristic (migration-period fallback) ----------------------------- #


def _heuristic_resolve(tool_name: str) -> ToolSource:
    """Migration-period heuristic for dynamically-named extension tools only.

    All static identities are populated by _bootstrap_registry() and reach
    _REGISTRY directly. This function only handles dynamic mcp_/skill_ prefix
    patterns that cannot be enumerated at bootstrap time. Truly unknown names
    raise ToolSourceUnknownError.

    Strict fail-closed: no native prefix fallback, no 'unknown' sentinel.
    """
    if tool_name.startswith("mcp_"):
        return ToolSource(source="mcp", category="mcp", canonical_name=tool_name)
    if tool_name.startswith("skill_"):
        return ToolSource(source="skill", category="skill", canonical_name=tool_name)
    raise ToolSourceUnknownError(
        f"Tool {tool_name!r} is not in _REGISTRY (bootstrap or factory) and "
        f"does not match any migration-period heuristic (mcp_/skill_ prefix). "
        f"Either (a) the factory forgot to call annotate_and_register_tool_source(), "
        f"(b) this is a new static tool that needs adding to _CANONICAL_TOOL_IDENTITIES, "
        f"or (c) the test is using an unknown fabricated name."
    )


# ---- Public API -------------------------------------------------------- #


def resolve_tool_source(tool_name: str) -> ToolSource:
    """CS1 contract: name-based lookup in module _REGISTRY.

    Identity only. Answers "what kind of tool is this name?".
    Does NOT answer "is this tool currently bound / available / invokable".

    Raises ToolSourceUnknownError if name is neither in registry nor resolvable
    via heuristic fallback.
    """
    if tool_name in _REGISTRY:
        return _REGISTRY[tool_name]
    return _heuristic_resolve(tool_name)


def resolve_tool_source_from_tool(tool) -> ToolSource:
    """Convenience: read _actus_source metadata directly from tool object.

    Prefer this ONLY when the caller already has a BaseTool that has been
    through a factory. Factory calls annotate_and_register_tool_source which
    writes both metadata AND _REGISTRY.

    Caveat: _bootstrap_registry() populates _REGISTRY but does NOT write tool
    metadata (bootstrap has no tool objects, only names). So for a canonical
    name whose factory has not yet run in this process, resolve_tool_source(name)
    works (bootstrap-covered) but resolve_tool_source_from_tool(tool) raises.

    Rule of thumb: when unsure, use name-based resolve_tool_source(tool.name).

    Raises ToolSourceUnknownError if metadata missing.
    """
    source = (tool.metadata or {}).get("_actus_source")
    if source is None:
        raise ToolSourceUnknownError(tool.name)
    return source
