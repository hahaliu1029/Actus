"""Section schema for B5 prompt modularization.

B5 C1: defines the data structures used by ``PromptAssembler`` to compose
the system prompt from independent sections.

Architecture:
    ``Section`` is a frozen-ish dataclass holding rendering metadata + a
    ``render(ctx)`` callable. Sections are grouped into ``SectionRegistry``
    instances (one per language × scenario, e.g. ``zh_executor``).

    ``SectionRegistry.__post_init__`` runs startup-time validation: each
    section is rendered against a canonical fixture context and the output
    is scanned for dangling skill tool references via
    ``invariants._assert_no_dangling_skill_tool_refs``. This catches
    section authoring bugs at startup, not at first user request.

    ``MINIMAL_MODE_ALLOWLIST`` is the central declaration of which sections
    survive in ``PromptMode.MINIMAL`` (sub-agent context). Sections do NOT
    declare ``include_in_minimal`` themselves — the allowlist is the single
    source of truth.

C1 only ships the data structures. Concrete sections, bundles, and the
PromptAssembler consumer come in C2/C3/C4/C5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

from app.domain.models.memory_recall import RecalledMemory
from app.domain.services.prompts.errors import SectionValidationError
from app.domain.services.prompts.invariants import _assert_no_dangling_skill_tool_refs
from app.domain.services.prompts.memory_snapshot import MemorySnapshot


# ---- Mode --------------------------------------------------------------- #


class PromptMode(Enum):
    """Assembly mode controlling which sections are included.

    - FULL: production main agent — all sections
    - MINIMAL: in-process sub-agent — sections in MINIMAL_MODE_ALLOWLIST only
    - NONE: extreme degradation — only critical-priority sections (>= 9)
    """

    FULL = "full"
    MINIMAL = "minimal"
    NONE = "none"


# ---- RenderContext ------------------------------------------------------ #


@dataclass(frozen=True)
class RenderContext:
    """Per-step context passed to each Section.render() call.

    Built once per node invocation by ``build_render_context`` (C5a) from
    ``state``, ``config``, and ``agent_config``. Sections must read from
    this dataclass — never reach into ``state`` directly.

    **Immutability**: ``frozen=True`` and all collection fields are immutable
    (``frozenset``/``tuple``) so a buggy section cannot mutate the context
    and leak state to subsequent sections in the same assemble() call.
    """

    lang: Literal["zh", "en"]
    provider: Literal["openai", "anthropic"] = "openai"
    step_description: str | None = None
    message: str | None = None
    attachments_text: str | None = None
    skill_context: str | None = None
    """The pre-built skill markdown blob from ``_build_runtime_system_context``.

    Sections can reference this verbatim, but tools_guide_dynamic must read
    tool names from ``bound_tool_names`` (not from this string).
    """
    skill_names_in_context: tuple[str, ...] = ()
    """Authoritative skill id list — for telemetry only, NOT for invariants."""
    bound_tool_names: frozenset[str] = field(default_factory=frozenset)
    """Authoritative LangChain tool name set — single source of truth for
    ``tools_guide_dynamic`` rendering and the ``_assert_no_dangling_skill_tool_refs``
    invariant scan.
    """
    conversation_summaries: tuple[str, ...] = ()
    has_file_view: bool = False
    has_memory_tools: bool = False
    has_vision: bool = True
    has_pdf: bool = True
    tool_categories: frozenset[str] = field(default_factory=frozenset)
    mcp_active: bool = False
    a2a_active: bool = False
    memory_snapshot: MemorySnapshot | None = None
    """M2-PR3: bucketed memory chunks for prompt-time injection.

    Built upstream (repo query happens there, see
    ``memory_snapshot.build_memory_snapshot``) and attached to the
    context before any section renders. When ``None``, the three memory
    sections (``memory_rules``, ``memory_user_profile``,
    ``memory_fact_index``) emit nothing — non-memory-aware call sites
    (legacy executor path, tests without memory fixtures) therefore
    remain unaffected.
    """
    team_members: tuple[tuple[str, str], ...] | None = None
    """S4 PR-4: agent-team ``(role, description)`` pairs for the planner
    teaching section. Loaded best-effort + STRUCTURAL-ONLY in
    ``planner_node`` (no §13 capability validation — that lives in
    ``_run_parallel_backend``). When ``None`` (flag-OFF / no team / load
    failure), the teaching section emits nothing, keeping all non-team
    call sites byte-identical (INV-0).
    """
    recalled_memory: RecalledMemory | None = None
    """B8: planner query-time 召回结果，仅 planner registry 的
    ``recalled_memory`` section 消费。由 recall provider 在两个 planner
    入口解析后经 ``build_render_context`` kwarg 传入——不进 graph state。
    ``None``（默认）= section 渲染 text=None，所有非召回调用点
    byte-identical（INV-B8-OFF）。executor/updater 永不填充此字段。
    """
    parallel_dispatch_allowed: bool = True
    """[child-pwu fix] Whether the invoking runner can actually dispatch
    ``parallel_work_units`` (True only when the coordinator subgraph is
    wired into configurable — root runners with real coord_deps).
    Coordinator children / legacy runners get False so the
    ``parallel_work_units_teaching`` section stops teaching a schema whose
    emissions would only be stripped at the parse boundary
    (``main_graph._parallel_dispatch_allowed``). Default True keeps every
    non-threaded call site (executor ctx, fixtures, tests) byte-identical.
    """
    sandbox_tools_enabled: bool = True
    """SPM Task 26: whether sandbox-face tools (file / shell / browser /
    skill-creation) are provisioned for this deployment.

    A **deployment constant** derived once from settings
    (``sandbox_provision_mode != "off"``) — NOT per-step state — so the
    cacheable sections that read it (``identity`` / ``behavior_core`` /
    ``output_format``) stay cacheable (DD-7). ``True`` (always / on_demand)
    selects the current sandbox-teaching
    prose, keeping every existing render byte-identical (INV-SPM-2). ``False``
    (off) selects the sandbox-free variant that removes shell/browser/takeover/
    skill-creation teaching and rewords the MCP-over-browser/terminal
    comparison. Default ``True`` keeps every non-threaded construction point
    (fixtures, tests) byte-identical.
    """


# ---- SectionOutput ------------------------------------------------------ #


@dataclass(frozen=True)
class SectionOutput:
    """Return value of Section.render().

    ``text=None`` (or empty string) means "this section does not output for
    the given context" — the assembler skips it without affecting other
    sections.

    ``metadata`` is for telemetry/observability only. It does NOT participate
    in the invariant scan (which operates on the rendered text directly).

    Frozen post-audit (MEDIUM #4) so render functions cannot accidentally
    mutate a returned output after the fact. ``metadata`` is still a
    mutable dict internally (Python's frozen dataclass is shallow);
    callers should treat it as read-only.
    """

    text: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- Section ------------------------------------------------------------ #


@dataclass(frozen=True)
class Section:
    """A composable prompt section.

    ``render`` is a pure function from RenderContext → SectionOutput.
    ``estimate_tokens`` is optional — if None, ``PromptAssembler`` uses its
    own configured ``TokenEstimator``. Sections only override this for very
    specific cases (e.g. fast-path counting).

    Frozen post-audit (MEDIUM #4) so module-level singletons can't be
    reassigned at runtime. The ``render`` and ``estimate_tokens``
    callables are themselves immutable (function references).
    """

    id: str
    priority: int
    """1-10 scale. >= ``CRITICAL_PRIORITY_MIN`` (8) sections are protected
    from budget-driven dropping. Lower priority is more droppable."""

    cacheable: bool
    """B5 metadata-only field. C1-C7 do not consume this. B5.5 (caching
    follow-up) will use it to decide where to insert the cache_control
    boundary marker."""

    dynamic: bool
    """True if the rendered output varies per step. Sections with
    dynamic=True are excluded from the cacheable prefix in B5.5."""

    render: Callable[[RenderContext], SectionOutput]
    estimate_tokens: Callable[[str], int] | None = None
    """Optional override of the token estimator. None means "use the
    PromptAssembler's default TokenEstimator instance"."""

    max_tokens: int | None = None
    """M2-PR0: per-section token cap used by the section's own render()
    for internal truncation.

    Contract:
    - ``None`` = no per-section cap; the section relies only on the
      global ``PromptAssembler`` budget + priority-DESC drop loop.
    - ``int`` = the section's render() is expected to truncate its
      own output at roughly this many tokens (sorting rules defined
      inside the section). The assembler does NOT enforce it — it
      only trusts the rendered text as-is.

    Used by M2 memory sections (``memory_user_profile``,
    ``memory_rules``, ``memory_fact_index``) to cap each category's
    contribution so the section can't dominate the global budget
    even when fact index returns hundreds of rows.

    Note: we deliberately do NOT add an ``allocate()`` method to
    ``PromptAssembler`` — the existing priority-DESC drop loop in
    ``assembler.py:140-151`` handles cross-section budget pressure.
    ``max_tokens`` is strictly the section-internal cap.
    """


# ---- MINIMAL_MODE_ALLOWLIST --------------------------------------------- #


MINIMAL_MODE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "identity",
        "behavior_core",
        "tools_guide_stable",
        "planner_identity",
    }
)
"""Sections that survive in PromptMode.MINIMAL (sub-agent context).

External allowlist instead of a per-section ``include_in_minimal`` field
so future minimal-mode tuning happens in one place, not by scanning every
section file. See B5 design doc Issue 5 (eng review)."""


NONE_MODE_PRIORITY_MIN = 9
"""Priority threshold for ``PromptMode.NONE`` (extreme degradation mode).

Intentionally STRICTER than ``CRITICAL_PRIORITY_MIN = 8`` (in assembler.py):
- ``CRITICAL_PRIORITY_MIN = 8``: under budget pressure in FULL mode, sections
  with priority >= 8 are protected from dropping
- ``NONE_MODE_PRIORITY_MIN = 9``: in NONE mode (extreme degradation, e.g.
  catastrophic budget exhaustion), only the most-critical sections survive

A section with priority=8 is "important enough to keep under normal budget
pressure" but "not critical enough to keep when the system is in extreme
degradation". Section authors should reserve priority 9-10 for sections
the agent **cannot function without**. As of C2:
- priority=10: identity, behavior_core (agent identity + behavior rules)
- priority=9: output_format (JSON return contract — without it the LLM
  output can't be parsed by ``SummarizerOutput``)

All three survive NONE mode. priority=8 and below sections (skill_context,
conversation_summaries, sandbox_state in C3) get dropped in NONE."""


# ---- SectionRegistry ---------------------------------------------------- #


@dataclass(frozen=True)
class SectionRegistry:
    """Ordered tuple of sections for one (language × scenario) bundle.

    The ``sections`` tuple order IS the assembly order. Sections are NOT
    reordered by priority — priority only governs budget-driven dropping.

    ``__post_init__`` runs **first-use validation**: each section is
    rendered against ``_FIXTURE_CTX`` and the output is scanned for
    dangling skill tool references. The validation fires when the
    registry is constructed — and because ``prompts/__init__.py`` uses
    a deferred import in ``get_prompt_section_bundle``, construction
    happens on the first call to that function (not at application
    startup). See the ``prompts`` package docstring for the rationale.

    Frozen because instances are module-level singletons (see
    ``bundles/zh.py``, ``bundles/en.py``) and must not be mutated after
    construction. Post-audit (MEDIUM #4) ``sections`` is a ``tuple`` so
    the underlying sequence is also immutable — the earlier ``list``
    allowed in-place mutation even on a frozen registry.
    """

    sections: tuple[Section, ...]
    name: str

    def __post_init__(self) -> None:
        """Coerce ``sections`` to a tuple, then first-use-validate each one.

        Coercion: callers may pass a list for ergonomics (test helpers
        often do); we convert to tuple so the stored field is actually
        immutable. ``object.__setattr__`` is required because the
        dataclass is frozen.

        Validation: each section is rendered against ``_FIXTURE_CTX`` and
        the output is scanned for dangling skill tool references. See
        class docstring for rationale on first-use vs startup timing.
        """
        if not isinstance(self.sections, tuple):
            object.__setattr__(self, "sections", tuple(self.sections))
        for section in self.sections:
            try:
                output = section.render(_FIXTURE_CTX)
            except Exception as exc:
                raise SectionValidationError(
                    f"Section '{section.id}' (registry '{self.name}') "
                    f"render() failed on fixture ctx: {exc}"
                ) from exc
            if output and output.text:
                _assert_no_dangling_skill_tool_refs(
                    output.text,
                    _FIXTURE_CTX.bound_tool_names,
                    section_id=section.id,
                )

    def filter(self, mode: PromptMode) -> list[Section]:
        """Return sections retained for the given assembly mode."""
        if mode == PromptMode.FULL:
            return list(self.sections)
        if mode == PromptMode.MINIMAL:
            return [s for s in self.sections if s.id in MINIMAL_MODE_ALLOWLIST]
        # NONE: only highest-priority sections (stricter than CRITICAL_PRIORITY_MIN)
        return [s for s in self.sections if s.priority >= NONE_MODE_PRIORITY_MIN]


# ---- PromptBundle ------------------------------------------------------- #


@dataclass(frozen=True)
class PromptBundle:
    """Group of registries for one language across multiple scenarios.

    A bundle holds the executor / planner / updater registries for a given
    language. The DI layer (service_dependencies in C5a) constructs one
    bundle per language and selects at runtime via ``state["language"]``.

    Frozen because ``ZH_BUNDLE`` and ``EN_BUNDLE`` are module-level
    singletons returned by ``get_prompt_section_bundle(lang)``. Any mutation
    on a returned instance would silently affect all subsequent callers.
    """

    lang: Literal["zh", "en"]
    executor: SectionRegistry
    planner: SectionRegistry
    updater: SectionRegistry


# ---- _FIXTURE_CTX (canonical startup-validation context) ---------------- #


_FIXTURE_CTX = RenderContext(
    lang="zh",
    provider="openai",
    step_description="fixture step",
    message="fixture user message",
    attachments_text="",
    skill_context="## Active Skills\n- example skill (fixture)",
    skill_names_in_context=("example-skill",),
    bound_tool_names=frozenset(
        {
            # Native tools (representative subset; sections shouldn't
            # hard-code these but they're allowed to mention them in prose)
            "file_view",
            "file_read",
            "file_write",
            "shell_execute",
            "browser_navigate",
            "message_notify_user",
            "message_ask_user",
            # Memory
            "memory_search",
            "memory_get",
            "memory_save",
            # Skill tools — both standard and double-underscore edge case
            "skill_example_action",
            "skill_foo__bar_tool",
            # MCP
            "mcp_amap_maps_weather",
            # A2A
            "get_remote_agent_cards",
            "call_remote_agent",
        }
    ),
    conversation_summaries=(),
    has_file_view=True,
    has_memory_tools=True,
    has_vision=True,
    has_pdf=True,
    tool_categories=frozenset({"file", "shell", "browser", "memory", "skill", "mcp", "a2a"}),
    mcp_active=True,
    a2a_active=True,
)
"""Canonical fixture for SectionRegistry startup validation.

Includes representative tool names from every category, plus the double-
underscore edge case (``skill_foo__bar_tool``) so the regex pattern in
``invariants.py`` is exercised at startup.

Sections rendered against this fixture must produce output where every
``skill_*`` token is in ``bound_tool_names``. If a section author hard-codes
a fake skill name like ``skill_my_demo``, the registry will refuse to
construct at startup.
"""
