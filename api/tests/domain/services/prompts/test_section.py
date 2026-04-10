"""B5 C1: Section / SectionRegistry / PromptMode / fixtures tests."""
from __future__ import annotations

import pytest

from app.domain.services.prompts.errors import (
    SectionValidationError,
    ToolBindingInvariantError,
)
from app.domain.services.prompts.section import (
    MINIMAL_MODE_ALLOWLIST,
    PromptBundle,
    PromptMode,
    RenderContext,
    Section,
    SectionOutput,
    SectionRegistry,
    _FIXTURE_CTX,
)


# ---- Helper to build a trivial section --------------------------------- #


def _make_section(
    section_id: str,
    *,
    text: str = "stub",
    priority: int = 5,
    cacheable: bool = False,
    dynamic: bool = False,
) -> Section:
    return Section(
        id=section_id,
        priority=priority,
        cacheable=cacheable,
        dynamic=dynamic,
        render=lambda ctx, _t=text: SectionOutput(text=_t),
    )


def _make_registry(*sections: Section, name: str = "test_registry") -> SectionRegistry:
    return SectionRegistry(sections=list(sections), name=name)


# ---- PromptMode --------------------------------------------------------- #


def test_prompt_mode_values() -> None:
    assert PromptMode.FULL.value == "full"
    assert PromptMode.MINIMAL.value == "minimal"
    assert PromptMode.NONE.value == "none"


# ---- SectionOutput ------------------------------------------------------ #


def test_section_output_default_metadata_empty_dict() -> None:
    out = SectionOutput(text="hello")
    assert out.metadata == {}


def test_section_output_with_metadata() -> None:
    out = SectionOutput(text="hi", metadata={"foo": "bar"})
    assert out.metadata == {"foo": "bar"}


def test_section_output_text_can_be_none() -> None:
    out = SectionOutput(text=None)
    assert out.text is None


# ---- RenderContext ------------------------------------------------------ #


def test_render_context_minimal_construction() -> None:
    ctx = RenderContext(lang="zh")
    assert ctx.lang == "zh"
    assert ctx.provider == "openai"
    assert ctx.bound_tool_names == frozenset()
    assert ctx.skill_names_in_context == ()
    assert ctx.has_file_view is False


def test_render_context_full_construction() -> None:
    ctx = RenderContext(
        lang="en",
        provider="anthropic",
        bound_tool_names=frozenset({"file_view", "skill_foo_bar"}),
        skill_names_in_context=("foo",),
        has_file_view=True,
    )
    assert ctx.lang == "en"
    assert ctx.provider == "anthropic"
    assert "skill_foo_bar" in ctx.bound_tool_names
    assert ctx.has_file_view is True


def test_render_context_is_frozen() -> None:
    """RenderContext must be immutable so sections cannot mutate shared state."""
    ctx = RenderContext(lang="zh")
    with pytest.raises(Exception):  # FrozenInstanceError
        ctx.lang = "en"  # type: ignore[misc]


def test_render_context_collection_fields_are_immutable() -> None:
    """Collection fields use frozenset/tuple — no .add()/.append() allowed."""
    ctx = RenderContext(lang="zh", bound_tool_names=frozenset({"x"}))
    assert isinstance(ctx.bound_tool_names, frozenset)
    assert isinstance(ctx.skill_names_in_context, tuple)
    assert isinstance(ctx.conversation_summaries, tuple)
    assert isinstance(ctx.tool_categories, frozenset)


def test_section_cannot_mutate_fixture_ctx() -> None:
    """A section that tries to mutate _FIXTURE_CTX must not affect later sections.

    Since RenderContext is frozen and uses immutable collections, any
    mutation attempt raises immediately. This test documents the contract.
    """
    def evil_render(ctx: RenderContext) -> SectionOutput:
        # Attempt to mutate — should raise
        with pytest.raises(Exception):
            ctx.bound_tool_names.add("skill_evil")  # type: ignore[attr-defined]
        return SectionOutput(text="harmless")

    s = Section(
        id="test_immutable",
        priority=10,
        cacheable=False,
        dynamic=False,
        render=evil_render,
    )
    # Construction should not raise — the assertion inside evil_render passes
    _make_registry(s)


# ---- Section ------------------------------------------------------------ #


def test_section_dataclass_construction() -> None:
    section = _make_section("identity", priority=10, cacheable=True)
    assert section.id == "identity"
    assert section.priority == 10
    assert section.cacheable is True
    assert section.dynamic is False
    assert section.estimate_tokens is None  # default


def test_section_render_returns_section_output() -> None:
    section = _make_section("test", text="hello world")
    output = section.render(RenderContext(lang="zh"))
    assert isinstance(output, SectionOutput)
    assert output.text == "hello world"


# ---- SectionRegistry: filter -------------------------------------------- #


def test_registry_filter_full_returns_all() -> None:
    s1 = _make_section("identity", priority=10)
    s2 = _make_section("custom_low_pri", priority=2)
    registry = _make_registry(s1, s2)
    assert [s.id for s in registry.filter(PromptMode.FULL)] == [
        "identity",
        "custom_low_pri",
    ]


def test_registry_filter_minimal_uses_allowlist() -> None:
    """MINIMAL mode keeps only sections whose id is in MINIMAL_MODE_ALLOWLIST."""
    s_in = _make_section("identity", priority=10)  # in allowlist
    s_out = _make_section("custom_section", priority=10)  # not in allowlist
    registry = _make_registry(s_in, s_out)
    filtered = registry.filter(PromptMode.MINIMAL)
    assert [s.id for s in filtered] == ["identity"]


def test_minimal_allowlist_contents() -> None:
    """Verify the canonical allowlist is what's documented."""
    assert "identity" in MINIMAL_MODE_ALLOWLIST
    assert "behavior_core" in MINIMAL_MODE_ALLOWLIST
    assert "tools_guide_stable" in MINIMAL_MODE_ALLOWLIST


def test_registry_filter_none_keeps_only_highest_priority() -> None:
    """NONE mode uses NONE_MODE_PRIORITY_MIN (9), strictly higher than
    CRITICAL_PRIORITY_MIN (8). A section with priority=8 is protected from
    budget dropping but excluded from NONE mode — see section.py docstring."""
    from app.domain.services.prompts.section import NONE_MODE_PRIORITY_MIN

    assert NONE_MODE_PRIORITY_MIN == 9  # documented contract

    s_high = _make_section("must_keep", priority=10)
    s_borderline = _make_section("borderline", priority=9)
    s_critical_but_excluded = _make_section("critical_8", priority=8)
    s_low = _make_section("low", priority=2)
    registry = _make_registry(s_high, s_borderline, s_critical_but_excluded, s_low)
    filtered = registry.filter(PromptMode.NONE)
    assert {s.id for s in filtered} == {"must_keep", "borderline"}
    assert "critical_8" not in {s.id for s in filtered}, (
        "priority=8 sections are 'critical' for budget purposes but NOT for NONE mode"
    )


def test_none_and_critical_thresholds_are_intentionally_different() -> None:
    """Document the divergence: NONE_MODE_PRIORITY_MIN > CRITICAL_PRIORITY_MIN."""
    from app.domain.services.prompts.assembler import CRITICAL_PRIORITY_MIN
    from app.domain.services.prompts.section import NONE_MODE_PRIORITY_MIN

    assert NONE_MODE_PRIORITY_MIN > CRITICAL_PRIORITY_MIN


# ---- SectionRegistry: __post_init__ validation ------------------------- #


def test_registry_construction_succeeds_for_clean_section() -> None:
    """A section that returns harmless text validates fine."""
    s = _make_section("identity", text="You are an agent.")
    # Should not raise
    _make_registry(s)


def test_registry_construction_passes_when_section_returns_none() -> None:
    """Sections returning text=None are skipped by the validator."""
    s = Section(
        id="empty",
        priority=5,
        cacheable=False,
        dynamic=True,
        render=lambda ctx: SectionOutput(text=None),
    )
    _make_registry(s)  # should not raise


def test_registry_construction_fails_on_section_render_exception() -> None:
    """If a section's render raises, the registry refuses to construct."""

    def boom(ctx: RenderContext) -> SectionOutput:
        raise RuntimeError("intentional test failure")

    s = Section(
        id="broken",
        priority=5,
        cacheable=False,
        dynamic=False,
        render=boom,
    )
    with pytest.raises(SectionValidationError) as exc:
        _make_registry(s)
    assert "broken" in str(exc.value)
    assert "intentional test failure" in str(exc.value)


def test_registry_construction_fails_on_dangling_skill_ref() -> None:
    """If a section emits a skill name not in the fixture's bound_tool_names,
    the registry refuses to construct."""
    s = _make_section("bad_section", text="Use `skill_imaginary_tool` to do X.")
    with pytest.raises(ToolBindingInvariantError) as exc:
        _make_registry(s)
    assert "skill_imaginary_tool" in str(exc.value)
    assert "bad_section" in str(exc.value)


def test_registry_construction_passes_for_fixture_known_skill() -> None:
    """Section emits a skill name that IS in _FIXTURE_CTX.bound_tool_names → OK."""
    # _FIXTURE_CTX contains 'skill_example_action'
    assert "skill_example_action" in _FIXTURE_CTX.bound_tool_names
    s = _make_section("ok_section", text="Try `skill_example_action`.")
    _make_registry(s)  # should not raise


def test_registry_construction_passes_for_double_underscore_fixture_skill() -> None:
    """Edge case: skill_foo__bar_tool (double underscore) is in fixture."""
    assert "skill_foo__bar_tool" in _FIXTURE_CTX.bound_tool_names
    s = _make_section("dunder", text="Use `skill_foo__bar_tool` for X.")
    _make_registry(s)


# ---- PromptBundle ------------------------------------------------------- #


def test_prompt_bundle_construction() -> None:
    s = _make_section("identity")
    executor = _make_registry(s, name="zh_executor")
    planner = _make_registry(s, name="zh_planner")
    updater = _make_registry(s, name="zh_updater")

    bundle = PromptBundle(
        lang="zh",
        executor=executor,
        planner=planner,
        updater=updater,
    )
    assert bundle.lang == "zh"
    assert bundle.executor.name == "zh_executor"
    assert bundle.planner.name == "zh_planner"
    assert bundle.updater.name == "zh_updater"


# ---- _FIXTURE_CTX sanity ----------------------------------------------- #


def test_fixture_ctx_has_representative_tools() -> None:
    """Fixture should include tools from every category section authors might
    reference: native, memory, skill, mcp, a2a."""
    bound = _FIXTURE_CTX.bound_tool_names
    assert "file_view" in bound  # native
    assert "memory_search" in bound  # memory
    assert "skill_example_action" in bound  # skill
    assert "mcp_amap_maps_weather" in bound  # mcp
    assert "get_remote_agent_cards" in bound  # a2a


def test_fixture_ctx_includes_double_underscore_edge_case() -> None:
    """Catches future regex regressions on the _normalize_function_part bug."""
    assert "skill_foo__bar_tool" in _FIXTURE_CTX.bound_tool_names


def test_fixture_ctx_flags_set() -> None:
    """has_* flags should be enabled so sections that gate on them get exercised."""
    assert _FIXTURE_CTX.has_file_view is True
    assert _FIXTURE_CTX.has_memory_tools is True
    assert _FIXTURE_CTX.mcp_active is True
    assert _FIXTURE_CTX.a2a_active is True
