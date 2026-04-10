"""B5 C1: PromptAssembler tests."""
from __future__ import annotations

from typing import Any

import pytest

from app.domain.services.prompts.assembler import (
    AssembleResult,
    CRITICAL_PRIORITY_MIN,
    PromptAssembler,
)
from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.section import (
    PromptMode,
    RenderContext,
    Section,
    SectionOutput,
    SectionRegistry,
)
from app.domain.services.graphs.token_estimator import TokenEstimator


# ---- Helpers ------------------------------------------------------------ #


class _FakeTelemetry:
    """Minimal PromptTelemetryPort fake recording all calls."""

    def __init__(self) -> None:
        self.assembly_calls: list[dict[str, Any]] = []

    def record_assembly(self, **kwargs: Any) -> None:
        self.assembly_calls.append(kwargs)

    def record_llm_invocation(self, **kwargs: Any) -> None:  # pragma: no cover - unused
        pass

    def record_lc_tools_degradation(self, *, reason: str) -> None:  # pragma: no cover
        pass


def _section(
    section_id: str,
    *,
    text: str,
    priority: int,
    metadata: dict[str, Any] | None = None,
    cacheable: bool = False,
    dynamic: bool = False,
) -> Section:
    return Section(
        id=section_id,
        priority=priority,
        cacheable=cacheable,
        dynamic=dynamic,
        render=lambda ctx, _t=text, _m=metadata or {}: SectionOutput(
            text=_t, metadata=dict(_m)
        ),
    )


def _ctx(lang: str = "zh") -> RenderContext:
    return RenderContext(lang=lang)  # type: ignore[arg-type]


def _make_assembler(
    *,
    max_tokens: int = 10_000,
    telemetry: _FakeTelemetry | None = None,
) -> PromptAssembler:
    budget = SystemPromptBudget(max_tokens=max_tokens)
    estimator = TokenEstimator(strategy="hybrid")
    return PromptAssembler(
        budget=budget,
        token_estimator=estimator,
        telemetry=telemetry,
    )


# ---- Basic assembly ---------------------------------------------------- #


def test_assemble_single_section() -> None:
    s = _section("identity", text="You are an agent.", priority=10)
    registry = SectionRegistry(sections=[s], name="test")
    assembler = _make_assembler()

    result = assembler.assemble(registry, _ctx(), PromptMode.FULL)
    assert isinstance(result, AssembleResult)
    assert result.text == "You are an agent."
    assert result.sections_included == ["identity"]
    assert result.sections_dropped == []
    assert result.tokens_used > 0


def test_assemble_multiple_sections_in_declaration_order() -> None:
    """Output order MUST match registry declaration order, NOT priority order."""
    s_low = _section("z_low_pri", text="LOW", priority=5)
    s_high = _section("a_high_pri", text="HIGH", priority=10)
    s_mid = _section("m_mid_pri", text="MID", priority=8)
    registry = SectionRegistry(sections=[s_low, s_high, s_mid], name="test")
    assembler = _make_assembler()

    result = assembler.assemble(registry, _ctx())
    # Declaration order = z_low_pri, a_high_pri, m_mid_pri
    assert result.sections_included == ["z_low_pri", "a_high_pri", "m_mid_pri"]
    assert result.text == "LOW\n\nHIGH\n\nMID"


def test_assemble_skips_section_returning_none_text() -> None:
    s_keep = _section("keep", text="kept text", priority=10)
    s_skip = Section(
        id="skip",
        priority=10,
        cacheable=False,
        dynamic=True,
        render=lambda ctx: SectionOutput(text=None),
    )
    registry = SectionRegistry(sections=[s_keep, s_skip], name="test")
    result = _make_assembler().assemble(registry, _ctx())
    assert result.sections_included == ["keep"]
    assert "kept text" in result.text


def test_assemble_skips_section_returning_empty_string() -> None:
    s_keep = _section("keep", text="kept", priority=10)
    s_empty = _section("empty", text="", priority=10)
    registry = SectionRegistry(sections=[s_keep, s_empty], name="test")
    result = _make_assembler().assemble(registry, _ctx())
    assert "empty" not in result.sections_included


# ---- Mode filtering ---------------------------------------------------- #


def test_assemble_minimal_mode_filters_via_allowlist() -> None:
    """MINIMAL mode keeps only sections whose id is in MINIMAL_MODE_ALLOWLIST."""
    s_in = _section("identity", text="ID", priority=10)  # in allowlist
    s_out = _section("custom", text="OUT", priority=10)  # not in allowlist
    registry = SectionRegistry(sections=[s_in, s_out], name="test")
    result = _make_assembler().assemble(registry, _ctx(), PromptMode.MINIMAL)
    assert result.sections_included == ["identity"]
    assert "OUT" not in result.text


def test_assemble_none_mode_keeps_only_critical_priority() -> None:
    s_high = _section("crit", text="C", priority=9)
    s_low = _section("normal", text="N", priority=8)
    registry = SectionRegistry(sections=[s_high, s_low], name="test")
    result = _make_assembler().assemble(registry, _ctx(), PromptMode.NONE)
    assert result.sections_included == ["crit"]


# ---- Budget enforcement ------------------------------------------------ #


def test_assemble_drops_low_priority_when_over_budget() -> None:
    """Sections with priority < CRITICAL_PRIORITY_MIN get dropped first."""
    big_text = "x" * 4000  # ~1000 tokens (hybrid: ASCII = 0.25/char)
    s_critical = _section("crit", text="critical", priority=10)
    s_drop = _section("droppable", text=big_text, priority=5)
    s_keep_big = _section("keep_big", text=big_text, priority=9)  # critical
    registry = SectionRegistry(
        sections=[s_critical, s_drop, s_keep_big],
        name="test",
    )
    assembler = _make_assembler(max_tokens=1500)
    result = assembler.assemble(registry, _ctx())

    assert "crit" in result.sections_included
    assert "keep_big" in result.sections_included
    assert "droppable" in result.sections_dropped


def test_assemble_protects_critical_even_when_over_budget() -> None:
    """Critical priority (>= 8) sections are NEVER dropped, even if total exceeds budget."""
    big = "x" * 4000  # ~1000 tokens
    s_crit_a = _section("crit_a", text=big, priority=10)
    s_crit_b = _section("crit_b", text=big, priority=10)
    registry = SectionRegistry(sections=[s_crit_a, s_crit_b], name="test")
    assembler = _make_assembler(max_tokens=500)  # tiny budget

    result = assembler.assemble(registry, _ctx())
    # Both critical sections kept; tokens_used > budget
    assert "crit_a" in result.sections_included
    assert "crit_b" in result.sections_included
    assert result.sections_dropped == []
    assert result.tokens_used > 500


def test_critical_priority_min_constant_matches_default() -> None:
    """CRITICAL_PRIORITY_MIN module constant should match SystemPromptBudget default."""
    assert CRITICAL_PRIORITY_MIN == SystemPromptBudget(max_tokens=1).critical_priority_min


# ---- Metadata aggregation ---------------------------------------------- #


def test_metadata_list_fields_concatenate_across_sections() -> None:
    s_a = _section("a", text="A", priority=10, metadata={"refs": ["x", "y"]})
    s_b = _section("b", text="B", priority=10, metadata={"refs": ["z"]})
    registry = SectionRegistry(sections=[s_a, s_b], name="test")
    result = _make_assembler().assemble(registry, _ctx())
    assert result.metadata["refs"] == ["x", "y", "z"]


def test_metadata_scalar_fields_last_wins() -> None:
    s_a = _section("a", text="A", priority=10, metadata={"version": 1})
    s_b = _section("b", text="B", priority=10, metadata={"version": 2})
    registry = SectionRegistry(sections=[s_a, s_b], name="test")
    result = _make_assembler().assemble(registry, _ctx())
    # 'b' is rendered after 'a' in declaration order → wins
    assert result.metadata["version"] == 2


# ---- version_hash ----------------------------------------------------- #


def test_version_hash_is_stable_for_same_input() -> None:
    s = _section("identity", text="hello", priority=10)
    registry = SectionRegistry(sections=[s], name="test")
    assembler = _make_assembler()
    r1 = assembler.assemble(registry, _ctx())
    r2 = assembler.assemble(registry, _ctx())
    assert r1.version_hash == r2.version_hash


def test_version_hash_changes_when_text_changes() -> None:
    s_a = _section("identity", text="hello", priority=10)
    s_b = _section("identity", text="world", priority=10)
    r_a = _make_assembler().assemble(SectionRegistry(sections=[s_a], name="t"), _ctx())
    r_b = _make_assembler().assemble(SectionRegistry(sections=[s_b], name="t"), _ctx())
    assert r_a.version_hash != r_b.version_hash


def test_version_hash_is_16_chars() -> None:
    s = _section("identity", text="x", priority=10)
    result = _make_assembler().assemble(SectionRegistry(sections=[s], name="t"), _ctx())
    assert len(result.version_hash) == 16


def test_version_hash_includes_dropped_sections() -> None:
    """C1 review fix: two assemblies with the same final text but different
    drops must produce different version hashes (replay-compat detection)."""
    big = "x" * 4000  # ~1000 tokens
    s_keep = _section("identity", text="hello", priority=10)
    s_drop = _section("droppable", text=big, priority=5)

    # Tight budget: drops `droppable`
    tight = _make_assembler(max_tokens=300).assemble(
        SectionRegistry(sections=[s_keep, s_drop], name="t"), _ctx()
    )
    assert "droppable" in tight.sections_dropped

    # Loose budget: same registry, same text "hello", but droppable kept too
    # → different content, naturally different hash. We need a more subtle case.

    # Real test: SAME final text "hello" but in one assembly droppable was
    # never registered, in another it was registered AND dropped.
    only_keep = _make_assembler(max_tokens=10000).assemble(
        SectionRegistry(sections=[s_keep], name="t"), _ctx()
    )
    # Both produce text == "hello", but `tight` has dropped=["droppable"]
    # while `only_keep` has dropped=[]
    assert tight.text == only_keep.text == "hello"
    assert tight.version_hash != only_keep.version_hash, (
        "version_hash should differ when dropped section lists differ "
        "even if final text is identical"
    )


# ---- Telemetry --------------------------------------------------------- #


def test_telemetry_record_assembly_called() -> None:
    s = _section("identity", text="hello", priority=10)
    registry = SectionRegistry(sections=[s], name="test")
    telemetry = _FakeTelemetry()
    assembler = _make_assembler(telemetry=telemetry)
    assembler.assemble(registry, _ctx())

    assert len(telemetry.assembly_calls) == 1
    call = telemetry.assembly_calls[0]
    assert call["sections_included"] == ["identity"]
    assert call["sections_dropped"] == []
    assert call["lang"] == "zh"
    assert call["provider"] == "openai"
    assert call["mode"] == "full"
    assert call["fallback_used"] is False


def test_telemetry_records_fallback_used_flag() -> None:
    s = _section("identity", text="hi", priority=10)
    registry = SectionRegistry(sections=[s], name="test")
    telemetry = _FakeTelemetry()
    assembler = _make_assembler(telemetry=telemetry)
    assembler.assemble(registry, _ctx(), fallback_used=True)
    assert telemetry.assembly_calls[0]["fallback_used"] is True


def test_telemetry_failure_does_not_propagate() -> None:
    """If telemetry raises, assemble() must still return normally."""

    class _BoomTelemetry:
        def record_assembly(self, **kwargs: Any) -> None:
            raise RuntimeError("telemetry exploded")

        def record_llm_invocation(self, **kwargs: Any) -> None:  # pragma: no cover
            pass

        def record_lc_tools_degradation(self, *, reason: str) -> None:  # pragma: no cover
            pass

    s = _section("identity", text="hi", priority=10)
    registry = SectionRegistry(sections=[s], name="test")
    budget = SystemPromptBudget(max_tokens=10_000)
    estimator = TokenEstimator(strategy="hybrid")
    assembler = PromptAssembler(
        budget=budget, token_estimator=estimator, telemetry=_BoomTelemetry()  # type: ignore[arg-type]
    )

    # Should not raise
    result = assembler.assemble(registry, _ctx())
    assert result.text == "hi"


# ---- Section.estimate_tokens override --------------------------------- #


def test_section_can_override_estimate_tokens() -> None:
    """Sections can specify their own token estimator function."""
    s = Section(
        id="custom",
        priority=10,
        cacheable=False,
        dynamic=False,
        render=lambda ctx: SectionOutput(text="hi"),
        estimate_tokens=lambda text: 9999,  # always returns 9999
    )
    registry = SectionRegistry(sections=[s], name="test")
    result = _make_assembler().assemble(registry, _ctx())
    assert result.tokens_used == 9999
