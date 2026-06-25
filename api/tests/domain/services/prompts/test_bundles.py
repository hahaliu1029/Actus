"""B5 C4 / M2-PR3: tests for the per-language PromptBundle assembly.

Verifies that the ZH and EN bundles from ``prompts/bundles/`` construct
correctly, pass ``SectionRegistry.__post_init__`` startup validation, and
that the executor registry renders the expected canonical section set
(C2 + C3 + M2-PR3 memory sections) when given a fully-populated
``RenderContext``.

These are structural sanity tests rather than byte-equivalence snapshots:

- The C2 sections (identity / behavior_core / output_format) already have
  byte-level snapshot coverage in ``test_sections_c2_snapshot.py``.
- The C3 sections (tools_guide_dynamic in particular) intentionally do
  NOT byte-match the legacy ``_build_available_tool_summary``. So here
  we just verify that the new bundle produces a coherent assembled
  prompt with all expected markers present.
"""
from __future__ import annotations

import pytest

from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts import (
    get_prompt_bundle,
    get_prompt_section_bundle,
)
from app.domain.services.prompts.assembler import PromptAssembler
from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.bundles.en import (
    EN_BUNDLE,
    EN_EXECUTOR_REGISTRY,
    EN_PLANNER_REGISTRY,
    EN_UPDATER_REGISTRY,
)
from app.domain.services.prompts.bundles.zh import (
    ZH_BUNDLE,
    ZH_EXECUTOR_REGISTRY,
    ZH_PLANNER_REGISTRY,
    ZH_UPDATER_REGISTRY,
)
from app.domain.services.prompts.section import (
    PromptBundle,
    PromptMode,
    RenderContext,
)


_EXPECTED_EXECUTOR_SECTION_IDS = [
    "identity",
    "behavior_core",
    "output_format",
    "tools_guide_stable",
    "tools_guide_dynamic",
    "memory_rules",
    "memory_user_profile",
    "skill_context",
    "conversation_summaries",
    "memory_fact_index",
    "sandbox_state",
]


def _make_assembler() -> PromptAssembler:
    return PromptAssembler(
        budget=SystemPromptBudget(max_tokens=10_000),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )


def _make_full_ctx(lang: str) -> RenderContext:
    """Build a RenderContext that exercises every C2+C3 section.

    **Fields left at default** (none of the C2/C3 sections read them yet):
    - ``has_vision``, ``has_pdf`` (RenderContext defaults to True)
    - ``mcp_active``, ``a2a_active`` (default False)
    - ``tool_categories`` (default empty frozenset)
    - ``provider`` (defaults to "openai")

    When a future section starts reading one of these, update this fixture
    explicitly so the defaults don't silently mask regressions.
    """
    return RenderContext(
        lang=lang,  # type: ignore[arg-type]
        has_file_view=True,
        has_memory_tools=True,
        bound_tool_names=frozenset(
            {
                "shell_execute",
                "file_read",
                "file_view",
                "browser_navigate",
                "memory_search",
                "skill_foo_bar",
            }
        ),
        skill_context="## Active Skills\n- skill_foo: do foo things",
        skill_names_in_context=("foo",),
        conversation_summaries=("round 1: user asked X",),
    )


# ---- Bundle construction & identity ------------------------------------ #


def test_zh_bundle_is_prompt_bundle_instance() -> None:
    assert isinstance(ZH_BUNDLE, PromptBundle)
    assert ZH_BUNDLE.lang == "zh"
    assert ZH_BUNDLE.executor is ZH_EXECUTOR_REGISTRY
    assert ZH_BUNDLE.planner is ZH_PLANNER_REGISTRY
    assert ZH_BUNDLE.updater is ZH_UPDATER_REGISTRY


def test_en_bundle_is_prompt_bundle_instance() -> None:
    assert isinstance(EN_BUNDLE, PromptBundle)
    assert EN_BUNDLE.lang == "en"
    assert EN_BUNDLE.executor is EN_EXECUTOR_REGISTRY
    assert EN_BUNDLE.planner is EN_PLANNER_REGISTRY
    assert EN_BUNDLE.updater is EN_UPDATER_REGISTRY


def test_executor_registries_declare_canonical_sections_in_order() -> None:
    for registry in (ZH_EXECUTOR_REGISTRY, EN_EXECUTOR_REGISTRY):
        ids = [s.id for s in registry.sections]
        assert ids == _EXPECTED_EXECUTOR_SECTION_IDS, (
            f"{registry.name}: wrong section order or count, got {ids}"
        )


def test_planner_and_updater_registries_populated_in_c6() -> None:
    """WS0: planner/updater registries now carry the flag-gated
    parallel_work_units_teaching section at index 1 (after planner_identity)."""
    expected_ids = [
        "planner_identity",
        "parallel_work_units_teaching",
        "agent_team_teaching",          # [S4] new — inserted after the S2 teaching section
        "planner_tool_summary_legacy",
        "conversation_summaries",
    ]
    for registry in (
        ZH_PLANNER_REGISTRY,
        ZH_UPDATER_REGISTRY,
        EN_PLANNER_REGISTRY,
        EN_UPDATER_REGISTRY,
    ):
        ids = [s.id for s in registry.sections]
        assert ids == expected_ids, (
            f"{registry.name}: expected {expected_ids}, got {ids}"
        )


# ---- get_prompt_section_bundle dispatch -------------------------------- #


def test_get_prompt_section_bundle_returns_zh_by_default() -> None:
    assert get_prompt_section_bundle(None) is ZH_BUNDLE
    assert get_prompt_section_bundle("") is ZH_BUNDLE
    bundle = get_prompt_section_bundle("zh")
    assert bundle is ZH_BUNDLE
    # Guard against a future refactor silently detaching planner/updater
    assert bundle.executor is ZH_EXECUTOR_REGISTRY
    assert bundle.planner is ZH_PLANNER_REGISTRY
    assert bundle.updater is ZH_UPDATER_REGISTRY


def test_get_prompt_section_bundle_returns_en_for_english() -> None:
    assert get_prompt_section_bundle("EN") is EN_BUNDLE
    assert get_prompt_section_bundle("  en  ") is EN_BUNDLE
    bundle = get_prompt_section_bundle("en")
    assert bundle is EN_BUNDLE
    assert bundle.executor is EN_EXECUTOR_REGISTRY
    assert bundle.planner is EN_PLANNER_REGISTRY
    assert bundle.updater is EN_UPDATER_REGISTRY


def test_get_prompt_section_bundle_falls_back_to_zh_for_unknown_lang() -> None:
    """Same fallback convention as get_prompt_bundle — see C0a rationale."""
    assert get_prompt_section_bundle("fr") is ZH_BUNDLE
    assert get_prompt_section_bundle("ja") is ZH_BUNDLE


def test_get_prompt_bundle_returns_simplenamespace_with_templates_only() -> None:
    """Post-C7.5: ``get_prompt_bundle`` still returns a SimpleNamespace,
    but only with HumanMessage template constants. System-prompt content
    is delivered via ``get_prompt_section_bundle``."""
    from types import SimpleNamespace

    zh = get_prompt_bundle("zh")
    en = get_prompt_bundle("en")
    assert isinstance(zh, SimpleNamespace)
    assert isinstance(en, SimpleNamespace)
    # Spot-check that the surviving HumanMessage-template constants
    # (EXECUTION_PROMPT, CREATE_PLAN_PROMPT) are still exposed post-C7.5.
    # System-prompt constants (REACT_SYSTEM_PROMPT etc.) were removed.
    assert hasattr(zh, "EXECUTION_PROMPT")
    assert hasattr(en, "EXECUTION_PROMPT")
    assert hasattr(zh, "CREATE_PLAN_PROMPT")
    assert hasattr(en, "CREATE_PLAN_PROMPT")
    # Negative: the deleted legacy system-prompt constants must NOT be exposed
    assert not hasattr(zh, "REACT_SYSTEM_PROMPT")
    assert not hasattr(en, "PLANNER_SYSTEM_PROMPT")


# ---- Executor assembly end-to-end -------------------------------------- #


@pytest.mark.parametrize(
    "lang,bundle",
    [("zh", ZH_BUNDLE), ("en", EN_BUNDLE)],
)
def test_executor_assembly_renders_expected_sections(
    lang: str, bundle: PromptBundle
) -> None:
    """Assembling the executor registry with a full ctx renders every
    section except ``sandbox_state`` (stub returning None) and the three
    M2-PR3 memory sections (``memory_rules`` / ``memory_user_profile`` /
    ``memory_fact_index``), which return None whenever
    ``ctx.memory_snapshot`` is not populated. ``_make_full_ctx`` leaves
    ``memory_snapshot`` at its ``None`` default so this test still
    reflects the non-memory legacy assembly.

    The expected list is derived from ``_EXPECTED_EXECUTOR_SECTION_IDS``
    so it stays accurate when new sections are added.
    """
    ctx = _make_full_ctx(lang)
    result = _make_assembler().assemble(bundle.executor, ctx, PromptMode.FULL)

    assert result.text
    # sandbox_state = stub → None; memory_* = snapshot absent → None
    not_emitted_without_memory_or_sandbox = {
        "sandbox_state",
        "memory_rules",
        "memory_user_profile",
        "memory_fact_index",
    }
    expected = [
        s
        for s in _EXPECTED_EXECUTOR_SECTION_IDS
        if s not in not_emitted_without_memory_or_sandbox
    ]
    assert result.sections_included == expected


def test_zh_and_en_executor_output_differ() -> None:
    """Language dispatch must produce materially different text in the two
    bundles. If this fails, lang dispatch is broken somewhere."""
    zh_result = _make_assembler().assemble(
        ZH_BUNDLE.executor, _make_full_ctx("zh"), PromptMode.FULL
    )
    en_result = _make_assembler().assemble(
        EN_BUNDLE.executor, _make_full_ctx("en"), PromptMode.FULL
    )
    assert zh_result.text != en_result.text
    # ZH has CJK characters; EN does not (in the identity/behavior_core prose)
    assert any("\u4e00" <= ch <= "\u9fff" for ch in zh_result.text)
    # The only CJK in EN output is in user-provided fields (none here), so
    # the assembled EN text should have no CJK at all.
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in en_result.text)


def test_executor_assembly_contains_conversation_summary_header_per_lang() -> None:
    """Verify the localized header fix from C3 flows through the bundle."""
    zh_result = _make_assembler().assemble(
        ZH_BUNDLE.executor, _make_full_ctx("zh"), PromptMode.FULL
    )
    en_result = _make_assembler().assemble(
        EN_BUNDLE.executor, _make_full_ctx("en"), PromptMode.FULL
    )
    assert "## 历史对话摘要" in zh_result.text
    assert "## Conversation History Summary" in en_result.text


def test_executor_assembly_version_hash_differs_between_langs() -> None:
    """ZH and EN assemblies must produce different version hashes so the
    C5a replay-compat check can detect language drift mid-session."""
    zh_result = _make_assembler().assemble(
        ZH_BUNDLE.executor, _make_full_ctx("zh"), PromptMode.FULL
    )
    en_result = _make_assembler().assemble(
        EN_BUNDLE.executor, _make_full_ctx("en"), PromptMode.FULL
    )
    assert zh_result.version_hash != en_result.version_hash


def test_executor_assembly_empty_ctx_still_produces_c2_sections() -> None:
    """With no flags / no summaries / no skill_context, only the 3 C2
    sections survive — all C3 sections drop out cleanly."""
    ctx = RenderContext(lang="zh")
    result = _make_assembler().assemble(
        ZH_BUNDLE.executor, ctx, PromptMode.FULL
    )
    assert result.sections_included == ["identity", "behavior_core", "output_format"]


def test_minimal_mode_consistency_across_bundles() -> None:
    """MINIMAL mode filters both ZH and EN bundles to the same allowlist IDs."""
    ctx_zh = _make_full_ctx("zh")
    ctx_en = _make_full_ctx("en")
    zh_result = _make_assembler().assemble(
        ZH_BUNDLE.executor, ctx_zh, PromptMode.MINIMAL
    )
    en_result = _make_assembler().assemble(
        EN_BUNDLE.executor, ctx_en, PromptMode.MINIMAL
    )
    assert zh_result.sections_included == en_result.sections_included
