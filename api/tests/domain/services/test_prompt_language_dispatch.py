"""B5 C0a: language dispatch + provider_name attribute tests.

This file tests:
1. ``prompts.get_prompt_bundle("zh"|"en")`` returns the correct constants
2. ``main_graph`` consumers can resolve all prompt fields via the bundle
3. Three LLM adapters expose ``provider_name`` attribute defaulting to "openai"
4. Unknown lang code falls back to ``zh`` without crashing
"""
from __future__ import annotations

import pytest

from app.domain.services.prompts import get_prompt_bundle


# ---- Bundle structure ---------------------------------------------------- #


# B5 C7.5: system-prompt constants (REACT_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT,
# FILE_VIEW_HINT, MEMORY_TOOLS_HINT) removed. The bundle now only exposes
# HumanMessage templates with .format() placeholders. Tests that previously
# asserted on those constants have been rewritten to use the section bundle
# (for system-prompt content) or dropped.
REQUIRED_FIELDS = (
    # react
    "EXECUTION_PROMPT",
    "SUMMARIZE_PROMPT",
    # planner
    "CREATE_PLAN_PROMPT",
    "UPDATE_PLAN_PROMPT",
    "EXECUTION_SUMMARY_NONE_FALLBACK",
    # summary
    "GENERATE_SUMMARY_PROMPT",
)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_bundle_exposes_all_required_fields(lang: str) -> None:
    bundle = get_prompt_bundle(lang)
    assert bundle.lang == lang
    for field in REQUIRED_FIELDS:
        value = getattr(bundle, field, None)
        assert isinstance(value, str), f"{lang}.{field} must be str, got {type(value)}"
        assert value.strip(), f"{lang}.{field} must be non-empty"


def test_zh_bundle_contains_chinese_text() -> None:
    bundle = get_prompt_bundle("zh")
    # ZH create-plan prompt should contain CJK characters
    assert any("\u4e00" <= ch <= "\u9fff" for ch in bundle.CREATE_PLAN_PROMPT)


def test_en_bundle_is_english() -> None:
    bundle = get_prompt_bundle("en")
    # EN execution prompt opening line should be ASCII instruction text
    first_line = bundle.EXECUTION_PROMPT.strip().split("\n", 1)[0]
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in first_line), (
        f"EN bundle's EXECUTION_PROMPT first line should be ASCII; got: {first_line!r}"
    )


def test_bundle_zh_and_en_are_different_instances() -> None:
    zh = get_prompt_bundle("zh")
    en = get_prompt_bundle("en")
    assert zh.CREATE_PLAN_PROMPT != en.CREATE_PLAN_PROMPT
    assert zh.GENERATE_SUMMARY_PROMPT != en.GENERATE_SUMMARY_PROMPT


@pytest.mark.parametrize("unknown", ["", "fr", "JP", "  ", "xx-YY"])
def test_unknown_lang_falls_back_to_zh(unknown: str) -> None:
    bundle = get_prompt_bundle(unknown)
    zh = get_prompt_bundle("zh")
    assert bundle.lang == "zh"
    assert bundle.CREATE_PLAN_PROMPT == zh.CREATE_PLAN_PROMPT


def test_none_lang_falls_back_to_zh() -> None:
    bundle = get_prompt_bundle(None)  # type: ignore[arg-type]
    assert bundle.lang == "zh"


# ---- Format placeholders preserved across languages ---------------------- #


def test_execution_prompt_placeholders_match() -> None:
    """Both ZH and EN EXECUTION_PROMPT must accept the same .format() args."""
    args = {
        "step": "test step",
        "message": "test message",
        "attachments": "[]",
        "language": "en",
    }
    for lang in ("zh", "en"):
        bundle = get_prompt_bundle(lang)
        rendered = bundle.EXECUTION_PROMPT.format(**args)
        assert "test step" in rendered
        assert "test message" in rendered


def test_create_plan_prompt_placeholders_match() -> None:
    args = {"message": "test msg", "attachments": "[]"}
    for lang in ("zh", "en"):
        bundle = get_prompt_bundle(lang)
        rendered = bundle.CREATE_PLAN_PROMPT.format(**args)
        assert "test msg" in rendered


def test_update_plan_prompt_placeholders_match() -> None:
    args = {
        "plan": '{"goal": "test"}',
        "step": '{"description": "step"}',
        "execution_summary": "result",
    }
    for lang in ("zh", "en"):
        bundle = get_prompt_bundle(lang)
        rendered = bundle.UPDATE_PLAN_PROMPT.format(**args)
        assert "test" in rendered


def test_generate_summary_prompt_placeholders_match() -> None:
    args = {
        "round_number": 1,
        "plan_goal": "do thing",
        "steps_summary": "- step 1: done",
    }
    for lang in ("zh", "en"):
        bundle = get_prompt_bundle(lang)
        rendered = bundle.GENERATE_SUMMARY_PROMPT.format(**args)
        assert "do thing" in rendered
        assert "step 1" in rendered


# ---- LLM adapter provider_name ------------------------------------------- #


def test_actus_chat_model_provider_name_default() -> None:
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

    model = ActusChatModel(api_key="test")
    assert model.provider_name == "openai"


def test_actus_responses_model_provider_name_default() -> None:
    from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

    model = ActusResponsesModel(api_key="test")
    assert model.provider_name == "openai"


def test_actus_fallback_chat_model_provider_name_default() -> None:
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
    from app.infrastructure.external.llm.actus_fallback_chat_model import (
        ActusFallbackChatModel,
    )
    from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

    primary = ActusChatModel(api_key="test")
    fallback = ActusResponsesModel(api_key="test")
    model = ActusFallbackChatModel(primary=primary, fallback=fallback)
    # Independent field — does NOT delegate to inner adapter
    assert model.provider_name == "openai"


def test_actus_chat_model_provider_name_can_be_set_to_anthropic() -> None:
    """B5.1 prerequisite: provider_name accepts 'anthropic' literal."""
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

    model = ActusChatModel(api_key="test", provider_name="anthropic")
    assert model.provider_name == "anthropic"


# ---- C0b: EN/ZH semantic parity ------------------------------------------ #


def test_en_react_skill_creation_has_pause_gating() -> None:
    """EN behavior_core section must mention the pause gating that matches
    ZH semantics (system pauses for explicit confirmation).

    Post-C7.5: content moved from ``bundle.REACT_SYSTEM_PROMPT`` to the
    ``behavior_core`` section. We render the section directly for the
    assertion.
    """
    from app.domain.services.prompts.section import RenderContext
    from app.domain.services.prompts.sections.behavior_core import (
        behavior_core_section,
    )

    ctx = RenderContext(lang="en")
    output = behavior_core_section.render(ctx)
    assert output.text is not None
    text = output.text
    # Step 2: blueprint pause
    assert "pause" in text.lower(), "EN behavior_core must describe pause gating"
    assert "confirmation" in text.lower() or "confirm" in text.lower()
    # Step 3: install pause — explicit mention of waiting for install confirm
    assert "install" in text.lower()


def test_en_planner_create_plan_has_image_hallucination_guard() -> None:
    """EN CREATE_PLAN_PROMPT must contain the image-hallucination prohibition
    that ZH version has."""
    bundle = get_prompt_bundle("en")
    text = bundle.CREATE_PLAN_PROMPT
    assert "cannot see" in text.lower() or "hallucination" in text.lower(), (
        "EN CREATE_PLAN_PROMPT must explicitly forbid image content hallucination"
    )
    # Should mention the right approach
    assert "filename" in text.lower() or "filenames" in text.lower()


def test_en_planner_create_plan_has_mcp_priority() -> None:
    """EN CREATE_PLAN_PROMPT must mention MCP/A2A tool priority."""
    bundle = get_prompt_bundle("en")
    text = bundle.CREATE_PLAN_PROMPT
    assert "mcp" in text.lower(), "EN CREATE_PLAN_PROMPT must mention MCP tools"
    assert "prefer" in text.lower() or "prioritize" in text.lower() or "priority" in text.lower()


def test_en_planner_create_plan_has_skill_creation_special_handling() -> None:
    """EN CREATE_PLAN_PROMPT must mention the skill-creation single-step rule."""
    bundle = get_prompt_bundle("en")
    text = bundle.CREATE_PLAN_PROMPT
    assert "brainstorm_skill" in text and "generate_skill" in text
    assert "single-step" in text.lower() or "single step" in text.lower()


def test_en_update_plan_prompt_has_execution_summary_placeholder() -> None:
    """BLOCKER: EN UPDATE_PLAN_PROMPT must accept execution_summary kwarg
    (previously silently dropped because the placeholder was missing)."""
    bundle = get_prompt_bundle("en")
    text = bundle.UPDATE_PLAN_PROMPT
    assert "{execution_summary}" in text, (
        "EN UPDATE_PLAN_PROMPT must contain {execution_summary} placeholder; "
        "otherwise main_graph.updater_node silently drops execution context"
    )
    # Verify .format() actually substitutes it
    rendered = text.format(
        plan="{}", step="{}", execution_summary="my summary text",
    )
    assert "my summary text" in rendered


def test_both_bundles_have_execution_summary_none_fallback() -> None:
    """Both ZH and EN must expose EXECUTION_SUMMARY_NONE_FALLBACK so main_graph
    no longer hard-codes a Chinese fallback string."""
    zh = get_prompt_bundle("zh")
    en = get_prompt_bundle("en")
    assert hasattr(zh, "EXECUTION_SUMMARY_NONE_FALLBACK")
    assert hasattr(en, "EXECUTION_SUMMARY_NONE_FALLBACK")
    assert zh.EXECUTION_SUMMARY_NONE_FALLBACK != en.EXECUTION_SUMMARY_NONE_FALLBACK
    # ZH should be Chinese, EN should not contain CJK
    assert any("\u4e00" <= ch <= "\u9fff" for ch in zh.EXECUTION_SUMMARY_NONE_FALLBACK)
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in en.EXECUTION_SUMMARY_NONE_FALLBACK)


def test_zh_and_en_react_step_count_parity() -> None:
    """ZH and EN behavior_core should have similar structure (length within 3x).

    Post-C7.5: content moved to section registry. We render the three
    C2 sections (identity + behavior_core + output_format) and compare
    the concatenated length between ZH and EN.
    """
    from app.domain.services.prompts.section import RenderContext
    from app.domain.services.prompts.sections.behavior_core import (
        behavior_core_section,
    )
    from app.domain.services.prompts.sections.identity import identity_section
    from app.domain.services.prompts.sections.output_format import (
        output_format_section,
    )

    def _render_combined(lang: str) -> str:
        ctx = RenderContext(lang=lang)  # type: ignore[arg-type]
        return "\n".join(
            s.render(ctx).text or ""
            for s in (identity_section, behavior_core_section, output_format_section)
        )

    zh = _render_combined("zh")
    en = _render_combined("en")
    ratio = len(en) / max(len(zh), 1)
    assert 0.8 < ratio < 3.0, (
        f"ZH/EN identity+behavior_core+output_format length ratio out of parity: "
        f"{ratio:.2f} (zh={len(zh)}, en={len(en)})"
    )


def test_zh_and_en_planner_create_plan_parity() -> None:
    """ZH and EN CREATE_PLAN_PROMPT should have similar structure."""
    zh = get_prompt_bundle("zh").CREATE_PLAN_PROMPT
    en = get_prompt_bundle("en").CREATE_PLAN_PROMPT
    ratio = len(en) / max(len(zh), 1)
    assert 0.8 < ratio < 3.0, (
        f"ZH/EN CREATE_PLAN_PROMPT length ratio out of parity: {ratio:.2f} "
        f"(zh={len(zh)}, en={len(en)})"
    )


def test_zh_and_en_bundles_expose_same_attribute_set() -> None:
    """Structural parity: ZH and EN bundles must expose the SAME attributes.

    Catches the case where a new constant is added to one language but
    forgotten on the other side — the bundle would silently expose `None`
    via SimpleNamespace and downstream code would AttributeError.
    """
    zh_attrs = set(vars(get_prompt_bundle("zh")))
    en_attrs = set(vars(get_prompt_bundle("en")))
    only_in_zh = zh_attrs - en_attrs
    only_in_en = en_attrs - zh_attrs
    assert not only_in_zh, f"Attrs only in ZH bundle: {sorted(only_in_zh)}"
    assert not only_in_en, f"Attrs only in EN bundle: {sorted(only_in_en)}"


# ---- C0a BLOCKER fix: planner_node writes language back to state --------- #


def test_planner_node_writes_language_to_state() -> None:
    """BLOCKER fix: planner_node MUST return `language=plan.language` so
    subsequent nodes (executor/updater/summarizer) read the LLM-detected
    language via state["language"].

    Without this, all C0a bundle dispatch is inert because state["language"]
    keeps the initial "zh" default from planner_react.invoke() initialization.
    """
    import inspect
    from app.domain.services.graphs import main_graph

    source = inspect.getsource(main_graph.build_main_graph)
    # Find the planner_node return block — must include "language": plan.language
    assert '"language": plan.language' in source, (
        "planner_node return dict must include `language=plan.language` "
        "so state['language'] reflects the LLM-detected language. "
        "Without this, executor/updater/summarizer always see the default 'zh'."
    )
