"""SPM Task 26: off-variant prompt sections (identity / behavior_core /
output_format) + RenderContext.sandbox_tools_enabled plumbing.

Two guarantees:

1. **off (sandbox_tools_enabled=False)** renders contain NONE of the
   ``FORBIDDEN_OFF_MARKERS`` (sandbox tool names ∪ sandbox-semantic words) —
   the off deployment must not teach an agent about a sandbox it does not have
   (INV-SPM-3).
2. **on (sandbox_tools_enabled=True, the default)** renders are byte-identical
   to the canonical pre-change section constants (INV-SPM-2). The on-path golden
   must never drift, even now that ``off`` is a first-class mode (unlocked in
   PR-4 / Task 32) and exercises the separate False path in production.

The planner single-step skill-creation teaching lives in the ``CREATE_PLAN_PROMPT``
HumanMessage template (``prompts/planner.py`` / ``prompts/en/planner.py``), NOT
in a RenderContext section — its off variant is delivered by Task 27's
``get_prompt_bundle(sandbox_tools_enabled=...)`` selector and is covered by
Task 27's bundle tests, so it is intentionally out of scope here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.services.prompts import get_prompt_bundle
from app.domain.services.prompts.section import RenderContext, Section
from app.domain.services.prompts.sections import behavior_core as behavior_core_mod
from app.domain.services.prompts.sections import identity as identity_mod
from app.domain.services.prompts.sections import output_format as output_format_mod

from tests.domain.services.prompts._off_markers import FORBIDDEN_OFF_MARKERS

# section_name -> (Section, {lang: canonical on-constant})
_SECTIONS: dict[str, tuple[Section, dict[str, str]]] = {
    "identity": (
        identity_mod.identity_section,
        {"zh": identity_mod._ZH_TEXT, "en": identity_mod._EN_TEXT},
    ),
    "behavior_core": (
        behavior_core_mod.behavior_core_section,
        {"zh": behavior_core_mod._ZH_TEXT, "en": behavior_core_mod._EN_TEXT},
    ),
    "output_format": (
        output_format_mod.output_format_section,
        {"zh": output_format_mod._ZH_TEXT, "en": output_format_mod._EN_TEXT},
    ),
}


def render_section(section_name: str, *, lang: str, sandbox_tools_enabled: bool) -> str:
    """Render a single C2 section for the given language + sandbox flag."""
    section, _ = _SECTIONS[section_name]
    ctx = RenderContext(lang=lang, sandbox_tools_enabled=sandbox_tools_enabled)  # type: ignore[arg-type]
    output = section.render(ctx)
    assert output.text is not None, f"{section_name}/{lang} rendered no text"
    return output.text


# ---- RenderContext plumbing ------------------------------------------- #


def test_render_context_default_sandbox_tools_enabled_true() -> None:
    """Deployment-constant default is True (all non-threaded call sites keep
    the on variant → byte-identity)."""
    assert RenderContext(lang="zh").sandbox_tools_enabled is True


# ---- off variant: no sandbox teaching --------------------------------- #


@pytest.mark.parametrize("lang", ["zh", "en"])
@pytest.mark.parametrize("section_name", ["identity", "behavior_core", "output_format"])
def test_off_section_has_no_sandbox_teaching(lang: str, section_name: str) -> None:
    text = render_section(section_name, lang=lang, sandbox_tools_enabled=False)
    for marker in FORBIDDEN_OFF_MARKERS:
        assert marker not in text, f"{section_name}/{lang} leaks {marker!r}"


@pytest.mark.parametrize("lang", ["zh", "en"])
@pytest.mark.parametrize("section_name", ["identity", "behavior_core", "output_format"])
def test_off_section_differs_from_on(lang: str, section_name: str) -> None:
    """The off branch must actually diverge from the on constant (guards
    against a no-op branch that would silently leak sandbox teaching)."""
    off = render_section(section_name, lang=lang, sandbox_tools_enabled=False)
    on = render_section(section_name, lang=lang, sandbox_tools_enabled=True)
    assert off != on


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_off_behavior_core_rewords_mcp_but_keeps_it(lang: str) -> None:
    """The MCP-vs-browser/terminal comparison is REWORDED (not deleted): off
    still prefers MCP tools, just without the browser/terminal comparison."""
    text = render_section("behavior_core", lang=lang, sandbox_tools_enabled=False)
    assert "MCP" in text
    # a2a discovery guidance is sandbox-agnostic and survives
    assert "get_remote_agent_cards" in text


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_off_output_format_keeps_attachments_field(lang: str) -> None:
    """The JSON parser contract is unchanged — only the /home/ubuntu example
    paths are dropped; the ``attachments`` field itself stays."""
    text = render_section("output_format", lang=lang, sandbox_tools_enabled=False)
    assert "attachments" in text


# ---- on variant: byte-identity (INV-SPM-2) ---------------------------- #


@pytest.mark.parametrize("lang", ["zh", "en"])
@pytest.mark.parametrize("section_name", ["identity", "behavior_core", "output_format"])
def test_on_section_byte_identical_to_canonical_constant(
    lang: str, section_name: str
) -> None:
    """on render (explicit True) == the canonical pre-change section constant."""
    _, canonical = _SECTIONS[section_name]
    on = render_section(section_name, lang=lang, sandbox_tools_enabled=True)
    assert on == canonical[lang]


@pytest.mark.parametrize("lang", ["zh", "en"])
@pytest.mark.parametrize("section_name", ["identity", "behavior_core", "output_format"])
def test_on_section_equals_default_context(lang: str, section_name: str) -> None:
    """Default RenderContext (no sandbox flag passed — what the golden suite and
    every non-threaded call site use) renders identically to explicit on."""
    section, _ = _SECTIONS[section_name]
    default_text = section.render(RenderContext(lang=lang)).text  # type: ignore[arg-type]
    on_text = render_section(section_name, lang=lang, sandbox_tools_enabled=True)
    assert default_text == on_text


def test_on_sections_retain_sandbox_teaching() -> None:
    """Sanity: the on path genuinely still teaches sandbox (proves the branch
    did not accidentally serve the off variant on the on path)."""
    assert "沙箱" in render_section("identity", lang="zh", sandbox_tools_enabled=True)
    assert "sandbox" in render_section("identity", lang="en", sandbox_tools_enabled=True)
    assert "/home/ubuntu" in render_section(
        "output_format", lang="zh", sandbox_tools_enabled=True
    )
    assert "浏览器" in render_section(
        "behavior_core", lang="zh", sandbox_tools_enabled=True
    )
    assert "browser" in render_section(
        "behavior_core", lang="en", sandbox_tools_enabled=True
    )


# ====================================================================== #
# Task 27: get_prompt_bundle selector + off HumanMessage-template variants
# ====================================================================== #

_GOLDEN_DIR = Path(__file__).parent / "golden"

# Templates the bundle exports that carry sandbox teaching → have an off variant.
# The remaining two (UPDATE_PLAN_PROMPT, EXECUTION_SUMMARY_NONE_FALLBACK) carry
# none and are reused byte-identically across modes.
_VARIANT_TEMPLATES = (
    "EXECUTION_PROMPT",
    "SUMMARIZE_PROMPT",
    "CREATE_PLAN_PROMPT",
    "GENERATE_SUMMARY_PROMPT",
)
_REUSED_TEMPLATES = ("UPDATE_PLAN_PROMPT", "EXECUTION_SUMMARY_NONE_FALLBACK")


def _dump_bundle(bundle) -> dict[str, str]:
    """Canonical snapshot: all str attributes, key-sorted (matches the golden
    generator + INV-SPM-2 baseline dump)."""
    return {k: v for k, v in sorted(vars(bundle).items()) if isinstance(v, str)}


def _load_golden(rel: str) -> dict[str, str]:
    return json.loads((_GOLDEN_DIR / rel).read_text(encoding="utf-8"))


# ---- signature / default ---------------------------------------------- #


def test_get_prompt_bundle_default_sandbox_tools_enabled_true() -> None:
    """Default (no kwarg) is the on variant — the production path."""
    assert _dump_bundle(get_prompt_bundle("zh")) == _dump_bundle(
        get_prompt_bundle("zh", sandbox_tools_enabled=True)
    )


# ---- off variant: no sandbox teaching (ALL templates) ----------------- #


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_off_bundle_all_templates_have_no_sandbox_teaching(lang: str) -> None:
    """r2 (codex planR1#13): do NOT name the three templates — iterate EVERY str
    attribute of the bundle. The bundle also exports the planner templates whose
    skill-creation teaching (brainstorm_skill / generate_skill / install_skill)
    a fixed triple would miss, teaching an off deployment tools it lacks. Reuses
    the shared FORBIDDEN_OFF_MARKERS (Task 26) — never a second hardcoded set."""
    b = get_prompt_bundle(lang, sandbox_tools_enabled=False)
    templates = {k: v for k, v in vars(b).items() if isinstance(v, str)}
    assert templates, "bundle 结构变化——先核对 __init__.py 导出"
    for name, tpl in templates.items():
        for marker in FORBIDDEN_OFF_MARKERS:
            assert marker not in tpl, f"{lang}/{name} leaks {marker!r}"


def test_zh_en_off_parity() -> None:
    """ZH/EN off variants expose an identical field structure (parity)."""
    zh = get_prompt_bundle("zh", sandbox_tools_enabled=False)
    en = get_prompt_bundle("en", sandbox_tools_enabled=False)
    assert set(vars(zh)) == set(vars(en))


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_off_bundle_matches_golden(lang: str) -> None:
    """off variant == committed off golden snapshot (drift lock)."""
    off = _dump_bundle(get_prompt_bundle(lang, sandbox_tools_enabled=False))
    assert off == _load_golden(f"off/off_bundle_{lang}.json")


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_off_bundle_diverges_from_on(lang: str) -> None:
    """Guard against a no-op selector: the four sandbox-teaching templates must
    actually change, and the two sandbox-agnostic ones must NOT (reused)."""
    on = _dump_bundle(get_prompt_bundle(lang, sandbox_tools_enabled=True))
    off = _dump_bundle(get_prompt_bundle(lang, sandbox_tools_enabled=False))
    for name in _VARIANT_TEMPLATES:
        assert on[name] != off[name], f"{lang}/{name} did not diverge in off"
    for name in _REUSED_TEMPLATES:
        assert on[name] == off[name], f"{lang}/{name} must be reused byte-for-byte"


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_off_create_plan_rewords_mcp_but_keeps_it(lang: str) -> None:
    """planner off variant REWORDS the MCP paragraph (drops the browser/terminal
    comparison) but still prefers MCP, and REMOVES the skill-creation teaching."""
    off = get_prompt_bundle(lang, sandbox_tools_enabled=False)
    assert "MCP" in off.CREATE_PLAN_PROMPT  # preference retained
    for tool in ("brainstorm_skill", "generate_skill", "install_skill"):
        assert tool not in off.CREATE_PLAN_PROMPT  # skill-creation teaching gone


# ---- on variant: byte-identity (INV-SPM-2 baseline) ------------------- #


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_default_true_bundle_byte_identical(lang: str) -> None:
    """INV-SPM-2: default (True) bundle == the committed baseline golden, dumped
    once from the pre-change constants. The on/True path golden must never drift,
    even now that off is a first-class mode (unlocked in PR-4 / Task 32) and drives
    the separate False path."""
    assert _dump_bundle(get_prompt_bundle(lang)) == _load_golden(
        f"baseline_bundle_{lang}.json"
    )


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_on_bundle_retains_sandbox_teaching(lang: str) -> None:
    """Sanity: the on path genuinely still teaches sandbox (proves the selector
    did not accidentally serve the off variant on the on path)."""
    on = get_prompt_bundle(lang, sandbox_tools_enabled=True)
    assert "file_read" in on.EXECUTION_PROMPT
    assert "/home/ubuntu" in on.SUMMARIZE_PROMPT
    assert "/home/ubuntu" in on.GENERATE_SUMMARY_PROMPT
    marker = "浏览器" if lang == "zh" else "browser"
    assert marker in on.CREATE_PLAN_PROMPT
