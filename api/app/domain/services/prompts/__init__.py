"""Prompt bundle dispatch by language.

Two parallel APIs, both post-C7.5:

1. ``get_prompt_bundle(lang) -> SimpleNamespace`` — HumanMessage template
   constants API. Returns the dynamic prompt strings that still use
   ``.format(...)``-style placeholders and are composed directly into
   ``HumanMessage`` by ``main_graph`` and ``planner_react``:
   ``EXECUTION_PROMPT``, ``SUMMARIZE_PROMPT``, ``CREATE_PLAN_PROMPT``,
   ``UPDATE_PLAN_PROMPT``, ``EXECUTION_SUMMARY_NONE_FALLBACK``,
   ``GENERATE_SUMMARY_PROMPT``. These are NOT system-prompt content —
   the section-based PromptAssembler handles system prompts.

2. ``get_prompt_section_bundle(lang) -> PromptBundle`` — sections API.
   Returns a ``PromptBundle`` with ``executor/planner/updater``
   ``SectionRegistry`` instances. Consumed by ``PromptAssembler`` in
   ``executor_node``, ``planner_node``, ``updater_node``, and
   ``_run_planner_for_detection``.

B5 C7.5 removed the legacy SystemMessage constants (``REACT_SYSTEM_PROMPT``,
``PLANNER_SYSTEM_PROMPT``, ``FILE_VIEW_HINT``, ``MEMORY_TOOLS_HINT``) from
both this namespace and from ``prompts/react.py`` / ``prompts/planner.py``.
Their content now lives in ``prompts/sections/`` and is composed via
``SectionRegistry``.

Known exceptions (single-language, not in bundle):
- ``prompts.continuation_classifier`` — caller has no language context at
  invocation time (runs before planner detects language).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

from app.domain.services.prompts.section import PromptBundle

# Canonical language slug for prompt dispatch. Other parts of the codebase
# (state["language"], plan.language, etc.) currently use plain `str` because
# the value originally came from a Pydantic field with no Literal constraint.
# This alias is the canonical type — new code SHOULD adopt it.
SupportedLang = Literal["zh", "en"]


def get_prompt_bundle(
    lang: str | SupportedLang | None,
    *,
    sandbox_tools_enabled: bool = True,
) -> SimpleNamespace:
    """Return a namespace of HumanMessage template constants for the given language.

    Post-C7.5, this namespace contains ONLY templates with ``{placeholder}``
    substitutions that are composed into ``HumanMessage`` by main_graph /
    planner_react. System-prompt content is delivered via
    ``get_prompt_section_bundle`` + ``PromptAssembler``.

    Falls back to ``zh`` for any unrecognized language code (or ``None``)
    so the system keeps working if a new language slug appears in state
    without a bundle. The fallback is intentional — see the design doc
    "B5 C0a known exceptions" for the rationale.

    SPM Task 27 (INV-SPM-2 / INV-SPM-3): ``sandbox_tools_enabled`` selects the
    off-deployment variant of the bundle. When False, templates that teach the
    agent about sandbox file/shell/browser tools (or the skill-creation tool
    chain, or ``/home/ubuntu`` paths) are swapped for sandbox-agnostic variants
    so an ``off`` deployment never advertises tools it does not have. The
    default True path returns the unchanged on-variant constants and MUST stay
    byte-identical (INV-SPM-2: the on-variant golden must never drift; ``off`` is
    a first-class mode since PR-4 / Task 32 and drives the False path). The
    bundle is rebuilt per call (no cache), so no
    keying by flag is needed — on/off namespaces never share state.
    """
    normalized = (lang or "zh").lower().strip()
    if normalized == "en":
        return _build_en_bundle(sandbox_tools_enabled=sandbox_tools_enabled)
    return _build_zh_bundle(sandbox_tools_enabled=sandbox_tools_enabled)


def get_prompt_section_bundle(
    lang: str | SupportedLang | None,
) -> PromptBundle:
    """Return the ``PromptBundle`` (executor/planner/updater registries).

    Added in B5 C4, sole path for system-prompt assembly since C7.5.
    Falls back to ``zh`` for unrecognized language codes — same convention
    as ``get_prompt_bundle`` above.
    """
    # Deferred import: bundle construction triggers
    # ``SectionRegistry.__post_init__`` first-use validation (each section
    # is rendered against ``_FIXTURE_CTX`` and scanned for dangling skill
    # tool refs). Keeping the import inside the function means:
    #   - ``import app.domain.services.prompts`` stays cheap for consumers
    #     that only need ``get_prompt_bundle`` HumanMessage templates.
    #   - Validation fires on the FIRST call to this function (typically
    #     during ``AgentTaskRunner.__init__`` when it builds the
    #     PromptAssembler), not at application bootstrap.
    # Audit MEDIUM #5: earlier docstrings claimed "startup fail-fast" —
    # that was misleading. The actual semantic is "first-bundle-access".
    from app.domain.services.prompts.bundles.en import EN_BUNDLE
    from app.domain.services.prompts.bundles.zh import ZH_BUNDLE

    normalized = (lang or "zh").lower().strip()
    if normalized == "en":
        return EN_BUNDLE
    return ZH_BUNDLE


def _build_zh_bundle(*, sandbox_tools_enabled: bool = True) -> SimpleNamespace:
    from app.domain.services.prompts import react as zh_react
    from app.domain.services.prompts import planner as zh_planner
    from app.domain.services.prompts import summary as zh_summary

    # SPM Task 27: pick the off variant for templates that carry sandbox
    # teaching; the other two (UPDATE_PLAN_PROMPT, EXECUTION_SUMMARY_NONE_FALLBACK)
    # carry none and are reused unchanged in both modes.
    return SimpleNamespace(
        lang="zh",
        # react
        EXECUTION_PROMPT=(
            zh_react.EXECUTION_PROMPT
            if sandbox_tools_enabled
            else zh_react.EXECUTION_PROMPT_OFF
        ),
        SUMMARIZE_PROMPT=(
            zh_react.SUMMARIZE_PROMPT
            if sandbox_tools_enabled
            else zh_react.SUMMARIZE_PROMPT_OFF
        ),
        # planner
        CREATE_PLAN_PROMPT=(
            zh_planner.CREATE_PLAN_PROMPT
            if sandbox_tools_enabled
            else zh_planner.CREATE_PLAN_PROMPT_OFF
        ),
        UPDATE_PLAN_PROMPT=zh_planner.UPDATE_PLAN_PROMPT,  # no sandbox teaching → reused
        EXECUTION_SUMMARY_NONE_FALLBACK=zh_planner.EXECUTION_SUMMARY_NONE_FALLBACK,  # reused
        # summary
        GENERATE_SUMMARY_PROMPT=(
            zh_summary.GENERATE_SUMMARY_PROMPT
            if sandbox_tools_enabled
            else zh_summary.GENERATE_SUMMARY_PROMPT_OFF
        ),
    )


def _build_en_bundle(*, sandbox_tools_enabled: bool = True) -> SimpleNamespace:
    from app.domain.services.prompts.en import react as en_react
    from app.domain.services.prompts.en import planner as en_planner
    from app.domain.services.prompts.en import summary as en_summary

    return SimpleNamespace(
        lang="en",
        # react
        EXECUTION_PROMPT=(
            en_react.EXECUTION_PROMPT
            if sandbox_tools_enabled
            else en_react.EXECUTION_PROMPT_OFF
        ),
        SUMMARIZE_PROMPT=(
            en_react.SUMMARIZE_PROMPT
            if sandbox_tools_enabled
            else en_react.SUMMARIZE_PROMPT_OFF
        ),
        # planner
        CREATE_PLAN_PROMPT=(
            en_planner.CREATE_PLAN_PROMPT
            if sandbox_tools_enabled
            else en_planner.CREATE_PLAN_PROMPT_OFF
        ),
        UPDATE_PLAN_PROMPT=en_planner.UPDATE_PLAN_PROMPT,  # no sandbox teaching → reused
        EXECUTION_SUMMARY_NONE_FALLBACK=en_planner.EXECUTION_SUMMARY_NONE_FALLBACK,  # reused
        # summary
        GENERATE_SUMMARY_PROMPT=(
            en_summary.GENERATE_SUMMARY_PROMPT
            if sandbox_tools_enabled
            else en_summary.GENERATE_SUMMARY_PROMPT_OFF
        ),
    )
