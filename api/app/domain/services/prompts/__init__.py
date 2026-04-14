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


def get_prompt_bundle(lang: str | SupportedLang | None) -> SimpleNamespace:
    """Return a namespace of HumanMessage template constants for the given language.

    Post-C7.5, this namespace contains ONLY templates with ``{placeholder}``
    substitutions that are composed into ``HumanMessage`` by main_graph /
    planner_react. System-prompt content is delivered via
    ``get_prompt_section_bundle`` + ``PromptAssembler``.

    Falls back to ``zh`` for any unrecognized language code (or ``None``)
    so the system keeps working if a new language slug appears in state
    without a bundle. The fallback is intentional — see the design doc
    "B5 C0a known exceptions" for the rationale.
    """
    normalized = (lang or "zh").lower().strip()
    if normalized == "en":
        return _build_en_bundle()
    return _build_zh_bundle()


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


def _build_zh_bundle() -> SimpleNamespace:
    from app.domain.services.prompts import react as zh_react
    from app.domain.services.prompts import planner as zh_planner
    from app.domain.services.prompts import summary as zh_summary

    return SimpleNamespace(
        lang="zh",
        # react
        EXECUTION_PROMPT=zh_react.EXECUTION_PROMPT,
        SUMMARIZE_PROMPT=zh_react.SUMMARIZE_PROMPT,
        # planner
        CREATE_PLAN_PROMPT=zh_planner.CREATE_PLAN_PROMPT,
        UPDATE_PLAN_PROMPT=zh_planner.UPDATE_PLAN_PROMPT,
        EXECUTION_SUMMARY_NONE_FALLBACK=zh_planner.EXECUTION_SUMMARY_NONE_FALLBACK,
        # summary
        GENERATE_SUMMARY_PROMPT=zh_summary.GENERATE_SUMMARY_PROMPT,
    )


def _build_en_bundle() -> SimpleNamespace:
    from app.domain.services.prompts.en import react as en_react
    from app.domain.services.prompts.en import planner as en_planner
    from app.domain.services.prompts.en import summary as en_summary

    return SimpleNamespace(
        lang="en",
        # react
        EXECUTION_PROMPT=en_react.EXECUTION_PROMPT,
        SUMMARIZE_PROMPT=en_react.SUMMARIZE_PROMPT,
        # planner
        CREATE_PLAN_PROMPT=en_planner.CREATE_PLAN_PROMPT,
        UPDATE_PLAN_PROMPT=en_planner.UPDATE_PLAN_PROMPT,
        EXECUTION_SUMMARY_NONE_FALLBACK=en_planner.EXECUTION_SUMMARY_NONE_FALLBACK,
        # summary
        GENERATE_SUMMARY_PROMPT=en_summary.GENERATE_SUMMARY_PROMPT,
    )
