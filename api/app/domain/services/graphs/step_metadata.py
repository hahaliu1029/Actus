"""Per-step metadata returned alongside the compiled react sub-graph.

B5 C5a: ``react_graph_provider`` changes its signature from returning a
bare ``CompiledStateGraph`` to returning ``(CompiledStateGraph, StepMetadata)``.
The metadata is the authoritative per-step truth about which tools are
bound, which skill context is in scope, and which skill ids contribute to
that context.

Splitting the stateful graph from the pure metadata lets tests mock the
metadata directly without constructing a real LangGraph: the executor's
prompt-assembly logic only needs the ``StepMetadata`` fields, not the
graph itself.

``RefreshedSkillsResult`` is the internal return type of
``_compute_refreshed_skills`` (the pure compute half of the refactored
``_refresh_skill_context_for_step``). The apply half
``_apply_refreshed_skills`` takes it and performs the atomic mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.models.skill import Skill


@dataclass(frozen=True)
class StepMetadata:
    """Authoritative per-step metadata produced by ``react_graph_provider``.

    - ``bound_tool_names``: tool names actually bound on the compiled react
      graph for THIS step. This is the LOWER BOUND on what the step_react
      can receive: partial skill-tool mutation may cause the actual bound
      set to be a superset. B5's correctness contract is "the LLM only
      acts on tools the prompt advertises", which is a weaker invariant
      than "bound == advertised".
    - ``skill_context``: the markdown skill context used for prompt
      assembly on this step. Distinct from ``state.skill_context`` (the
      updater-written fallback) — see two-clock architecture in the B5
      design doc.
    - ``skill_ids``: ordered tuple of skill ids that produced
      ``skill_context``. Used by telemetry and by the PromptAssembler
      metadata aggregation.
    """

    bound_tool_names: frozenset[str]
    skill_context: str
    skill_ids: tuple[str, ...]


@dataclass(frozen=True)
class RefreshedSkillsResult:
    """Pure-compute output of ``_compute_refreshed_skills``.

    Returned by the pure function; consumed by ``_apply_refreshed_skills``
    which performs the single atomic mutation point (with rollback on
    partial failure). A ``None`` return from ``_compute_refreshed_skills``
    means "sticky: keep the previous skill selection" (low embedding score)
    — it is NOT a ``RefreshedSkillsResult`` with empty fields.
    """

    skills: tuple["Skill", ...]
    context: str
    skill_ids: tuple[str, ...]
    scores: tuple[float, ...] | None
