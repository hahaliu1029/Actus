"""PromptAssembler — composes a system prompt from a SectionRegistry.

B5 C1: takes a SectionRegistry, a RenderContext, and a PromptMode; renders
each section, applies the global token budget, and returns the final
assembled text plus aggregated metadata.

Key design points:
- **Output order = registry declaration order** (NOT priority order). Priority
  only governs which sections get DROPPED when over budget.
- **Critical sections protected**: priority >= ``CRITICAL_PRIORITY_MIN``
  sections never get dropped, even if the assembled total exceeds budget
  (a warn telemetry event is emitted instead).
- **Single-pass token accounting**: estimator runs once per section. The
  per-section ``estimate_tokens`` callable overrides the assembler's default
  if specified.
- **Metadata aggregation**: list-typed metadata fields concatenate across
  sections; other types are last-write-wins. Used by telemetry, NOT by the
  ``_assert_no_dangling_skill_tool_refs`` invariant scan (which runs at
  startup against the rendered text directly).

**Sync vs async**: ``assemble()`` is sync. The telemetry call is synchronous
and may block on file I/O if the implementation is ``JsonlPromptTelemetry``.
For B5 this is an accepted tradeoff — the JSONL append is microsecond-scale
in practice. If telemetry becomes a hot-path bottleneck, the port can be
made async or the implementation can offload to a thread executor. See
TODOS.md for the follow-up.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.section import (
    PromptMode,
    RenderContext,
    Section,
    SectionRegistry,
)

if TYPE_CHECKING:
    from app.domain.external.telemetry import PromptTelemetryPort
    from app.domain.services.graphs.token_estimator import TokenEstimator


logger = logging.getLogger(__name__)


CRITICAL_PRIORITY_MIN = 8
"""Sections with priority >= this value are protected from budget-driven
dropping. Must match ``SystemPromptBudget.critical_priority_min`` default."""


@dataclass
class AssembleResult:
    """Result of PromptAssembler.assemble().

    ``version_hash`` is a stable identifier for the assembled output —
    written to state by C5a so checkpointer replay can detect when the
    section definitions have changed mid-session.

    ``metadata`` aggregates metadata from each rendered section (telemetry only).
    """

    text: str
    tokens_used: int
    sections_included: list[str]
    sections_dropped: list[str]
    version_hash: str
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptAssembler:
    """Composes a system prompt from a SectionRegistry within a token budget.

    Constructor accepts a budget (hard cap), a TokenEstimator (used as the
    default for sections that don't specify their own), and an optional
    telemetry port for observability.
    """

    def __init__(
        self,
        budget: SystemPromptBudget,
        token_estimator: "TokenEstimator",
        telemetry: "PromptTelemetryPort | None" = None,
    ):
        self._budget = budget
        self._token_estimator = token_estimator
        self._telemetry = telemetry

    def assemble(
        self,
        registry: SectionRegistry,
        context: RenderContext,
        mode: PromptMode = PromptMode.FULL,
        *,
        fallback_used: bool = False,
    ) -> AssembleResult:
        """Render and assemble all sections into a single system prompt string.

        Algorithm:
        1. Filter sections by mode (full / minimal / none)
        2. Render each section against the context
        3. Walk in priority-DESC order to decide which to drop when over budget
        4. Output in registry-declaration order (NOT priority order)
        5. Aggregate metadata
        6. Compute version_hash for replay-compat checks
        7. Emit telemetry event
        """
        candidates = registry.filter(mode)

        # Step 1: render every section once. Cache the (output, tokens) result
        # so we can decide which to drop without re-rendering.
        rendered: dict[str, tuple[Section, str, int, dict[str, Any]] | None] = {}
        for section in candidates:
            output = section.render(context)
            if not output or not output.text:
                rendered[section.id] = None
                continue
            text = output.text
            estimate = section.estimate_tokens or self._token_estimator.estimate
            tokens = estimate(text)
            rendered[section.id] = (section, text, tokens, dict(output.metadata))

        # Step 2: walk in priority-DESC order to decide drops.
        # Priority >= CRITICAL_PRIORITY_MIN sections are always kept; their
        # tokens count first.
        dropped: list[str] = []
        kept_ids: set[str] = set()
        total = 0

        # Sort by priority desc, stable on declaration order
        priority_order = sorted(
            (s for s in candidates if rendered.get(s.id) is not None),
            key=lambda s: -s.priority,
        )
        for section in priority_order:
            entry = rendered[section.id]
            assert entry is not None  # Already filtered above
            _, _, tokens, _ = entry
            if (
                total + tokens > self._budget.max_tokens
                and section.priority < self._budget.critical_priority_min
            ):
                dropped.append(section.id)
                continue
            kept_ids.add(section.id)
            total += tokens

        # Step 3: build output in registry declaration order
        output_texts: list[str] = []
        included: list[str] = []
        aggregated_metadata: dict[str, Any] = {}
        for section in candidates:
            if section.id not in kept_ids:
                continue
            entry = rendered[section.id]
            assert entry is not None
            _, text, _, metadata = entry
            output_texts.append(text)
            included.append(section.id)
            for key, value in metadata.items():
                if isinstance(value, list):
                    existing = aggregated_metadata.setdefault(key, [])
                    if isinstance(existing, list):
                        existing.extend(value)
                else:
                    aggregated_metadata[key] = value

        text = "\n\n".join(output_texts).strip()
        # Hash includes the dropped section ids so two assemblies with the
        # same final text but different drop reasons (e.g. budget A vs budget B)
        # produce different version hashes. This is required for replay-compat
        # detection in C5a — see B5 design doc Risk #2.
        hash_input = f"{text}|dropped:{','.join(sorted(dropped))}"
        version_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

        if self._telemetry is not None:
            try:
                self._telemetry.record_assembly(
                    sections_included=included,
                    sections_dropped=dropped,
                    tokens_used=total,
                    lang=context.lang,
                    provider=context.provider,
                    mode=mode.value,
                    version_hash=version_hash,
                    fallback_used=fallback_used,
                )
            except Exception as exc:
                # Telemetry must never propagate errors, but the swallowed
                # exception is logged at WARN so production debugging is
                # possible — bare `except: pass` would be invisible.
                logger.warning(
                    "PromptAssembler: telemetry.record_assembly failed: %s", exc
                )

        return AssembleResult(
            text=text,
            tokens_used=total,
            sections_included=included,
            sections_dropped=dropped,
            version_hash=version_hash,
            metadata=aggregated_metadata,
        )

    # ----------------------------------------------------------------------
    # C2 PR-4 §8.7 — Coordinator child minimal prompt
    # ----------------------------------------------------------------------
    @staticmethod
    def build_minimal_for_coordinator_child(
        *,
        objective: str,
        phase: "Literal['exploration', 'write']",
        allowed_paths: list[str],
        work_unit_id: str,
        expected_result_schema: str | None = None,
    ) -> str:
        """Static helper — composes the restricted prompt for a coordinator
        child (spec §8.7) without going through the registry pipeline.

        Why staticmethod instead of an ``assemble()`` call:
        - The child prompt is fixed-shape and does not need priority-based
          section budget arbitration.
        - The child must NOT receive skill_context / tool_summary /
          conversation_summaries — those are root-only concerns.
        - Bypassing the registry keeps the child's prompt deterministic and
          minimal; the parent reducer (PR-5) compares observed child
          behavior against the spec'd allowlist, so prompt minimality is
          a security invariant, not just a UX choice.

        Returns the assembled prompt text. Caller (CoordinatorChildRunner)
        feeds the string directly to the inner runner."""
        from app.domain.services.prompts.sections.coordinator_work_unit import (
            build_coordinator_work_unit_section,
        )
        work_unit_block = build_coordinator_work_unit_section(
            objective=objective,
            phase=phase,
            allowed_paths=allowed_paths,
            work_unit_id=work_unit_id,
            expected_result_schema=expected_result_schema,
        )
        # Identity + restricted-behavior preamble. Kept inline (not pulled
        # from existing Section classes) because those classes wire to the
        # full root prompt schema (memory, skills, output_format JSON shape)
        # which the coordinator child must NOT inherit — see docstring above.
        identity = (
            "# Coordinator Step Worker\n\n"
            "You are a restricted worker spawned by a coordinator. Your scope "
            "is bounded by the work unit below. Stay within the authorized "
            "paths and allowed tools."
        )
        behavior = (
            "## Behavior\n\n"
            "- Read what you need to understand the objective.\n"
            "- For WRITE phase: make the necessary file changes, then return.\n"
            "- For EXPLORATION phase: analyze and return a proposed_write_plan.\n"
            "- Do NOT attempt tools or paths outside your authorization.\n"
            "- Stop as soon as the objective is met. No exploratory tangents."
        )
        return "\n\n---\n\n".join([identity, behavior, work_unit_block])
