"""M2-PR3: memory_rules section — injects rule-category memories.

Priority 8 (= ``CRITICAL_PRIORITY_MIN``) — survives budget pressure drops
in FULL mode because rules are hard behavioral constraints the agent
must honor. Losing one silently ("don't force-push to main") has
substantially higher blast radius than losing a user-profile preference
or a fact-index entry, so rules sit at the priority floor that
``PromptAssembler`` protects.

Sorted by ``updated_at DESC`` (applied upstream in ``MemorySnapshot``).
Hard-capped at ``RULE_CHUNK_CAP = 20`` chunks by the snapshot layer.
Self-truncates the rendered bullet list at
``MEMORY_RULE_SECTION_BUDGET = 2500`` tokens per design doc §597.

Returns ``SectionOutput(text=None)`` when ``ctx.memory_snapshot`` is
None (snapshot not populated) or has no rule chunks.
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)
from app.domain.services.prompts.sections._memory_section_helpers import (
    assemble_bullets_within_budget,
    pick_header,
    sanitize_bullet_content,
)


MEMORY_RULE_SECTION_BUDGET = 2500

_ZH_HEADER = (
    "## 项目规则\n"
    "以下是用户为本项目（或所有项目）设定的永久约束，每一条都必须遵守。"
)
_EN_HEADER = (
    "## Project Rules\n"
    "Permanent constraints the user has set for this project (or all "
    "projects). Every rule must be respected."
)


def _render(ctx: RenderContext) -> SectionOutput:
    snapshot = ctx.memory_snapshot
    if snapshot is None or not snapshot.rule_chunks:
        return SectionOutput(text=None)

    header = pick_header(ctx.lang, zh=_ZH_HEADER, en=_EN_HEADER)
    # Sanitize: flatten internal newlines + drop empties. Rule chunks with
    # all-whitespace content would otherwise emit a bare ``"- "`` bullet.
    bullets = [
        f"- {sanitized}"
        for c in snapshot.rule_chunks
        if (sanitized := sanitize_bullet_content(c.content))
    ]
    if not bullets:
        return SectionOutput(text=None)

    result = assemble_bullets_within_budget(
        header=header,
        bullets=bullets,
        max_tokens=MEMORY_RULE_SECTION_BUDGET,
    )
    if result is None:
        return SectionOutput(text=None)
    text, emitted = result

    # Count denominator uses the pre-sanitized total so "dropped" reflects
    # what the caller passed in, not what survived sanitation.
    total = len(snapshot.rule_chunks)
    return SectionOutput(
        text=text,
        metadata={
            "memory_rule_count": total,
            "memory_rule_emitted": emitted,
            "memory_rule_dropped": total - emitted,
        },
    )


memory_rules_section = Section(
    id="memory_rules",
    priority=8,
    cacheable=False,
    dynamic=True,
    render=_render,
    max_tokens=MEMORY_RULE_SECTION_BUDGET,
)
