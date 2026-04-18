"""M2-PR3: memory_fact_index section — compact index of fact-category memories.

Priority 5 — lowest of the three memory sections, first to drop under
budget pressure. Facts are verifiable world/project references
("DB is PostgreSQL 17", "staging endpoint at actus.internal:8443/v2");
missing one causes the agent to ask the user again, but doesn't change
behavior, so fact index is the safest memory section to drop.

Format is deliberately compact: ``- [{id}] {first 15 chars}…`` — a
table of contents, not an encyclopedia. The agent sees what facts
exist and can call ``memory_get`` with the id for full content when
needed. This keeps the per-fact token cost tiny so we can surface up
to ``FACT_CHUNK_CAP = 50`` pointers within a 1000-token budget.

Sorted by ``updated_at DESC`` (applied upstream in ``MemorySnapshot``).
Self-truncates at ``MEMORY_FACT_INDEX_BUDGET = 1000`` tokens per design
doc §598.
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
)


MEMORY_FACT_INDEX_BUDGET = 1000
_FACT_PREVIEW_CHARS = 15

_ZH_HEADER = (
    "## 事实索引\n"
    "历史会话里沉淀下的项目事实（代码位置、接口、技术栈）。"
    "如需具体内容，调用 memory_get 传入对应 id。"
)
_EN_HEADER = (
    "## Fact Index\n"
    "Project facts recorded from past sessions (code pointers, endpoints, "
    "stack). Call memory_get with the id for full content."
)


def _preview(text: str) -> str:
    """Collapse whitespace and return the first ``_FACT_PREVIEW_CHARS`` of
    the content, appending an ellipsis only when truncation happens.
    Ellipsis uses the Chinese one-char '…' so the visual width matches
    both zh and en rendering without special-casing."""
    flat = " ".join(text.split())
    if len(flat) <= _FACT_PREVIEW_CHARS:
        return flat
    return flat[:_FACT_PREVIEW_CHARS] + "…"


def _render(ctx: RenderContext) -> SectionOutput:
    snapshot = ctx.memory_snapshot
    if snapshot is None or not snapshot.fact_chunks:
        return SectionOutput(text=None)

    header = pick_header(ctx.lang, zh=_ZH_HEADER, en=_EN_HEADER)
    # ``_preview`` already collapses whitespace via ``" ".join(text.split())``
    # so newline sanitation is baked in — but still filter empties so a
    # chunk with all-whitespace content doesn't emit ``- [id] ``.
    bullets = [
        f"- [{c.id}] {preview}"
        for c in snapshot.fact_chunks
        if (preview := _preview(c.content))
    ]
    if not bullets:
        return SectionOutput(text=None)

    result = assemble_bullets_within_budget(
        header=header,
        bullets=bullets,
        max_tokens=MEMORY_FACT_INDEX_BUDGET,
    )
    if result is None:
        return SectionOutput(text=None)
    text, emitted = result

    total = len(snapshot.fact_chunks)
    return SectionOutput(
        text=text,
        metadata={
            "memory_fact_count": total,
            "memory_fact_emitted": emitted,
            "memory_fact_dropped": total - emitted,
        },
    )


memory_fact_index_section = Section(
    id="memory_fact_index",
    priority=5,
    cacheable=False,
    dynamic=True,
    render=_render,
    max_tokens=MEMORY_FACT_INDEX_BUDGET,
)
