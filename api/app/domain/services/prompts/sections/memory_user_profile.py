"""M2-PR3: memory_user_profile section — injects user-category memories.

Priority 7 — below rules (8) but above fact index (5). User-profile
items ("prefers Go, 10 years experience"; "reply in Chinese") bias the
agent's communication style and defaults. Losing one degrades
personalization but doesn't break correctness, so it stays below
``CRITICAL_PRIORITY_MIN = 8``.

Sorted by (pinned DESC, updated_at DESC) upstream in ``MemorySnapshot``,
so tail-drop under budget pressure preserves pinned items. Hard-capped
at ``USER_CHUNK_CAP = 10`` chunks. Self-truncates at
``MEMORY_USER_SECTION_BUDGET = 1500`` tokens per design doc §596.

Pinned items are rendered with a leading ``★`` marker so the agent can
distinguish "the user has pinned this — do not lose track of it"
from "this is a recent preference the user expressed". The marker is
visual only; no behavior hinges on parsing it.
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


MEMORY_USER_SECTION_BUDGET = 1500

_ZH_HEADER = (
    "## 用户画像\n"
    "以下是用户在历史对话里主动表达过的偏好/身份/工作风格，用来 personalize 你的回答。"
    "★ 标记表示用户已 pin，务必记住。"
)
_EN_HEADER = (
    "## User Profile\n"
    "Preferences, identity, and working style the user has expressed in "
    "past sessions — use to personalize responses. Items marked ★ are "
    "pinned by the user and must be retained."
)


def _render(ctx: RenderContext) -> SectionOutput:
    snapshot = ctx.memory_snapshot
    if snapshot is None or not snapshot.user_chunks:
        return SectionOutput(text=None)

    header = pick_header(ctx.lang, zh=_ZH_HEADER, en=_EN_HEADER)
    # Sanitize: flatten internal newlines + drop empties. User-profile
    # chunks with all-whitespace content would otherwise emit a bare
    # ``"- "`` or ``"- ★ "`` bullet.
    bullets: list[str] = []
    for c in snapshot.user_chunks:
        sanitized = sanitize_bullet_content(c.content)
        if not sanitized:
            continue
        marker = "★ " if c.pinned else ""
        bullets.append(f"- {marker}{sanitized}")
    if not bullets:
        return SectionOutput(text=None)

    result = assemble_bullets_within_budget(
        header=header,
        bullets=bullets,
        max_tokens=MEMORY_USER_SECTION_BUDGET,
    )
    if result is None:
        return SectionOutput(text=None)
    text, emitted = result

    total = len(snapshot.user_chunks)
    pinned_count = sum(1 for c in snapshot.user_chunks if c.pinned)
    return SectionOutput(
        text=text,
        metadata={
            "memory_user_count": total,
            "memory_user_pinned_count": pinned_count,
            "memory_user_emitted": emitted,
            "memory_user_dropped": total - emitted,
        },
    )


memory_user_profile_section = Section(
    id="memory_user_profile",
    priority=7,
    cacheable=False,
    dynamic=True,
    render=_render,
    max_tokens=MEMORY_USER_SECTION_BUDGET,
)
