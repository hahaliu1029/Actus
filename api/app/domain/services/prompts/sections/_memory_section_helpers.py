"""M2-PR3: shared rendering helpers for the three memory prompt sections.

Underscore-prefixed module: these helpers are implementation details
of ``memory_rules`` / ``memory_user_profile`` / ``memory_fact_index``
and are not intended to be imported outside that trio (or their tests).
Kept separate from ``memory_snapshot`` because the snapshot is a data
concern (async fetch, sort order) while these are rendering concerns
(token estimation, bullet assembly, language dispatch).
"""
from __future__ import annotations

from typing import Literal


# ---- Token estimation (CJK-aware) --------------------------------------- #


def estimate_tokens_cjk_aware(text: str) -> int:
    """Cheap CJK-aware token estimator used by memory sections for their
    own internal ``max_tokens`` self-truncation.

    Mirrors the weights AND rounding of
    ``domain/services/graphs/token_estimator.TokenEstimator.estimate``
    (hybrid strategy): CJK = 1.5 tokens/char, ASCII = 0.25 tokens/char,
    other = 1.0. Final tokens = ``max(round(total), 1)`` for non-empty
    input, 0 for empty. Matching the canonical rounding means this
    helper and the ``PromptAssembler``'s own estimator agree on whether
    a given bullet fits — two different rounding rules (``int`` vs
    ``round``) would silently drift by ±1 token per bullet and let
    sections overshoot their cap or under-utilize the budget.

    Reimplemented locally rather than importing the full
    ``TokenEstimator`` because that class pulls in LangChain ``BaseMessage``
    machinery at import time; memory sections should stay cheap to render
    during startup validation.
    """
    if not text:
        return 0
    total = 0.0
    for ch in text:
        cp = ord(ch)
        if _is_cjk(cp):
            total += 1.5
        elif cp < 0x80:
            total += 0.25
        else:
            total += 1.0
    return max(round(total), 1)


def _is_cjk(cp: int) -> bool:
    """Reproduces ``token_estimator._is_cjk`` locally. Matches its ranges
    exactly so memory section truncation decisions align with
    ``PromptAssembler``'s outer budget accounting."""
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x2E80 <= cp <= 0x2EFF
        or 0x3000 <= cp <= 0x303F
        or 0xFF00 <= cp <= 0xFFEF
        or 0xF900 <= cp <= 0xFAFF
    )


# ---- Shared bullet assembly with token cap ------------------------------ #


def assemble_bullets_within_budget(
    *,
    header: str,
    bullets: list[str],
    max_tokens: int,
) -> tuple[str, int] | None:
    """Glue a header + bullet list, dropping tail bullets once
    ``max_tokens`` would be exceeded. Returns ``(text, kept_count)`` or
    ``None`` if nothing fits.

    Returns ``None`` when ``bullets`` is empty OR when even the first
    bullet would exceed ``max_tokens``. Callers translate ``None`` into
    ``SectionOutput(text=None)`` to signal "skip this section". We
    deliberately never emit "header-only" output: if the budget is too
    tight to fit even the header + one bullet, the section is dropped
    entirely rather than emitting a confusing orphan header.

    Returning ``kept_count`` alongside the text prevents the caller
    from having to re-derive it via brittle string scans (e.g.
    ``text.count("\\n- ")``) that break when bullet content or headers
    contain similar substrings.

    The bullets list's own ordering is the authority — this function
    only drops tail entries, never reorders. Callers (memory_rules /
    memory_user_profile / memory_fact_index) are responsible for passing
    bullets in priority order (pinned first, most recent first, etc.) so
    tail-drop preserves the most important items.
    """
    if not bullets:
        return None

    header_tokens = estimate_tokens_cjk_aware(header)
    kept_bullets: list[str] = []
    running = header_tokens
    for bullet in bullets:
        delta = estimate_tokens_cjk_aware("\n" + bullet)
        if running + delta > max_tokens:
            break
        kept_bullets.append(bullet)
        running += delta

    if not kept_bullets:
        return None
    text = header + "\n" + "\n".join(kept_bullets)
    return text, len(kept_bullets)


# ---- Content sanitation ------------------------------------------------- #


def sanitize_bullet_content(raw: str) -> str:
    """Flatten a user-authored chunk to a single-line bullet body.

    Collapses all whitespace runs (including newlines and tabs) to a
    single space and strips leading/trailing whitespace. This defends
    two invariants the section output relies on:

    1. **Bullet structure**: each ``- {content}`` bullet occupies one
       line. An embedded ``\\n`` in content would break the bullet list
       visually (and, less importantly, any naive ``text.count("\\n- ")``
       scan). Users can legitimately save multi-line memory content
       (e.g. a rule with sub-points); we sacrifice fidelity for
       prompt-layout stability.

    2. **Injection hardening (best-effort)**: content can't forge its
       own header (``\\n## 用户画像``) or break out of the bullet list
       with a blank line. The attacker and victim are the same user
       (memories are user-scoped), so the worst case is self-confusion;
       still worth defending so the LLM receives a well-structured
       prompt regardless of what the user persisted.

    Returns an empty string unchanged — callers filter those out so we
    never emit a bare ``"- "`` bullet.
    """
    if not raw:
        return ""
    return " ".join(raw.split())


# ---- Language dispatch -------------------------------------------------- #


def pick_header(lang: Literal["zh", "en"], *, zh: str, en: str) -> str:
    """Tiny sugar so section files read `pick_header(ctx.lang, zh=..., en=...)`
    instead of an inline ternary. English fallback for any non-zh lang."""
    return en if lang == "en" else zh
