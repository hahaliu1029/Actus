"""conversation_summaries section — joined history summaries with localized header.

B5 C3: replaces the legacy hardcoded Chinese header in ``main_graph.py``:

    system_content += "\\n\\n## 历史对话摘要\\n" + "\\n\\n".join(conversation_summaries)

The legacy version hardcoded ``## 历史对话摘要`` even for English users
(a known C0a/C0b miss). This section dispatches the header by ``ctx.lang``.

Returns ``SectionOutput(text=None)`` when ``ctx.conversation_summaries`` is
empty.
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


_ZH_HEADER = "## 历史对话摘要"
_EN_HEADER = "## Conversation History Summary"


def _render(ctx: RenderContext) -> SectionOutput:
    """Render conversation summaries with a language-appropriate header."""
    summaries = ctx.conversation_summaries
    if not summaries:
        return SectionOutput(text=None)

    header = _EN_HEADER if ctx.lang == "en" else _ZH_HEADER
    body = "\n\n".join(summaries)
    text = f"{header}\n{body}"
    return SectionOutput(
        text=text,
        metadata={"summary_count": len(summaries)},
    )


conversation_summaries_section = Section(
    id="conversation_summaries",
    priority=6,
    cacheable=False,
    dynamic=True,
    render=_render,
)
