"""file_truncated reminder — fires after ContextAssembler truncates tool output.

**B5 C8**: stub only. The condition returns False unconditionally; when
rolled out post-B5.1, this reminder would alert the LLM that the tool
output it's reading was truncated and avoid hallucinating the missing
content.

TODO(B5.1 rollout): wire the condition to a state flag like
``ctx.last_tool_output_truncated`` (new RenderContext field). The
``_phase1_compress_tool_content`` path in ``context_assembler.py`` is
the natural producer of this signal.
"""
from __future__ import annotations

from app.domain.services.prompts.reminders.registry import Reminder
from app.domain.services.prompts.section import RenderContext


def _condition(ctx: RenderContext) -> bool:
    """Stub: always False until B5.1 rollout."""
    return False


def _render(ctx: RenderContext) -> str:
    """Stub reminder body. Not emitted in B5."""
    return (
        "Note: one or more recent tool outputs were truncated to fit the "
        "context window. Do NOT assume the missing content; re-read with "
        "the file_read offset parameter if you need specific sections."
    )


file_truncated_reminder = Reminder(
    id="file_truncated",
    condition=_condition,
    render=_render,
    provider_aware=True,
)
