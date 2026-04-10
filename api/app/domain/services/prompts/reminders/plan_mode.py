"""plan_mode reminder — fires when the agent is in plan-mode replan step.

**B5 C8**: stub only. The condition returns False unconditionally so
this reminder never fires until the B5.1 provider bench confirms
whether ``<system-reminder>`` tags are worth rolling out. When enabled,
this reminder would nudge the LLM to stay focused on plan updates
rather than executing tools.

TODO(B5.1 rollout): wire the condition to a state flag like
``ctx.mode == "plan_mode"`` once plan mode is a real thing. Right now
``RenderContext`` doesn't carry a mode field — if B5.1 decides to roll
out this reminder, add a ``plan_mode: bool`` field to ``RenderContext``
and have the condition read it.
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
        "You are in plan mode. Do NOT call execution tools; only output "
        "plan updates via the structured JSON format."
    )


plan_mode_reminder = Reminder(
    id="plan_mode",
    condition=_condition,
    render=_render,
    provider_aware=True,
)
