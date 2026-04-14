"""skill_install_confirm reminder — fires between generate_skill and install_skill.

**B5 C8**: stub only. The condition returns False unconditionally; when
rolled out post-B5.1, this reminder would remind the LLM that it has
just completed ``generate_skill`` and must wait for explicit user
confirmation before calling ``install_skill``. This is currently enforced
by ``behavior_core`` section prose, not by a dedicated reminder — the
B5.1 bench might show that a top-level reminder outperforms inlined
prose for attention.

TODO(B5.1 rollout): wire the condition to a state flag like
``ctx.pending_skill_install_confirmation`` (new RenderContext field),
populated by the react_graph tool node after ``generate_skill`` returns
a successful blueprint.
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
        "You just generated a skill blueprint. Do NOT call install_skill "
        "until the user explicitly confirms. Show the blueprint and wait "
        "for confirmation."
    )


skill_install_confirm_reminder = Reminder(
    id="skill_install_confirm",
    condition=_condition,
    render=_render,
    provider_aware=True,
)
