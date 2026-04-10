"""Reminder registry + provider-aware rendering helper.

B5 C8: data classes only, no rollout. See ``reminders/__init__.py`` for
context on why this is skeleton-only.

Architecture:
- ``Reminder`` is a frozen dataclass describing one conditional reminder
  (id, condition callable, render callable, provider_aware flag).
- ``ReminderRegistry`` is a frozen dataclass holding a tuple of
  ``Reminder`` instances (mirrors ``SectionRegistry`` which was frozen
  in C4 review). Register by constructing a new registry with the
  updated tuple — no mutable ``register()`` method.
- ``render_reminder_block(text, ctx)`` wraps raw reminder text in the
  provider-appropriate envelope:
  - Anthropic: ``<system-reminder>...</system-reminder>`` ephemeral tag
  - OpenAI (and any unknown provider): ``## 重要提醒\n\n...`` markdown
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.domain.services.prompts.section import RenderContext


@dataclass(frozen=True)
class Reminder:
    """A conditional reminder that can be injected into a system prompt.

    Fields:
    - ``id``: unique identifier (used for logging / telemetry dedup)
    - ``condition``: pure function ``RenderContext -> bool``; the reminder
      fires iff this returns True. Must NOT mutate the context.
    - ``render``: pure function ``RenderContext -> str``; returns the raw
      reminder body (no envelope). Envelope is applied by
      ``render_reminder_block`` if ``provider_aware=True``.
    - ``provider_aware``: when True (default), the rendered body is wrapped
      by ``render_reminder_block``. When False, the ``render`` output is
      emitted as-is — used for reminders that embed their own formatting.
    """

    id: str
    condition: Callable[[RenderContext], bool]
    render: Callable[[RenderContext], str]
    provider_aware: bool = True


@dataclass(frozen=True)
class ReminderRegistry:
    """Ordered collection of ``Reminder`` instances.

    Frozen because instances are module-level singletons (mirrors
    ``SectionRegistry``). To "register" a reminder, construct a new
    registry with the updated tuple.

    ``active(ctx)`` returns the list of rendered reminder bodies whose
    condition fires, in registration order. Caller decides how to place
    the results into the assembled prompt (append, prepend, etc.).
    """

    reminders: tuple[Reminder, ...]

    def active(self, ctx: RenderContext) -> list[str]:
        """Return rendered text for every reminder whose condition fires.

        Each rendered body is wrapped by ``render_reminder_block`` if the
        reminder opts into provider-aware rendering. The list preserves
        the registration order of the registry.
        """
        out: list[str] = []
        for reminder in self.reminders:
            if not reminder.condition(ctx):
                continue
            body = reminder.render(ctx)
            if reminder.provider_aware:
                out.append(render_reminder_block(body, ctx))
            else:
                out.append(body)
        return out


def render_reminder_block(text: str, ctx: RenderContext) -> str:
    """Wrap ``text`` in the provider-appropriate envelope.

    - ``ctx.provider == "anthropic"``: returns
      ``<system-reminder>text</system-reminder>`` (Anthropic-specific
      ephemeral tag that claude-5.5-sonnet attenuates on subsequent
      turns — see B5.1 bench TODO).
    - Otherwise (``"openai"`` or any unrecognized provider): returns
      ``## 重要提醒\n\ntext`` markdown header that non-Anthropic models
      recognize as a strong reminder.
    """
    if ctx.provider == "anthropic":
        return f"<system-reminder>{text}</system-reminder>"
    return f"## 重要提醒\n\n{text}"
