"""Reminder injection infrastructure — B5 C8 skeleton, NOT rolled out.

This package holds the ``ReminderRegistry`` data classes and provider-aware
rendering helper. No reminder is wired into any system prompt assembly
path in B5 itself — the ``EXECUTION_PROMPT`` hardcoded "提醒：..." string
remains unchanged.

Rollout is blocked on the **B5.1 Provider Bench** follow-up (TODOS #31):
we need to verify whether ``<system-reminder>`` tags meaningfully affect
GPT-5.4's attention the same way they do claude-5.5-sonnet before
globally replacing the hardcoded reminder.

Consumers (future, post-bench):
- ``executor_node`` could call ``registry.active(ctx)`` and append the
  results to the assembled system prompt.
- Per-step conditional reminders (e.g. file_truncated when a large file
  was just read) could be triggered from inside a react_graph tool node.
"""
from __future__ import annotations

from app.domain.services.prompts.reminders.registry import (
    Reminder,
    ReminderRegistry,
    render_reminder_block,
)

__all__ = ["Reminder", "ReminderRegistry", "render_reminder_block"]
