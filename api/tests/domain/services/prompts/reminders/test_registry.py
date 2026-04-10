"""B5 C8: ReminderRegistry + render_reminder_block unit tests.

These tests exercise the data structures and provider-aware rendering
helper. The 3 stub reminders (plan_mode, file_truncated,
skill_install_confirm) are verified to be inert (condition returns
False for any reasonable RenderContext) but to still produce a
non-empty render body when called directly — that's the contract for
future B5.1 rollout.
"""
from __future__ import annotations

import pytest

from app.domain.services.prompts.reminders import (
    Reminder,
    ReminderRegistry,
    render_reminder_block,
)
from app.domain.services.prompts.reminders.file_truncated import (
    file_truncated_reminder,
)
from app.domain.services.prompts.reminders.plan_mode import plan_mode_reminder
from app.domain.services.prompts.reminders.skill_install_confirm import (
    skill_install_confirm_reminder,
)
from app.domain.services.prompts.section import RenderContext


# ---- Reminder dataclass ------------------------------------------------ #


class TestReminderDataclass:
    def test_reminder_is_frozen(self) -> None:
        r = Reminder(
            id="test",
            condition=lambda ctx: True,
            render=lambda ctx: "body",
        )
        with pytest.raises((AttributeError, Exception)):
            r.id = "mutated"  # type: ignore[misc]

    def test_reminder_default_provider_aware_is_true(self) -> None:
        r = Reminder(
            id="test",
            condition=lambda ctx: True,
            render=lambda ctx: "body",
        )
        assert r.provider_aware is True

    def test_reminder_provider_aware_can_be_disabled(self) -> None:
        r = Reminder(
            id="test",
            condition=lambda ctx: True,
            render=lambda ctx: "body",
            provider_aware=False,
        )
        assert r.provider_aware is False


# ---- ReminderRegistry -------------------------------------------------- #


class TestReminderRegistry:
    def test_empty_registry_returns_empty_active_list(self) -> None:
        registry = ReminderRegistry(reminders=())
        ctx = RenderContext(lang="zh")
        assert registry.active(ctx) == []

    def test_registry_is_frozen(self) -> None:
        registry = ReminderRegistry(reminders=())
        with pytest.raises((AttributeError, Exception)):
            registry.reminders = ()  # type: ignore[misc]

    def test_only_matching_reminders_returned(self) -> None:
        r_match = Reminder(
            id="match",
            condition=lambda ctx: True,
            render=lambda ctx: "matched body",
        )
        r_skip = Reminder(
            id="skip",
            condition=lambda ctx: False,
            render=lambda ctx: "should not appear",
        )
        registry = ReminderRegistry(reminders=(r_match, r_skip))
        ctx = RenderContext(lang="zh")
        active = registry.active(ctx)
        assert len(active) == 1
        assert "matched body" in active[0]
        assert "should not appear" not in " ".join(active)

    def test_active_preserves_registration_order(self) -> None:
        r1 = Reminder(id="first", condition=lambda ctx: True, render=lambda ctx: "ONE")
        r2 = Reminder(id="second", condition=lambda ctx: True, render=lambda ctx: "TWO")
        r3 = Reminder(id="third", condition=lambda ctx: True, render=lambda ctx: "THREE")
        registry = ReminderRegistry(reminders=(r1, r2, r3))
        ctx = RenderContext(lang="zh")
        active = registry.active(ctx)
        assert len(active) == 3
        assert "ONE" in active[0]
        assert "TWO" in active[1]
        assert "THREE" in active[2]

    def test_provider_aware_false_bypasses_envelope(self) -> None:
        r = Reminder(
            id="raw",
            condition=lambda ctx: True,
            render=lambda ctx: "<custom-tag>raw body</custom-tag>",
            provider_aware=False,
        )
        registry = ReminderRegistry(reminders=(r,))
        ctx = RenderContext(lang="zh", provider="anthropic")
        active = registry.active(ctx)
        assert len(active) == 1
        # provider_aware=False → render output emitted as-is, no envelope
        assert active[0] == "<custom-tag>raw body</custom-tag>"
        assert "<system-reminder>" not in active[0]
        assert "## 重要提醒" not in active[0]


# ---- render_reminder_block --------------------------------------------- #


class TestRenderReminderBlock:
    def test_anthropic_wraps_in_system_reminder_tag(self) -> None:
        ctx = RenderContext(lang="zh", provider="anthropic")
        result = render_reminder_block("be careful", ctx)
        assert result == "<system-reminder>be careful</system-reminder>"

    def test_openai_wraps_in_markdown_header(self) -> None:
        ctx = RenderContext(lang="zh", provider="openai")
        result = render_reminder_block("be careful", ctx)
        assert result == "## 重要提醒\n\nbe careful"

    def test_unknown_provider_defaults_to_openai_format(self) -> None:
        """RenderContext.provider is Literal["openai", "anthropic"] but the
        helper must tolerate any string and default to the markdown format
        for safety — it's the conservative choice."""
        # We can't construct RenderContext(provider="other") because the
        # Literal constraint rejects it. Instead we call the helper with
        # a minimal fake ctx that exposes the provider attribute.
        from types import SimpleNamespace

        fake_ctx = SimpleNamespace(provider="mystery-provider")
        result = render_reminder_block("note", fake_ctx)  # type: ignore[arg-type]
        assert result == "## 重要提醒\n\nnote"


# ---- Stub reminders ---------------------------------------------------- #


class TestStubReminders:
    @pytest.mark.parametrize(
        "reminder",
        [
            plan_mode_reminder,
            file_truncated_reminder,
            skill_install_confirm_reminder,
        ],
    )
    def test_stub_condition_always_false(self, reminder: Reminder) -> None:
        """All 3 stub reminders must be inert in B5 — condition returns
        False regardless of the context passed in."""
        for ctx in (
            RenderContext(lang="zh"),
            RenderContext(lang="en"),
            RenderContext(lang="zh", has_file_view=True, has_memory_tools=True),
            RenderContext(lang="en", provider="anthropic"),
        ):
            assert reminder.condition(ctx) is False, (
                f"{reminder.id} fired for ctx {ctx!r} but should be inert in B5"
            )

    @pytest.mark.parametrize(
        "reminder",
        [
            plan_mode_reminder,
            file_truncated_reminder,
            skill_install_confirm_reminder,
        ],
    )
    def test_stub_render_produces_non_empty_body(self, reminder: Reminder) -> None:
        """Stub render functions still produce a non-empty body so future
        B5.1 rollout can smoke-test them without rewriting."""
        ctx = RenderContext(lang="zh")
        body = reminder.render(ctx)
        assert isinstance(body, str)
        assert body.strip()
        assert len(body) > 10  # meaningful content, not just whitespace

    def test_stubs_have_unique_ids(self) -> None:
        """No id collision among the 3 stub reminders."""
        ids = {
            plan_mode_reminder.id,
            file_truncated_reminder.id,
            skill_install_confirm_reminder.id,
        }
        assert ids == {"plan_mode", "file_truncated", "skill_install_confirm"}

    def test_stubs_are_all_provider_aware(self) -> None:
        """All 3 stubs default to provider-aware rendering (the expected
        rollout mode). If a future stub needs raw rendering, it must
        opt out explicitly."""
        assert plan_mode_reminder.provider_aware is True
        assert file_truncated_reminder.provider_aware is True
        assert skill_install_confirm_reminder.provider_aware is True


# ---- Registry with all 3 stubs (integration-style smoke test) --------- #


def test_registry_with_all_stubs_is_inert() -> None:
    """Even when all 3 stub reminders are registered, ``active()`` returns
    an empty list because every stub's condition is False."""
    registry = ReminderRegistry(
        reminders=(
            plan_mode_reminder,
            file_truncated_reminder,
            skill_install_confirm_reminder,
        )
    )
    for ctx in (
        RenderContext(lang="zh"),
        RenderContext(lang="en", provider="anthropic"),
    ):
        assert registry.active(ctx) == []
