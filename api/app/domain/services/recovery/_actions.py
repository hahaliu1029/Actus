"""Typed rewrite actions registered by B2 Recovery rules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StripResponseFormat:
    code: str = "strip_response_format"

    async def apply(self, messages, kwargs, ctx):
        if kwargs.get("response_format") is None:
            return None
        new_kwargs = dict(kwargs)
        new_kwargs["response_format"] = None
        return list(messages), new_kwargs


@dataclass(frozen=True)
class DisableThinking:
    code: str = "disable_thinking"

    async def apply(self, messages, kwargs, ctx):
        style = ctx.profile.thinking_toggle_style
        new_kwargs = dict(kwargs)
        if style == "extra_body_enable_thinking":
            extra = dict(new_kwargs.get("extra_body", {}))
            if extra.get("enable_thinking") is False:
                return None
            extra["enable_thinking"] = False
            new_kwargs["extra_body"] = extra
        elif style == "extra_body_thinking":
            extra = dict(new_kwargs.get("extra_body", {}))
            if extra.get("thinking") is False:
                return None
            extra["thinking"] = False
            new_kwargs["extra_body"] = extra
        elif style == "openai_reasoning_effort":
            if new_kwargs.get("reasoning_effort") == "minimal":
                return None
            new_kwargs["reasoning_effort"] = "minimal"
        else:
            return None
        return list(messages), new_kwargs


@dataclass(frozen=True)
class DowngradeToolChoiceToAuto:
    code: str = "downgrade_tool_choice_to_auto"

    async def apply(self, messages, kwargs, ctx):
        # Round 16 P1 #2: must NOT short-circuit when current is None.
        # ActusChatModel resolves tool_choice via
        # `resolve_tool_choice(per_call=kwargs.pop("tool_choice", None),
        #                      bound_value=self._bound_tool_choice, ...)`
        # (`actus_chat_model.py:670-683` for _agenerate, :837-850 for _astream).
        # Per-call beats bound. So if the user did
        # `bind_tools(..., tool_choice="required")` the per-call kwarg is None
        # but the bound value is "required" — Recovery's classify still hits
        # because the resolved value at the wire is "required". If this action
        # short-circuits on None, the rule self-skips, R2/R3 collapse to
        # candidate-exhaustion, and budget_exhausted fires without any actual
        # rewrite. Writing per-call "auto" overrides the bound value at the
        # exact resolution point.
        current = kwargs.get("tool_choice")
        if current in ("auto", "none"):
            return None  # already safe at the per-call layer
        new_kwargs = dict(kwargs)
        new_kwargs["tool_choice"] = "auto"
        return list(messages), new_kwargs


@dataclass(frozen=True)
class TriggerRecompact:
    code: str = "trigger_recompact"

    async def apply(self, messages, kwargs, ctx):
        if ctx.on_context_overflow is None:
            return None
        new_messages = await ctx.on_context_overflow(
            list(messages), dict(kwargs),  # shallow copy; I5
        )
        if new_messages is None:
            return None
        return new_messages, dict(kwargs)


# Placeholder — not registered in PR-2. Reserved for future DashScope sub-profiles.
@dataclass(frozen=True)
class StripExtraBodyField:
    field_name: str

    @property
    def code(self) -> str:
        return f"strip_extra_body_field_{self.field_name}"

    async def apply(self, messages, kwargs, ctx):
        extra = dict(kwargs.get("extra_body", {}))
        if self.field_name not in extra:
            return None
        del extra[self.field_name]
        new_kwargs = dict(kwargs)
        new_kwargs["extra_body"] = extra
        return list(messages), new_kwargs
