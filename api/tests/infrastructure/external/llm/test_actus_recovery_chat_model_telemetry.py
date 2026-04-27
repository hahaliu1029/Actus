"""PR-3 T50-T58 + T19g/h/h2/h3/h4: telemetry emission contract.

Uses Helper A/B/C from the plan header. Rule-registered tests require
package-import to fire register_all (PR-2), so the autouse snapshot fixture
restores RECOVERY_RULES between tests.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


# ---------- Helpers A / B / C ----------


def _bad_request(body_str: str, *, status: int = 400) -> openai.BadRequestError:
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req, text=body_str)
    return openai.BadRequestError(message=body_str, response=resp, body={"message": body_str})


class _StubBaseModel(BaseChatModel):
    def __init__(self, llm_type: str = "stub", **data):
        super().__init__(**data)
        object.__setattr__(self, "_llm_type_override", llm_type)
        self._agenerate = AsyncMock()
        self._astream = None

    @property
    def _llm_type(self) -> str:
        return getattr(self, "_llm_type_override", "stub")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def _generate(self, *a, **kw):
        raise NotImplementedError("async-only stub")

    def bind_tools(self, tools, **kw):
        return self


def _make_recovery_wrapper(inner, profile, **kw):
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
    )
    return ActusRecoveryChatModel.model_construct(
        inner=inner, profile=profile,
        api_mode=kw.get("api_mode", "chat_completions"),
        on_context_overflow=kw.get("on_context_overflow"),
        max_rewrite_attempts=kw.get("max_rewrite_attempts", 2),
    )


class _TelemetryProbe:
    """Minimal telemetry port — captures emit_recovery_event calls."""

    def __init__(self):
        self.events = []

    def emit_recovery_event(self, event):
        self.events.append(event)


@pytest.fixture(autouse=True)
def _recovery_rules_snapshot():
    from app.domain.services.recovery._registry import RECOVERY_RULES
    saved = dict(RECOVERY_RULES)
    try:
        yield
    finally:
        RECOVERY_RULES.clear()
        RECOVERY_RULES.update(saved)


# ---------- T50, T53, T55 (basic emission contracts) ----------


async def test_T50_recovery_event_emitted_for_retry_sent():
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401  (PR-2 register_all)

    inner = _StubBaseModel()
    call = {"n": 0}

    async def _agenerate(*a, **kw):
        call["n"] += 1
        if call["n"] == 1:
            raise _bad_request("Json mode response is not supported when enable_thinking is true")
        return ChatResult(generations=[ChatGeneration(message=HumanMessage(content="ok"))])

    inner._agenerate = _agenerate

    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    await wrapper._agenerate([HumanMessage(content="x")])
    outcomes = [e.outcome for e in probe.events]
    assert "retry_sent" in outcomes
    assert "success" in outcomes


async def test_T53_attempt_zero_success_emits_no_event():
    from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE

    inner = _StubBaseModel()
    inner._agenerate = AsyncMock(
        return_value=ChatResult(generations=[ChatGeneration(message=HumanMessage(content="ok"))])
    )
    wrapper = _make_recovery_wrapper(inner, GENERIC_OPENAI_PROFILE)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    await wrapper._agenerate([HumanMessage(content="x")])
    assert probe.events == []


async def test_T55_same_call_id_across_attempts():
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401

    inner = _StubBaseModel()
    call = {"n": 0}

    async def _agenerate(*a, **kw):
        call["n"] += 1
        if call["n"] == 1:
            raise _bad_request("Json mode response is not supported when enable_thinking is true")
        return ChatResult(generations=[ChatGeneration(message=HumanMessage(content="ok"))])

    inner._agenerate = _agenerate

    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    await wrapper._agenerate([HumanMessage(content="x")])
    call_ids = {e.call_id for e in probe.events}
    assert len(call_ids) == 1


# ---------- T51, T52, T54 (rule_missed / budget_exhausted / no-PII) ----------


async def test_T51_recovery_event_emitted_for_rule_missed():
    from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE
    import app.domain.services.recovery  # noqa: F401

    inner = _StubBaseModel()
    inner._agenerate = AsyncMock(side_effect=_bad_request("unknown quirk"))
    wrapper = _make_recovery_wrapper(inner, GENERIC_OPENAI_PROFILE)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    with pytest.raises(openai.BadRequestError):
        await wrapper._agenerate([HumanMessage(content="x")])
    assert any(e.outcome == "rule_missed" for e in probe.events)


async def test_T52_recovery_event_emitted_for_budget_exhausted():
    """Audit Round 7 P1 #1: must trigger BOTH R1 candidates so all
    max+1=3 inner calls fire and the third hits the budget_exhausted branch.
    """
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401

    inner = _StubBaseModel()
    inner._agenerate = AsyncMock(side_effect=_bad_request(
        "Json mode response is not supported when enable_thinking is true"
    ))
    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE, max_rewrite_attempts=2)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    with pytest.raises(openai.BadRequestError):
        await wrapper._agenerate(
            [HumanMessage(content="x")],
            response_format={"type": "json_schema"},
        )
    assert any(e.outcome == "budget_exhausted" for e in probe.events), (
        f"Expected budget_exhausted; got outcomes={[e.outcome for e in probe.events]}"
    )


def test_T54_recovery_event_has_no_kwargs_snapshot():
    """RecoveryEvent must NOT embed kwargs / messages / body (PII defense)."""
    import dataclasses
    from app.domain.services.recovery._event import RecoveryEvent
    names = {f.name for f in dataclasses.fields(RecoveryEvent)}
    forbidden = {"kwargs", "messages", "body", "payload", "request_body"}
    assert not (names & forbidden), (
        f"RecoveryEvent must not contain {forbidden & names}"
    )


# ---------- T19g/h/h2/h3/h4 (error_class/fingerprint_code contracts) ----------


async def test_T19g_retry_sent_event_carries_error_class():
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401

    inner = _StubBaseModel()
    call = {"n": 0}

    async def _agenerate(*a, **kw):
        call["n"] += 1
        if call["n"] == 1:
            raise _bad_request("Json mode response is not supported when enable_thinking is true")
        return ChatResult(generations=[ChatGeneration(message=HumanMessage(content="ok"))])

    inner._agenerate = _agenerate

    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    await wrapper._agenerate([HumanMessage(content="x")])
    retry_events = [e for e in probe.events if e.outcome == "retry_sent"]
    assert retry_events
    for e in retry_events:
        assert e.error_class is not None, f"retry_sent event missing error_class: {e}"


async def test_T19h_budget_exhausted_event_carries_error_class():
    """Audit Round 7 P1 #1: same root cause as T52."""
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401

    inner = _StubBaseModel()
    inner._agenerate = AsyncMock(side_effect=_bad_request(
        "Json mode response is not supported when enable_thinking is true"
    ))
    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE, max_rewrite_attempts=2)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    with pytest.raises(openai.BadRequestError):
        await wrapper._agenerate(
            [HumanMessage(content="x")],
            response_format={"type": "json_schema"},
        )
    exhausted = [e for e in probe.events if e.outcome == "budget_exhausted"]
    assert exhausted, (
        f"no budget_exhausted event; outcomes={[e.outcome for e in probe.events]}"
    )
    for e in exhausted:
        assert e.error_class is not None, (
            f"budget_exhausted event missing error_class: {e}"
        )


async def test_T19h3_R4_compact_then_still_overflow_emits_budget_exhausted_not_rule_missed():
    """Audit Round 14 P2 #1 regression-lock: R4 single-candidate exhaustion
    on attempt 1 must emit budget_exhausted, NOT rule_missed.

    Trace:
      attempt=0: 400 CONTEXT_OVERFLOW → R4 → TriggerRecompact applies → retry_sent
      attempt=1: 400 CONTEXT_OVERFLOW → R4 candidate already attempted (I2 dedup)
                 → loop exhausts → budget_exhausted (Round 14: was rule_missed)
                 → wrapper raises
    """
    from app.domain.services.provider_profiles._base import ErrorClass, ErrorDiagnostic
    from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE
    from app.infrastructure.external.llm import actus_recovery_chat_model as wrapper_mod
    import app.domain.services.recovery  # noqa: F401

    def _fake_classify(exc, profile):
        return ErrorDiagnostic(
            error_class=ErrorClass.CONTEXT_OVERFLOW,
            fingerprint_code="context_length_exceeded",
        )

    import pytest as _pytest
    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(wrapper_mod, "classify_error_diagnostic", _fake_classify)
    try:
        compact_calls = {"n": 0}

        async def _cb(messages, kwargs):
            compact_calls["n"] += 1
            return list(messages)

        inner = _StubBaseModel()
        inner._agenerate = AsyncMock(side_effect=_bad_request("context_length_exceeded"))

        wrapper = _make_recovery_wrapper(
            inner, GENERIC_OPENAI_PROFILE, on_context_overflow=_cb,
            max_rewrite_attempts=2,
        )
        probe = _TelemetryProbe()
        wrapper.attach_telemetry(probe, lang="zh")

        with pytest.raises(openai.BadRequestError):
            await wrapper._agenerate([HumanMessage(content="x")])

        outcomes = [e.outcome for e in probe.events]
        assert "retry_sent" in outcomes, f"expected retry_sent on attempt 0; got {outcomes}"
        assert "budget_exhausted" in outcomes, (
            f"R4 single-candidate exhaustion must emit budget_exhausted "
            f"(spec §7.9); got outcomes={outcomes}."
        )
        assert "rule_missed" not in outcomes, (
            f"R4 path must NOT emit rule_missed when candidates were tried "
            f"and exhausted; got {outcomes}"
        )
        assert compact_calls["n"] == 1
    finally:
        monkeypatch.undo()


async def test_T19h4_final_attempt_unknown_emits_rule_missed_not_budget_exhausted():
    """Audit Round 19 P1 #2 regression-lock: at the cap, UNKNOWN /
    TRANSIENT_* / PERMANENT_4XX errors emit rule_missed, NOT budget_exhausted.

    Trace:
      attempt 0: 400 COMPAT_QUIRK → DowngradeToolChoiceToAuto → retry_sent
      attempt 1: ServerRequestsError → classify → UNKNOWN → in I9 set
                 → final_outcome = rule_missed → wrapper raises
    """
    from app.application.errors.exceptions import ServerRequestsError
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401

    inner = _StubBaseModel()
    call = {"n": 0}

    async def _agenerate(*a, **kw):
        call["n"] += 1
        if call["n"] == 1:
            raise _bad_request(
                "Invalid `tool_choice` field. tool_choice is one of the strings: "
                "'auto', 'none' or an object."
            )
        raise ServerRequestsError("rate limited (post-translate)")

    inner._agenerate = _agenerate

    wrapper = _make_recovery_wrapper(
        inner, DASHSCOPE_QWEN_PROFILE, max_rewrite_attempts=1,
    )
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    with pytest.raises(ServerRequestsError):
        await wrapper._agenerate(
            [HumanMessage(content="x")],
            tool_choice="required",
        )

    outcomes = [e.outcome for e in probe.events]
    assert "retry_sent" in outcomes, (
        f"expected retry_sent from attempt 0 R2 rewrite; got {outcomes}"
    )
    assert "budget_exhausted" not in outcomes, (
        f"Final-attempt UNKNOWN/TRANSIENT must emit rule_missed, NOT "
        f"budget_exhausted (E contract + spec §7.7). Got outcomes={outcomes}."
    )
    assert "rule_missed" in outcomes, (
        f"Final-attempt UNKNOWN must emit rule_missed; got {outcomes}"
    )


async def test_T19h2_retry_sent_event_with_wildcard_match_has_none_fingerprint_code(monkeypatch):
    """Spec §8.2 edge case: R4 wildcard match with fingerprint_code=None
    diagnostic still emits retry_sent with fingerprint_code=None.
    """
    from app.domain.services.provider_profiles._base import ErrorClass, ErrorDiagnostic
    from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE
    from app.infrastructure.external.llm import actus_recovery_chat_model as wrapper_mod
    import app.domain.services.recovery  # noqa: F401

    def _fake_classify(exc, profile):
        return ErrorDiagnostic(
            error_class=ErrorClass.CONTEXT_OVERFLOW, fingerprint_code=None,
        )

    monkeypatch.setattr(
        wrapper_mod, "classify_error_diagnostic", _fake_classify,
    )

    compact_called = {"n": 0}

    async def _cb(messages, kwargs):
        compact_called["n"] += 1
        return list(messages)

    inner = _StubBaseModel()
    call = {"n": 0}

    async def _agenerate(*a, **kw):
        call["n"] += 1
        if call["n"] == 1:
            raise _bad_request("arbitrary body — classify is monkeypatched")
        return ChatResult(generations=[ChatGeneration(message=HumanMessage(content="ok"))])

    inner._agenerate = _agenerate
    wrapper = _make_recovery_wrapper(
        inner, GENERIC_OPENAI_PROFILE, on_context_overflow=_cb,
    )
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    await wrapper._agenerate([HumanMessage(content="x")])
    retry_events = [e for e in probe.events if e.outcome == "retry_sent"]
    assert retry_events, "R4 wildcard did not emit retry_sent"
    for e in retry_events:
        assert e.error_class == ErrorClass.CONTEXT_OVERFLOW
        assert e.fingerprint_code is None, (
            f"R4 wildcard retry_sent must preserve fingerprint_code=None, got {e.fingerprint_code!r}"
        )
    assert compact_called["n"] == 1, "R4 TriggerRecompact callback was not invoked"


# ---------- T56_T57, T58 (success last_action_code; astream order) ----------


async def test_T56_T57_success_event_carries_last_action_code():
    """Round 22 P2 #2 fix: must exercise the S1 cumulative chain.

    Trace under R1:
      attempt 0: 400 quirk + per-call response_format → StripResponseFormat → retry_sent
      attempt 1: 400 quirk again (rf=None now, thinking still on)
                 → StripRF skips → DisableThinking applies → retry_sent
      attempt 2: 200 ok → success event with last_action_code = 'disable_thinking'
    """
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401

    inner = _StubBaseModel()
    call = {"n": 0}

    async def _agenerate(*a, **kw):
        call["n"] += 1
        if call["n"] <= 2:
            raise _bad_request(
                "Json mode response is not supported when enable_thinking is true"
            )
        return ChatResult(generations=[ChatGeneration(message=HumanMessage(content="ok"))])

    inner._agenerate = _agenerate

    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE, max_rewrite_attempts=2)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    await wrapper._agenerate(
        [HumanMessage(content="x")],
        response_format={"type": "json_schema"},
    )

    retries = [e for e in probe.events if e.outcome == "retry_sent"]
    assert [e.action_code for e in retries] == [
        "strip_response_format", "disable_thinking",
    ], (
        f"Round 22 P2 #2: must see both candidates fire in order; "
        f"got {[e.action_code for e in retries]}"
    )

    successes = [e for e in probe.events if e.outcome == "success"]
    assert len(successes) == 1
    assert successes[0].action_code == "disable_thinking", (
        f"success.action_code must equal LAST applied action's code; "
        f"got {successes[0].action_code!r}"
    )
    assert successes[0].error_class is None
    assert successes[0].fingerprint_code is None


async def test_T58_astream_success_event_before_first_yield():
    """Round 22 P2 #3: success event must be emitted BEFORE first chunk yields."""
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401

    inner = _StubBaseModel()
    call = {"n": 0}

    async def inner_astream(messages, **kw):
        call["n"] += 1
        if call["n"] == 1:
            raise _bad_request("Json mode response is not supported when enable_thinking is true")
        yield ChatGenerationChunk(message=AIMessageChunk(content="chunk-1"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="chunk-2"))

    inner._astream = inner_astream

    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE)
    probe = _TelemetryProbe()
    wrapper.attach_telemetry(probe, lang="zh")

    chunks = []
    async for c in wrapper._astream([HumanMessage(content="x")]):
        if not chunks:
            success_now = [e for e in probe.events if e.outcome == "success"]
            assert len(success_now) == 1, (
                f"spec §8.2 / Round 22 P2 #3: success event must be emitted "
                f"BEFORE first chunk yields. Probe at first-chunk receipt: "
                f"{[e.outcome for e in probe.events]}"
            )
        chunks.append(c)

    success_events = [e for e in probe.events if e.outcome == "success"]
    assert len(success_events) == 1
    assert len(chunks) == 2
