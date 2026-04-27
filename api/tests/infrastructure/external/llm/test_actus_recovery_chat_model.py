"""PR-1 skeleton tests (T01-T09).

Uses Helper A (_bad_request) + Helper B (_StubBaseModel / _make_recovery_wrapper)
+ Helper C (_recovery_rules_snapshot) from the plan's "Common Test Helpers"
section. See plan header for rationale.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.domain.services.provider_profiles._base import ErrorClass, ProviderProfile


pytestmark = pytest.mark.anyio


# ---------- Helpers A / B / C (copied from plan header — mandatory) ----------


def _bad_request(body_str: str, *, status: int = 400) -> openai.BadRequestError:
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req, text=body_str)
    return openai.BadRequestError(message=body_str, response=resp, body={"message": body_str})


class _StubBaseModel(BaseChatModel):
    """Real BaseChatModel subclass so wrap_with_recovery / bind_tools
    internal ctor validation accepts it. See plan header Helper B rationale
    (Audit Round 3 P1 #3).

    Constructor preserves the pre-fix `_StubBaseModel("primary")` call
    signature via a positional `llm_type` kwarg so existing test code in
    later tasks doesn't need mechanical rewrites.
    """

    def __init__(self, llm_type: str = "stub", **data):
        super().__init__(**data)
        object.__setattr__(self, "_llm_type_override", llm_type)
        # Tests assign _agenerate / _astream directly — BaseChatModel uses
        # extra='ignore' and allows setattr on undeclared fields.
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


def _make_recovery_wrapper(
    inner,
    profile,
    *,
    api_mode: str = "chat_completions",
    on_context_overflow=None,
    max_rewrite_attempts: int = 2,
):
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
    )
    return ActusRecoveryChatModel.model_construct(
        inner=inner,
        profile=profile,
        api_mode=api_mode,
        on_context_overflow=on_context_overflow,
        max_rewrite_attempts=max_rewrite_attempts,
    )


@pytest.fixture(autouse=True)
def _recovery_rules_snapshot():
    from app.domain.services.recovery._registry import RECOVERY_RULES
    saved = dict(RECOVERY_RULES)
    try:
        yield
    finally:
        RECOVERY_RULES.clear()
        RECOVERY_RULES.update(saved)


# ---------- Fixtures ----------


def _fake_profile(provider_id: str = "test_provider") -> ProviderProfile:
    return ProviderProfile(
        provider_id=provider_id,
        human_name="Test",
        default_api_mode="chat_completions",
        api_mode_fallback_enabled=True,
    )


def _stub_success(text: str = "ok") -> _StubBaseModel:
    inner = _StubBaseModel()
    inner._agenerate = AsyncMock(
        return_value=ChatResult(
            generations=[ChatGeneration(message=HumanMessage(content=text))]
        )
    )

    async def _astream_once(*a, **kw):
        yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    inner._astream = _astream_once
    return inner


# ---------- Tests ----------


async def test_T01_empty_registry_behaves_as_noop():
    """T01: empty rule table → all calls equivalent to inner._agenerate."""
    from app.domain.services.recovery._registry import RECOVERY_RULES
    RECOVERY_RULES.clear()

    inner = _stub_success("hello")
    wrapper = _make_recovery_wrapper(inner, _fake_profile())
    result = await wrapper._agenerate([HumanMessage(content="hi")])
    assert result.generations[0].message.content == "hello"
    inner._agenerate.assert_awaited_once()


def test_T02_api_mode_required_at_construction():
    """Non-model_construct path MUST validate api_mode.

    Use a real BaseChatModel subclass (via .model_construct()) so the
    validation error comes from the missing `api_mode` kwarg, not from
    inner-type validation against BaseChatModel.
    """
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
    )
    inner = ActusChatModel.model_construct()
    with pytest.raises((TypeError, ValueError)):
        ActusRecoveryChatModel(
            inner=inner,
            profile=_fake_profile(),
            on_context_overflow=None,
            # api_mode intentionally missing
        )


def test_T03_clone_bind_tools_propagates_fields():
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
    )
    wrapper = _make_recovery_wrapper(
        _stub_success(), _fake_profile(), max_rewrite_attempts=3,
    )
    clone = wrapper.bind_tools([])
    assert isinstance(clone, ActusRecoveryChatModel)
    assert clone.api_mode == "chat_completions"
    assert clone.profile is wrapper.profile
    assert clone.max_rewrite_attempts == 3


def test_T04_with_structured_output_preserves_recovery_via_bind_tools():
    """Audit Round 16 P2 #3: spec §10.1 T04 / I11 require that
    `wrapper.with_structured_output(schema)` produces a Runnable whose
    underlying model is the Recovery wrapper. Spec §4.3 (line 720-735)
    explicitly does NOT override `with_structured_output`; it relies on
    BaseChatModel's default implementation, which internally calls
    `self.bind_tools([schema_as_tool], tool_choice=...)` and then wraps
    the bound model with a parser. Because Recovery's `bind_tools` returns
    a new `ActusRecoveryChatModel`, the structured-output path must STILL
    route through Recovery's _agenerate/_astream.

    Production callsites that rely on this:
      - main_graph.py:312 (planner structured output)
      - main_graph.py:831 (updater structured output)

    Without this test, a future refactor that overrides
    `with_structured_output` to short-circuit `bind_tools` would silently
    bypass Recovery on every planner/updater LLM call.
    """
    from langchain_core.runnables import Runnable
    from pydantic import BaseModel

    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
    )

    class _Schema(BaseModel):
        answer: str

    wrapper = _make_recovery_wrapper(_stub_success(), _fake_profile())
    structured = wrapper.with_structured_output(_Schema)

    assert isinstance(structured, Runnable)

    steps = getattr(structured, "steps", None) or getattr(structured, "first", None)
    if steps is None:
        steps = [structured]
    elif not isinstance(steps, list):
        steps = [steps]

    found_recovery = False
    for step in steps:
        if isinstance(step, ActusRecoveryChatModel):
            found_recovery = True
            break
        bound = getattr(step, "bound", None)
        if isinstance(bound, ActusRecoveryChatModel):
            found_recovery = True
            break

    assert found_recovery, (
        "with_structured_output must preserve Recovery wrapper through bind_tools. "
        "If this fails, either the wrapper now overrides with_structured_output "
        "(violating spec §4.3 / I11) or bind_tools no longer returns "
        "ActusRecoveryChatModel."
    )


async def test_T05_budget_override_zero_disables_recovery(monkeypatch):
    """max_rewrite_attempts=0 → pure pass-through: first failure raises,
    no classify, no match_rule, no RecoveryEvent emission.

    Pre-fix bug: the cap-hit branch (``attempt == max_rewrite_attempts``)
    fired on the first failure when max=0, calling
    ``classify_error_diagnostic`` and ``_emit_recovery_event`` before the
    re-raise. PR-3 telemetry would then leak events on the disable path.
    Post-fix: ``_agenerate`` short-circuits to inner BEFORE entering the
    recovery loop when ``max_rewrite_attempts <= 0``.
    """
    from app.domain.services.recovery._registry import RECOVERY_RULES
    from app.infrastructure.external.llm import actus_recovery_chat_model as recovery_mod
    RECOVERY_RULES.clear()

    classify_calls: list[Exception] = []
    real_classify = recovery_mod.classify_error_diagnostic

    def _spy_classify(exc, profile):
        classify_calls.append(exc)
        return real_classify(exc, profile)

    monkeypatch.setattr(recovery_mod, "classify_error_diagnostic", _spy_classify)

    inner = _StubBaseModel()
    inner._agenerate = AsyncMock(side_effect=RuntimeError("boom"))
    wrapper = _make_recovery_wrapper(inner, _fake_profile(), max_rewrite_attempts=0)

    emit_calls: list[dict] = []
    monkeypatch.setattr(
        wrapper,
        "_emit_recovery_event",
        lambda **kw: emit_calls.append(kw),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await wrapper._agenerate([HumanMessage(content="hi")])
    inner._agenerate.assert_awaited_once()
    assert classify_calls == [], (
        "max_rewrite_attempts=0 must short-circuit BEFORE classify_error_diagnostic"
    )
    assert emit_calls == [], (
        "max_rewrite_attempts=0 must NOT emit any RecoveryEvent"
    )


async def test_T05b_budget_override_zero_disables_recovery_for_astream(monkeypatch):
    """Same disable contract as T05 but for the streaming path."""
    from app.domain.services.recovery._registry import RECOVERY_RULES
    from app.infrastructure.external.llm import actus_recovery_chat_model as recovery_mod
    RECOVERY_RULES.clear()

    classify_calls: list[Exception] = []
    real_classify = recovery_mod.classify_error_diagnostic

    def _spy_classify(exc, profile):
        classify_calls.append(exc)
        return real_classify(exc, profile)

    monkeypatch.setattr(recovery_mod, "classify_error_diagnostic", _spy_classify)

    inner = _StubBaseModel()

    async def _failing_stream(*a, **kw):
        raise RuntimeError("stream boom")
        yield  # pragma: no cover — make this an async generator

    inner._astream = _failing_stream
    wrapper = _make_recovery_wrapper(inner, _fake_profile(), max_rewrite_attempts=0)

    emit_calls: list[dict] = []
    monkeypatch.setattr(
        wrapper,
        "_emit_recovery_event",
        lambda **kw: emit_calls.append(kw),
    )

    with pytest.raises(RuntimeError, match="stream boom"):
        async for _ in wrapper._astream([HumanMessage(content="hi")]):
            pass
    assert classify_calls == [], (
        "max_rewrite_attempts=0 must short-circuit BEFORE classify_error_diagnostic"
    )
    assert emit_calls == [], (
        "max_rewrite_attempts=0 must NOT emit any RecoveryEvent"
    )


async def test_T09_permanent_4xx_passes_through_unchanged():
    """I9: PERMANENT_4XX stays in I9 pass-through set — no retry."""
    from app.domain.services.recovery._registry import RECOVERY_RULES
    RECOVERY_RULES.clear()

    inner = _StubBaseModel()
    # 403 → no fingerprint match → exception-class fallback → BadRequestError
    # isinstance check → PERMANENT_4XX (via openai.BadRequestError branch of
    # _classify.py:82-90).
    inner._agenerate = AsyncMock(side_effect=_bad_request("forbidden", status=403))
    wrapper = _make_recovery_wrapper(inner, _fake_profile())

    with pytest.raises(openai.BadRequestError):
        await wrapper._agenerate([HumanMessage(content="hi")])
    inner._agenerate.assert_awaited_once()


def test_T17_build_llm_signature_unchanged():
    import inspect
    from app.interfaces.service_dependencies import _build_llm
    sig = inspect.signature(_build_llm)
    params = list(sig.parameters.keys())
    assert params[0] == "llm_config"
    assert "supports_pdf_input" in sig.parameters
    assert "on_context_overflow" not in sig.parameters
    assert "profile" not in sig.parameters


def test_T18_wrap_with_recovery_preserves_fallback_topology():
    from app.infrastructure.external.llm.actus_fallback_chat_model import (
        ActusFallbackChatModel,
    )
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
        wrap_with_recovery,
    )
    primary = _stub_success("P")
    fallback = _stub_success("F")
    # Use a non-default provider_name to catch the Round 4 P2 #4 regression:
    # if wrap_with_recovery drops provider_name, this would revert to "openai".
    fb = ActusFallbackChatModel.model_construct(
        primary=primary, fallback=fallback, provider_name="anthropic", profile=None,
    )
    wrapped = wrap_with_recovery(fb, profile=_fake_profile(), on_context_overflow=None)

    assert isinstance(wrapped, ActusFallbackChatModel)
    assert isinstance(wrapped.primary, ActusRecoveryChatModel)
    assert isinstance(wrapped.fallback, ActusRecoveryChatModel)
    assert wrapped.primary.api_mode == "chat_completions"
    assert wrapped.fallback.api_mode == "responses"
    # Audit Round 4 P2 #4 regression gate
    assert wrapped.provider_name == "anthropic", (
        "wrap_with_recovery must preserve provider_name; otherwise downstream "
        "routing / telemetry silently reverts to the 'openai' default."
    )


def test_T19_wrap_with_recovery_two_sessions_independent_callbacks():
    """Audit Round 3 P2 #4 fix: remove `object.__setattr__(cached, "_llm_type", ...)`.
    `_llm_type` is a read-only @property on ActusChatModel (`actus_chat_model.py:170`),
    overriding it raises `AttributeError: property ... has no setter`. The test
    doesn't need to customize _llm_type to verify callback independence — just
    use the default model_construct() instance.
    """
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        wrap_with_recovery,
    )
    cached = ActusChatModel.model_construct()

    async def cb_a(m, k): return None
    async def cb_b(m, k): return None

    w1 = wrap_with_recovery(cached, profile=_fake_profile(), on_context_overflow=cb_a)
    w2 = wrap_with_recovery(cached, profile=_fake_profile(), on_context_overflow=cb_b)
    assert w1.on_context_overflow is cb_a
    assert w2.on_context_overflow is cb_b


def test_T19b_wrap_with_recovery_inherits_telemetry_from_inner():
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        wrap_with_recovery,
    )
    inner = ActusChatModel.model_construct()
    tel = object()  # anonymous sentinel — Helper B doesn't need a MagicMock
    object.__setattr__(inner, "_telemetry", tel)
    object.__setattr__(inner, "_telemetry_lang", "en")
    wrapper = wrap_with_recovery(inner, profile=_fake_profile(), on_context_overflow=None)
    assert getattr(wrapper, "_telemetry", None) is tel
    assert getattr(wrapper, "_telemetry_lang", None) == "en"


def test_T19i_bind_tools_propagates_telemetry():
    wrapper = _make_recovery_wrapper(_stub_success(), _fake_profile())
    tel = object()
    wrapper.attach_telemetry(tel, lang="en")
    clone = wrapper.bind_tools([])
    assert getattr(clone, "_telemetry", None) is tel
    assert getattr(clone, "_telemetry_lang", None) == "en"


def test_T19p_wrapper_identifying_params_forwards_inner_model_and_provider():
    """Audit Round 19 P1 #1 regression-lock: ActusRecoveryChatModel must
    expose `model` + `provider_id` in `_identifying_params` so B4's
    CostCallbackHandler reads them via
    `BaseChatModel._get_invocation_params(...)["invocation_params"]`
    (`cost_callback_handler.py:159`). Without this override, every
    Recovery-wrapped LLM call lands in CostRecord with `model='unknown'`
    and heuristic provider attribution.

    Verified manually: a BaseChatModel subclass without _identifying_params
    override exposes only `{'_type': 'wrapper', 'stop': None}` to
    invocation_params (uv run python -c).
    """
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

    inner = ActusChatModel.model_construct(
        model_name="qwen-max-2025",
    )
    object.__setattr__(inner, "profile", _fake_profile("dashscope_qwen"))

    wrapper = _make_recovery_wrapper(inner, _fake_profile("dashscope_qwen"))
    params = wrapper._identifying_params

    assert params.get("model") == "qwen-max-2025", (
        f"wrapper._identifying_params['model'] must forward inner.model_name; "
        f"got {params.get('model')!r}. Without this, CostRecord ends up with "
        f"model='unknown' and B4 cost ledger loses model attribution."
    )
    assert params.get("provider_id") == "dashscope_qwen", (
        f"wrapper._identifying_params['provider_id'] must forward "
        f"profile.provider_id (or inner's identifying_params); "
        f"got {params.get('provider_id')!r}."
    )

    # Sanity: the keys are actually visible through BaseChatModel's
    # canonical accessor — that's the path CostCallbackHandler walks.
    invocation = wrapper._get_invocation_params({})
    assert invocation.get("model") == "qwen-max-2025"
    assert invocation.get("provider_id") == "dashscope_qwen"


def test_T19j_inherit_telemetry_reads_correct_fields():
    """Field name `_telemetry` per _telemetry_mixin.py:67-68 (NOT `_prompt_telemetry`)."""
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        _inherit_telemetry,
    )
    inner = _stub_success()
    tel = object()
    object.__setattr__(inner, "_telemetry", tel)
    object.__setattr__(inner, "_telemetry_lang", "zh")
    wrapper = _make_recovery_wrapper(inner, _fake_profile())
    _inherit_telemetry(wrapper, inner)
    assert getattr(wrapper, "_telemetry", None) is tel


def test_T19n_wrap_with_recovery_responses_adapter_api_mode_is_responses():
    """Dispatch on inner adapter TYPE, not profile.default_api_mode (v10 #2)."""
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
    from app.infrastructure.external.llm.actus_responses_model import (
        ActusResponsesModel,
    )
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        wrap_with_recovery,
    )
    # model_construct bypasses required-field validation so the stand-in
    # adapter can be used as the `inner` arg without a full config.
    chat = ActusChatModel.model_construct()
    wrapped_chat = wrap_with_recovery(chat, profile=_fake_profile(), on_context_overflow=None)
    assert wrapped_chat.api_mode == "chat_completions"

    resp = ActusResponsesModel.model_construct()
    wrapped_resp = wrap_with_recovery(resp, profile=_fake_profile(), on_context_overflow=None)
    assert wrapped_resp.api_mode == "responses"
