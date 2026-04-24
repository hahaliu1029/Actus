"""Task 3.8 — ActusFallbackChatModel profile guard regression tests.

T12: api_mode_fallback_enabled=False → re-raise without invoking fallback.
T12 regression direction: api_mode_fallback_enabled=True → fallback DOES fire.

Uses ``model_construct`` to bypass Pydantic BaseChatModel validation when
substituting MagicMock primary/fallback (mirrors test_fallback_api_mode_gate.py).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.domain.services.provider_profiles import get_profile
from app.infrastructure.external.llm.actus_fallback_chat_model import (
    ActusFallbackChatModel,
)


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _bad_request() -> openai.BadRequestError:
    req = httpx.Request("POST", "https://x/v1/chat")
    resp = httpx.Response(400, request=req, text="bad")
    return openai.BadRequestError("bad", response=resp, body={"message": "bad"})


def _ok_chat_result(content: str = "ok") -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content=content))]
    )


async def test_fallback_bypasses_when_profile_disables() -> None:
    """T12: api_mode_fallback_enabled=False → re-raise directly, no fallback"""
    primary = MagicMock()
    primary._llm_type = "primary-stub"
    primary._agenerate = AsyncMock(side_effect=_bad_request())
    fallback = MagicMock()
    fallback._llm_type = "fallback-stub"
    fallback._agenerate = AsyncMock()

    wrapper = ActusFallbackChatModel.model_construct(
        primary=primary,
        fallback=fallback,
        provider_name="openai",
        profile=get_profile("kimi_k2"),  # api_mode_fallback_enabled=False
    )

    with pytest.raises(openai.BadRequestError):
        await wrapper._agenerate([HumanMessage("q")])
    fallback._agenerate.assert_not_called()


async def test_fallback_triggers_when_profile_enables() -> None:
    """generic_openai profile → fallback 正常升级"""
    primary = MagicMock()
    primary._llm_type = "primary-stub"
    primary._agenerate = AsyncMock(side_effect=_bad_request())
    fallback = MagicMock()
    fallback._llm_type = "fallback-stub"
    fallback._agenerate = AsyncMock(return_value=_ok_chat_result("fallback-result"))

    wrapper = ActusFallbackChatModel.model_construct(
        primary=primary,
        fallback=fallback,
        provider_name="openai",
        profile=get_profile("generic_openai"),  # api_mode_fallback_enabled=True
    )

    result = await wrapper._agenerate([HumanMessage("q")])
    fallback._agenerate.assert_called_once()
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "fallback-result"


# ---------- COMPAT_QUIRK re-raise branch (actus_fallback_chat_model.py:176) ----------
#
# Reaching this branch requires a profile that:
#   1. has ``api_mode_fallback_enabled=True`` (so the first guard is bypassed), AND
#   2. has an error_fingerprints entry that classifies the raised exception as
#      ``ErrorClass.COMPAT_QUIRK``.
# No shipped profile satisfies both (Kimi / DeepSeek have the fingerprints but
# disable fallback; generic_openai / openai_official enable fallback but have
# no COMPAT_QUIRK fingerprints). Use ``dataclasses.replace`` on DeepSeek
# Reasoner to get a synthetic profile that isolates just this branch.


def _compat_quirk_bad_request() -> openai.BadRequestError:
    """BadRequestError whose body matches DeepSeek Reasoner's COMPAT_QUIRK
    fingerprint (``"Missing reasoning_content"``)."""
    req = httpx.Request("POST", "https://x/v1/chat")
    body_text = "Missing reasoning_content in assistant message at index 2"
    resp = httpx.Response(400, request=req, text=body_text)
    return openai.BadRequestError(
        "bad", response=resp, body={"message": body_text},
    )


def _profile_with_fallback_enabled_and_compat_fingerprint():
    """DeepSeek Reasoner fingerprints + api_mode_fallback_enabled=True.

    Isolates the COMPAT_QUIRK guard from the api_mode_fallback guard so a
    failure in this test points directly at the COMPAT_QUIRK branch.
    """
    from dataclasses import replace
    return replace(
        get_profile("deepseek_reasoner"),
        api_mode_fallback_enabled=True,
    )


async def test_fallback_reraises_on_compat_quirk_even_when_fallback_enabled() -> None:
    """Task 3.8 T12: COMPAT_QUIRK-classified primary error must skip the
    Chat→Responses escalation even when the profile has
    ``api_mode_fallback_enabled=True`` — escalation would surface the same
    quirk on Responses (same payload shape). B2 / upper layer consumes the
    typed COMPAT_QUIRK signal instead.
    """
    profile = _profile_with_fallback_enabled_and_compat_fingerprint()
    assert profile.api_mode_fallback_enabled is True  # first guard bypassed

    primary = MagicMock()
    primary._llm_type = "primary-stub"
    primary._agenerate = AsyncMock(side_effect=_compat_quirk_bad_request())
    fallback = MagicMock()
    fallback._llm_type = "fallback-stub"
    fallback._agenerate = AsyncMock(
        return_value=_ok_chat_result("should-not-be-reached"),
    )

    wrapper = ActusFallbackChatModel.model_construct(
        primary=primary,
        fallback=fallback,
        provider_name="openai",
        profile=profile,
    )

    with pytest.raises(openai.BadRequestError):
        await wrapper._agenerate([HumanMessage("q")])
    fallback._agenerate.assert_not_called()


async def test_fallback_still_escalates_on_non_compat_quirk_when_enabled() -> None:
    """Regression sibling: when fallback is enabled AND the error does NOT
    classify as COMPAT_QUIRK, escalation still happens. Same synthetic profile,
    different error body so the fingerprint does not match (falls through to
    the exception-class PERMANENT_4XX branch in ``_classify``).
    """
    profile = _profile_with_fallback_enabled_and_compat_fingerprint()

    primary = MagicMock()
    primary._llm_type = "primary-stub"
    primary._agenerate = AsyncMock(side_effect=_bad_request())  # body="bad"
    fallback = MagicMock()
    fallback._llm_type = "fallback-stub"
    fallback._agenerate = AsyncMock(
        return_value=_ok_chat_result("fallback-result"),
    )

    wrapper = ActusFallbackChatModel.model_construct(
        primary=primary,
        fallback=fallback,
        provider_name="openai",
        profile=profile,
    )

    result = await wrapper._agenerate([HumanMessage("q")])
    fallback._agenerate.assert_called_once()
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "fallback-result"


async def test_astream_reraises_on_compat_quirk_even_when_fallback_enabled() -> None:
    """Mirror of the ``_agenerate`` COMPAT_QUIRK guard on the streaming path
    (actus_fallback_chat_model.py around line 233)."""
    profile = _profile_with_fallback_enabled_and_compat_fingerprint()
    fallback_iters = 0

    async def _primary_raises(*_args, **_kwargs):
        raise _compat_quirk_bad_request()
        yield  # pragma: no cover — keeps this an async generator

    async def _fallback_stream(*_args, **_kwargs):
        nonlocal fallback_iters
        fallback_iters += 1
        yield None  # pragma: no cover — guarded by assertion below

    primary = MagicMock()
    primary._llm_type = "primary-stub"
    primary._astream = _primary_raises
    fallback = MagicMock()
    fallback._llm_type = "fallback-stub"
    fallback._astream = _fallback_stream

    wrapper = ActusFallbackChatModel.model_construct(
        primary=primary,
        fallback=fallback,
        provider_name="openai",
        profile=profile,
    )

    with pytest.raises(openai.BadRequestError):
        async for _chunk in wrapper._astream([HumanMessage("q")]):
            pass  # pragma: no cover — primary raises before any chunk

    assert fallback_iters == 0, (
        "_astream must not invoke fallback when primary error is COMPAT_QUIRK"
    )


async def test_fallback_bypasses_when_profile_is_glm() -> None:
    """T-P1-SMOKE-3: GLM profile api_mode_fallback_enabled=False 生效 (Spec §8.4).

    验证 Finding 2 的 GLM 迁移路径：今天通过 _build_llm try/except silent-fallback
    到 generic_openai (api_mode_fallback_enabled=True)，本 PR 之后切到 glm profile
    (api_mode_fallback_enabled=False)。ActusFallbackChatModel 的 guard 把
    previously-silent 404-on-fallback 转成 typed re-raise，不误升到 Responses API。
    """
    primary = MagicMock()
    primary._llm_type = "primary-stub"
    primary._agenerate = AsyncMock(side_effect=_bad_request())
    fallback = MagicMock()
    fallback._llm_type = "fallback-stub"
    fallback._agenerate = AsyncMock()

    wrapper = ActusFallbackChatModel.model_construct(
        primary=primary,
        fallback=fallback,
        provider_name="openai",
        profile=get_profile("glm"),   # A7 P1: api_mode_fallback_enabled=False
    )

    with pytest.raises(openai.BadRequestError):
        await wrapper._agenerate([HumanMessage("q")])
    fallback._agenerate.assert_not_called()
