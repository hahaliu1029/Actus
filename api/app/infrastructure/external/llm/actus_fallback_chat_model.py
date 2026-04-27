"""ActusFallbackChatModel — BaseChatModel with primary/fallback delegation.

Replaces ``primary.with_fallbacks([fallback])`` to avoid a langchain-core bug
where ``RunnableWithFallbacks.__getattr__`` calls ``typing.get_type_hints()``
on ``BaseChatModel.with_structured_output``, which fails because ``builtins``
is imported under ``TYPE_CHECKING`` only in langchain-core.

Semantics: this wrapper is for **cross-protocol escalation**
(chat.completions → responses), not same-protocol retry. It only falls
through on exceptions that specifically indicate "primary doesn't accept
this protocol/payload" — currently:

- ``openai.BadRequestError`` (400)
- ``openai.UnprocessableEntityError`` (422)

``openai.NotFoundError`` is deliberately NOT in the trigger set:
``NotFoundError`` is a generic 404 that also fires on wrong model name
or wrong base_url path, which are permanent user-config errors that
would be silently papered over by an escalation hop. If a future
provider really needs "chat endpoint missing → try responses" routing,
replace this tuple with a predicate that inspects ``exc.request.url``
or ``exc.body["code"]`` rather than widening the class match.

All other exceptions (``ServerRequestsError`` — timeout/empty-response
wrapper translated by ``_timeout_helpers``, 5xx, rate limits, auth
failures, arbitrary ``Exception`` subclasses) propagate so LangGraph
node ``RetryPolicy`` or upper layers can retry same endpoint. Funneling
transient errors through fallback hides the real retry path and, on
providers without a Responses API (e.g. Zhipu ``glm-*`` on
``/api/paas/v4``), converts a retryable error into a hard 404.

Both ``bind_tools`` and ``with_structured_output`` propagate to both
inner models so tool schemas stay consistent.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, List, Literal, Optional

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

logger = logging.getLogger(__name__)


def _default_generic_profile() -> Any:
    """Default ProviderProfile factory — generic_openai.

    Used by Pydantic Field(default_factory=...) so existing test fixtures
    constructing ActusFallbackChatModel(...) without profile still work.
    _build_llm passes an explicit profile in the production path.
    """
    from app.domain.services.provider_profiles import get_profile
    return get_profile("generic_openai")

# Exception types that mean "primary does not accept this protocol/payload".
# Only these trigger the cross-protocol escalation to fallback. Anything
# else propagates — see module docstring for why ``NotFoundError`` is NOT
# included.
_FALLBACK_TRIGGER_EXCEPTIONS: tuple[type[BaseException], ...] = (
    openai.BadRequestError,
    openai.UnprocessableEntityError,
)


async def _stamp_fallback_escalation(
    stream: AsyncIterator[ChatGenerationChunk],
    fallback_adapter: BaseChatModel,
) -> AsyncIterator[ChatGenerationChunk]:
    """Tag the first chunk from fallback with escalation metadata.

    Streaming path can't use ``_notify_fallback_escalation`` because
    ``BaseChatModel.astream`` doesn't thread ``run_manager`` down to subclass
    ``_astream`` (see ``langchain_core/language_models/chat_models.py``
    line ~668). We encode ``attempt_ix`` / ``model`` / ``provider_id`` into
    the chunk's ``response_metadata``; ``merge_chat_generation_chunks``
    propagates that into the merged message passed to ``on_llm_end``, and
    the handler applies it to the pending entry there.
    """
    try:
        id_params = dict(fallback_adapter._identifying_params or {})
    except Exception:  # noqa: BLE001
        id_params = {}
    stamped = False
    async for chunk in stream:
        if not stamped:
            stamped = True
            try:
                msg = chunk.message
                rm = dict(getattr(msg, "response_metadata", None) or {})
                rm.setdefault("actus_fallback_attempt_ix", 1)
                fb_model = id_params.get("model")
                fb_provider = id_params.get("provider_id")
                if fb_model:
                    rm.setdefault("actus_fallback_model", fb_model)
                if fb_provider:
                    rm.setdefault("actus_fallback_provider", fb_provider)
                msg.response_metadata = rm
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Failed to stamp fallback escalation metadata: %s", exc
                )
        yield chunk


def _notify_fallback_escalation(
    run_manager: Optional[AsyncCallbackManagerForLLMRun],
    fallback_adapter: BaseChatModel,
) -> None:
    """Duck-typed hook: tell cost/telemetry handlers that fallback engaged.

    Any handler attached to this run that exposes
    ``mark_fallback_escalation(run_id, *, attempt_ix=..., model=..., provider=...)``
    gets notified. ``CostCallbackHandler`` uses this to stamp
    ``attempt_ix=1`` on the CostRecord **and** to swap the pending entry's
    ``model`` / ``provider`` over to the fallback adapter's identity — so
    the CostRecord reflects the adapter that actually billed instead of the
    primary whose ``_identifying_params`` was captured on start.

    Errors from handlers are swallowed — a buggy handler must not break
    the LLM call path (same invariant as ``_persist_safely``).
    """
    if run_manager is None:
        return
    # Extract fallback's model/provider so handlers can correct the
    # attribution. ``_identifying_params`` is a property on BaseChatModel;
    # Actus adapters override it to expose real values.
    try:
        id_params = dict(fallback_adapter._identifying_params or {})
    except Exception:
        id_params = {}
    fb_model = id_params.get("model")
    fb_provider = id_params.get("provider_id")

    handlers = getattr(run_manager, "handlers", None) or []
    inline_handlers = getattr(run_manager, "inheritable_handlers", None) or []
    # Inheritable handlers may duplicate entries in ``handlers``; a set of
    # ids avoids double-notification.
    seen: set[int] = set()
    for h in list(handlers) + list(inline_handlers):
        if id(h) in seen:
            continue
        seen.add(id(h))
        hook = getattr(h, "mark_fallback_escalation", None)
        if hook is None:
            continue
        try:
            hook(
                run_manager.run_id,
                attempt_ix=1,
                model=fb_model,
                provider=fb_provider,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Handler %r.mark_fallback_escalation failed: %s",
                type(h).__name__, exc,
            )


class ActusFallbackChatModel(BaseChatModel):
    """BaseChatModel that tries *primary* first and falls back to *fallback*.

    Both ``primary`` and ``fallback`` must be ``BaseChatModel`` instances.
    """

    primary: BaseChatModel
    fallback: BaseChatModel
    # B5 C0a: provider identification — independent field, NOT delegated to inner adapters.
    # Currently all Actus LLM adapters target OpenAI-compatible endpoints; B5.1 may
    # introduce real Anthropic routing via LLMConfig.provider field.
    provider_name: Literal["openai", "anthropic"] = "openai"
    # A7 P0.1: profile held on wrapper for symmetry with inner adapters. Inner
    # primary/fallback each carry their own profile (propagated via their
    # bind_tools), so this wrapper field is informational. Typed Any to
    # override LangChain BaseChatModel.profile (ModelProfile | None).
    profile: Any = Field(default_factory=_default_generic_profile)

    @property
    def _llm_type(self) -> str:
        return "actus-fallback"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Delegate ``model`` + ``provider_id`` to the primary adapter.

        Fallback is transparent by design — the primary's model/provider is
        what the caller "asked for", and since a successful primary
        invocation is the common case, attributing cost to the primary is
        the correct default. If fallback engages, ``attempt_ix`` should
        reflect that (tracked separately as a post-M0 follow-up).
        """
        primary = self.primary
        model = getattr(primary, "model_name", None) or "unknown"
        profile = getattr(primary, "profile", None)
        provider_id = getattr(profile, "provider_id", None) or "unknown"
        return {"model": model, "provider_id": provider_id}

    # ---- B5 C11: telemetry hook ----------------------------------------- #

    def attach_telemetry(self, telemetry: Any, lang: str = "zh") -> None:
        """Forward telemetry attachment to both primary and fallback.

        When the primary succeeds, ``primary._agenerate`` emits one
        event. When the primary fails and the fallback path runs,
        ``fallback._agenerate`` emits a second event. The two events
        for a single logical call are distinguishable by the
        ``provider`` field (and by the ``tools_hash`` if they differ).

        Downstream analysis can dedup adjacent events with the same
        ``(system_prompt_hash, tools_hash)`` if a single-event view
        is desired.

        ``lang`` is forwarded identically to both inner adapters so the
        two events carry the same language attribution.
        """
        if hasattr(self.primary, "attach_telemetry"):
            self.primary.attach_telemetry(telemetry, lang=lang)
        if hasattr(self.fallback, "attach_telemetry"):
            self.fallback.attach_telemetry(telemetry, lang=lang)

    # ---- sync (not used — project is async-only) ------------------------- #

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async interface. Project is async-only.")

    # ---- async ----------------------------------------------------------- #

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 检测多模态内容用于调试
        has_multimodal = any(
            isinstance(m.content, list) for m in messages
            if hasattr(m, "content")
        )
        if has_multimodal:
            logger.info(
                "[MULTIMODAL] FallbackModel dispatching with multimodal content, "
                "primary=%s, fallback=%s",
                getattr(self.primary, "model_name", self.primary._llm_type),
                getattr(self.fallback, "model_name", self.fallback._llm_type),
            )
        try:
            return await self.primary._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )
        except _FALLBACK_TRIGGER_EXCEPTIONS as primary_exc:
            # A7 P1: profile-driven gate. Providers without a Responses API
            # (e.g. Kimi) declare api_mode_fallback_enabled=False; for those,
            # a 400/422 from Chat Completions is a real payload error and
            # must propagate — escalating to Responses will just produce a
            # 404 and hide the original diagnostic. Re-raise the primary
            # exception untouched so LangGraph RetryPolicy / upper layers
            # see the real cause.
            profile = getattr(self, "profile", None)
            if profile is not None and not getattr(
                profile, "api_mode_fallback_enabled", True
            ):
                logger.info(
                    "Primary LLM (%s) raised %s but profile.provider_id=%s "
                    "has api_mode_fallback_enabled=False; skipping Chat->"
                    "Responses escalation and propagating original exception",
                    self.primary._llm_type,
                    type(primary_exc).__name__,
                    getattr(profile, "provider_id", "?"),
                )
                raise
            # A7 Task 3.8 (T12): even when api_mode_fallback_enabled=True, if
            # the primary error classifies as COMPAT_QUIRK (provider-specific
            # known quirk like DeepSeek 'Missing reasoning_content'), a Chat
            # → Responses escalation will not help — the same payload shape
            # would surface the same quirk on Responses. Re-raise so B2 /
            # upper recovery can act on the typed signal.
            if profile is not None:
                from app.domain.services.provider_profiles._classify import (
                    classify_error,
                )
                from app.domain.services.provider_profiles._base import ErrorClass
                err_class = classify_error(primary_exc, profile)
                # B2 PR-2 (T12 extension): COMPAT_QUIRK and CONTEXT_OVERFLOW
                # both stay terminal at the Fallback layer — escalating to
                # Responses can't help (same payload shape would re-trigger).
                # B2 Recovery handles both via typed rules (R1-R3 for QUIRK,
                # R4 for CONTEXT_OVERFLOW). LangGraph RetryPolicy still owns
                # the TRANSIENT_* classes via the existing escalation path.
                if err_class in (ErrorClass.COMPAT_QUIRK, ErrorClass.CONTEXT_OVERFLOW):
                    logger.info(
                        "Primary LLM (%s) raised %s classified as %s "
                        "for profile.provider_id=%s; skipping Chat->Responses "
                        "escalation (B2 Recovery terminal)",
                        self.primary._llm_type,
                        type(primary_exc).__name__,
                        err_class.name,
                        getattr(profile, "provider_id", "?"),
                    )
                    raise
            logger.warning(
                "Primary LLM (%s) protocol incompatible, escalating to %s: %s",
                self.primary._llm_type, self.fallback._llm_type, primary_exc,
            )
            _notify_fallback_escalation(run_manager, self.fallback)
            return await self.fallback._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # B2 PR-2: peek-first-chunk semantics. Once the primary stream has
        # yielded a chunk downstream, mid-stream errors propagate unchanged
        # (B2 contract: cross-protocol escalation must not partial-replay).
        # Only failures that occur BEFORE the first chunk trigger the
        # Chat->Responses fallback path.
        stream = self.primary._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs,
        )
        try:
            first_chunk = await stream.__anext__()
        except StopAsyncIteration:
            # Primary produced an empty stream — no chunks, no error.
            return
        except _FALLBACK_TRIGGER_EXCEPTIONS as primary_exc:
            # A7 P1 + B2 PR-2 T12: profile-driven gate. Mirror _agenerate's
            # gate at lines ~274-308. Any class in (COMPAT_QUIRK,
            # CONTEXT_OVERFLOW) → re-raise so B2 Recovery (one layer up)
            # gets the typed signal; CHAT->Responses can't help.
            profile = getattr(self, "profile", None)
            if profile is not None and not getattr(
                profile, "api_mode_fallback_enabled", True
            ):
                logger.info(
                    "Primary LLM stream (%s) raised %s but profile.provider_id=%s "
                    "has api_mode_fallback_enabled=False; skipping Chat->"
                    "Responses escalation and propagating original exception",
                    self.primary._llm_type,
                    type(primary_exc).__name__,
                    getattr(profile, "provider_id", "?"),
                )
                raise
            if profile is not None:
                from app.domain.services.provider_profiles._classify import (
                    classify_error,
                )
                from app.domain.services.provider_profiles._base import ErrorClass
                err_class = classify_error(primary_exc, profile)
                if err_class in (ErrorClass.COMPAT_QUIRK, ErrorClass.CONTEXT_OVERFLOW):
                    logger.info(
                        "Primary LLM stream (%s) raised %s classified as %s "
                        "for profile.provider_id=%s; skipping Chat->Responses "
                        "escalation (B2 Recovery terminal)",
                        self.primary._llm_type,
                        type(primary_exc).__name__,
                        err_class.name,
                        getattr(profile, "provider_id", "?"),
                    )
                    raise
            logger.warning(
                "Primary LLM stream (%s) protocol incompatible, escalating to %s: %s",
                self.primary._llm_type, self.fallback._llm_type, primary_exc,
            )
            # Audit Round 16 P1 #1 — preserve fallback escalation stamping.
            # Streaming path: ``BaseChatModel.astream`` (langchain-core
            # chat_models.py line 668) does NOT forward ``run_manager`` into
            # the subclass ``_astream``, so ``_notify_fallback_escalation``
            # often sees ``run_manager=None`` and can't reach the handler. We
            # also stamp the first fallback chunk's ``response_metadata``
            # with escalation info; the merged AIMessage on ``on_llm_end``
            # carries it and cost-ledger / telemetry attribute the call to
            # the fallback adapter.
            _notify_fallback_escalation(run_manager, self.fallback)
            async for chunk in _stamp_fallback_escalation(
                self.fallback._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs,
                ),
                self.fallback,
            ):
                yield chunk
            return

        # Primary produced first chunk: lock to primary stream. Mid-stream
        # errors propagate unchanged (NO try/except below).
        yield first_chunk
        async for chunk in stream:
            yield chunk

    # ---- bind_tools / with_structured_output ----------------------------- #

    def bind_tools(self, tools: list, **kwargs: Any) -> "ActusFallbackChatModel":
        # provider_name is an independent field on the wrapper (not delegated
        # to children) — see line 40 comment. Propagate it to the clone so
        # any non-default value set on the wrapper survives bind_tools. The
        # children's bind_tools handles their own provider_name + telemetry +
        # timeout_seconds propagation independently.
        return ActusFallbackChatModel(
            primary=self.primary.bind_tools(tools, **kwargs),
            fallback=self.fallback.bind_tools(tools, **kwargs),
            provider_name=self.provider_name,
            profile=self.profile,  # A7 P0.1: propagate profile to clone
        )
