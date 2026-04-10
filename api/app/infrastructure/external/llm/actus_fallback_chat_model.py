"""ActusFallbackChatModel — BaseChatModel with primary/fallback delegation.

Replaces ``primary.with_fallbacks([fallback])`` to avoid a langchain-core bug
where ``RunnableWithFallbacks.__getattr__`` calls ``typing.get_type_hints()``
on ``BaseChatModel.with_structured_output``, which fails because ``builtins``
is imported under ``TYPE_CHECKING`` only in langchain-core.

This model delegates to ``primary`` first; on any exception it retries with
``fallback``. Both ``bind_tools`` and ``with_structured_output`` propagate to
both inner models so that tool schemas stay consistent.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, List, Literal, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)


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

    @property
    def _llm_type(self) -> str:
        return "actus-fallback"

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
        except Exception as primary_exc:
            logger.warning(
                "Primary LLM (%s) failed, falling back: %s",
                self.primary._llm_type, primary_exc,
            )
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
        try:
            async for chunk in self.primary._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            ):
                yield chunk
        except Exception as primary_exc:
            logger.warning(
                "Primary LLM stream (%s) failed, falling back: %s",
                self.primary._llm_type, primary_exc,
            )
            async for chunk in self.fallback._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            ):
                yield chunk

    # ---- bind_tools / with_structured_output ----------------------------- #

    def bind_tools(self, tools: list, **kwargs: Any) -> "ActusFallbackChatModel":
        return ActusFallbackChatModel(
            primary=self.primary.bind_tools(tools, **kwargs),
            fallback=self.fallback.bind_tools(tools, **kwargs),
        )
