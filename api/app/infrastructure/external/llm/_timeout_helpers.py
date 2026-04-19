"""Shared timeout + transient-error translation for Actus LLM adapters.

Both ``ActusChatModel`` and ``ActusResponsesModel`` need the same
per-call hard timeout wrap around their provider ``create()`` awaitable,
and the same translation of transient transport errors so LangGraph
``RetryPolicy(retry_on=ServerRequestsError)`` at ``react_graph.llm_node``
and ``main_graph.planner_node`` can actually retry them.

Rather than duplicate the try/except block in each adapter (and risk
message-format drift between them), we expose ``with_llm_timeout`` as a
free function that reads ``adapter.timeout_seconds`` and
``adapter.model_name`` via duck typing.

Why not a base class method? LangChain's ``BaseChatModel`` uses Pydantic
v2 which makes inheritance for behavior-sharing awkward. Free functions
taking the adapter as the first argument is the existing Actus idiom --
see ``_telemetry_mixin.py`` which follows the same pattern for telemetry
attachment.

What is translated to ``ServerRequestsError``:

- ``asyncio.TimeoutError`` — D5.1 hard timeout (``wait_for``)
- ``openai.APITimeoutError`` — SDK-level HTTP read/connect timeout
- ``openai.APIConnectionError`` — TCP/TLS/DNS failures
- ``openai.InternalServerError`` — upstream 5xx
- ``openai.RateLimitError`` — 429

What is NOT translated (propagates as-is):

- ``openai.BadRequestError`` / ``UnprocessableEntityError`` — protocol
  incompatibility; ``ActusFallbackChatModel`` escalates these to the
  Responses API
- ``openai.AuthenticationError`` / ``PermissionDeniedError`` — permanent
  failures; no retry is helpful
- ``openai.NotFoundError`` — propagates; could be wrong model name
  (permanent) or missing endpoint (protocol), decided upstream
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable

import openai

from app.application.errors.exceptions import ServerRequestsError

# Transient transport errors that the SDK itself no longer retries
# (because ``max_retries=0`` is set at client init to keep LangGraph
# as the single retry authority). Translate them to
# ``ServerRequestsError`` so ``RetryPolicy(retry_on=ServerRequestsError)``
# can actually retry at the graph layer.
TRANSIENT_OPENAI_EXCEPTIONS: tuple[type[BaseException], ...] = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
)


def translate_transient(adapter: Any, exc: BaseException) -> ServerRequestsError:
    """Build a ``ServerRequestsError`` for a transient openai transport error.

    Same message format that ``with_llm_timeout`` produces, extracted so
    callers that can't route their I/O through ``with_llm_timeout`` (e.g.
    ``ActusChatModel._astream`` which iterates the stream AFTER the
    ``wait_for`` wrap has released) can still translate transient errors
    consistently. Caller is expected to ``raise ... from exc``.
    """
    return ServerRequestsError(
        f"LLM ({adapter.model_name}) transient transport error "
        f"({type(exc).__name__}): {exc}"
    )


async def with_llm_timeout(adapter: Any, coro: Awaitable[Any]) -> Any:
    """Wrap *coro* in ``asyncio.wait_for`` and translate transient errors.

    Timeout behavior: if ``adapter.timeout_seconds <= 0``, the ``wait_for``
    wrap is skipped (escape hatch for debugging / unlimited budget).
    Otherwise bounded by ``timeout_seconds`` seconds. ``asyncio.TimeoutError``
    becomes ``ServerRequestsError`` with message
    ``"LLM ({model_name}) call exceeded {N}s hard timeout"``.

    Transient-error translation: ``openai.APITimeoutError``,
    ``APIConnectionError``, ``InternalServerError`` and ``RateLimitError``
    become ``ServerRequestsError`` preserving the original message. This
    applies regardless of whether ``wait_for`` is active. Other
    ``openai`` exceptions (``BadRequestError``, ``AuthenticationError``,
    ``NotFoundError``, etc.) are left as-is so upper layers can route
    them (fallback wrapper, permanent failure, etc.).
    """
    timeout = adapter.timeout_seconds
    try:
        if timeout <= 0:
            return await coro
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise ServerRequestsError(
            f"LLM ({adapter.model_name}) call exceeded "
            f"{timeout}s hard timeout"
        ) from exc
    except TRANSIENT_OPENAI_EXCEPTIONS as exc:
        raise translate_transient(adapter, exc) from exc
