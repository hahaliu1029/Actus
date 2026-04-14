"""Shared timeout helper for Actus LLM adapters (D5.1).

Both ``ActusChatModel`` and ``ActusResponsesModel`` need the same
per-call hard timeout wrap around their provider ``create()`` awaitable.
Rather than duplicate the ~7-line try/except block in each adapter (and
risk message-format drift between them), we expose ``with_llm_timeout``
as a free function that reads ``adapter.timeout_seconds`` and
``adapter.model_name`` via duck typing.

Why not a base class method? LangChain's ``BaseChatModel`` uses Pydantic
v2 which makes inheritance for behavior-sharing awkward. Free functions
taking the adapter as the first argument is the existing Actus idiom --
see ``_telemetry_mixin.py`` which follows the same pattern for telemetry
attachment.

D5.1: ``asyncio.TimeoutError`` is translated to ``ServerRequestsError``
so LangGraph ``RetryPolicy(max_attempts=3)`` at ``react_graph.llm_node``
and ``main_graph.planner_node`` can handle retries uniformly. SDK-level
retry is disabled via ``AsyncOpenAI(max_retries=0)`` in each adapter's
``_get_client``, so this ``wait_for`` window maps 1:1 to a single HTTP
attempt.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable

from app.application.errors.exceptions import ServerRequestsError


async def with_llm_timeout(adapter: Any, coro: Awaitable[Any]) -> Any:
    """Wrap *coro* in ``asyncio.wait_for`` using ``adapter.timeout_seconds``.

    If ``adapter.timeout_seconds <= 0``, pass through unchanged (escape
    hatch for debugging -- allows an adapter to be configured with
    unlimited per-call budget). Otherwise bound the coroutine by
    ``timeout_seconds`` seconds and translate any ``asyncio.TimeoutError``
    to a ``ServerRequestsError`` whose message embeds the adapter's
    ``model_name`` and the configured budget -- callers' test assertions
    match the format ``"LLM ({model_name}) call exceeded {N}s hard timeout"``.
    """
    timeout = adapter.timeout_seconds
    if timeout <= 0:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise ServerRequestsError(
            f"LLM ({adapter.model_name}) call exceeded "
            f"{timeout}s hard timeout"
        ) from exc
