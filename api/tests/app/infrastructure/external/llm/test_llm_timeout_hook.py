"""D5.1: per-call LLM timeout tests for ActusChatModel / ActusResponsesModel / ActusFallbackChatModel.

Mirrors the organization of ``test_llm_telemetry_hook.py``:
- single file covering all three adapters
- fake clients via ``unittest.mock.AsyncMock``
- async test functions (pytest-asyncio)
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from app.application.errors.exceptions import ServerRequestsError
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
from app.infrastructure.external.llm.actus_fallback_chat_model import (
    ActusFallbackChatModel,
)
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_chat_completion(content: str = "ok") -> SimpleNamespace:
    """Minimal Chat Completions response matching the fields _agenerate reads."""
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        id="chatcmpl-test",
        choices=[choice],
        model="test-model",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _make_responses_api_response(content: str = "ok") -> SimpleNamespace:
    """Minimal Responses API response matching _normalize_response + _agenerate."""
    content_item = {"type": "output_text", "text": content}
    output_item = {"type": "message", "role": "assistant", "content": [content_item]}
    return SimpleNamespace(
        model_dump=lambda: {"output": [output_item], "usage": {}},
    )


# ---------------------------------------------------------------------------
# ChatModel: field + _get_client
# ---------------------------------------------------------------------------


class TestChatModelFieldAndClient:
    """D5.1: timeout_seconds field + max_retries=0 on AsyncOpenAI."""

    def test_default_timeout_seconds_is_120(self) -> None:
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
        )
        assert model.timeout_seconds == 120.0

    def test_explicit_timeout_seconds_set(self) -> None:
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
            timeout_seconds=45.0,
        )
        assert model.timeout_seconds == 45.0

    def test_get_client_passes_max_retries_zero(self) -> None:
        """_get_client must construct AsyncOpenAI with max_retries=0.

        D5.1: LangGraph RetryPolicy is the single retry authority.
        Without max_retries=0, worst case is 3 graph x 3 SDK = 9 HTTP calls.
        """
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
        )
        with patch(
            "app.infrastructure.external.llm.actus_chat_model.AsyncOpenAI"
        ) as mock_cls:
            model._get_client()
            mock_cls.assert_called_once()
            kwargs = mock_cls.call_args.kwargs
            assert kwargs.get("max_retries") == 0


class TestChatModelAgenerateTimeout:
    """D5.1: wait_for wrap on _agenerate."""

    async def test_agenerate_timeout_fires(self) -> None:
        """When create() hangs beyond timeout_seconds, raise ServerRequestsError."""
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0.2,  # 200ms budget
        )

        async def slow_create(**_kwargs):
            await asyncio.sleep(5.0)  # way beyond 200ms
            return _make_chat_completion()

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=slow_create)

        with patch.object(model, "_get_client", return_value=mock_client):
            with pytest.raises(ServerRequestsError, match=r"exceeded 0\.2s hard timeout"):
                await model._agenerate([HumanMessage(content="hi")])

    async def test_agenerate_zero_unlimited_bypasses_wait_for(self) -> None:
        """timeout_seconds=0 must NOT call asyncio.wait_for at all."""
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0,
        )
        mock_resp = _make_chat_completion(content="hello")
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            with patch(
                "app.infrastructure.external.llm._timeout_helpers.asyncio.wait_for"
            ) as mock_wait_for:
                result = await model._agenerate([HumanMessage(content="hi")])

        assert mock_wait_for.call_count == 0
        assert len(result.generations) == 1

    async def test_agenerate_normal_path_unaffected(self) -> None:
        """With default timeout_seconds=120 and immediate create(), behavior is unchanged."""
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
        )
        mock_resp = _make_chat_completion(content="hello")
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "hello"


class TestChatModelAstreamTimeout:
    """D5.1: wait_for wrap covers the initial stream-obtain step via _obtain_stream() helper.

    The wrap must bound *both* of _astream's branches at their convergence
    point — whether create() returns an async iterator directly (mocks) or
    an awaitable (real AsyncOpenAI). Chunk iteration (async for chunk in
    stream) remains intentionally unwrapped; D5 ExecutionWatchdog handles
    mid-stream stalls at the graph level.
    """

    async def test_astream_awaitable_path_timeout_fires(self) -> None:
        """Path 2 (real AsyncOpenAI): awaitable resolving to iterator hangs, raise."""
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0.2,
        )

        async def slow_create(**_kwargs):
            await asyncio.sleep(5.0)
            # never reached
            async def _never_yields():
                if False:
                    yield None
            return _never_yields()

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=slow_create)

        with patch.object(model, "_get_client", return_value=mock_client):
            with pytest.raises(ServerRequestsError, match=r"exceeded 0\.2s hard timeout"):
                async for _chunk in model._astream([HumanMessage(content="hi")]):
                    pass

    async def test_astream_direct_iterator_path_completes_fast(self) -> None:
        """Path 1 (mock async gen): direct iterator returns from helper instantly.

        The _obtain_stream() helper sees __aiter__ and returns immediately.
        wait_for awaits that near-instantaneous coroutine and passes. Chunks
        then iterate normally outside the wait_for window.

        Verification strategy: spy on asyncio.wait_for to prove the helper
        was called exactly once. Without the spy, the test would pass
        trivially even if wait_for were never invoked at all.
        """
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=1.0,
        )

        async def chunk_generator(*_args, **_kwargs):
            delta1 = SimpleNamespace(content="hello", role=None, tool_calls=None)
            yield SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=delta1, finish_reason=None)]
            )
            final = SimpleNamespace(content=None, role=None, tool_calls=None)
            yield SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=final, finish_reason="stop")]
            )

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        # Direct assignment: calling mock.create(**params) invokes
        # chunk_generator(**params) which returns an async generator object.
        # Inside _obtain_stream() this goes through the __aiter__ branch.
        mock_client.chat.completions.create = chunk_generator

        with patch.object(model, "_get_client", return_value=mock_client):
            with patch(
                "app.infrastructure.external.llm._timeout_helpers.asyncio.wait_for",
                wraps=asyncio.wait_for,
            ) as spy:
                collected: list[str] = []
                async for gen_chunk in model._astream([HumanMessage(content="hi")]):
                    if gen_chunk.message.content:
                        collected.append(gen_chunk.message.content)

        assert "".join(collected) == "hello"
        # Spy assertion: wait_for was called exactly once — on the _obtain_stream()
        # coroutine. Proves the helper was wrapped by wait_for, not bypassed.
        assert spy.call_count == 1, (
            f"Expected wait_for to be called exactly once (on _obtain_stream()); "
            f"got {spy.call_count}. This test exists to prove the __aiter__ "
            f"branch of _obtain_stream actually goes through wait_for."
        )

    async def test_astream_mid_stream_slowness_not_wrapped(self) -> None:
        """Chunk iteration must NOT be wrapped — D5 watchdog territory.

        Once _obtain_stream() returns the iterator, the async-for loop
        over chunks is outside the wait_for window. A 500ms stall between
        chunks must NOT trigger a 200ms adapter-level timeout.

        See also: ``TestMidStreamWatchdogContract.test_chat_mid_stream_stall_delegated_to_watchdog``
        — a sibling test that pins the same contract at the class level
        (named after the D5.1/D5 互补契约) as a named regression anchor.
        These two tests exercise the same production behavior and should
        be updated together if the mid-stream semantics change.
        """
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0.2,
        )

        async def chunk_generator(*_args, **_kwargs):
            delta1 = SimpleNamespace(content="a", role=None, tool_calls=None)
            yield SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=delta1, finish_reason=None)]
            )
            await asyncio.sleep(0.5)  # mid-stream stall, 2.5x the 0.2s timeout
            delta2 = SimpleNamespace(content="b", role=None, tool_calls=None)
            yield SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=delta2, finish_reason=None)]
            )
            final = SimpleNamespace(content=None, role=None, tool_calls=None)
            yield SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=final, finish_reason="stop")]
            )

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = chunk_generator

        with patch.object(model, "_get_client", return_value=mock_client):
            collected: list[str] = []
            async for gen_chunk in model._astream([HumanMessage(content="hi")]):
                if gen_chunk.message.content:
                    collected.append(gen_chunk.message.content)

        assert "".join(collected) == "ab"


class TestChatModelBindToolsClone:
    """Codex review BLOCK 4: bind_tools clone must preserve timeout_seconds.

    Without this, react_graph.py:180 ``llm.bind_tools(tools)`` returns a
    clone with the default timeout_seconds=120, silently dropping any
    user-configured value. Same for planner_react.py:501
    ``self._summary_llm.with_structured_output(...)`` which internally
    calls bind_tools.
    """

    def test_bind_tools_preserves_custom_timeout_seconds(self) -> None:
        @tool
        def dummy_tool(x: int) -> int:
            """A dummy tool for testing."""
            return x

        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=45.0,  # non-default
        )
        cloned = model.bind_tools([dummy_tool])
        assert cloned.timeout_seconds == 45.0, (
            "bind_tools clone lost timeout_seconds — would cause "
            "react_graph.llm_node to silently drop user config"
        )

    def test_bind_tools_preserves_zero_escape_hatch(self) -> None:
        @tool
        def dummy_tool(x: int) -> int:
            """A dummy tool for testing."""
            return x

        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0,
        )
        cloned = model.bind_tools([dummy_tool])
        assert cloned.timeout_seconds == 0


# ---------------------------------------------------------------------------
# ResponsesModel: field + _get_client
# ---------------------------------------------------------------------------


class TestResponsesModelFieldAndClient:
    """D5.1: timeout_seconds field + max_retries=0 on AsyncOpenAI (Responses API)."""

    def test_default_timeout_seconds_is_120(self) -> None:
        model = ActusResponsesModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
        )
        assert model.timeout_seconds == 120.0

    def test_explicit_timeout_seconds_set(self) -> None:
        model = ActusResponsesModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
            timeout_seconds=45.0,
        )
        assert model.timeout_seconds == 45.0

    def test_get_client_passes_max_retries_zero(self) -> None:
        model = ActusResponsesModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
        )
        with patch(
            "app.infrastructure.external.llm.actus_responses_model.AsyncOpenAI"
        ) as mock_cls:
            model._get_client()
            mock_cls.assert_called_once()
            kwargs = mock_cls.call_args.kwargs
            assert kwargs.get("max_retries") == 0


class TestResponsesModelAgenerateTimeout:
    """D5.1: wait_for wrap on _agenerate for Responses API."""

    async def test_responses_agenerate_timeout_fires(self) -> None:
        model = ActusResponsesModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0.2,
        )

        async def slow_create(**_kwargs):
            await asyncio.sleep(5.0)
            return _make_responses_api_response()

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=slow_create)

        with patch.object(model, "_get_client", return_value=mock_client):
            with pytest.raises(ServerRequestsError, match=r"exceeded 0\.2s hard timeout"):
                await model._agenerate([HumanMessage(content="hi")])

    async def test_responses_agenerate_zero_unlimited_bypasses_wait_for(self) -> None:
        model = ActusResponsesModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0,
        )
        mock_resp = _make_responses_api_response(content="hello")
        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            with patch(
                "app.infrastructure.external.llm._timeout_helpers.asyncio.wait_for"
            ) as mock_wait_for:
                result = await model._agenerate([HumanMessage(content="hi")])

        assert mock_wait_for.call_count == 0
        assert len(result.generations) == 1

    async def test_responses_agenerate_normal_path_unaffected(self) -> None:
        model = ActusResponsesModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
        )
        mock_resp = _make_responses_api_response(content="hello")
        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        assert len(result.generations) == 1


class TestResponsesModelAstreamInheritance:
    """D5.1: ResponsesModel._astream is a fallback over _agenerate (not real streaming).

    It calls await self._agenerate(...) and yields a single chunk. The
    timeout wrap on _agenerate therefore applies to _astream automatically
    -- no independent wait_for needed.
    """

    async def test_astream_inherits_timeout_via_agenerate(self) -> None:
        model = ActusResponsesModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0.2,
        )

        async def slow_create(**_kwargs):
            await asyncio.sleep(5.0)
            return _make_responses_api_response()

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=slow_create)

        with patch.object(model, "_get_client", return_value=mock_client):
            with pytest.raises(ServerRequestsError, match=r"exceeded 0\.2s hard timeout"):
                async for _chunk in model._astream([HumanMessage(content="hi")]):
                    pass


class TestResponsesModelBindToolsClone:
    """Codex review BLOCK 4 (ResponsesModel side)."""

    def test_bind_tools_preserves_custom_timeout_seconds(self) -> None:
        @tool
        def dummy_tool(x: int) -> int:
            """A dummy tool for testing."""
            return x

        model = ActusResponsesModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=45.0,
        )
        cloned = model.bind_tools([dummy_tool])
        assert cloned.timeout_seconds == 45.0


# ---------------------------------------------------------------------------
# Task 10: ActusFallbackChatModel passive budget tests
# ---------------------------------------------------------------------------


class TestFallbackModelBudgetIndependence:
    """D5.1: FallbackChatModel inherits timeout behavior from children.

    ActusFallbackChatModel._agenerate (line 82-112) catches ``Exception``
    (which includes ``ServerRequestsError`` from D5.1's ``with_llm_timeout``
    helper) and falls through to fallback. Each child's wait_for wrap is
    independent — primary's 120s budget is separate from fallback's 120s
    budget. The ``test_primary_timeout_triggers_fallback`` test pins this
    specifically via a caplog assertion on the ``"exceeded Ns hard timeout"``
    log message.
    """

    async def test_primary_timeout_triggers_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When primary._agenerate raises ServerRequestsError (timeout), fallback is invoked."""
        primary = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="primary-model",
            timeout_seconds=0.2,
        )
        fallback = ActusChatModel(
            base_url="https://y.test/v1",
            api_key="k",
            model_name="fallback-model",
            timeout_seconds=0.2,  # also tight, but will succeed because mock returns fast
        )
        fallback_model = ActusFallbackChatModel(primary=primary, fallback=fallback)

        # Primary hangs
        async def slow_create(**_kwargs):
            await asyncio.sleep(5.0)
            return _make_chat_completion()

        primary_client = MagicMock()
        primary_client.chat = MagicMock()
        primary_client.chat.completions = MagicMock()
        primary_client.chat.completions.create = AsyncMock(side_effect=slow_create)

        # Fallback returns immediately
        fallback_client = MagicMock()
        fallback_client.chat = MagicMock()
        fallback_client.chat.completions = MagicMock()
        fallback_client.chat.completions.create = AsyncMock(
            return_value=_make_chat_completion(content="from-fallback")
        )

        with patch.object(primary, "_get_client", return_value=primary_client):
            with patch.object(fallback, "_get_client", return_value=fallback_client):
                with caplog.at_level(logging.WARNING):
                    result = await fallback_model._agenerate([HumanMessage(content="hi")])

        # Fallback path should have succeeded
        assert result.generations[0].message.content == "from-fallback"
        # Primary should have been attempted (and timed out)
        assert primary_client.chat.completions.create.await_count == 1
        assert fallback_client.chat.completions.create.await_count == 1

        # Proves the SPECIFIC timeout triggered fallback, not just any exception.
        # ActusFallbackChatModel._agenerate logs "Primary LLM ... failed, falling back"
        # at WARNING level when primary raises. The log message includes the original
        # exception str, which for D5.1 timeouts is "LLM ({model}) call exceeded {N}s
        # hard timeout". Without this assertion, the test would pass even if the
        # primary raised any other Exception subclass.
        assert any(
            "exceeded 0.2s hard timeout" in record.message
            for record in caplog.records
        ), (
            f"Expected ServerRequestsError with 'exceeded 0.2s hard timeout' "
            f"in fallback warning log; got: {[r.message for r in caplog.records]}"
        )

    async def test_fallback_bind_tools_propagates_timeout_seconds_to_children(self) -> None:
        """When FallbackChatModel.bind_tools clones, both children keep timeout_seconds."""
        @tool
        def dummy_tool(x: int) -> int:
            """A dummy tool."""
            return x

        primary = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="primary",
            timeout_seconds=33.0,
        )
        fallback = ActusChatModel(
            base_url="https://y.test/v1",
            api_key="k",
            model_name="fallback",
            timeout_seconds=77.0,
        )
        fallback_model = ActusFallbackChatModel(primary=primary, fallback=fallback)
        cloned = fallback_model.bind_tools([dummy_tool])

        assert cloned.primary.timeout_seconds == 33.0
        assert cloned.fallback.timeout_seconds == 77.0

    async def test_fallback_bind_tools_preserves_wrapper_provider_name(self) -> None:
        """Codex review residual: ``ActusFallbackChatModel.bind_tools`` must
        propagate the wrapper's own ``provider_name`` to the clone.

        ``provider_name`` is an independent field on the wrapper (not delegated
        to children — see the comment at ``actus_fallback_chat_model.py:38-40``).
        Currently the main path always uses the default ``"openai"`` so this
        wasn't a runtime bug, but if a non-default provider is ever set on the
        wrapper layer (e.g. for B5.1 Anthropic routing), the clone would lose
        it. Pin the invariant now so a future regression fails loudly.
        """
        @tool
        def dummy_tool(x: int) -> int:
            """A dummy tool."""
            return x

        primary = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="primary",
        )
        fallback = ActusChatModel(
            base_url="https://y.test/v1",
            api_key="k",
            model_name="fallback",
        )
        # Construct the wrapper with a non-default provider_name to expose
        # the propagation gap. (Default is "openai"; "anthropic" is the only
        # other Literal value currently allowed.)
        fallback_model = ActusFallbackChatModel(
            primary=primary,
            fallback=fallback,
            provider_name="anthropic",
        )
        assert fallback_model.provider_name == "anthropic"

        cloned = fallback_model.bind_tools([dummy_tool])
        assert cloned.provider_name == "anthropic", (
            f"FallbackChatModel.bind_tools clone lost wrapper provider_name; "
            f"got {cloned.provider_name!r}, expected 'anthropic'. The clone "
            f"is constructed without explicit provider_name, falling back to "
            f"the default 'openai' — fix bind_tools to pass "
            f"provider_name=self.provider_name."
        )


# ---------------------------------------------------------------------------
# Task 11: with_structured_output regression test
# ---------------------------------------------------------------------------


class TestWithStructuredOutputClone:
    """Codex review BLOCK 4 (third regression test).

    LangChain's BaseChatModel.with_structured_output calls self.bind_tools()
    internally (langchain_core/language_models/chat_models.py:1697-1704),
    so the Task 5 bind_tools clone fix should transparently cover this.
    This test PINS the invariant: structured.first is the ActusChatModel
    clone with timeout_seconds preserved.

    This test is intentionally NOT skippable. If the access path breaks
    (e.g. future langchain-core changes the composition structure), the
    failure is a regression signal, not a reason to loosen the test.
    """

    def test_chat_with_structured_output_preserves_timeout_seconds(self) -> None:
        from langchain_core.runnables import RunnableSequence
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: str

        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=55.0,
        )
        structured = model.with_structured_output(Answer)

        # Pinned invariant 1: the return is a RunnableSequence
        # (llm | output_parser in chat_models.py:1723)
        assert isinstance(structured, RunnableSequence), (
            f"with_structured_output return type changed: got "
            f"{type(structured).__name__}, expected RunnableSequence. "
            f"If langchain-core upgraded, update the pinned access path "
            f"in this test and in the spec."
        )

        # Pinned invariant 2: .first is the ActusChatModel clone
        # (the llm operand on the left of `|`)
        inner = structured.first
        assert isinstance(inner, ActusChatModel), (
            f"RunnableSequence.first is not an ActusChatModel: got "
            f"{type(inner).__name__}. The composition shape changed."
        )

        # Core assertion: timeout_seconds propagated through bind_tools
        assert inner.timeout_seconds == 55.0, (
            f"bind_tools clone inside with_structured_output lost "
            f"timeout_seconds (got {inner.timeout_seconds}, expected 55.0)."
        )


# ---------------------------------------------------------------------------
# Task 12: Mid-stream watchdog contract test
# ---------------------------------------------------------------------------


class TestMidStreamWatchdogContract:
    """D5.1 vs D5 ExecutionWatchdog互补契约的可执行见证。

    Contract: D5.1 only wraps the INITIAL create() call. Mid-stream stalls
    are intentionally left for D5 ExecutionWatchdog (idle_timeout_seconds=
    120s at the graph level) to handle. This test pins the contract by
    asserting that a mid-stream stall is NOT caught by D5.1 adapter-level
    timeout.
    """

    async def test_chat_mid_stream_stall_delegated_to_watchdog(self) -> None:
        """Once the first chunk arrives, chunk iteration is unwrapped forever.

        See also: ``TestChatModelAstreamTimeout.test_astream_mid_stream_slowness_not_wrapped``
        — a sibling test that exercises the same production behavior from
        the ``_astream`` implementation angle. Both tests must stay in sync
        if the mid-stream semantics change. This test is the named contract
        anchor (class name documents the D5.1/D5 互补契约); the sibling test
        is the implementation-angle anchor (class name groups _astream
        timeout behaviors).
        """
        model = ActusChatModel(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="test-model",
            timeout_seconds=0.2,  # very tight — would fire if we wrapped chunks
        )

        # Initial create() returns an async generator IMMEDIATELY (bypasses
        # wait_for via __aiter__ branch). Then generator yields one chunk,
        # then stalls for 0.5s (2.5x the timeout).
        async def chunk_generator(*_args, **_kwargs):
            delta = SimpleNamespace(content="first", role=None, tool_calls=None)
            yield SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)]
            )
            await asyncio.sleep(0.5)  # mid-stream stall (2.5× the timeout) — should NOT trigger D5.1
            final_delta = SimpleNamespace(content=None, role=None, tool_calls=None)
            yield SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=final_delta, finish_reason="stop")]
            )

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = chunk_generator

        with patch.object(model, "_get_client", return_value=mock_client):
            collected: list[str] = []
            async for gen_chunk in model._astream([HumanMessage(content="hi")]):
                if gen_chunk.message.content:
                    collected.append(gen_chunk.message.content)

        # Contract holds: we completed both chunks despite the 0.5s mid-stream stall
        # and 0.2s adapter timeout. D5 ExecutionWatchdog would handle this at
        # the graph level instead.
        assert "first" in collected
