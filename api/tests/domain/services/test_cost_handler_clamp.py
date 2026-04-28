"""B4 M1: CostCallbackHandler model/provider clamp contract.

Pins the schema-bounds clamp added in ``_build_record``:

- model exceeding 128 chars → truncated to 128
- provider exceeding 64 chars → truncated to 64
- WARNING logged once per (session, kind), not once per LLM call
- per-call degraded-marker path inherits the clamp via ``replace()``
- pricing lookup runs against the ORIGINAL value so suffix truncation
  doesn't accidentally flip a row from priced→unpriced
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Awaitable, Callable, List
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.domain.models.cost_record import CostRecord
from app.domain.services.cost_callback_handler import (
    CostCallbackHandler,
    _MAX_MODEL_LEN,
    _MAX_PROVIDER_LEN,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_llm_result(
    usage_metadata: dict | None = None,
    content: str = "hi",
) -> LLMResult:
    msg = AIMessage(content=content, usage_metadata=usage_metadata)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _make_capture_persister() -> tuple[
    Callable[[CostRecord], Awaitable[None]], List[CostRecord]
]:
    captured: List[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    return persist, captured


async def _drive_one_call(
    handler: CostCallbackHandler,
    *,
    model: str,
    provider_id: str | None = None,
) -> None:
    """Push one start+end pair through the handler so _build_record runs."""
    run_id = uuid4()
    invocation_params: dict = {"model": model}
    if provider_id is not None:
        invocation_params["provider_id"] = provider_id
    await handler.on_chat_model_start(
        serialized={"id": ["ActusChatModel"]},
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params=invocation_params,
    )
    await handler.on_llm_end(
        _make_llm_result(
            usage_metadata={
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }
        ),
        run_id=run_id,
    )
    await handler.flush_pending(timeout=1.0)


class TestModelClamp:
    async def test_overlong_model_truncated_to_max(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-clamp-1", user_id="u", persister=persist
        )
        overlong = "x" * 200  # 200 > 128
        await _drive_one_call(handler, model=overlong, provider_id="openai_official")

        assert len(captured) == 1
        assert len(captured[0].model) == _MAX_MODEL_LEN
        assert captured[0].model == "x" * _MAX_MODEL_LEN

    async def test_exact_max_length_model_not_clamped(self) -> None:
        """Boundary: len == max should pass through unchanged, no log."""
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-clamp-2", user_id="u", persister=persist
        )
        boundary = "y" * _MAX_MODEL_LEN
        await _drive_one_call(handler, model=boundary, provider_id="openai_official")

        assert captured[0].model == boundary
        assert "model" not in handler._seen_overlong  # noqa: SLF001

    async def test_short_model_passes_through_unchanged(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-clamp-3", user_id="u", persister=persist
        )
        await _drive_one_call(handler, model="gpt-4o", provider_id="openai_official")

        assert captured[0].model == "gpt-4o"


class TestProviderClamp:
    async def test_overlong_provider_truncated_to_max(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-prov-1", user_id="u", persister=persist
        )
        overlong_provider = "p" * 100  # 100 > 64
        await _drive_one_call(
            handler, model="gpt-4o", provider_id=overlong_provider
        )

        assert len(captured) == 1
        assert len(captured[0].provider) == _MAX_PROVIDER_LEN
        assert captured[0].provider == "p" * _MAX_PROVIDER_LEN

    async def test_exact_max_length_provider_not_clamped(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-prov-2", user_id="u", persister=persist
        )
        boundary_provider = "q" * _MAX_PROVIDER_LEN
        await _drive_one_call(
            handler, model="gpt-4o", provider_id=boundary_provider
        )

        assert captured[0].provider == boundary_provider
        assert "provider" not in handler._seen_overlong  # noqa: SLF001


class TestWarningDedup:
    async def test_overlong_logs_once_per_kind_per_session(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """3 LLM calls all overlong on model → exactly 1 model WARNING."""
        persist, _ = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-dedup", user_id="u", persister=persist
        )

        with caplog.at_level(
            logging.WARNING, logger="app.domain.services.cost_callback_handler"
        ):
            for _ in range(3):
                await _drive_one_call(
                    handler,
                    model="m" * 200,
                    provider_id="openai_official",
                )

        clamp_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "exceeds" in r.getMessage()
        ]
        assert len(clamp_warnings) == 1, (
            f"expected exactly 1 clamp WARNING, got {len(clamp_warnings)}: "
            f"{[r.getMessage() for r in clamp_warnings]!r}"
        )
        assert "model" in clamp_warnings[0].getMessage()

    async def test_model_and_provider_log_independently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Distinct kinds dedup independently — overlong on both → 2 WARNINGs."""
        persist, _ = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-twokind", user_id="u", persister=persist
        )

        with caplog.at_level(
            logging.WARNING, logger="app.domain.services.cost_callback_handler"
        ):
            await _drive_one_call(
                handler, model="m" * 200, provider_id="p" * 100
            )

        clamp_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "exceeds" in r.getMessage()
        ]
        assert len(clamp_warnings) == 2
        kinds_logged = {
            "model" if "model" in r.getMessage() else "provider"
            for r in clamp_warnings
        }
        assert kinds_logged == {"model", "provider"}


class TestPricingUsesOriginalValues:
    """Pin the docstring claim: ``get_price`` runs against UNCLAMPED values.

    If a future refactor moved the clamp to entry-assignment time (rather
    than at _build_record output), the pricing lookup would silently start
    using the truncated model/provider — flipping a row from priced→unpriced
    when the table happens to key on the original full name. This test
    locks the read-point split: clamp at output, pricing on the original.
    """

    async def test_get_price_called_with_unclamped_model_and_provider(
        self,
    ) -> None:
        persist, _ = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-pricing", user_id="u", persister=persist
        )
        overlong_model = "m" * 200  # 200 > _MAX_MODEL_LEN (128)
        overlong_provider = "p" * 100  # 100 > _MAX_PROVIDER_LEN (64)

        # Patch the symbol bound inside cost_callback_handler. Returning
        # None makes get_price look "unpriced" so cost_status falls to
        # UNKNOWN — this test is about the call-args, not the return value.
        with patch(
            "app.domain.services.cost_callback_handler.get_price",
            return_value=None,
        ) as mock_get_price:
            await _drive_one_call(
                handler,
                model=overlong_model,
                provider_id=overlong_provider,
            )

        assert mock_get_price.call_count == 1
        called_model, called_provider = mock_get_price.call_args.args
        assert called_model == overlong_model, (
            f"get_price must receive the ORIGINAL model (len={len(overlong_model)}); "
            f"got len={len(called_model)} (clamp leaked into pricing path)"
        )
        assert called_provider == overlong_provider, (
            f"get_price must receive the ORIGINAL provider "
            f"(len={len(overlong_provider)}); got len={len(called_provider)} "
            f"(clamp leaked into pricing path)"
        )


class TestMarkerInheritsClamp:
    async def test_persist_failure_marker_uses_clamped_values(self) -> None:
        """``_persist_safely`` builds the marker via ``replace(record, ...)``
        on the already-clamped record — so the marker fits the schema too.

        Without _build_record clamping, the marker would inherit the
        original overlong model/provider and the marker insert would also
        fail with string-data-right-truncation, leaving the ledger
        completely missing this LLM call.
        """
        attempts: List[CostRecord] = []
        first_call = {"failed": False}

        async def first_fail_then_capture(record: CostRecord) -> None:
            attempts.append(record)
            if not first_call["failed"]:
                first_call["failed"] = True
                raise RuntimeError("simulated main-write failure")

        handler = CostCallbackHandler(
            session_id="sess-marker",
            user_id="u",
            persister=first_fail_then_capture,
        )
        await _drive_one_call(
            handler,
            model="m" * 300,
            provider_id="p" * 200,
        )

        # Two attempts: original write (raises) + degraded marker (succeeds).
        assert len(attempts) == 2
        for attempt in attempts:
            assert len(attempt.model) <= _MAX_MODEL_LEN
            assert len(attempt.provider) <= _MAX_PROVIDER_LEN
        # Marker inherits clamped values, just with persist_degraded node.
        assert attempts[1].node_name == "persist_degraded"
        assert attempts[1].total_usd == Decimal(0)
