"""B3-core PR-3b: SupervisorAwareCallbackHandler contract tests."""

from __future__ import annotations

from typing import Awaitable, Callable
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.domain.models.cost_record import CostRecord
from app.domain.services.execution_supervisor import ExecutionSupervisor
from app.domain.services.cost_callback_handler import (
    CostCallbackHandler,
    SupervisorAwareCallbackHandler,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSupervisor:
    def __init__(self) -> None:
        self.inc_calls: list[tuple[str, str]] = []
        self.dec_calls: list[tuple[str, str]] = []

    async def inflight_inc(self, *, session_id: str, kind: str) -> None:
        self.inc_calls.append((session_id, kind))

    async def inflight_dec(self, *, session_id: str, kind: str) -> None:
        self.dec_calls.append((session_id, kind))


class _BrokenRedis:
    async def hincrby(self, key: str, field: str, amount: int) -> int:
        raise RuntimeError("redis bounced")


class _LeakySupervisor:
    async def inflight_inc(self, *, session_id: str, kind: str) -> None:
        raise RuntimeError("redis bounced")

    async def inflight_dec(self, *, session_id: str, kind: str) -> None:
        raise RuntimeError("redis bounced")


def _make_persister() -> tuple[
    Callable[[CostRecord], Awaitable[None]], list[CostRecord]
]:
    captured: list[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    return persist, captured


def _make_llm_result() -> LLMResult:
    msg = AIMessage(content="done")
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _make_handler(
    supervisor: _FakeSupervisor,
) -> tuple[SupervisorAwareCallbackHandler, list[CostRecord]]:
    persist, captured = _make_persister()
    return (
        SupervisorAwareCallbackHandler(
            supervisor=supervisor,
            session_id="sess-A",
            user_id="user-A",
            persister=persist,
        ),
        captured,
    )


async def test_subclasses_cost_callback_handler() -> None:
    supervisor = _FakeSupervisor()
    handler, _ = _make_handler(supervisor)

    assert isinstance(handler, CostCallbackHandler)
    assert handler.session_id == "sess-A"
    assert handler.user_id == "user-A"


async def test_chat_model_start_increments_and_preserves_base_pending_entry() -> None:
    supervisor = _FakeSupervisor()
    handler, _ = _make_handler(supervisor)
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"id": ["test"]},
        messages=[[]],
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )

    assert supervisor.inc_calls == [("sess-A", "llm")]
    assert supervisor.dec_calls == []
    assert run_id in handler.pending_keys()


async def test_llm_start_increments_plain_llm_anchor_path() -> None:
    supervisor = _FakeSupervisor()
    handler, _ = _make_handler(supervisor)

    await handler.on_llm_start(
        serialized={"id": ["test"]},
        prompts=["hi"],
        run_id=uuid4(),
        invocation_params={"model": "gpt-4o"},
    )

    assert supervisor.inc_calls == [("sess-A", "llm")]


async def test_broken_redis_inflight_does_not_break_callback_path() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_BrokenRedis(), session_repository=object()
    )
    handler, _ = _make_handler(supervisor)
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"id": ["test"]},
        messages=[[]],
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )

    assert run_id in handler.pending_keys()


async def test_leaky_supervisor_inc_does_not_break_callback_path() -> None:
    handler, _ = _make_handler(_LeakySupervisor())
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"id": ["test"]},
        messages=[[]],
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )

    assert run_id in handler.pending_keys()


async def test_leaky_supervisor_dec_does_not_break_callback_path() -> None:
    handler, _ = _make_handler(_LeakySupervisor())

    await handler.on_llm_end(_make_llm_result(), run_id=uuid4())


async def test_llm_end_decrements_and_preserves_cost_persist() -> None:
    supervisor = _FakeSupervisor()
    handler, captured = _make_handler(supervisor)
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={"id": ["test"]},
        messages=[[]],
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )

    await handler.on_llm_end(_make_llm_result(), run_id=run_id)
    await handler.flush_pending()

    assert supervisor.inc_calls == [("sess-A", "llm")]
    assert supervisor.dec_calls == [("sess-A", "llm")]
    assert len(captured) == 1


async def test_llm_error_decrements_even_without_partial_response() -> None:
    supervisor = _FakeSupervisor()
    handler, captured = _make_handler(supervisor)
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={"id": ["test"]},
        messages=[[]],
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )

    await handler.on_llm_error(RuntimeError("upstream 5xx"), run_id=run_id)

    assert supervisor.dec_calls == [("sess-A", "llm")]
    assert captured == []


async def test_llm_end_decrements_even_when_base_handler_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _FakeSupervisor()
    handler, _ = _make_handler(supervisor)

    async def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("base handler exploded")

    monkeypatch.setattr(CostCallbackHandler, "on_llm_end", _boom)

    with pytest.raises(RuntimeError, match="base handler exploded"):
        await handler.on_llm_end(_make_llm_result(), run_id=uuid4())

    assert supervisor.dec_calls == [("sess-A", "llm")]
