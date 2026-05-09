"""PR-3b Task 7: supervisor inflight counter integration coverage."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.domain.models.cost_record import CostRecord
from app.domain.services.cost_callback_handler import SupervisorAwareCallbackHandler
from app.domain.services.tools._supervisor_tool_wrapper import (
    SupervisorAwareToolWrapper,
)

pytestmark = pytest.mark.anyio


class _ToolArgs(BaseModel):
    x: int = Field(..., description="probe input")


async def _noop_persist(_record: CostRecord) -> None:
    return None


def _llm_result() -> LLMResult:
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="done"))]]
    )


async def test_llm_callback_tracks_inflight_count_and_ttl(
    agent_service_with_redis,
    redis_client,
    sample_session,
) -> None:
    supervisor = agent_service_with_redis._supervisor
    sid = sample_session.id
    hot_key = f"supervisor:hot:{sid}"
    handler = SupervisorAwareCallbackHandler(
        supervisor=supervisor,
        session_id=sid,
        user_id=sample_session.user_id,
        persister=_noop_persist,
    )
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"id": ["integration", "chat"]},
        messages=[[]],
        run_id=run_id,
        invocation_params={"model": "integration-test-model"},
    )

    assert await redis_client.hget(hot_key, "inflight_llm_count") == "1"
    ttl_after_start = await redis_client.ttl(hot_key)
    assert 290 <= ttl_after_start <= 300

    await handler.on_llm_end(response=_llm_result(), run_id=run_id)

    assert await redis_client.hget(hot_key, "inflight_llm_count") == "0"
    ttl_after_end = await redis_client.ttl(hot_key)
    assert 290 <= ttl_after_end <= 300


async def test_tool_wrapper_tracks_inflight_during_async_tool_execution(
    agent_service_with_redis,
    redis_client,
    sample_session,
) -> None:
    supervisor = agent_service_with_redis._supervisor
    sid = sample_session.id
    hot_key = f"supervisor:hot:{sid}"

    async def _probe(x: int) -> str:
        assert await redis_client.hget(hot_key, "inflight_tool_count") == "1"
        return f"ok:{x}"

    inner = StructuredTool.from_function(
        coroutine=_probe,
        name="inflight_probe",
        description="probe tool inflight accounting",
        args_schema=_ToolArgs,
    )
    wrapped = SupervisorAwareToolWrapper(inner=inner, supervisor=supervisor)

    result = await wrapped.ainvoke(
        {"x": 7},
        config={"configurable": {"session_id": sid}},
    )

    assert result == "ok:7"
    assert await redis_client.hget(hot_key, "inflight_tool_count") == "0"


async def test_negative_inflight_dec_preserves_redis_value_and_records_metric(
    agent_service_with_redis,
    redis_client,
    sample_session,
    monkeypatch,
) -> None:
    supervisor = agent_service_with_redis._supervisor
    sid = sample_session.id
    hot_key = f"supervisor:hot:{sid}"
    meter_calls: list[tuple[str, dict[str, Any]]] = []

    def _record_meter(name: str, **attrs: Any) -> None:
        meter_calls.append((name, attrs))

    monkeypatch.setattr(supervisor, "_meter_inc", _record_meter)

    value = await supervisor.inflight_dec(session_id=sid, kind="llm")

    assert value == -1
    assert await redis_client.hget(hot_key, "inflight_llm_count") == "-1"
    assert meter_calls == [("inflight_negative", {"kind": "llm"})]
