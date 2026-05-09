"""B3-core PR-3b: SupervisorAwareToolWrapper unit tests."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, tool as lc_tool
from pydantic import BaseModel, Field

from app.domain.models.tool_result import AllowError, DecisionReason
from app.domain.services.graphs.react_graph import _invoke_wrapper
from app.domain.services.tools._supervisor_tool_wrapper import (
    SupervisorAwareToolWrapper,
    wrap_tool_list_for_supervisor,
)
from app.domain.services.tools.tool_source_resolver import ToolSource

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _ArgsSchema(BaseModel):
    x: int = Field(..., description="must be int")


class _ProxyProbeTool(BaseTool):
    name: str = "proxy_probe"
    description: str = "proxy probe"
    args_schema: type[BaseModel] = _ArgsSchema

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return f"sync:{args}:{kwargs}"

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        session_id = ""
        if config:
            session_id = str(config.get("configurable", {}).get("session_id", ""))
        return f"async:{args}:{kwargs}:sid={session_id}"


class _FakeSupervisor:
    def __init__(self) -> None:
        self.inc_calls: list[tuple[str, str]] = []
        self.dec_calls: list[tuple[str, str]] = []

    async def inflight_inc(self, *, session_id: str, kind: str) -> None:
        self.inc_calls.append((session_id, kind))

    async def inflight_dec(self, *, session_id: str, kind: str) -> None:
        self.dec_calls.append((session_id, kind))


def _make_inner_tool(name: str = "test_tool") -> BaseTool:
    async def _coro(x: int) -> str:
        return f"got {x}"

    return StructuredTool.from_function(
        coroutine=_coro,
        name=name,
        description="proxied desc",
        args_schema=_ArgsSchema,
    )


async def test_wrapper_instantiates_as_base_tool() -> None:
    wrapper = SupervisorAwareToolWrapper(
        inner=_make_inner_tool(),
        supervisor=_FakeSupervisor(),
    )

    assert isinstance(wrapper, BaseTool)


async def test_wrapper_proxies_metadata() -> None:
    inner = _make_inner_tool(name="proxy_me")
    inner.metadata = {"risk_level": "high"}
    wrapper = SupervisorAwareToolWrapper(inner=inner, supervisor=_FakeSupervisor())

    assert wrapper.name == "proxy_me"
    assert wrapper.description == "proxied desc"
    assert wrapper.args_schema is _ArgsSchema
    assert wrapper.metadata == {"risk_level": "high"}


async def test_wrapper_preserves_content_and_artifact_outcome_path() -> None:
    @lc_tool(response_format="content_and_artifact")
    async def typed_fail(x: str) -> tuple[str, AllowError]:
        """Typed failure tool."""
        outcome = AllowError(
            content=f"bad {x}",
            reason=DecisionReason(
                type="exception",
                code="typed_failure",
                message="typed failure",
            ),
            retryable=False,
        )
        return outcome.content, outcome

    wrapper = SupervisorAwareToolWrapper(
        inner=typed_fail,
        supervisor=_FakeSupervisor(),
    )
    outcome = await _invoke_wrapper(
        wrapper,
        {
            "id": "call-1",
            "name": "typed_fail",
            "args": {"x": "input"},
            "type": "tool_call",
        },
        ToolSource(source="native", category="file", canonical_name="typed_fail"),
    )

    assert wrapper.response_format == "content_and_artifact"
    assert isinstance(outcome, AllowError)
    assert outcome.reason.code == "typed_failure"


async def test_ainvoke_increments_then_decrements_on_success() -> None:
    supervisor = _FakeSupervisor()
    wrapper = SupervisorAwareToolWrapper(
        inner=_make_inner_tool(),
        supervisor=supervisor,
    )

    result = await wrapper.ainvoke(
        {"x": 42},
        config={"configurable": {"session_id": "sess-X"}},
    )

    assert result == "got 42"
    assert supervisor.inc_calls == [("sess-X", "tool")]
    assert supervisor.dec_calls == [("sess-X", "tool")]


async def test_ainvoke_decrements_on_validation_error() -> None:
    supervisor = _FakeSupervisor()
    wrapper = SupervisorAwareToolWrapper(
        inner=_make_inner_tool(),
        supervisor=supervisor,
    )

    with pytest.raises(Exception):
        await wrapper.ainvoke(
            {"x": "not an int"},
            config={"configurable": {"session_id": "sess-X"}},
        )

    assert supervisor.inc_calls == [("sess-X", "tool")]
    assert supervisor.dec_calls == [("sess-X", "tool")]


async def test_ainvoke_decrements_on_inner_exception() -> None:
    async def _boom(x: int) -> str:
        raise RuntimeError("inner exploded")

    inner = StructuredTool.from_function(
        coroutine=_boom,
        name="boom",
        description="raises",
        args_schema=_ArgsSchema,
    )
    supervisor = _FakeSupervisor()
    wrapper = SupervisorAwareToolWrapper(inner=inner, supervisor=supervisor)

    with pytest.raises(RuntimeError, match="inner exploded"):
        await wrapper.ainvoke(
            {"x": 1},
            config={"configurable": {"session_id": "sess-X"}},
        )

    assert supervisor.inc_calls == [("sess-X", "tool")]
    assert supervisor.dec_calls == [("sess-X", "tool")]


async def test_ainvoke_without_session_id_is_passthrough() -> None:
    supervisor = _FakeSupervisor()
    wrapper = SupervisorAwareToolWrapper(
        inner=_make_inner_tool(),
        supervisor=supervisor,
    )

    result = await wrapper.ainvoke({"x": 7}, config={"configurable": {}})

    assert result == "got 7"
    assert supervisor.inc_calls == []
    assert supervisor.dec_calls == []


async def test_ainvoke_without_config_is_passthrough() -> None:
    supervisor = _FakeSupervisor()
    wrapper = SupervisorAwareToolWrapper(
        inner=_make_inner_tool(),
        supervisor=supervisor,
    )

    result = await wrapper.ainvoke({"x": 7})

    assert result == "got 7"
    assert supervisor.inc_calls == []
    assert supervisor.dec_calls == []


async def test_run_and_arun_proxy_inner_tool() -> None:
    wrapper = SupervisorAwareToolWrapper(
        inner=_ProxyProbeTool(),
        supervisor=_FakeSupervisor(),
    )

    assert wrapper._run("a", y=1) == "sync:('a',):{'y': 1}"
    assert await wrapper._arun(
        "a",
        y=1,
        config={"configurable": {"session_id": "sess-X"}},
    ) == "async:('a',):{'y': 1}:sid=sess-X"


async def test_wrap_tool_list_without_supervisor_preserves_identity() -> None:
    tools = [_make_inner_tool(name="plain")]

    wrapped = wrap_tool_list_for_supervisor(tools, None)

    assert wrapped is tools


async def test_wrap_tool_list_wraps_plain_tools_and_preserves_existing_wrappers() -> None:
    supervisor = _FakeSupervisor()
    plain = _make_inner_tool(name="plain")
    existing = SupervisorAwareToolWrapper(
        inner=_make_inner_tool(name="existing"),
        supervisor=supervisor,
    )

    wrapped = wrap_tool_list_for_supervisor([plain, existing], supervisor)

    assert isinstance(wrapped[0], SupervisorAwareToolWrapper)
    assert wrapped[1] is existing
    assert [tool.name for tool in wrapped] == ["plain", "existing"]
