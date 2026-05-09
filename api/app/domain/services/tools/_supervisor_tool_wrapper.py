"""B3-core PR-3b: supervisor-aware LangChain tool wrapper."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


class SupervisorAwareToolWrapper(BaseTool):
    """Instrument an inner BaseTool's async invocation with tool inflight counts."""

    _inner: BaseTool = PrivateAttr()
    _supervisor: object = PrivateAttr()

    def __init__(self, *, inner: BaseTool, supervisor: object) -> None:
        super().__init__(
            name=inner.name,
            description=inner.description,
            args_schema=inner.args_schema,
            return_direct=getattr(inner, "return_direct", False),
            response_format=getattr(inner, "response_format", "content"),
            metadata=getattr(inner, "metadata", None),
        )
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_supervisor", supervisor)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        session_id = self._session_id_from_config(config)
        if session_id:
            await self._supervisor.inflight_inc(session_id=session_id, kind="tool")
        try:
            return await self._inner.ainvoke(input, config=config, **kwargs)
        finally:
            if session_id:
                await self._supervisor.inflight_dec(session_id=session_id, kind="tool")

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner._run(*args, **kwargs)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._inner._arun(*args, config=config, **kwargs)

    @staticmethod
    def _session_id_from_config(config: RunnableConfig | None) -> str:
        if not config:
            return ""
        configurable = config.get("configurable") or {}
        value = configurable.get("session_id")
        return str(value) if value else ""


def wrap_tool_list_for_supervisor(
    tools: list[BaseTool],
    supervisor: object | None,
) -> list[BaseTool]:
    """Wrap BaseTool instances with supervisor inflight accounting."""
    if supervisor is None:
        return tools
    return [
        tool
        if isinstance(tool, SupervisorAwareToolWrapper)
        else SupervisorAwareToolWrapper(inner=tool, supervisor=supervisor)
        for tool in tools
    ]
