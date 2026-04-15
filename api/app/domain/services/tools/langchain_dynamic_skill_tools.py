"""Bridge between SkillTool (OpenAI function-calling schemas) and LangChain StructuredTool.

Dynamically converts each tool schema returned by ``SkillTool.get_tools()``
into a LangChain ``StructuredTool`` so it can be used inside the LangGraph
react_graph alongside native tools.

Usage::

    from app.domain.services.tools.langchain_dynamic_skill_tools import (
        create_dynamic_skill_langchain_tools,
    )

    lc_tools = create_dynamic_skill_langchain_tools(skill_tool)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.tools import StructuredTool, ToolException
from pydantic import Field, create_model

from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)

if TYPE_CHECKING:
    from app.domain.services.tools.skill import SkillTool

logger = logging.getLogger(__name__)

# Maps JSON Schema type strings to Python types used by ``pydantic.create_model``.
TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _make_skill_invoke(skill_tool: "SkillTool", tool_name: str):
    """Factory that returns an async callable bound to *tool_name*.

    Using a factory avoids the classic Python closure-by-reference pitfall
    when creating callables inside a ``for`` loop.
    """

    async def _invoke(**kwargs: Any) -> str:
        result = await skill_tool.invoke(tool_name, **kwargs)
        if not result.success:
            raise ToolException(result.message or f"Tool '{tool_name}' failed")
        return result.model_dump_json()

    return _invoke


def create_dynamic_skill_langchain_tools(
    skill_tool: "SkillTool",
) -> list[StructuredTool]:
    """Convert every tool advertised by *skill_tool* into a LangChain ``StructuredTool``.

    Parameters
    ----------
    skill_tool:
        An initialised :class:`SkillTool` whose ``get_tools()`` returns a list
        of OpenAI function-calling dicts.

    Returns
    -------
    list[StructuredTool]
        One ``StructuredTool`` per successfully converted schema.  Malformed
        schemas are skipped with a warning log.
    """

    tools: list[StructuredTool] = []

    for schema in skill_tool.get_tools():
        try:
            func_def = schema["function"]
            name: str = func_def["name"]
            description: str = func_def.get("description", "")
            parameters: dict = func_def.get("parameters", {})
            properties: dict = parameters.get("properties", {})
            required: list[str] = parameters.get("required", [])

            # Build Pydantic field definitions: (type, Field(...))
            field_definitions: dict[str, Any] = {}
            for prop_name, prop_schema in properties.items():
                python_type = TYPE_MAP.get(
                    prop_schema.get("type", "string"), str
                )
                prop_description = prop_schema.get("description", "")
                if prop_name in required:
                    field_definitions[prop_name] = (
                        python_type,
                        Field(description=prop_description),
                    )
                else:
                    field_definitions[prop_name] = (
                        Optional[python_type],
                        Field(default=None, description=prop_description),
                    )

            args_model = create_model(f"{name}_args", **field_definitions)

            invoke_fn = _make_skill_invoke(skill_tool, name)

            lc_tool = StructuredTool.from_function(
                coroutine=invoke_fn,
                name=name,
                description=description,
                args_schema=args_model,
            )
            tools.append(lc_tool)

        except Exception:
            logger.warning(
                "Skipping malformed skill tool schema: %s",
                schema,
                exc_info=True,
            )

    for tool in tools:
        annotate_and_register_tool_source(tool, source="skill", category="skill")

    return tools
