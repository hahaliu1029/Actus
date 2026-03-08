"""基于 OpenAI Responses API 的 LLM 调用类。

适用于 gpt-5.4-pro 等仅支持 Responses API 的模型。
请求/响应格式与 Chat Completions 不同，本类负责将 LLM Protocol 的
统一接口映射到 Responses API，并将响应归一化为 Chat Completions 兼容字典。
"""

import json
import logging
from typing import Any, Dict, List

from app.application.errors.exceptions import ServerRequestsError
from app.domain.external.llm import LLM
from app.domain.models.app_config import LLMConfig
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIResponsesLLM(LLM):
    """基于 OpenAI Responses API (client.responses.create) 的 LLM 调用类"""

    def __init__(self, llm_config: LLMConfig, **kwargs) -> None:
        self._client = AsyncOpenAI(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key,
            **kwargs,
        )
        self._model_name = llm_config.model_name
        self._temperature = llm_config.temperature
        self._max_tokens = llm_config.max_tokens
        self._timeout = 3600

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    # ------------------------------------------------------------------
    # 格式转换：Chat Completions tools → Responses API tools
    # ------------------------------------------------------------------

    @staticmethod
    def _schema_declares_array(schema_type: Any) -> bool:
        if schema_type == "array":
            return True
        if isinstance(schema_type, list):
            return "array" in schema_type
        return False

    @classmethod
    def _sanitize_json_schema(cls, schema: Any, path: str = "root") -> Any:
        """递归修正 Responses API 更严格要求下的不完整 JSON Schema。"""
        if isinstance(schema, list):
            return [cls._sanitize_json_schema(item, f"{path}[]") for item in schema]

        if not isinstance(schema, dict):
            return schema

        sanitized = {
            key: cls._sanitize_json_schema(value, f"{path}.{key}")
            for key, value in schema.items()
        }

        if cls._schema_declares_array(sanitized.get("type")) and "items" not in sanitized:
            logger.warning("检测到数组 schema 缺少 items，已自动补齐: %s", path)
            sanitized["items"] = {}

        return sanitized

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将 Chat Completions 格式的工具定义转为 Responses API 格式。

        Chat Completions: {"type": "function", "function": {"name": ..., "parameters": ...}}
        Responses API:    {"type": "function", "name": ..., "parameters": ..., "strict": False}
        """
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            func = tool.get("function", {})
            parameters = OpenAIResponsesLLM._sanitize_json_schema(func.get("parameters", {}))
            converted.append({
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": parameters,
                "strict": False,
            })
        return converted

    @staticmethod
    def _convert_input_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将 Chat 风格消息转换为 Responses API 的 input items。

        当前 Agent 记忆以 Chat Completions 风格存储：
        - 普通消息：{role, content}
        - assistant 工具调用：{role: "assistant", tool_calls: [...]}
        - 工具结果：{role: "tool", tool_call_id, content}

        Responses API 的多轮工具调用需要显式回放：
        - function_call
        - function_call_output
        """
        converted: List[Dict[str, Any]] = []

        for message in messages:
            role = message.get("role")

            if role == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": message.get("content", ""),
                })
                continue

            if role == "assistant" and message.get("tool_calls"):
                content = message.get("content")
                if content is not None:
                    converted.append({
                        "role": "assistant",
                        "content": content,
                    })

                for tool_call in message.get("tool_calls", []):
                    function = tool_call.get("function", {})
                    converted.append({
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    })
                continue

            converted.append(message)

        return converted

    # ------------------------------------------------------------------
    # 响应归一化：Responses API output → Chat Completions message dict
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_response(response: Any) -> Dict[str, Any]:
        """将 Responses API 响应归一化为 Chat Completions 兼容的 message dict。

        Responses API output 数组可能包含:
        - type="message": 文本消息（content 中包含 type="output_text" 的项）
        - type="function_call": 工具调用
        """
        dumped = response.model_dump() if hasattr(response, "model_dump") else response
        output_items = dumped.get("output", [])

        content_text = ""
        tool_calls: List[Dict[str, Any]] = []

        for item in output_items:
            item_type = item.get("type")

            if item_type == "message":
                # 提取文本内容
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content_text += part.get("text", "")

            elif item_type == "function_call":
                # 映射为 Chat Completions 的 tool_calls 格式
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                })

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content_text or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return message

    # ------------------------------------------------------------------
    # 核心调用
    # ------------------------------------------------------------------

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        response_format: Dict[str, Any] = None,
        tool_choice: str = None,
    ) -> Dict[str, Any]:
        """通过 Responses API 调用 LLM，接口与 Chat Completions 版本保持一致。"""
        try:
            params: Dict[str, Any] = {
                "model": self._model_name,
                "temperature": self._temperature,
                "max_output_tokens": self._max_tokens,
                "input": self._convert_input_messages(messages),
                "timeout": self._timeout,
            }

            # response_format → text.format 映射
            if response_format is not None:
                params["text"] = {"format": response_format}

            if tools:
                params["tools"] = self._convert_tools(tools)
                logger.info("调用Responses API并携带工具信息: %s", self._model_name)
            else:
                logger.info("调用Responses API未携带工具: %s", self._model_name)

            if tool_choice is not None:
                params["tool_choice"] = tool_choice

            response = await self._client.responses.create(**params)

            dumped = response.model_dump() if hasattr(response, "model_dump") else response
            logger.info("Responses API返回内容: %s", dumped)

            return self._normalize_response(response)
        except ServerRequestsError:
            raise
        except Exception as e:
            logger.error("调用Responses API发生错误: %s: %s", type(e).__name__, str(e))
            raise ServerRequestsError(f"调用Responses API发起请求出错: {str(e)}")
