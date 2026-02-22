import logging
from typing import Any, Dict, List

from app.application.errors.exceptions import ServerRequestsError
from app.domain.external.llm import LLM
from app.domain.models.app_config import LLMConfig
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """基于OpenAI SDK/兼容OpenAI格式的LLM调用类"""

    def __init__(self, llm_config: LLMConfig, **kwargs) -> None:
        """构造函数，完成异步OpenAI客户端的创建和参数初始化"""
        # 1.初始化异步客户端
        self._client = AsyncOpenAI(
            base_url=str(llm_config.base_url),
            api_key=llm_config.api_key,
            **kwargs,
        )

        # 2.完成其他参数的存储
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

    @staticmethod
    def _extract_text(payload: Any) -> str | None:
        """从不同响应结构中尽量提取文本。"""
        if payload is None:
            return None

        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()

        if isinstance(payload, str):
            return payload

        if isinstance(payload, list):
            text_parts: List[str] = []
            for item in payload:
                part = OpenAILLM._extract_text(item)
                if part:
                    text_parts.append(part)
            return "".join(text_parts) if text_parts else None

        if isinstance(payload, dict):
            if isinstance(payload.get("text"), str):
                return payload["text"]
            if isinstance(payload.get("content"), str):
                return payload["content"]

            # 兼容 content/parts 为数组的结构
            for key in ("content", "parts"):
                if key in payload:
                    part = OpenAILLM._extract_text(payload[key])
                    if part:
                        return part

        return None

    @staticmethod
    def _normalize_message(message: Any) -> Dict[str, Any]:
        """将LLM消息统一转换为字典格式，兼容字符串和Pydantic对象。"""
        if hasattr(message, "model_dump"):
            message = message.model_dump()

        if isinstance(message, str):
            return {"role": "assistant", "content": message}

        if not isinstance(message, dict):
            raise ValueError(f"LLM返回的message类型非法: {type(message).__name__}")

        normalized = dict(message)
        if "role" not in normalized:
            normalized["role"] = "assistant"
        if "content" in normalized and not isinstance(normalized["content"], str):
            content = OpenAILLM._extract_text(normalized["content"])
            if content is not None:
                normalized["content"] = content

        return normalized

    @staticmethod
    def _extract_message(response: Any, dumped: Any) -> Any:
        """从响应对象或序列化字典中提取首条message。"""
        def _from_choice(choice: Any) -> Any:
            if hasattr(choice, "model_dump"):
                choice = choice.model_dump()

            if isinstance(choice, dict):
                if "message" in choice:
                    return choice["message"]

                text = OpenAILLM._extract_text(choice.get("text"))
                if text is not None:
                    return {"role": "assistant", "content": text}

                text = OpenAILLM._extract_text(choice.get("content"))
                if text is not None:
                    return {"role": "assistant", "content": text}
            else:
                message = getattr(choice, "message", None)
                if message is not None:
                    return message

                text = OpenAILLM._extract_text(getattr(choice, "text", None))
                if text is not None:
                    return {"role": "assistant", "content": text}

                text = OpenAILLM._extract_text(getattr(choice, "content", None))
                if text is not None:
                    return {"role": "assistant", "content": text}

            return None

        # 尝试从原始响应对象中提取
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            extracted = _from_choice(choices[0])
            if extracted is not None:
                return extracted

        # 尝试从序列化字典中提取
        if isinstance(dumped, dict):
            dumped_choices = dumped.get("choices")
            if isinstance(dumped_choices, list) and dumped_choices:
                extracted = _from_choice(dumped_choices[0])
                if extracted is not None:
                    return extracted

            # 兼容部分供应商直接返回 message/content
            if "message" in dumped:
                return dumped["message"]
            text = OpenAILLM._extract_text(dumped.get("content"))
            if text is not None:
                return {"role": "assistant", "content": text}

        # 提取失败，打印详细的响应结构以辅助排查
        logger.error(
            f"无法从LLM响应中提取choices.message, "
            f"response type={type(response).__name__}, "
            f"choices={choices}, "
            f"dumped={dumped}"
        )
        raise ValueError("LLM响应缺少choices.message字段")

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        response_format: Dict[str, Any] = None,
        tool_choice: str = None,
    ) -> Dict[str, Any]:
        """使用异步OpenAI客户端发起块响应（该步骤可以切换成流式响应）"""
        try:
            # 1.构建请求参数
            params: Dict[str, Any] = {
                "model": self._model_name,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "messages": messages,
                "timeout": self._timeout,
            }

            # 仅在有值时传递可选参数，避免None导致兼容性问题
            if response_format is not None:
                params["response_format"] = response_format
            if tools:
                params["tools"] = tools
                logger.info(
                    f"调用OpenAI客户端向LLM发起请求并携带工具信息: {self._model_name}"
                )
            else:
                logger.info(f"调用OpenAI客户端向LLM发起请求未携带工具: {self._model_name}")
            if tool_choice is not None:
                params["tool_choice"] = tool_choice

            response = await self._client.chat.completions.create(**params)

            # 3.处理响应数据并返回
            dumped = response.model_dump() if hasattr(response, "model_dump") else response
            logger.info(f"OpenAI客户端返回内容: {dumped}")
            raw_message = self._extract_message(response=response, dumped=dumped)
            return self._normalize_message(raw_message)
        except ServerRequestsError:
            raise
        except Exception as e:
            logger.error(f"调用OpenAI客户端发生错误: {type(e).__name__}: {str(e)}")
            raise ServerRequestsError(f"调用OpenAI客户端向LLM发起请求出错: {str(e)}")


