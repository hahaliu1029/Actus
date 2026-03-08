"""带兼容性回退的 LLM 包装器。

先走主实现，遇到"API 面不兼容"类错误时自动切到备实现。
仅对兼容性失败回退，不对 401/403、真正的 rate limit、网络超时、5xx 回退。
"""

import logging
import re
from typing import Any, Dict, List, Optional

from app.application.errors.exceptions import ServerRequestsError
from app.domain.external.llm import LLM

logger = logging.getLogger(__name__)

# 用于判断"API 面不兼容"的错误特征（不区分大小写）
_INCOMPATIBILITY_PATTERNS: list[re.Pattern] = [
    re.compile(r"不合法的\s*response_format", re.IGNORECASE),
    re.compile(r"invalid.*response_format", re.IGNORECASE),
    re.compile(r"invalid_request_error.*(?:response_format|json_object|text\.format)", re.IGNORECASE),
    re.compile(r"unsupported.*(?:response_format|json_object|text\.format)", re.IGNORECASE),
    re.compile(r"not.*supported.*(?:response_format|json_object)", re.IGNORECASE),
    re.compile(r"model.*does not support.*(?:response_format|json_object|responses api|chat completions)", re.IGNORECASE),
    re.compile(r"Unrecognized request argument.*response_format", re.IGNORECASE),
]


def _is_incompatibility_error(error: Exception) -> bool:
    """判断异常是否属于 API 兼容性问题，而非真正的运行时故障。"""
    msg = str(error)

    # 排除真正的认证/权限/限流/服务端错误
    # 注意：用户遇到的案例是 HTTP 429 但 error.type 是 invalid_request_error，
    # 所以不能单纯按 HTTP 状态码过滤，而是基于错误消息内容判断
    for pattern in _INCOMPATIBILITY_PATTERNS:
        if pattern.search(msg):
            return True

    return False


class FallbackLLM(LLM):
    """先走主实现，遇到兼容性错误时自动切到备实现。

    一旦主实现在某次调用中触发兼容性回退，后续所有调用直接走备实现，
    避免每次都浪费一次失败请求的延迟。
    """

    def __init__(self, primary: LLM, fallback: LLM) -> None:
        self._primary = primary
        self._fallback = fallback
        self._use_fallback = False

    @property
    def model_name(self) -> str:
        if self._use_fallback:
            return self._fallback.model_name
        return self._primary.model_name

    @property
    def temperature(self) -> float:
        if self._use_fallback:
            return self._fallback.temperature
        return self._primary.temperature

    @property
    def max_tokens(self) -> int:
        if self._use_fallback:
            return self._fallback.max_tokens
        return self._primary.max_tokens

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        response_format: Dict[str, Any] = None,
        tool_choice: str = None,
    ) -> Dict[str, Any]:
        # 已确认需要走备实现，直接跳过主实现
        if self._use_fallback:
            return await self._fallback.invoke(
                messages=messages,
                tools=tools,
                response_format=response_format,
                tool_choice=tool_choice,
            )

        try:
            return await self._primary.invoke(
                messages=messages,
                tools=tools,
                response_format=response_format,
                tool_choice=tool_choice,
            )
        except (ServerRequestsError, Exception) as e:
            if not _is_incompatibility_error(e):
                raise

            # 兼容性错误，切到备实现
            logger.warning(
                "主LLM实现触发兼容性错误，切换到备实现: primary=%s fallback=%s error=%s",
                self._primary.model_name,
                self._fallback.model_name,
                str(e)[:200],
            )
            self._use_fallback = True
            return await self._fallback.invoke(
                messages=messages,
                tools=tools,
                response_format=response_format,
                tool_choice=tool_choice,
            )
