"""FallbackLLM 单元测试。"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.application.errors.exceptions import ServerRequestsError
from app.infrastructure.external.llm.fallback_llm import FallbackLLM, _is_incompatibility_error

pytestmark = pytest.mark.anyio


class _StubLLM:
    """可配置的 LLM 桩实现。"""

    def __init__(
        self,
        name: str = "stub",
        response: Dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self._response = response or {"role": "assistant", "content": "ok"}
        self._error = error
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def temperature(self) -> float:
        return 0.7

    @property
    def max_tokens(self) -> int:
        return 4096

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        response_format: Dict[str, Any] = None,
        tool_choice: str = None,
    ) -> Dict[str, Any]:
        self.call_count += 1
        if self._error:
            raise self._error
        return self._response


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ------------------------------------------------------------------
# _is_incompatibility_error 判断测试
# ------------------------------------------------------------------


class TestIsIncompatibilityError:
    def test_detects_chinese_response_format_error(self) -> None:
        err = ServerRequestsError("调用OpenAI客户端出错: Error code: 429 - {'error': {'message': '不合法的response_format'}}")
        assert _is_incompatibility_error(err) is True

    def test_detects_english_response_format_error(self) -> None:
        err = Exception("invalid response_format parameter for this model")
        assert _is_incompatibility_error(err) is True

    def test_detects_response_format_invalid_request_error(self) -> None:
        err = ServerRequestsError("Error: invalid_request_error - unsupported response_format parameter")
        assert _is_incompatibility_error(err) is True

    def test_rejects_generic_invalid_request_error(self) -> None:
        err = ServerRequestsError("Error: invalid_request_error - malformed tool arguments")
        assert _is_incompatibility_error(err) is False

    def test_detects_model_not_support(self) -> None:
        err = Exception("model does not support response_format")
        assert _is_incompatibility_error(err) is True

    def test_rejects_auth_error(self) -> None:
        err = Exception("Error code: 401 - Unauthorized")
        assert _is_incompatibility_error(err) is False

    def test_rejects_rate_limit_without_format_mention(self) -> None:
        err = Exception("Error code: 429 - Rate limit exceeded")
        assert _is_incompatibility_error(err) is False

    def test_rejects_network_timeout(self) -> None:
        err = Exception("Connection timeout after 30s")
        assert _is_incompatibility_error(err) is False

    def test_rejects_server_error(self) -> None:
        err = Exception("Error code: 500 - Internal server error")
        assert _is_incompatibility_error(err) is False


# ------------------------------------------------------------------
# FallbackLLM 行为测试
# ------------------------------------------------------------------


class TestFallbackLLM:
    async def test_uses_primary_when_no_error(self) -> None:
        primary = _StubLLM(name="primary", response={"role": "assistant", "content": "from primary"})
        fallback = _StubLLM(name="fallback")

        llm = FallbackLLM(primary=primary, fallback=fallback)
        result = await llm.invoke(messages=[])

        assert result["content"] == "from primary"
        assert primary.call_count == 1
        assert fallback.call_count == 0

    async def test_falls_back_on_incompatibility_error(self) -> None:
        primary = _StubLLM(
            name="primary",
            error=ServerRequestsError("调用出错: 不合法的response_format"),
        )
        fallback = _StubLLM(name="fallback", response={"role": "assistant", "content": "from fallback"})

        llm = FallbackLLM(primary=primary, fallback=fallback)
        result = await llm.invoke(messages=[])

        assert result["content"] == "from fallback"
        assert primary.call_count == 1
        assert fallback.call_count == 1

    async def test_sticks_to_fallback_after_first_switch(self) -> None:
        primary = _StubLLM(
            name="primary",
            error=ServerRequestsError("invalid_request_error: unsupported response_format"),
        )
        fallback = _StubLLM(name="fallback")

        llm = FallbackLLM(primary=primary, fallback=fallback)
        await llm.invoke(messages=[])
        await llm.invoke(messages=[])

        # 第二次调用不再尝试 primary
        assert primary.call_count == 1
        assert fallback.call_count == 2

    async def test_does_not_fallback_on_auth_error(self) -> None:
        primary = _StubLLM(
            name="primary",
            error=ServerRequestsError("Error code: 401 - Unauthorized"),
        )
        fallback = _StubLLM(name="fallback")

        llm = FallbackLLM(primary=primary, fallback=fallback)

        with pytest.raises(ServerRequestsError):
            await llm.invoke(messages=[])

        assert fallback.call_count == 0

    async def test_does_not_fallback_on_real_rate_limit(self) -> None:
        primary = _StubLLM(
            name="primary",
            error=ServerRequestsError("Error code: 429 - Rate limit exceeded"),
        )
        fallback = _StubLLM(name="fallback")

        llm = FallbackLLM(primary=primary, fallback=fallback)

        with pytest.raises(ServerRequestsError):
            await llm.invoke(messages=[])

        assert fallback.call_count == 0

    async def test_model_name_switches_after_fallback(self) -> None:
        primary = _StubLLM(
            name="gpt-5.4-pro",
            error=ServerRequestsError("不合法的response_format"),
        )
        fallback = _StubLLM(name="gpt-5.4-pro-responses")

        llm = FallbackLLM(primary=primary, fallback=fallback)
        assert llm.model_name == "gpt-5.4-pro"

        await llm.invoke(messages=[])
        assert llm.model_name == "gpt-5.4-pro-responses"
