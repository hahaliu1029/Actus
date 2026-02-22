import pytest

from app.domain.models.app_config import LLMConfig
from app.infrastructure.external.llm.openai_llm import OpenAILLM

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]

    def model_dump(self):
        return {"choices": [{"message": self.choices[0].message}]}


class _FakeTextChoice:
    def __init__(self, text):
        self.text = text


class _FakeTextResponse:
    def __init__(self, text):
        self.choices = [_FakeTextChoice(text)]

    def model_dump(self):
        return {"choices": [{"text": self.choices[0].text}]}


class _FakeCompletions:
    def __init__(self, response):
        self._response = response

    async def create(self, **kwargs):
        return self._response


class _FakeChat:
    def __init__(self, response):
        self.completions = _FakeCompletions(response)


class _FakeClient:
    def __init__(self, response):
        self.chat = _FakeChat(response)


async def test_invoke_accepts_string_message_payload_from_compatible_api() -> None:
    llm = OpenAILLM(
        LLMConfig(
            base_url="https://api.deepseek.com",
            api_key="test-key",
            model_name="test-model",
        )
    )
    llm._client = _FakeClient(_FakeResponse("plain text message"))  # type: ignore[attr-defined]

    result = await llm.invoke(messages=[{"role": "user", "content": "hi"}], tools=None)

    assert result == {"role": "assistant", "content": "plain text message"}


async def test_invoke_accepts_choice_text_when_message_field_missing() -> None:
    llm = OpenAILLM(
        LLMConfig(
            base_url="https://api.deepseek.com",
            api_key="test-key",
            model_name="test-model",
        )
    )
    llm._client = _FakeClient(_FakeTextResponse("text from compatible api"))  # type: ignore[attr-defined]

    result = await llm.invoke(messages=[{"role": "user", "content": "hi"}], tools=None)

    assert result == {"role": "assistant", "content": "text from compatible api"}
