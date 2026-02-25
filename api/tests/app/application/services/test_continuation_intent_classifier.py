from __future__ import annotations

import asyncio
import json

import pytest
from app.application.services.continuation_intent_classifier import (
    ContinuationIntentClassifier,
)


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _FakeLLM:
    def __init__(self, content: str, delay_seconds: float = 0.0) -> None:
        self._content = content
        self._delay_seconds = delay_seconds

    async def invoke(self, **kwargs):  # noqa: ANN003
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        return {"role": "assistant", "content": self._content}


class _FakeParser:
    async def invoke(self, text: str, default_value=None):  # noqa: ANN001
        if not text.strip():
            return default_value
        return json.loads(text)


class _FailingParser:
    async def invoke(self, text: str, default_value=None):  # noqa: ANN001
        raise ValueError("invalid json")


async def test_classifier_parses_true_false_payload() -> None:
    classifier_true = ContinuationIntentClassifier(
        llm=_FakeLLM('{"is_continuation": true}'),
        json_parser=_FakeParser(),
        timeout_seconds=1.0,
    )
    classifier_false = ContinuationIntentClassifier(
        llm=_FakeLLM('{"is_continuation": false}'),
        json_parser=_FakeParser(),
        timeout_seconds=1.0,
    )

    assert (
        await classifier_true.classify(
            current_message="好的，继续",
            previous_substantive_message="请帮我优化 SQL 查询",
        )
        is True
    )
    assert (
        await classifier_false.classify(
            current_message="sql优化",
            previous_substantive_message="请帮我优化 SQL 查询",
        )
        is False
    )


async def test_classifier_returns_false_when_json_invalid() -> None:
    classifier = ContinuationIntentClassifier(
        llm=_FakeLLM("not-json"),
        json_parser=_FailingParser(),
        timeout_seconds=1.0,
    )

    assert (
        await classifier.classify(
            current_message="好的，继续",
            previous_substantive_message="请帮我优化 SQL 查询",
        )
        is False
    )


async def test_classifier_returns_false_when_timeout() -> None:
    classifier = ContinuationIntentClassifier(
        llm=_FakeLLM('{"is_continuation": true}', delay_seconds=0.2),
        json_parser=_FakeParser(),
        timeout_seconds=0.05,
    )

    assert (
        await classifier.classify(
            current_message="好的，继续",
            previous_substantive_message="请帮我优化 SQL 查询",
        )
        is False
    )
