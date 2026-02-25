from __future__ import annotations

import asyncio
import logging

from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.services.prompts.continuation_classifier import (
    CONTINUATION_CLASSIFIER_SYSTEM_PROMPT,
    CONTINUATION_CLASSIFIER_USER_PROMPT,
)

logger = logging.getLogger(__name__)


class ContinuationIntentClassifier:
    """基于LLM的续写意图二分类器。"""

    def __init__(
        self,
        llm: LLM,
        json_parser: JSONParser,
        timeout_seconds: float = 3.0,
    ) -> None:
        self._llm = llm
        self._json_parser = json_parser
        self._timeout_seconds = max(0.1, float(timeout_seconds))

    async def classify(
        self,
        current_message: str,
        previous_substantive_message: str,
    ) -> bool:
        if not current_message or not previous_substantive_message:
            return False

        try:
            async with asyncio.timeout(self._timeout_seconds):
                message = await self._llm.invoke(
                    messages=[
                        {
                            "role": "system",
                            "content": CONTINUATION_CLASSIFIER_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": CONTINUATION_CLASSIFIER_USER_PROMPT.format(
                                previous_substantive_message=previous_substantive_message,
                                current_message=current_message,
                            ),
                        },
                    ],
                    tools=[],
                    response_format={"type": "json_object"},
                    tool_choice="none",
                )
        except Exception as exc:
            logger.warning("续写意图LLM判定失败，回退false: %s", str(exc))
            return False

        content = ""
        if isinstance(message, dict):
            content = str(message.get("content") or "")

        try:
            parsed = await self._json_parser.invoke(
                content,
                default_value={"is_continuation": False},
            )
        except Exception as exc:
            logger.warning("续写意图JSON解析失败，回退false: %s", str(exc))
            return False

        if not isinstance(parsed, dict):
            return False
        return parsed.get("is_continuation") is True
