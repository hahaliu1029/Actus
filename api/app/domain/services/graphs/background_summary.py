"""Standalone background summary generation, extracted from summarizer_node.

Preserves the original summarizer_node contract:
- Streams raw LLM chunks as partial MessageEvents
- Parses final output as JSON {message, attachments} via SummarizerOutput
- Emits final MessageEvent with parsed text + File attachments
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from app.domain.models.event import BaseEvent, MessageEvent
from app.domain.models.file import File
from app.domain.models.llm_responses import SummarizerOutput
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


async def run_background_summary(
    messages: list[BaseMessage],
    summary_llm: BaseChatModel,
    on_event: Callable[[BaseEvent], Awaitable[None]],
) -> str | None:
    """Generate a user-visible streaming summary, independent of the graph.

    Matches the original summarizer_node contract:
    1. Stream raw LLM chunks as partial MessageEvents
    2. Parse final output as JSON {message, attachments} via SummarizerOutput
    3. Emit final MessageEvent with parsed text + File attachments

    Returns the parsed summary text, or None if LLM produced no content.
    """
    from app.domain.services.prompts.react import SUMMARIZE_PROMPT

    chunks: list[str] = []
    stream_id = str(uuid4())

    async for chunk in summary_llm.astream(
        messages + [HumanMessage(content=SUMMARIZE_PROMPT)]
    ):
        if chunk.content:
            chunks.append(chunk.content)
            await on_event(MessageEvent(
                role="assistant",
                message="".join(chunks),
                stream_id=stream_id,
                partial=True,
            ))

    full_text = "".join(chunks)
    if not full_text:
        return None

    # Parse {message, attachments} JSON from LLM output
    summary_text = full_text
    file_attachments: list[File] = []
    try:
        parsed = SummarizerOutput.model_validate_json(full_text)
        if parsed.text:
            summary_text = parsed.text
        for path in parsed.attachments:
            if isinstance(path, str) and path.strip():
                filename = path.rsplit("/", 1)[-1]
                ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
                file_attachments.append(File(
                    filename=filename,
                    filepath=path,
                    extension=ext,
                ))
    except (ValueError, Exception):
        # LLM returned non-JSON or malformed — use raw text, no attachments
        logger.debug("Summary output is not valid JSON, using raw text")

    # Final MessageEvent with parsed text + attachments
    await on_event(MessageEvent(
        role="assistant",
        message=summary_text,
        attachments=file_attachments,
        stream_id=stream_id,
        partial=False,
    ))
    return summary_text
