"""Standalone background summary generation, extracted from summarizer_node.

Preserves the original summarizer_node contract:
- Streams raw LLM chunks as partial MessageEvents
- Parses final output as JSON {message, attachments} via unwrap_message_envelope
  (tolerant four-tier fallback: direct → code fence → brace slice → json_repair)
- Emits final MessageEvent with parsed text + File attachments
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.domain.models.event import BaseEvent, MessageEvent
from app.domain.models.file import File
from app.domain.services.json_envelope import unwrap_message_envelope
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage


async def run_background_summary(
    messages: list[BaseMessage],
    summary_llm: BaseChatModel,
    on_event: Callable[[BaseEvent], Awaitable[None]],
    lang: str = "zh",
    *,
    callbacks: list[Any] | None = None,
) -> str | None:
    """Generate a user-visible streaming summary, independent of the graph.

    Matches the original summarizer_node contract:
    1. Stream raw LLM chunks as partial MessageEvents
    2. Parse final output as JSON {message, attachments} via unwrap_message_envelope
    3. Emit final MessageEvent with parsed text + File attachments

    Returns the parsed summary text, or None if LLM produced no content.

    B4 M0: ``callbacks`` lets the runner plumb the session-scoped
    CostCallbackHandler into this graph-external LLM call. Without it the
    summary's tokens don't hit the cost ledger (LangGraph metadata can't
    propagate via context when the call is outside the graph).
    ``metadata.langgraph_node`` is stamped to ``"background_summary"`` so
    the aggregate's ``by_node`` breakdown attributes this call correctly.
    """
    from app.domain.services.prompts import get_prompt_bundle

    bundle = get_prompt_bundle(lang)
    chunks: list[str] = []
    stream_id = str(uuid4())

    astream_kwargs: dict[str, Any] = {}
    if callbacks:
        astream_kwargs["config"] = {
            "callbacks": callbacks,
            "metadata": {
                "langgraph_node": "background_summary",
                "langgraph_step": 0,
            },
        }

    async for chunk in summary_llm.astream(
        messages + [HumanMessage(content=bundle.SUMMARIZE_PROMPT)],
        **astream_kwargs,
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

    # Tolerant unwrap — handles pseudo-JSON with unescaped newlines in string
    # values (common when markdown content is long) via json_repair fallback.
    summary_text, attachment_paths = unwrap_message_envelope(full_text)
    file_attachments: list[File] = []
    for path in attachment_paths:
        filename = path.rsplit("/", 1)[-1]
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        file_attachments.append(File(
            filename=filename,
            filepath=path,
            extension=ext,
        ))

    # Final MessageEvent with parsed text + attachments
    await on_event(MessageEvent(
        role="assistant",
        message=summary_text,
        attachments=file_attachments,
        stream_id=stream_id,
        partial=False,
    ))
    return summary_text
