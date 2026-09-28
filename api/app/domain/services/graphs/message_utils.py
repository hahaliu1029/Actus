"""Message conversion utilities between LangChain BaseMessage and Actus dict format.

Used at the boundary between LangGraph (BaseMessage) and Memory/raw-LLM (dict).
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


_PERSISTED_PROTOCOL_FIELDS = (
    "reasoning_content",
    "responses_output_items",
    "refusal",
)


def _copy_protocol_fields(additional_kwargs: Any) -> dict[str, Any]:
    """Persist only assistant fields required for provider message round-trips.

    Copy nested Responses items so Memory compaction or restored state cannot
    mutate the live graph's message metadata through a shared reference.
    """
    if not isinstance(additional_kwargs, dict):
        return {}
    return {
        key: deepcopy(additional_kwargs[key])
        for key in _PERSISTED_PROTOCOL_FIELDS
        if key in additional_kwargs
    }


_IMAGE_VISION_HINT = (
    '\n\n【图片分析提示】上述附件中的图片已直接嵌入本消息中，你可以直接看到图片内容。'
    '**请直接使用你的视觉能力分析图片**，无需调用任何外部图片分析工具（如 understand_image 等 MCP 工具）。'
    '你的原生视觉能力足以理解图片内容，直接分析即可，这比通过工具转发更快更准确。'
    '无需使用 file_read 或浏览器打开图片。'
)

_IMAGE_TOOL_MODE_HINT = (
    '\n\n【图片分析提示】上述附件包含图片，但当前模型不支持直接查看图片。'
    '**你必须使用 MCP 图片分析工具（如 understand_image）来分析图片内容。**'
    '不要尝试自行描述或猜测图片内容，所有图片理解必须通过工具完成。'
    '\n【MCP工具注意】将图片传递给 MCP 工具时，必须使用附件中标注的 external_url（而非沙箱路径），'
    '因为 MCP 服务运行在沙箱外部，无法访问沙箱文件系统。'
    '\n如果没有可用的 MCP 图片分析工具，可使用终端工具（如 python3 + PIL）提取图片基本信息。'
)

_IMAGE_PLANNER_VISION_HINT = (
    '\n\n【图片附件提示】上述附件包含图片，但你（规划者）无法直接查看图片内容。'
    '执行者可以直接看到图片内容（模型原生多模态能力），无需借助外部工具。'
    '请据此规划步骤：不要生成"等待用户描述图片"之类的步骤，'
    '也不要在步骤中指定使用特定的图片分析工具（如 understand_image），执行者会自行读取图片。'
)

_IMAGE_PLANNER_TOOL_HINT = (
    '\n\n【图片附件提示】上述附件包含图片，但你（规划者）无法直接查看图片内容。'
    '执行者也无法直接查看图片，需通过 MCP 图片分析工具（如 understand_image）理解图片内容。'
    '请据此规划步骤，不要生成"等待用户描述图片"之类的步骤。'
)


def format_attachments_text(
    attachments: list[str],
    has_image_blocks: bool = False,
    for_planner: bool = False,
    supports_vision: bool = True,
) -> str:
    """Format attachments list as prompt text, with conditional image hint.

    Args:
        has_image_blocks: Whether image content blocks exist for this message.
        for_planner: If True, use planner-specific hint.
        supports_vision: If False, use tool-mode hint (model can't see images).
    """
    text = ", ".join(attachments) if attachments else "无"
    if has_image_blocks:
        if for_planner:
            text += _IMAGE_PLANNER_VISION_HINT if supports_vision else _IMAGE_PLANNER_TOOL_HINT
        elif supports_vision:
            text += _IMAGE_VISION_HINT
        else:
            text += _IMAGE_TOOL_MODE_HINT
    return text


def build_multimodal_content(
    text: str, image_blocks: list[dict] | None = None,
) -> str | list[dict]:
    """Build HumanMessage content, optionally with image content blocks.

    Returns plain text string if no image blocks, or a list of content blocks
    (OpenAI Chat Completions multimodal format) when images are present.
    """
    if not image_blocks:
        return text
    return [{"type": "text", "text": text}] + list(image_blocks)


def dicts_to_messages(dicts: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert Actus dict messages to LangChain BaseMessage list."""
    messages: list[BaseMessage] = []
    for d in dicts:
        role = d.get("role", "user")
        content = d.get("content", "")

        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls_raw = d.get("tool_calls") or []
            tool_calls = []
            for tc in tool_calls_raw:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "args": args,
                })
            messages.append(AIMessage(
                content=content,
                tool_calls=tool_calls if tool_calls else [],
                additional_kwargs=_copy_protocol_fields(d.get("additional_kwargs")),
            ))
        elif role == "tool":
            messages.append(ToolMessage(
                content=content,
                tool_call_id=d.get("tool_call_id", ""),
                name=d.get("function_name", ""),
            ))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def _content_hash(content: Any) -> str:
    """Return a short fingerprint of message content for dedup.

    B5 C10: hashes ``content[:500] + content[-200:]`` — this avoids the
    cost of hashing multi-kB tool outputs in full while still
    distinguishing messages that share an opening but diverge in the
    tail. Multimodal content is flattened via
    ``_flatten_multimodal_content`` first so a text-only change isn't
    masked by identical base64 image bytes.
    """
    flat = _flatten_multimodal_content(content)
    if len(flat) <= 700:
        key = flat
    else:
        key = flat[:500] + flat[-200:]
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def dedup_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Deduplicate messages by ID and by ``(role, content_hash)``.

    B5 C10: two-pass "last one wins" dedup.

    **Pass 1 — id dedup**: later message with the same id replaces the
    earlier one at the same position. This preserves the
    ``langgraph.graph.message.add_messages`` contract.

    **Pass 2 — (role, content_hash) dedup**: for messages without an id
    (or whose id was not a duplicate), messages sharing both role and
    content_hash are deduplicated with later-wins semantics — the
    earlier copy is dropped and the later copy stays at its original
    position. This catches cases where the same message body is
    appended twice without ids (common in retry loops and resume
    paths).

    The role is part of the dedup key so a HumanMessage and an
    AIMessage with identical text are NOT merged — they represent
    different speakers.
    """
    # Pass 1: id dedup (unchanged pre-C10 behavior)
    id_seen: dict[str, int] = {}
    id_result: list[BaseMessage] = []
    for msg in messages:
        msg_id = getattr(msg, "id", None)
        if msg_id and msg_id in id_seen:
            id_result[id_seen[msg_id]] = msg  # replace in place
        else:
            if msg_id:
                id_seen[msg_id] = len(id_result)
            id_result.append(msg)

    # Pass 2: (role, content_hash) dedup for messages that passed pass 1.
    # Walk in order, remember the latest index for each (role, hash), and
    # on collision drop the earlier index from the result.
    #
    # We build the final list by walking the indices we want to keep.
    content_seen: dict[tuple[type, str], int] = {}
    keep: list[bool] = [True] * len(id_result)
    for idx, msg in enumerate(id_result):
        key = (type(msg), _content_hash(msg.content))
        prev_idx = content_seen.get(key)
        if prev_idx is not None:
            keep[prev_idx] = False  # drop the earlier copy (later wins)
        content_seen[key] = idx
    return [msg for idx, msg in enumerate(id_result) if keep[idx]]


# B5 C10: regex patterns for extracting the static skeleton of EXECUTION_PROMPT
# from both ZH and EN variants. The patterns strip the first `{step}`
# substitution (which is the only piece that varies across adjacent repeated
# execution prompts) so the fingerprint reflects the template, not the step.
_EXECUTION_PROMPT_STEP_PATTERN_ZH = re.compile(
    r"你正在执行任务：\n.*?\n\n", flags=re.DOTALL
)
_EXECUTION_PROMPT_STEP_PATTERN_EN = re.compile(
    r"You are executing the task:\n.*?\n\n", flags=re.DOTALL
)


def _execution_prompt_fingerprint(text: str) -> str | None:
    """Return a fingerprint for an EXECUTION_PROMPT body, or None.

    The fingerprint is the SHA256 of the first 300 chars of the text
    with the step-description block stripped out. Returns None if the
    text does not look like an EXECUTION_PROMPT (neither the ZH nor
    the EN step-description header is present).
    """
    stripped: str | None = None
    if _EXECUTION_PROMPT_STEP_PATTERN_ZH.search(text):
        stripped = _EXECUTION_PROMPT_STEP_PATTERN_ZH.sub("", text, count=1)
    elif _EXECUTION_PROMPT_STEP_PATTERN_EN.search(text):
        stripped = _EXECUTION_PROMPT_STEP_PATTERN_EN.sub("", text, count=1)
    if stripped is None:
        return None
    return hashlib.sha256(stripped[:300].encode("utf-8")).hexdigest()


def dedup_execution_prompts(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Collapse **adjacent** duplicate ``EXECUTION_PROMPT`` HumanMessages.

    B5 C10: during long sessions with retries, ``executor_node`` can
    inject multiple back-to-back ``HumanMessage(EXECUTION_PROMPT...)``
    entries whose only difference is the step description. When the
    same step is retried, the prompts become byte-identical (modulo
    the stripped step block). This function detects byte-identical
    fingerprints of **strictly adjacent** ``HumanMessage`` instances
    and keeps only the LAST copy.

    Non-adjacent duplicates (separated by any other message, even
    another ``HumanMessage`` with a different fingerprint) are
    preserved — this keeps the function narrowly scoped to the retry
    collapse case and leaves semantic judgements about broader
    conversation history to higher layers.

    Only ``HumanMessage`` instances whose content looks like an
    EXECUTION_PROMPT (has the ZH or EN step-description header) are
    considered for dedup. Every other message type passes through
    unchanged.

    This function is not yet wired into any production code path as
    of C10 — it's published for future ``executor_node`` optimization.
    """
    if len(messages) < 2:
        return list(messages)

    result: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            result.append(msg)
            continue
        text = _flatten_multimodal_content(msg.content)
        fp = _execution_prompt_fingerprint(text)
        if fp is None:
            # Not an EXECUTION_PROMPT-shaped HumanMessage — leave alone.
            result.append(msg)
            continue
        # Strictly adjacent: check if the immediately previous entry in
        # the result is also an EXECUTION_PROMPT HumanMessage with the
        # same fingerprint.
        if result:
            prev = result[-1]
            if isinstance(prev, HumanMessage):
                prev_text = _flatten_multimodal_content(prev.content)
                prev_fp = _execution_prompt_fingerprint(prev_text)
                if prev_fp == fp:
                    # Drop the earlier copy; later wins.
                    result[-1] = msg
                    continue
        result.append(msg)
    return result


def _flatten_multimodal_content(content: Any) -> str:
    """Extract text from multimodal content blocks for Memory persistence.

    When HumanMessage.content is a list[dict] (multimodal), extract only
    the text blocks and discard image blocks to avoid storing large base64
    payloads in Memory/database.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts) if text_parts else str(content)
    return str(content)


def messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain BaseMessage list to Actus dict format (for Memory/raw-LLM).

    Note: Multimodal content (image blocks) in HumanMessage is flattened to
    text-only to avoid persisting large base64 image data in Memory/database.
    """
    dicts: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            dicts.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            dicts.append({"role": "user", "content": _flatten_multimodal_content(msg.content)})
        elif isinstance(msg, AIMessage):
            d: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            protocol_fields = _copy_protocol_fields(msg.additional_kwargs)
            if protocol_fields:
                d["additional_kwargs"] = protocol_fields
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                            if isinstance(tc["args"], dict)
                            else tc["args"],
                        },
                    }
                    for tc in msg.tool_calls
                ]
            dicts.append(d)
        elif isinstance(msg, ToolMessage):
            dicts.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
                "function_name": msg.name or "",
            })
        else:
            dicts.append({"role": "user", "content": str(msg.content)})
    return dicts


def truncate_tool_content(content: str, max_chars: int = 8000) -> str:
    """对超长工具结果执行 head+tail 截断，保证返回长度 <= max_chars。

    当 len(content) <= max_chars 时原样返回。
    超限时先扣除截断标记开销，再将剩余预算均分给 head 和 tail。
    """
    if len(content) <= max_chars:
        return content
    # 用 len(content) 作为被截断字符数的上界，确保标记位数足够
    marker_overhead = len(f"\n...(已截断 {len(content)} 字符)\n")
    budget = max(0, max_chars - marker_overhead)
    head = budget // 2
    tail = budget - head
    removed = len(content) - head - tail
    return f"{content[:head]}\n...(已截断 {removed} 字符)\n{content[-tail:]}"
