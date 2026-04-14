"""Tolerant LLM JSON envelope parser.

LLM 回复经常被包装在 JSON 信封里（见 ``prompts/sections/output_format.py`` 和
``prompts/react.py::SUMMARIZE_PROMPT``）——例如
``{"success": true, "result": "...markdown...", "attachments": [...]}`` 或
``{"message": "...markdown...", "attachments": [...]}``。

当 markdown 内容较长、包含未转义的换行时，严格 ``json.loads`` 会失败，导致信封
被原样发送到前端。本模块提供四级兜底解析，顺序与
``skill_creator_service._parse_llm_json`` 保持一致：

1. ``json.loads`` 直接解析
2. 从 markdown 代码围栏（``` ```json ... ``` ```）提取后再解析
3. 截取第一个 ``{`` 到最后一个 ``}`` 的子串再解析
4. ``json_repair`` 库兜底——修复未转义换行、缺失引号等常见 LLM 错误

``react_graph.llm_node``（中间步骤回复）和 ``main_graph.summarizer_node``（最终
总结）都走这里，保持两条路径的行为一致。
"""
from __future__ import annotations

import json
import re
from typing import Any

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def parse_llm_json_envelope(text: str) -> dict[str, Any] | None:
    """从 LLM 文本输出中宽松提取 JSON 对象。

    返回解析得到的 dict，失败时返回 ``None``（**不抛异常**）。调用方可以据此
    决定是否降级为原文显示。

    仅对返回值为 ``dict`` 的情况视为成功——数组、字符串、数字等都会返回
    ``None``，因为信封约定总是一个对象。
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    # Tier 1: 直接解析
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier 2: markdown 代码围栏
    fence_match = _CODE_FENCE_RE.search(stripped)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 3: 首 { ... 末 }
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            parsed = json.loads(stripped[first_brace : last_brace + 1])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 4: json_repair 兜底（修复未转义换行等）
    try:
        from json_repair import repair_json

        repaired = repair_json(stripped, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:  # noqa: BLE001 — 兜底路径，任何异常都应降级
        pass

    return None


def unwrap_message_envelope(text: str) -> tuple[str, list[str]]:
    """从可能为 JSON 信封的 LLM 输出中提取 ``(display_text, attachments)``。

    接受两种 key 形状——``{"result": "...", "attachments": [...]}`` 和
    ``{"message": "...", "attachments": [...]}``，匹配 ``SummarizerOutput``
    模型的容忍度（见 ``domain/models/llm_responses.py``）。

    语义：
    - 解析失败 → 返回 ``(原文, [])``
    - 解析成功且 ``message`` 或 ``result`` 有有效内容 → 返回 ``(提取文本, 附件列表)``
    - 解析成功但两个 key 都为空 → 返回 ``(原文, 附件列表)``——避免静默吞掉附件

    ``attachments`` 只保留非空字符串项。
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else "", []

    parsed = parse_llm_json_envelope(text)
    if parsed is None:
        return text, []

    raw_attachments = parsed.get("attachments")
    if isinstance(raw_attachments, list):
        attachments = [
            item for item in raw_attachments
            if isinstance(item, str) and item.strip()
        ]
    else:
        attachments = []

    raw_text = parsed.get("message") or parsed.get("result")
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text, attachments

    return text, attachments
