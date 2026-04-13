"""Tests for domain.services.json_envelope — tolerant LLM JSON envelope parser.

Covers the primary bug scenario: LLM returns pseudo-JSON with **unescaped
newlines** inside string values (common when markdown content is long), and
strict ``json.loads`` fails. The four-tier fallback (direct → code fence →
brace slice → ``json_repair``) must still unwrap the content.
"""

from __future__ import annotations

from app.domain.services.json_envelope import (
    parse_llm_json_envelope,
    unwrap_message_envelope,
)


# ---------------------------------------------------------------------------
# parse_llm_json_envelope
# ---------------------------------------------------------------------------


class TestParseLlmJsonEnvelope:
    def test_valid_json_returns_dict(self) -> None:
        assert parse_llm_json_envelope('{"a": 1}') == {"a": 1}

    def test_whitespace_around_json(self) -> None:
        assert parse_llm_json_envelope('  \n {"a": 1}\n\t ') == {"a": 1}

    def test_empty_string_returns_none(self) -> None:
        assert parse_llm_json_envelope("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert parse_llm_json_envelope("   \n\t  ") is None

    def test_non_string_input_returns_none(self) -> None:
        assert parse_llm_json_envelope(None) is None  # type: ignore[arg-type]
        assert parse_llm_json_envelope(123) is None  # type: ignore[arg-type]

    def test_json_array_returns_none(self) -> None:
        # 信封约定必须是 object，数组不算
        assert parse_llm_json_envelope("[1, 2, 3]") is None

    def test_json_string_returns_none(self) -> None:
        assert parse_llm_json_envelope('"hello"') is None

    def test_plain_text_returns_none(self) -> None:
        assert parse_llm_json_envelope("just some plain text no braces") is None

    def test_markdown_code_fence_with_language(self) -> None:
        text = '```json\n{"result": "hi"}\n```'
        assert parse_llm_json_envelope(text) == {"result": "hi"}

    def test_markdown_code_fence_without_language(self) -> None:
        text = '```\n{"result": "hi"}\n```'
        assert parse_llm_json_envelope(text) == {"result": "hi"}

    def test_markdown_code_fence_with_prefix(self) -> None:
        text = 'Here is the response:\n```json\n{"result": "hi"}\n```'
        assert parse_llm_json_envelope(text) == {"result": "hi"}

    def test_prefix_text_before_json_object(self) -> None:
        text = 'Here you go:\n{"result": "hi"}'
        # Tier 3 (first { to last }) 应该处理
        assert parse_llm_json_envelope(text) == {"result": "hi"}

    def test_unescaped_newlines_in_string_value(self) -> None:
        """主 bug 场景：LLM 在 result 字段里塞了真正的换行。"""
        text = '{\n  "success": true,\n  "result": "line one\nline two\nline three",\n  "attachments": []\n}'
        parsed = parse_llm_json_envelope(text)
        assert parsed is not None
        assert parsed.get("success") is True
        # json_repair 应能还原多行文本
        assert "line one" in parsed.get("result", "")
        assert "line two" in parsed.get("result", "")

    def test_unescaped_newlines_with_markdown_content(self) -> None:
        """CDU 截图的原始场景：JSON 里塞了一大段 markdown。"""
        text = (
            '{\n'
            '  "success": true,\n'
            '  "attachments": ["/home/ubuntu/upload/CDU.html"],\n'
            '  "result": "## 设计图分析\n\n### 布局结构\n- **整体**：1200x400 卡片"\n'
            '}'
        )
        parsed = parse_llm_json_envelope(text)
        assert parsed is not None
        assert "设计图分析" in parsed.get("result", "")
        assert parsed.get("attachments") == ["/home/ubuntu/upload/CDU.html"]

    def test_summarizer_shape_with_unescaped_newlines(self) -> None:
        """CDU 截图的另一种形状：{message, attachments}，message 里有真换行。"""
        text = (
            '{\n'
            '  "message": "## ✅ CDU 概览图 1:1 还原完成\n\n我已根据您上传的设计图...",\n'
            '  "attachments": ["/home/ubuntu/upload/CDU.html"]\n'
            '}'
        )
        parsed = parse_llm_json_envelope(text)
        assert parsed is not None
        assert "CDU 概览图" in parsed.get("message", "")
        assert parsed.get("attachments") == ["/home/ubuntu/upload/CDU.html"]


# ---------------------------------------------------------------------------
# unwrap_message_envelope
# ---------------------------------------------------------------------------


class TestUnwrapMessageEnvelope:
    def test_clean_json_with_result_key(self) -> None:
        text, attachments = unwrap_message_envelope(
            '{"success": true, "result": "hello", "attachments": []}'
        )
        assert text == "hello"
        assert attachments == []

    def test_clean_json_with_message_key(self) -> None:
        text, attachments = unwrap_message_envelope(
            '{"message": "hi there", "attachments": []}'
        )
        assert text == "hi there"
        assert attachments == []

    def test_message_key_preferred_over_result(self) -> None:
        """SummarizerOutput.text 语义：message 非空时优先于 result。"""
        text, _ = unwrap_message_envelope(
            '{"message": "primary", "result": "secondary"}'
        )
        assert text == "primary"

    def test_empty_message_falls_back_to_result(self) -> None:
        text, _ = unwrap_message_envelope(
            '{"message": "", "result": "fallback"}'
        )
        assert text == "fallback"

    def test_attachments_extracted(self) -> None:
        text, attachments = unwrap_message_envelope(
            '{"result": "done", "attachments": ["/home/a.md", "/home/b.md"]}'
        )
        assert text == "done"
        assert attachments == ["/home/a.md", "/home/b.md"]

    def test_attachments_filter_non_string_items(self) -> None:
        text, attachments = unwrap_message_envelope(
            '{"result": "done", "attachments": ["/a.md", null, 123, ""]}'
        )
        assert text == "done"
        assert attachments == ["/a.md"]

    def test_plain_text_unchanged(self) -> None:
        text, attachments = unwrap_message_envelope("just a normal sentence")
        assert text == "just a normal sentence"
        assert attachments == []

    def test_empty_string(self) -> None:
        text, attachments = unwrap_message_envelope("")
        assert text == ""
        assert attachments == []

    def test_non_string_input(self) -> None:
        text, attachments = unwrap_message_envelope(None)  # type: ignore[arg-type]
        assert text == ""
        assert attachments == []

    def test_parsed_but_no_text_keeps_raw(self) -> None:
        """解析成功但 message/result 都为空时，保留原文避免信息丢失。"""
        raw = '{"success": true, "attachments": ["/a.md"]}'
        text, attachments = unwrap_message_envelope(raw)
        assert text == raw
        assert attachments == ["/a.md"]

    def test_pseudo_json_with_unescaped_newlines_in_result(self) -> None:
        """react_graph 路径：output_format 形状，message 里有真换行。"""
        raw = (
            '{\n'
            '  "success": true,\n'
            '  "result": "## 标题\n\n- 列表项 1\n- 列表项 2",\n'
            '  "attachments": ["/home/ubuntu/report.md"]\n'
            '}'
        )
        text, attachments = unwrap_message_envelope(raw)
        assert "## 标题" in text
        assert "列表项 1" in text
        assert attachments == ["/home/ubuntu/report.md"]
        # 关键：不能把原始 JSON 信封漏给前端
        assert not text.strip().startswith('{')

    def test_pseudo_json_with_unescaped_newlines_in_message(self) -> None:
        """summarizer_node 路径：SUMMARIZE_PROMPT 形状，message 里有真换行。"""
        raw = (
            '{\n'
            '  "message": "## ✅ 任务完成\n\n我已经完成了 HTML 还原工作。",\n'
            '  "attachments": ["/home/ubuntu/upload/CDU.html"]\n'
            '}'
        )
        text, attachments = unwrap_message_envelope(raw)
        assert "任务完成" in text
        assert attachments == ["/home/ubuntu/upload/CDU.html"]
        assert not text.strip().startswith('{')

    def test_markdown_fence_wrapped_envelope(self) -> None:
        raw = '```json\n{"message": "fenced content", "attachments": []}\n```'
        text, attachments = unwrap_message_envelope(raw)
        assert text == "fenced content"
        assert attachments == []
