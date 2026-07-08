"""Integration tests covering the 4 verification points from design review.

VP1: Multi-tool-call with file_view mixed — tool-call pairing intact
VP2: Multi-image file_view → cross-step compaction keeps only text block
VP3: messages_to_dicts() persistence keeps only text summary
VP4: @lc_tool FileProcessResult passthrough (covered in test_file_view_passthrough.py)
"""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.domain.external.file_processor import FileProcessResult
from app.domain.services.graphs.context_assembler import group_messages
from app.domain.services.graphs.main_graph import _compact_messages
from app.domain.services.graphs.message_utils import messages_to_dicts


class TestVP1MultiToolCallPairing:
    def test_three_tool_calls_one_file_view(self):
        """3 tool calls (file_view + 2 others). All ToolMessages grouped, HumanMessage separate."""
        ai = AIMessage(content="", tool_calls=[
            {"id": "c1", "name": "file_view", "args": {}},
            {"id": "c2", "name": "shell_execute", "args": {}},
            {"id": "c3", "name": "file_read", "args": {}},
        ])
        tm1 = ToolMessage(content="[Image: a.png]", tool_call_id="c1", name="file_view")
        tm2 = ToolMessage(content="output", tool_call_id="c2", name="shell_execute")
        tm3 = ToolMessage(content="file content", tool_call_id="c3", name="file_read")
        hm = HumanMessage(content=[
            {"type": "text", "text": "[file_view: file_view — 1 image(s) loaded]"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ])

        groups = group_messages([ai, tm1, tm2, tm3, hm])
        tool_call_groups = [g for g in groups if g.kind == "tool_call"]
        human_groups = [g for g in groups if g.kind == "human"]
        assert len(tool_call_groups) == 1
        assert len(tool_call_groups[0].messages) == 4  # AI + 3 ToolMessages
        assert len(human_groups) == 1


class TestVP2CrossStepCompaction:
    def test_multi_image_compacted_to_text(self):
        """5 images from PDF → after compaction, only text remains."""
        blocks = [
            {"type": "text", "text": "[file_view: file_view — 5 image(s) loaded]"},
        ]
        for i in range(5):
            blocks.append({"type": "image_url", "image_url": {"url": f"https://example.com/page_{i}.png"}})

        hm = HumanMessage(content=blocks)
        compacted = _compact_messages([hm])
        assert len(compacted) == 1
        content = compacted[0].content
        assert isinstance(content, str)
        assert "file_view" in content
        assert "image_url" not in content
        assert "example.com" not in content


class TestVP3Persistence:
    def test_persistence_strips_images(self):
        """messages_to_dicts must strip image blocks from HumanMessage."""
        hm = HumanMessage(content=[
            {"type": "text", "text": "[file_view: file_view — 3 image(s) loaded]"},
            {"type": "image_url", "image_url": {"url": "https://example.com/1.png"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/2.png"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/3.png"}},
        ])
        dicts = messages_to_dicts([hm])
        content = dicts[0]["content"]
        assert isinstance(content, str)
        assert "file_view" in content
        assert "example.com" not in content


class TestVPRealGraphToolNode:
    """通过真实的 build_react_graph + tool_node 验证 FileProcessResult 处理。"""

    def test_file_view_through_real_tool_node(self):
        """绑定 file_view 和 shell_execute，构造双工具 AIMessage，
        验证 tool_node 输出中 ToolMessages 在前、HumanMessage 在后。"""
        from unittest.mock import AsyncMock, MagicMock
        from langchain_core.tools import tool as lc_tool
        from app.domain.services.graphs.react_graph import build_react_graph

        # 1. fake file_view（返回 FileProcessResult）
        @lc_tool
        async def file_view(filepath: str) -> FileProcessResult:
            """View file."""
            return FileProcessResult(
                text="[Image: test.png, 100x200]",
                image_blocks=({"type": "image_url", "image_url": {"url": "https://example.com/test.png"}},),
            )

        @lc_tool
        async def shell_execute(command: str) -> str:
            """Run shell."""
            return "file1.txt"

        # 2. fake LLM — 第一轮返回双工具调用，第二轮返回最终回答
        call_count = 0

        async def fake_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content="", tool_calls=[
                    {"id": "call_fv", "name": "file_view", "args": {"filepath": "/tmp/test.png"}},
                    {"id": "call_sh", "name": "shell_execute", "args": {"command": "ls"}},
                ])
            return AIMessage(content='{"success": true, "result": "done", "attachments": []}')

        fake_llm = MagicMock()
        # B12 flip: MagicMock auto-creates a truthy `.profile`; with P1 materialize
        # now default-ON, pin None so the reshape branch is skipped (no real
        # provider profile configured — this test exercises the file_view→deferred
        # HumanMessage flow, not provider-specific image reshaping).
        fake_llm.profile = None
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)
        fake_llm.ainvoke = fake_ainvoke

        # 3. 构建并执行 react_graph
        graph = build_react_graph(llm=fake_llm, tools=[file_view, shell_execute])

        initial_state = {
            "messages": [HumanMessage(content="analyze image")],
            "events": [],
            "attempt_count": 0,
            "failure_count": 0,
            "soft_hint_sent": False,
            "should_interrupt": False,
            "llm_input_messages": [],
        }

        final_state = asyncio.run(
            graph.ainvoke(initial_state, config={"recursion_limit": 10})
        )

        # 4. 验证输出消息序列
        msgs = final_state["messages"]

        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        human_multimodal = [
            m for m in msgs
            if isinstance(m, HumanMessage) and isinstance(m.content, list)
        ]

        assert len(tool_msgs) >= 2, f"Expected 2+ ToolMessages, got {len(tool_msgs)}"
        assert len(human_multimodal) >= 1, f"Expected 1+ multimodal HumanMessage, got {len(human_multimodal)}"

        # HumanMessage 包含 text block（不是纯 image）
        hm = human_multimodal[0]
        text_blocks = [b for b in hm.content if b.get("type") == "text"]
        image_blocks = [b for b in hm.content if b.get("type") == "image_url"]
        assert len(text_blocks) >= 1, "HumanMessage must contain text block for compaction"
        assert len(image_blocks) >= 1, "HumanMessage must contain image block"

        # 消息顺序：所有 ToolMessage 在 multimodal HumanMessage 之前
        hm_idx = msgs.index(hm)
        for tm in tool_msgs:
            assert msgs.index(tm) < hm_idx, "ToolMessage should come before multimodal HumanMessage"
