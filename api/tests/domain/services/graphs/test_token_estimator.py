"""Tests for TokenEstimator."""
import pytest

pytestmark = pytest.mark.anyio  # required for async tests appended in Task 4


class TestEstimateText:
    """Test estimate() for char and hybrid strategies."""

    # --- char strategy ---

    def test_char_strategy_ascii(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="char")
        assert est.estimate("hello world") == len("hello world") // 3  # 11 // 3 = 3

    def test_char_strategy_chinese(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="char")
        assert est.estimate("你好世界") == len("你好世界") // 3  # 4 // 3 = 1

    # --- hybrid strategy ---

    def test_hybrid_pure_ascii(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        result = est.estimate("hello world")
        assert result == max(round(11 * 0.25), 1)  # 3

    def test_hybrid_pure_chinese(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        assert est.estimate("你好世界") == 6  # 4 * 1.5

    def test_hybrid_mixed(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        text = "Hello你好World世界"
        result = est.estimate(text)
        # ASCII: H,e,l,l,o,W,o,r,l,d = 10 → 10*0.25 = 2.5
        # CJK: 你,好,世,界 = 4 → 4*1.5 = 6.0
        # total = 8.5 → round = 8
        assert result == round(8.5)

    def test_hybrid_empty_string(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        assert est.estimate("") == 0

    def test_hybrid_emoji(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        result = est.estimate("😀🎉")
        assert result == 2  # 2 non-ASCII non-CJK → 2 * 1.0

    def test_minimum_one_token(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        assert est.estimate("a") >= 1

    def test_effective_strategy_hybrid(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        assert est.effective_strategy == "hybrid"

    def test_effective_strategy_char(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="char")
        assert est.effective_strategy == "char"


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class TestEstimateMessage:
    """Test estimate_message() for various message types."""

    def test_human_message_str(self):
        from app.domain.services.graphs.token_estimator import (
            MESSAGE_OVERHEAD_TOKENS,
            TokenEstimator,
        )

        est = TokenEstimator(strategy="hybrid")
        msg = HumanMessage(content="test")
        expected = est.estimate("test") + MESSAGE_OVERHEAD_TOKENS
        assert est.estimate_message(msg) == expected

    def test_ai_message_with_tool_calls(self):
        from app.domain.services.graphs.token_estimator import (
            MESSAGE_OVERHEAD_TOKENS,
            TokenEstimator,
        )

        est = TokenEstimator(strategy="hybrid")
        msg = AIMessage(
            content="I'll search for that.",
            tool_calls=[
                {"id": "c1", "name": "search", "args": {"query": "test query"}},
            ],
        )
        result = est.estimate_message(msg)
        text_tokens = est.estimate("I'll search for that.")
        args_tokens = est.estimate(str({"query": "test query"}))
        assert result == MESSAGE_OVERHEAD_TOKENS + text_tokens + args_tokens

    def test_tool_message(self):
        from app.domain.services.graphs.token_estimator import (
            MESSAGE_OVERHEAD_TOKENS,
            TokenEstimator,
        )

        est = TokenEstimator(strategy="hybrid")
        msg = ToolMessage(content="search result here", tool_call_id="c1")
        expected = est.estimate("search result here") + MESSAGE_OVERHEAD_TOKENS
        assert est.estimate_message(msg) == expected

    def test_system_message(self):
        from app.domain.services.graphs.token_estimator import (
            MESSAGE_OVERHEAD_TOKENS,
            TokenEstimator,
        )

        est = TokenEstimator(strategy="hybrid")
        msg = SystemMessage(content="You are a helpful assistant.")
        expected = est.estimate("You are a helpful assistant.") + MESSAGE_OVERHEAD_TOKENS
        assert est.estimate_message(msg) == expected


class TestEstimateMessages:
    """Test estimate_messages() batch estimation."""

    def test_multiple_messages_sum(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]
        expected = sum(est.estimate_message(m) for m in msgs)
        assert est.estimate_messages(msgs) == expected

    def test_empty_list(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        assert est.estimate_messages([]) == 0


import logging
from unittest.mock import MagicMock, patch


class TestProviderApiStrategy:
    """Test provider_api strategy with tiktoken integration."""

    def test_tiktoken_available(self):
        """Real tiktoken test — skipped in CI if tiktoken not installed."""
        tiktoken = pytest.importorskip("tiktoken")
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="provider_api", model_name="gpt-4o")
        assert est.effective_strategy == "provider_api"

        text = "Hello, 你好世界!"
        expected = len(tiktoken.encoding_for_model("gpt-4o").encode(text))
        assert est.estimate(text) == expected

    def test_tiktoken_fallback_unknown_model(self, caplog):
        """Unknown model → _try_load_tiktoken hits KeyError → hybrid."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.side_effect = KeyError("no encoding")

        with caplog.at_level(logging.INFO):
            with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
                est = TokenEstimator(
                    strategy="provider_api", model_name="deepseek-reasoner"
                )
        assert est.effective_strategy == "hybrid"
        assert "falling back to hybrid" in caplog.text
        assert est.estimate("你好") == round(2 * 1.5)  # 3

    def test_tiktoken_not_installed(self, caplog):
        """tiktoken ImportError → _try_load_tiktoken catches it → hybrid."""
        import sys
        from app.domain.services.graphs.token_estimator import TokenEstimator

        saved = sys.modules.pop("tiktoken", None)
        try:
            with caplog.at_level(logging.INFO):
                with patch.dict("sys.modules", {"tiktoken": None}):
                    est = TokenEstimator(
                        strategy="provider_api", model_name="gpt-4o"
                    )
            assert est.effective_strategy == "hybrid"
            assert "falls back to hybrid" in caplog.text
        finally:
            if saved is not None:
                sys.modules["tiktoken"] = saved

    def test_effective_strategy_no_fallback(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        assert est.effective_strategy == "hybrid"


from app.domain.models.memory import Memory
from app.domain.services.graphs.message_utils import (
    dicts_to_messages,
    messages_to_dicts,
)


class TestIntegrationWithCheckOverflow:
    """End-to-end tests matching the real _check_overflow path.
    Real flow: BaseMessage → messages_to_dicts → Memory → compact → dicts_to_messages → estimate.
    Tests cover both the full roundtrip (starting from BaseMessage) and the post-dict path.
    """

    def test_full_roundtrip_from_base_messages(self):
        """Full roundtrip: BaseMessage → messages_to_dicts → Memory → compact → estimate."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")

        base_msgs = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="请帮我搜索这个问题" * 50),
            AIMessage(
                content="我来帮你搜索。",
                tool_calls=[
                    {"id": "c1", "name": "search", "args": {"query": "测试查询"}},
                ],
            ),
            ToolMessage(content="搜索结果：" + "结果内容" * 100, tool_call_id="c1"),
        ]

        # Step 1: messages_to_dicts (real serialization path)
        dict_msgs = messages_to_dicts(base_msgs)

        # Verify serialization happened: tool_calls args should be JSON string
        ai_dict = [m for m in dict_msgs if m["role"] == "assistant"][0]
        assert isinstance(ai_dict["tool_calls"][0]["function"]["arguments"], str)

        # Step 2: Memory + compact
        memory = Memory(messages=dict_msgs)
        memory.compact(keep_summary=True)

        # Step 3: dicts_to_messages + estimate
        lc_msgs = dicts_to_messages(memory.messages)
        result = est.estimate_messages(lc_msgs)

        # Should be significantly higher than old len/3 due to Chinese content
        total_content = "".join(m.get("content", "") for m in memory.messages)
        old_estimate = len(total_content) // 3
        assert result > old_estimate

    def test_full_roundtrip_multimodal_flattened(self):
        """BaseMessage with multimodal content → messages_to_dicts flattens it → estimate."""
        from app.domain.services.graphs.token_estimator import (
            IMAGE_TOKEN_ESTIMATE,
            TokenEstimator,
        )

        est = TokenEstimator(strategy="hybrid")

        base_msgs = [
            HumanMessage(content=[
                {"type": "text", "text": "Look at this image:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ]),
        ]

        dict_msgs = messages_to_dicts(base_msgs)
        assert isinstance(dict_msgs[0]["content"], str)

        lc_msgs = dicts_to_messages(dict_msgs)
        result = est.estimate_messages(lc_msgs)
        assert result < IMAGE_TOKEN_ESTIMATE

    def test_chinese_not_underestimated(self):
        """Core regression: 1000 Chinese chars must estimate >> old len/3."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        chinese_text = "你" * 1000
        dict_msgs = [{"role": "user", "content": chinese_text}]
        lc_msgs = dicts_to_messages(dict_msgs)
        result = est.estimate_messages(lc_msgs)

        old_estimate = len(chinese_text) // 3  # 333
        assert result > old_estimate * 3

    def test_real_path_after_compact(self):
        """Full compact → estimate path: tokens decrease after compact."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        dict_msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Search the web."},
            {
                "role": "assistant",
                "content": "I'll browse that.",
                "reasoning_content": "Let me think step by step..." * 100,
            },
            {
                "role": "tool",
                "function_name": "browser_view",
                "content": "<html>" + "x" * 5000 + "</html>",
            },
        ]
        memory_before = Memory(messages=[m.copy() for m in dict_msgs])
        tokens_before = est.estimate_messages(dicts_to_messages(memory_before.messages))

        memory_after = Memory(messages=[m.copy() for m in dict_msgs])
        memory_after.compact(keep_summary=False)
        tokens_after = est.estimate_messages(dicts_to_messages(memory_after.messages))

        assert tokens_after > 0
        assert tokens_after < tokens_before

    def test_real_path_browser_tool_compacted_default(self):
        """Production default: keep_summary=True → browser_view gets summary."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        html = "<html><head><title>搜索结果</title></head><body>" + "大量网页内容" * 500 + "</body></html>"
        dict_msgs = [
            {
                "role": "tool",
                "function_name": "browser_view",
                "content": html,
            },
        ]
        memory = Memory(messages=dict_msgs)
        memory.compact(keep_summary=True)

        assert memory.messages[0]["content"] != html
        assert "browser_view" in memory.messages[0]["content"]

        lc_msgs = dicts_to_messages(memory.messages)
        result = est.estimate_messages(lc_msgs)
        original_tokens = est.estimate_messages(dicts_to_messages([{"role": "tool", "content": html}]))
        assert result < original_tokens

    def test_real_path_browser_tool_compacted_no_summary(self):
        """keep_summary=False → browser_view becomes '(removed)'."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        dict_msgs = [
            {
                "role": "tool",
                "function_name": "browser_view",
                "content": "<html><body>" + "大量网页内容" * 500 + "</body></html>",
            },
        ]
        memory = Memory(messages=dict_msgs)
        memory.compact(keep_summary=False)

        assert memory.messages[0]["content"] == "(removed)"

        lc_msgs = dicts_to_messages(memory.messages)
        result = est.estimate_messages(lc_msgs)
        assert result < 20

    def test_real_path_reasoning_content_removed_by_compact(self):
        """compact() deletes reasoning_content."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        dict_msgs = [
            {
                "role": "assistant",
                "content": "Final answer.",
                "reasoning_content": "Internal reasoning " * 200,
            },
        ]
        memory = Memory(messages=[m.copy() for m in dict_msgs])
        memory.compact(keep_summary=False)

        assert "reasoning_content" not in memory.messages[0]

        lc_msgs = dicts_to_messages(memory.messages)
        result = est.estimate_messages(lc_msgs)
        expected = est.estimate("Final answer.") + 3
        assert result == expected


class TestRobustness:
    """Defensive boundaries — not standard runtime path."""

    def test_corrupted_tool_call_args(self):
        """Invalid JSON in tool_calls args → dicts_to_messages degrades to {}."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        est = TokenEstimator(strategy="hybrid")
        dict_msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "NOT VALID JSON {{{",
                        },
                    }
                ],
            },
        ]
        lc_msgs = dicts_to_messages(dict_msgs)
        result = est.estimate_messages(lc_msgs)
        assert result > 0

    def test_image_only_content_stringified(self):
        """image-only content after _flatten_multimodal_content → str(list)."""
        from app.domain.services.graphs.token_estimator import (
            IMAGE_TOKEN_ESTIMATE,
            TokenEstimator,
        )

        est = TokenEstimator(strategy="hybrid")
        stringified = str([{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}])
        dict_msgs = [{"role": "user", "content": stringified}]
        lc_msgs = dicts_to_messages(dict_msgs)
        result = est.estimate_messages(lc_msgs)
        assert result > 0
        assert result < IMAGE_TOKEN_ESTIMATE


class TestEstimateMessageImageBlock:
    """Future path (B2 reserved) — raw BaseMessage with image blocks."""

    def test_image_block_estimation(self):
        from app.domain.services.graphs.token_estimator import (
            IMAGE_TOKEN_ESTIMATE,
            MESSAGE_OVERHEAD_TOKENS,
            TokenEstimator,
        )

        est = TokenEstimator(strategy="hybrid")
        # Use a presigned URL (not data URL) to get the fixed IMAGE_TOKEN_ESTIMATE fallback
        msg = HumanMessage(
            content=[{"type": "image_url", "image_url": {"url": "https://minio.example.com/img.png"}}]
        )
        assert est.estimate_message(msg) == IMAGE_TOKEN_ESTIMATE + MESSAGE_OVERHEAD_TOKENS

    def test_mixed_text_image_blocks(self):
        from app.domain.services.graphs.token_estimator import (
            IMAGE_TOKEN_ESTIMATE,
            MESSAGE_OVERHEAD_TOKENS,
            TokenEstimator,
        )

        est = TokenEstimator(strategy="hybrid")
        # Use a presigned URL (not data URL) to get the fixed IMAGE_TOKEN_ESTIMATE fallback
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Look at this image:"},
                {"type": "image_url", "image_url": {"url": "https://minio.example.com/img.png"}},
            ]
        )
        text_tokens = est.estimate("Look at this image:")
        assert est.estimate_message(msg) == text_tokens + IMAGE_TOKEN_ESTIMATE + MESSAGE_OVERHEAD_TOKENS


class TestCheckOverflowWiring:
    """Verify _check_overflow actually uses TokenEstimator when guard is enabled."""

    async def test_check_overflow_uses_token_estimator(self):
        """With guard enabled, _check_overflow should use the estimator, not len/3."""
        from unittest.mock import AsyncMock, MagicMock

        from langgraph.checkpoint.memory import MemorySaver

        from app.domain.models.app_config import AgentConfig
        from app.domain.models.context_overflow_config import ContextOverflowConfig
        from app.domain.models.memory import Memory
        from app.application.services.sandbox_accessors import (
            EagerBrowserAccessor,
            EagerSandboxAccessor,
        )
        from app.domain.services.flows.planner_react import PlannerReActFlow

        overflow = ContextOverflowConfig(
            context_overflow_guard_enabled=True,
            context_window=2048,
            soft_trigger_ratio=0.5,
            hard_trigger_ratio=0.5,
            token_estimator="hybrid",
        )

        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=False)
        mock_uow.session = AsyncMock()
        mock_uow.session.save_memory = AsyncMock()

        flow = PlannerReActFlow(
            uow_factory=MagicMock(return_value=mock_uow),
            llm=MagicMock(),
            agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
            session_id="test-session",
            browser_accessor=EagerBrowserAccessor(AsyncMock()),
            sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
            search_engine=AsyncMock(),
            mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
            a2a_tool=MagicMock(manager=None),
            skill_tool=MagicMock(),
            overflow_config=overflow,
            checkpointer=MemorySaver(),
        )

        assert hasattr(flow, "_token_estimator")
        assert flow._token_estimator.effective_strategy == "hybrid"

        # 800 Chinese chars: hybrid ≈ 800*1.5=1200 * 1.15 safety ≈ 1380 > hard_limit 1024
        # Old len/3: 800/3*1.15 ≈ 307 < 1024 → would NOT trigger
        chinese_msgs = [{"role": "user", "content": "你" * 800}]
        memory = Memory(messages=chinese_msgs)

        await flow._check_overflow(memory)

        mock_uow.session.save_memory.assert_called_once()


class TestImageTokenDynamicEstimation:
    """Tests for dynamic image token estimation (M1b)."""

    def test_data_url_uses_base64_length(self):
        from app.domain.services.graphs.token_estimator import _estimate_image_tokens
        import math
        b64_data = "A" * 1000
        block = {"image_url": {"url": f"data:image/png;base64,{b64_data}"}}
        result = _estimate_image_tokens(block)
        assert result == math.ceil(1000 * 0.125)

    def test_presigned_url_uses_fallback(self):
        from app.domain.services.graphs.token_estimator import IMAGE_TOKEN_ESTIMATE, _estimate_image_tokens
        block = {"image_url": {"url": "https://minio.example.com/presigned/img.png"}}
        result = _estimate_image_tokens(block)
        assert result == IMAGE_TOKEN_ESTIMATE

    def test_empty_url_uses_fallback(self):
        from app.domain.services.graphs.token_estimator import IMAGE_TOKEN_ESTIMATE, _estimate_image_tokens
        block = {"image_url": {"url": ""}}
        result = _estimate_image_tokens(block)
        assert result == IMAGE_TOKEN_ESTIMATE

    def test_estimate_message_uses_dynamic_for_data_url(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from langchain_core.messages import HumanMessage
        import math
        b64_data = "B" * 800
        msg = HumanMessage(content=[
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
        ])
        est = TokenEstimator(strategy="char")
        tokens = est.estimate_message(msg)
        expected_image = math.ceil(800 * 0.125)
        assert tokens < 2000 + 50  # far less than old fixed estimate

    def test_file_block_estimation_scales_with_size(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from langchain_core.messages import HumanMessage
        import base64
        # ~320KB base64 → ceil(320000/81920) = 4 pages → 8000 tokens
        raw_bytes = b"\x00" * 240_000
        b64_data = base64.b64encode(raw_bytes).decode()
        msg = HumanMessage(content=[
            {"type": "file", "file": {"filename": "doc.pdf", "file_data": f"data:application/pdf;base64,{b64_data}"}},
        ])
        est = TokenEstimator(strategy="char")
        tokens = est.estimate_message(msg)
        # ceil(320000/81920) = 4 pages → 4 * 2000 = 8000 (+ overhead)
        assert tokens >= 8000

    def test_file_block_estimation_uses_ceil_not_floor(self):
        """Boundary test: just over 1 page threshold should estimate 2 pages, not 1."""
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from langchain_core.messages import HumanMessage
        import base64
        # 80KB + 1 byte raw → base64 slightly > 80KB → ceil says 2 pages, floor says 1
        raw_bytes = b"\x00" * (80 * 1024 + 1)
        b64_data = base64.b64encode(raw_bytes).decode()
        msg = HumanMessage(content=[
            {"type": "file", "file": {"filename": "doc.pdf", "file_data": f"data:application/pdf;base64,{b64_data}"}},
        ])
        est = TokenEstimator(strategy="char")
        tokens = est.estimate_message(msg)
        # ceil says 2 pages → 4000+ tokens; floor would say 1 → only 2000+
        assert tokens >= 4000

    def test_file_block_no_data_uses_fallback(self):
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content=[
            {"type": "file", "file": {"file_id": "file-abc123"}},
        ])
        est = TokenEstimator(strategy="char")
        tokens = est.estimate_message(msg)
        assert tokens >= 5000
