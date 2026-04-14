"""B5 C10: ``dedup_messages`` (role, content_hash) dedup + ``dedup_execution_prompts``.

The existing id-based dedup is already covered by
``test_message_utils.py``. These tests exercise the C10 additions:

1. id-less messages are now deduped by (role, content_hash)
2. later-wins semantic matches id dedup behavior
3. multimodal content is flattened before hashing
4. ``dedup_execution_prompts`` collapses strictly adjacent retry loops
"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.domain.services.graphs.message_utils import (
    _content_hash,
    dedup_execution_prompts,
    dedup_messages,
)


# ---- _content_hash helper --------------------------------------------- #


class TestContentHash:
    def test_empty_string(self) -> None:
        h = _content_hash("")
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex

    def test_stable_for_identical_input(self) -> None:
        assert _content_hash("hello world") == _content_hash("hello world")

    def test_different_for_different_input(self) -> None:
        assert _content_hash("hello") != _content_hash("world")

    def test_short_content_hashed_in_full(self) -> None:
        """Content ≤ 700 chars is hashed in full (no head+tail split)."""
        short = "abc" * 100  # 300 chars
        long = "abc" * 100 + "d"  # 301 chars, still ≤ 700
        assert _content_hash(short) != _content_hash(long)

    def test_long_content_head_tail_boundary(self) -> None:
        """For content > 700 chars, only head[:500] + tail[-200:] is hashed.
        Two messages with different middles but identical head+tail should
        collide — this is the documented tradeoff for tool_call perf."""
        middle_a = "a" * 500 + "MIDDLE_A" * 100 + "z" * 200
        middle_b = "a" * 500 + "MIDDLE_B" * 100 + "z" * 200
        # Both have length > 700 and same head[:500] and same tail[-200:]
        assert len(middle_a) > 700
        assert len(middle_b) > 700
        assert middle_a[:500] == middle_b[:500]
        assert middle_a[-200:] == middle_b[-200:]
        # → documented collision
        assert _content_hash(middle_a) == _content_hash(middle_b)

    def test_long_content_distinguishes_different_tails(self) -> None:
        """Two messages with identical head but different tails (within the
        last 200 chars) must hash differently."""
        same_head = "a" * 500
        padding = "m" * 100  # in the middle, NOT in head[:500] or tail[-200:]
        # Distinguishing bytes must land inside tail[-200:] to be observed
        tail_a = "q" * 100 + "TAIL_A" + "z" * 94  # 200 chars
        tail_b = "q" * 100 + "TAIL_B" + "z" * 94
        msg_a = same_head + padding + tail_a  # len 800
        msg_b = same_head + padding + tail_b
        assert len(msg_a) == 800 and len(msg_b) == 800
        assert _content_hash(msg_a) != _content_hash(msg_b)

    def test_multimodal_content_flattened_before_hashing(self) -> None:
        """A list[dict] content with text+image blocks should hash to the
        same value as a plain-text content of the extracted text."""
        multimodal = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "world"},
        ]
        # _flatten_multimodal_content joins text blocks with \n
        plain = "hello\nworld"
        assert _content_hash(multimodal) == _content_hash(plain)


# ---- dedup_messages (id + content) ------------------------------------ #


class TestDedupByContent:
    def test_identical_human_messages_without_id_collapse(self) -> None:
        """Two HumanMessage with identical text and no id → one result."""
        h1 = HumanMessage(content="same body")
        h2 = HumanMessage(content="same body")
        result = dedup_messages([h1, h2])
        assert len(result) == 1

    def test_later_wins_semantic(self) -> None:
        """Consistent with id dedup: the LATER occurrence replaces the earlier."""
        h_old = HumanMessage(content="same body")
        h_new = HumanMessage(content="same body")
        result = dedup_messages([h_old, h_new])
        # Identity check: the kept message is the latest one
        assert len(result) == 1
        assert result[0] is h_new

    def test_same_text_different_roles_not_merged(self) -> None:
        """HumanMessage and AIMessage with identical text must both survive —
        the content_hash key is (type, hash), not just hash."""
        h = HumanMessage(content="same body")
        a = AIMessage(content="same body")
        result = dedup_messages([h, a])
        assert len(result) == 2

    def test_mixed_id_and_content_dedup(self) -> None:
        """id dedup runs first; remaining id-less duplicates are dedup'd by content."""
        h1 = HumanMessage(content="first", id="id-1")
        h2 = HumanMessage(content="body A")
        h3 = HumanMessage(content="body A")  # dup of h2
        h4 = HumanMessage(content="first updated", id="id-1")  # replaces h1
        result = dedup_messages([h1, h2, h3, h4])
        # h1 replaced by h4 in-place; h2 dropped by h3 (later wins)
        assert len(result) == 2
        # The id-1 slot holds the updated copy
        assert any(
            getattr(m, "id", None) == "id-1" and m.content == "first updated"
            for m in result
        )
        # And exactly one copy of "body A" survives (the later one)
        body_a_copies = [m for m in result if m.content == "body A"]
        assert len(body_a_copies) == 1
        assert body_a_copies[0] is h3

    def test_three_copies_collapse_to_one(self) -> None:
        h1 = HumanMessage(content="repeat")
        h2 = HumanMessage(content="repeat")
        h3 = HumanMessage(content="repeat")
        result = dedup_messages([h1, h2, h3])
        assert len(result) == 1
        assert result[0] is h3

    def test_multimodal_dedup(self) -> None:
        """Two multimodal messages with the same text blocks should collapse
        even if the image_url wrapper objects are different instances."""
        h1 = HumanMessage(
            content=[
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]
        )
        h2 = HumanMessage(
            content=[
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]
        )
        result = dedup_messages([h1, h2])
        # Multimodal flattening extracts only text blocks → both hash to
        # the same value → later wins.
        assert len(result) == 1
        assert result[0] is h2

    def test_interleaved_dedup_preserves_order(self) -> None:
        """[A1, B1, A2] → [B1, A2] — A1 is dropped, B1 stays at its index."""
        a1 = HumanMessage(content="alpha")
        b1 = HumanMessage(content="bravo")
        a2 = HumanMessage(content="alpha")
        result = dedup_messages([a1, b1, a2])
        assert len(result) == 2
        assert result[0] is b1
        assert result[1] is a2

    def test_empty_list(self) -> None:
        assert dedup_messages([]) == []

    def test_single_message_passthrough(self) -> None:
        h = HumanMessage(content="only")
        result = dedup_messages([h])
        assert result == [h]

    def test_long_content_near_700_boundary(self) -> None:
        """Messages just past the 700-char boundary that differ only in
        head[:500] must still hash differently. Same-content messages
        near the boundary may legitimately collide (documented tradeoff
        in ``_content_hash``)."""
        # Both > 700 chars so head+tail mode is used.
        padding = "z" * 200  # identical tail
        head_a = "x" * 450 + "HEAD_A" + "y" * 44  # 500 chars with distinguishing bytes
        head_b = "x" * 450 + "HEAD_B" + "y" * 44
        msg_a = head_a + "m" * 100 + padding  # len 800
        msg_b = head_b + "m" * 100 + padding
        h1 = HumanMessage(content=msg_a)
        h2 = HumanMessage(content=msg_b)
        result = dedup_messages([h1, h2])
        # head[:500] differs → different hash → both preserved
        assert len(result) == 2


# ---- dedup_execution_prompts ------------------------------------------ #


# A minimal ZH EXECUTION_PROMPT-shaped body for fixture purposes. The
# regex stripper looks for the line "你正在执行任务：\n{step}\n\n".
def _zh_exec_prompt(step: str, tail: str = "rest of prompt") -> str:
    return (
        f"\n你正在执行任务：\n{step}\n\n"
        f"用户消息(message):\ntest\n\n"
        f"附件(attachments):\n[]\n\n"
        f"工作语言(language):\nzh\n\n"
        f"{tail}"
    )


def _en_exec_prompt(step: str, tail: str = "rest of prompt") -> str:
    return (
        f"\nYou are executing the task:\n{step}\n\n"
        f"User Message:\ntest\n\n"
        f"Attachments:\n[]\n\n"
        f"Working Language:\nen\n\n"
        f"{tail}"
    )


class TestDedupExecutionPrompts:
    def test_empty_list(self) -> None:
        assert dedup_execution_prompts([]) == []

    def test_single_message(self) -> None:
        h = HumanMessage(content=_zh_exec_prompt("step one"))
        result = dedup_execution_prompts([h])
        assert result == [h]

    def test_different_steps_all_preserved(self) -> None:
        """5 prompts with different step descriptions all survive — the
        step-stripping regex removes the variable part, but the remaining
        fingerprint (template prose) is identical, so wait — this needs
        to be thought through.

        Actually: the fingerprint IS the stripped skeleton. Different
        steps produce the SAME stripped body → same fingerprint → adjacent
        dedup WOULD collapse them. The test case in the design doc is
        wrong in that regard; the correct test is:
        - 5 adjacent identical prompts collapse to 1 (last wins)
        - 5 adjacent DIFFERENT-step prompts collapse to 1 (last wins,
          because step is stripped)
        - non-adjacent prompts (separated by AIMessage/ToolMessage) are
          preserved

        So: the real test for "different steps preserved" is the
        non-adjacent case — where an AIMessage splits them up.
        """
        h1 = HumanMessage(content=_zh_exec_prompt("step one"))
        a1 = AIMessage(content="tool call between")
        h2 = HumanMessage(content=_zh_exec_prompt("step two"))
        a2 = AIMessage(content="another between")
        h3 = HumanMessage(content=_zh_exec_prompt("step three"))
        result = dedup_execution_prompts([h1, a1, h2, a2, h3])
        # All 5 preserved: each HumanMessage is separated by a non-Human.
        assert len(result) == 5

    def test_three_adjacent_identical_prompts_collapse_to_one(self) -> None:
        h1 = HumanMessage(content=_zh_exec_prompt("same step"))
        h2 = HumanMessage(content=_zh_exec_prompt("same step"))
        h3 = HumanMessage(content=_zh_exec_prompt("same step"))
        result = dedup_execution_prompts([h1, h2, h3])
        assert len(result) == 1
        assert result[0] is h3  # later wins

    def test_adjacent_different_steps_collapse_because_skeleton_matches(
        self,
    ) -> None:
        """The regex strips the step block → identical skeleton → same
        fingerprint → adjacent pairs collapse. This is the retry-loop
        optimization the function targets."""
        h1 = HumanMessage(content=_zh_exec_prompt("do A"))
        h2 = HumanMessage(content=_zh_exec_prompt("do B"))
        result = dedup_execution_prompts([h1, h2])
        assert len(result) == 1
        assert result[0] is h2

    def test_interrupted_by_ai_message_preserves_both(self) -> None:
        h1 = HumanMessage(content=_zh_exec_prompt("do A"))
        ai = AIMessage(content="thinking...")
        h2 = HumanMessage(content=_zh_exec_prompt("do A"))
        result = dedup_execution_prompts([h1, ai, h2])
        assert len(result) == 3
        assert result == [h1, ai, h2]

    def test_interrupted_by_tool_message_preserves_both(self) -> None:
        h1 = HumanMessage(content=_zh_exec_prompt("do A"))
        tool = ToolMessage(content="result", tool_call_id="tc1")
        h2 = HumanMessage(content=_zh_exec_prompt("do A"))
        result = dedup_execution_prompts([h1, tool, h2])
        assert len(result) == 3

    def test_non_execution_prompt_human_messages_preserved(self) -> None:
        """Regular HumanMessages (without the EXECUTION_PROMPT skeleton)
        are left untouched — the function is narrowly scoped."""
        h1 = HumanMessage(content="just a regular message")
        h2 = HumanMessage(content="another regular message")
        result = dedup_execution_prompts([h1, h2])
        assert len(result) == 2

    def test_system_message_passes_through(self) -> None:
        sys = SystemMessage(content="system context")
        h = HumanMessage(content=_zh_exec_prompt("step"))
        result = dedup_execution_prompts([sys, h])
        assert len(result) == 2
        assert result[0] is sys

    def test_en_execution_prompt_collapses(self) -> None:
        """Both ZH and EN patterns are recognized."""
        h1 = HumanMessage(content=_en_exec_prompt("step one"))
        h2 = HumanMessage(content=_en_exec_prompt("step two"))
        result = dedup_execution_prompts([h1, h2])
        assert len(result) == 1
        assert result[0] is h2

    def test_zh_and_en_do_not_cross_collapse(self) -> None:
        """Adjacent ZH and EN prompts should still collapse IF their
        stripped-skeleton fingerprints happen to match. They should NOT
        collapse because the template prose differs between languages."""
        h_zh = HumanMessage(content=_zh_exec_prompt("step"))
        h_en = HumanMessage(content=_en_exec_prompt("step"))
        result = dedup_execution_prompts([h_zh, h_en])
        assert len(result) == 2

    def test_mixed_with_regular_human_message(self) -> None:
        h_exec1 = HumanMessage(content=_zh_exec_prompt("A"))
        h_regular = HumanMessage(content="user chatting")
        h_exec2 = HumanMessage(content=_zh_exec_prompt("B"))
        result = dedup_execution_prompts([h_exec1, h_regular, h_exec2])
        # h_regular is not an EXECUTION_PROMPT, it breaks adjacency.
        assert len(result) == 3
