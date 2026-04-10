"""B5 C5b: parity test for executor_node with PromptAssembler feature flag.

With ``use_prompt_assembler`` OFF, ``executor_node`` uses the legacy
string concatenation path (C0a bundle + C5a fresh_skill_context).
With the flag ON, it goes through ``PromptAssembler.assemble`` with
the C2+C3 section registry.

The two outputs are NOT byte-identical — ``tools_guide_dynamic`` generates
a cleaner tool summary than the legacy ``_build_available_tool_summary``.
What we assert instead is **content equivalence**: both paths produce
a non-empty SystemMessage containing the key structural markers
(identity prose, tool hints when applicable, conversation summaries).

Rather than spin up the full LangGraph executor_node (which requires
mocks for dozens of dependencies), these tests exercise the system
prompt assembly logic directly via ``PromptAssembler.assemble`` and
the legacy concatenation code path. We verify both produce sane output
for representative state and that the version_hash + tokens_used are
populated on the assembler path.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts import get_prompt_section_bundle
from app.domain.services.prompts.assembler import PromptAssembler
from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.render_context import build_render_context
from app.domain.services.prompts.section import PromptMode


def _make_prompt_assembler() -> PromptAssembler:
    return PromptAssembler(
        budget=SystemPromptBudget(max_tokens=10_000),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )


def _representative_state() -> dict:
    """State fixture that exercises every C2+C3 section."""
    return {
        "language": "zh",
        "message": "帮我写一个 Python 脚本",
        "attachments": [],
        "conversation_summaries": ["第一轮：用户问了 X"],
        "skill_context": "## Active Skills\n- skill_foo: do foo things",
        "skill_names_in_context": ["foo"],
    }


def _representative_config() -> dict:
    """Config fixture matching what C5a's fresh_config would produce."""
    llm = MagicMock()
    llm.provider_name = "openai"
    return {
        "configurable": {
            "llm": llm,
            "agent_config": MagicMock(supports_vision=True, supports_pdf_input=True),
            "has_file_view": True,
            "has_memory_tools": True,
            "bound_tool_names": frozenset(
                {
                    "shell_execute",
                    "file_view",
                    "file_read",
                    "memory_search",
                    "skill_foo_bar",
                }
            ),
        }
    }


# ---- Assembler-only structural tests (post-C7.5) ---------------------- #


def _assembler_system_content(
    state: dict, config: dict
) -> tuple[str, str, int]:
    """Run the executor_node PromptAssembler path in isolation.

    Returns (text, version_hash, tokens_used).
    """
    language = state.get("language", "zh")
    section_bundle = get_prompt_section_bundle(language)
    ctx = build_render_context(state, config, config["configurable"]["agent_config"])
    result = _make_prompt_assembler().assemble(
        section_bundle.executor, ctx, PromptMode.FULL, fallback_used=False
    )
    return result.text, result.version_hash, result.tokens_used


# ---- Tests ------------------------------------------------------------- #


def test_assembler_produces_non_empty_system_content() -> None:
    """Sanity: the assembler-only path produces a non-empty SystemMessage body."""
    state = _representative_state()
    config = _representative_config()

    assembler_text, _, _ = _assembler_system_content(state, config)

    assert assembler_text
    assert assembler_text.strip()


def test_assembler_includes_identity_prose() -> None:
    """The assembled text must include the core identity prose (5-step loop)."""
    state = _representative_state()
    config = _representative_config()

    assembler_text, _, _ = _assembler_system_content(state, config)
    # The 5-step loop opens with "任务执行智能体" in ZH.
    assert "任务执行智能体" in assembler_text


def test_assembler_includes_file_view_hint_when_available() -> None:
    """With has_file_view=True, the assembler must mention file_view guidance."""
    state = _representative_state()
    config = _representative_config()

    assembler_text, _, _ = _assembler_system_content(state, config)
    assert "文件理解" in assembler_text


def test_assembler_includes_memory_hint_when_available() -> None:
    """With has_memory_tools=True, the assembler must mention memory tools."""
    state = _representative_state()
    config = _representative_config()

    assembler_text, _, _ = _assembler_system_content(state, config)
    assert "记忆工具" in assembler_text


def test_assembler_includes_conversation_summaries_header() -> None:
    """The assembled text must emit the Chinese conversation summary header."""
    state = _representative_state()
    config = _representative_config()

    assembler_text, _, _ = _assembler_system_content(state, config)
    assert "## 历史对话摘要" in assembler_text
    assert "第一轮：用户问了 X" in assembler_text


def test_assembler_includes_skill_context_body() -> None:
    """The assembled text must emit the skill_context body (Active Skills markdown)."""
    state = _representative_state()
    config = _representative_config()

    assembler_text, _, _ = _assembler_system_content(state, config)
    assert "Active Skills" in assembler_text
    assert "skill_foo" in assembler_text


def test_assembler_path_produces_version_hash_and_tokens() -> None:
    """Flag ON path must populate version_hash (16-char sha) and tokens_used."""
    state = _representative_state()
    config = _representative_config()

    _, version_hash, tokens_used = _assembler_system_content(state, config)
    assert isinstance(version_hash, str)
    assert len(version_hash) == 16  # sha256 truncated to 16 chars
    assert tokens_used > 0


def test_assembler_path_includes_available_tool_summary_header() -> None:
    """The new tools_guide_dynamic section emits its hardcoded English marker."""
    state = _representative_state()
    config = _representative_config()

    assembler_text, _, _ = _assembler_system_content(state, config)
    assert "## Available Tool Summary" in assembler_text


def test_en_language_dispatches_to_en_bundle() -> None:
    """Flag ON path honors state.language for section registry dispatch."""
    state = _representative_state()
    state["language"] = "en"
    state["conversation_summaries"] = ["Round 1: user asked X"]
    state["skill_context"] = "## Active Skills\n- skill_foo: do foo things"
    config = _representative_config()

    assembler_text, _, _ = _assembler_system_content(state, config)
    # English header from conversation_summaries section
    assert "## Conversation History Summary" in assembler_text
    # Chinese markers should be absent from the structural headers
    assert "## 历史对话摘要" not in assembler_text
