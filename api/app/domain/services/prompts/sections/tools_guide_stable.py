"""tools_guide_stable section — file_view + memory tools static guidance.

B5 C3: wraps the legacy ``FILE_VIEW_HINT`` and ``MEMORY_TOOLS_HINT`` constants,
conditionally rendered based on ``ctx.has_file_view`` and ``ctx.has_memory_tools``.

The text content does NOT depend on the live tool list — only on the boolean
capability flags. This is what makes the section ``cacheable=True``: under
B5.5 prompt caching, the rendered output is stable across steps as long as
the agent capability flags stay the same.

Returns ``SectionOutput(text=None)`` when neither flag is set, so the
assembler skips the section entirely (no empty paragraph in the output).
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


# ---- file_view hint ----------------------------------------------------- #


_ZH_FILE_VIEW = (
    "- **文件理解**：遇到图片、PDF、音频、视频等非文本文件时，**必须使用 `file_view` 工具**而非 `file_read`。\n"
    "  `file_view` 会自动识别文件类型并返回你能理解的内容（图片直接展示、PDF 提取文本、音频转录、视频提取关键帧等）。\n"
    "  `file_read` 仅用于文本文件（代码、配置、日志等），对二进制文件会返回乱码。"
)


_EN_FILE_VIEW = (
    "- **File understanding**: For images, PDFs, audio, and video files, **use `file_view`** instead of `file_read`.\n"
    "  `file_view` automatically detects the file type and returns content you can understand (images displayed, PDFs extracted, audio transcribed, video keyframes extracted).\n"
    "  `file_read` is only for text files (code, config, logs) — binary files will return garbage."
)


# ---- memory tools hint -------------------------------------------------- #


# Always-on portion: search/get guidance keyed on ``has_memory_tools`` flag
# (true whenever any memory-category tool is bound).
_ZH_MEMORY_BASE = (
    "## 记忆工具\n"
    "你可以使用 memory_search 搜索之前对话中的信息。"
    "当用户提到\"之前\"\"上次\"\"以前讨论过\"等暗示历史上下文时，"
    "优先使用 memory_search 查找相关记忆。"
    "搜索结果包含 ID，可用 memory_get 获取完整内容。"
)


_EN_MEMORY_BASE = (
    "## Memory Tools\n"
    "You can use memory_search to find information from previous conversations. "
    "When the user mentions \"previously\", \"last time\", \"we discussed before\", "
    "or other phrases that hint at historical context, prefer memory_search to look up "
    "relevant memories. Search results contain IDs that you can pass to memory_get to "
    "retrieve full content."
)


# Conditional portion: save guidance keyed on ``memory_save in bound_tool_names``.
# ``create_memory_tools`` only adds ``memory_save`` when session_id + write_service
# + session_redis are all wired; this gate mirrors that so we don't teach the LLM
# a tool it cannot actually call (which would bounce through the unknown-tool path).
_ZH_MEMORY_SAVE = (
    "遇到**用户偏好 / 永久性规则 / 可复用事实**时，用 memory_save 记录："
    "category=user 表示用户身份或偏好，rule 表示硬约束，fact 表示事实。"
    "当前 session 最多保存 20 次，重复内容会被自动去重；"
    "不要拿来存对话片段或临时笔记。"
)


_EN_MEMORY_SAVE = (
    "When you encounter **user preferences / permanent rules / reusable facts**, "
    "use memory_save to persist them: category=user for identity/preferences, "
    "rule for hard constraints, fact for durable truths. The current session is "
    "capped at 20 saves and duplicate content is auto-deduplicated; don't use "
    "this for conversational snippets or throwaway notes."
)


# ---- render ------------------------------------------------------------- #


def _render(ctx: RenderContext) -> SectionOutput:
    """Conditionally render file_view + memory tools hints.

    Memory section has two independent gates:
      * ``has_memory_tools`` — any memory-category tool bound → search/get base.
      * ``"memory_save" in bound_tool_names`` — only when the save tool is
        actually bound (requires session_id + write_service + redis). Teaching
        ``memory_save`` when it isn't bound would funnel the LLM into the
        unknown-tool path.
    """
    parts: list[str] = []

    if ctx.has_file_view:
        parts.append(_EN_FILE_VIEW if ctx.lang == "en" else _ZH_FILE_VIEW)

    has_memory_save = "memory_save" in ctx.bound_tool_names
    if ctx.has_memory_tools:
        memory_text = _EN_MEMORY_BASE if ctx.lang == "en" else _ZH_MEMORY_BASE
        if has_memory_save:
            save_text = _EN_MEMORY_SAVE if ctx.lang == "en" else _ZH_MEMORY_SAVE
            memory_text = memory_text + "\n\n" + save_text
        parts.append(memory_text)

    if not parts:
        return SectionOutput(text=None)

    return SectionOutput(
        text="\n\n".join(parts),
        metadata={
            "has_file_view": ctx.has_file_view,
            "has_memory_tools": ctx.has_memory_tools,
            "has_memory_save": has_memory_save,
        },
    )


tools_guide_stable_section = Section(
    id="tools_guide_stable",
    priority=8,
    cacheable=True,
    dynamic=False,
    render=_render,
)
