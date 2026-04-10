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


_ZH_MEMORY = (
    "## 记忆工具\n"
    "你可以使用 memory_search 搜索之前对话中的信息。"
    "当用户提到\"之前\"\"上次\"\"以前讨论过\"等暗示历史上下文时，"
    "优先使用 memory_search 查找相关记忆。"
    "搜索结果包含 ID，可用 memory_get 获取完整内容。"
)


_EN_MEMORY = (
    "## Memory Tools\n"
    "You can use memory_search to find information from previous conversations. "
    "When the user mentions \"previously\", \"last time\", \"we discussed before\", "
    "or other phrases that hint at historical context, prefer memory_search to look up "
    "relevant memories. Search results contain IDs that you can pass to memory_get to "
    "retrieve full content."
)


# ---- render ------------------------------------------------------------- #


def _render(ctx: RenderContext) -> SectionOutput:
    """Conditionally render file_view + memory tools hints.

    - If neither flag is set → return ``SectionOutput(text=None)`` so the
      assembler skips this section entirely.
    - If only file_view → file_view hint only.
    - If only memory_tools → memory tools hint only.
    - If both → file_view hint, blank line, memory tools hint.
    """
    parts: list[str] = []

    if ctx.has_file_view:
        parts.append(_EN_FILE_VIEW if ctx.lang == "en" else _ZH_FILE_VIEW)

    if ctx.has_memory_tools:
        parts.append(_EN_MEMORY if ctx.lang == "en" else _ZH_MEMORY)

    if not parts:
        return SectionOutput(text=None)

    return SectionOutput(
        text="\n\n".join(parts),
        metadata={
            "has_file_view": ctx.has_file_view,
            "has_memory_tools": ctx.has_memory_tools,
        },
    )


tools_guide_stable_section = Section(
    id="tools_guide_stable",
    priority=8,
    cacheable=True,
    dynamic=False,
    render=_render,
)
