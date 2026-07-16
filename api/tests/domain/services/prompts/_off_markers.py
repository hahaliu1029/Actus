"""Shared FORBIDDEN_OFF_MARKERS constant for SPM off-variant prompt scans.

SPM Task 26 (plan §Task 26 / R28-META3): the sandbox-forbidden marker set is a
**single source of truth** shared by the off-variant prompt-text scanning tests
(Task 26 sections + Task 27 bundle templates). It is **derived** from the
production constant ``SANDBOX_FACE_TOOL_NAMES`` (the four-family sandbox tool
name set that Task 24 gates off) — never a hand-maintained tuple — so that if a
future static template leaks a sandbox tool name, the marker set already covers
it (R23 fixed a hand-written tuple that had silently dropped
``file_str_replace`` / ``file_find_in_content`` / ``file_find_by_name`` /
``file_list``).

To that derived tool-name set we union a small set of sandbox-**semantic** words
(bilingual): the sandbox itself, its canonical workspace path, the file-tools /
takeover / browser / terminal vocabulary. The skill-creation tool names
(``generate_skill`` / ``brainstorm_skill`` / ``install_skill``) are already in
``SANDBOX_FACE_TOOL_NAMES`` and need no separate entry.

Scanning discipline (frozen): scan only the **static** prompt templates /
sections. Do NOT scan dynamic skill/memory/user-authored text, and do NOT add
over-broad bare tokens like ``"file"`` / ``"skill"`` that would false-positive on
legitimate non-sandbox prose.
"""
from __future__ import annotations

from app.domain.services.tools.langchain_tools import SANDBOX_FACE_TOOL_NAMES

# Derived (not hard-coded): full sandbox tool-name closure ∪ sandbox-semantic
# words. Any static off-variant template/section must contain none of these.
FORBIDDEN_OFF_MARKERS: frozenset[str] = frozenset(SANDBOX_FACE_TOOL_NAMES) | {
    "沙箱",
    "sandbox",
    "/home/ubuntu",
    "文件工具",
    "file tools",
    "takeover",
    "浏览器",
    "终端",
    "browser",
    "terminal",
}
