"""Canonical YAML frontmatter serialization for memory files.

Design ref: `liuyixuan-develop-design-20260417-094857.md` §P2 Canonical frontmatter:

    id: 01HXYZ...
    title: "..."
    category: user | rule | fact
    source: session_flush | manual | memory_save | file
    created_at: 2026-04-17T09:48:57Z
    updated_at: 2026-04-17T09:48:57Z
    tags: [...]
    pinned: false

PR-5A added ``serialize_memory_file``; PR-5B adds ``parse_memory_file`` (inverse
for FsReconciler orphan detection) plus the canonical ``build_memory_frontmatter``
builder, shared by application layer (MemoryManagementService writes) and
infrastructure layer (FsReconciler rebuilds). Previously the builder lived on
``MemoryManagementService`` as a staticmethod — reconciler couldn't import it
without application → infrastructure coupling, and a copy would drift.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import yaml

if TYPE_CHECKING:
    from app.domain.models.memory_chunk import MemoryChunk


# Canonical key order. FsMemoryWriter is the only writer of memory files,
# so every file on disk comes out of this module with stable ordering — that
# matters for deterministic `git diff` when users version-control the memory
# directory, and for byte-for-byte testing of the writer.
FRONTMATTER_KEYS: tuple[str, ...] = (
    "id",
    "title",
    "category",
    "source",
    "created_at",
    "updated_at",
    "tags",
    "pinned",
)

TITLE_MAX_LENGTH = 80


def _canonical_frontmatter(frontmatter: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with keys in canonical order.

    Unknown keys are appended in their incoming order after the canonical set,
    so additive extensions (e.g. PR-4+8 could add ``auto_promoted_at``) don't
    silently get dropped. But they lose their canonical slot — writers should
    extend ``FRONTMATTER_KEYS`` when adding a first-class field.
    """
    canonical: dict[str, Any] = {}
    for key in FRONTMATTER_KEYS:
        if key in frontmatter:
            canonical[key] = frontmatter[key]
    for key, value in frontmatter.items():
        if key not in canonical:
            canonical[key] = value
    return canonical


def serialize_memory_file(frontmatter: Mapping[str, Any], body: str) -> str:
    """Render ``---\\n<yaml>\\n---\\n\\n<body>\\n`` in canonical order.

    Newline semantics: exactly one blank line between closing ``---`` and body;
    a single trailing ``\\n`` at EOF (POSIX text file convention). If ``body``
    itself ends in ``\\n`` we do not add a second — avoids ballooning trailing
    whitespace on repeated writes.
    """
    ordered = _canonical_frontmatter(frontmatter)
    yaml_block = yaml.dump(
        ordered,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip()

    parts = ["---", yaml_block, "---", ""]
    stripped = body.rstrip("\n")
    if stripped:
        parts.append(stripped)
    return "\n".join(parts) + "\n"


def parse_memory_file(text: str) -> tuple[dict[str, Any], str]:
    """Inverse of ``serialize_memory_file``: split ``---<yaml>---<body>`` doc.

    Returns ``(frontmatter_dict, body_str)``. Tolerant to:
    - missing trailing ``\\n`` at EOF
    - body starting directly after closing fence (no blank line) — still parses
    - CRLF line endings (normalized to LF for the split, preserved in body)

    Raises ``ValueError`` on malformed input:
    - no opening ``---`` fence on the first non-empty line
    - no closing ``---`` fence
    - YAML parse error

    FsReconciler uses this only to extract ``id`` for orphan detection. Callers
    should treat unknown frontmatter values as opaque — don't rely on parse
    round-tripping when the file was hand-edited (power-user path, design L690).
    """
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        raise ValueError("memory file 缺少开头 --- 分隔符")
    start = i + 1

    end = None
    for j in range(start, len(lines)):
        if lines[j].strip() == "---":
            end = j
            break
    if end is None:
        raise ValueError("memory file 缺少结尾 --- 分隔符")

    yaml_block = "\n".join(lines[start:end])
    try:
        loaded = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"memory file frontmatter YAML 解析失败: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("memory file frontmatter 不是 YAML mapping")

    body_lines = lines[end + 1 :]
    # 跳过 closing fence 后的单个空行（canonical 格式）; 多个空行全保留。
    if body_lines and body_lines[0] == "":
        body_lines = body_lines[1:]
    body = "\n".join(body_lines)
    return loaded, body


def derive_title(content: str, *, max_length: int = TITLE_MAX_LENGTH) -> str:
    """M1 降级版 title：内容首行 strip + 截断。

    设计 L66：``title`` = "一行摘要，用户可手写或 LLM 自动生成"。M1 阶段没有 UI
    title 输入也没有 LLM summary，取内容首行做 best-effort 衍生——保证
    frontmatter canonical 完整性。后续 PR 接 LLM 版本时替换本函数即可。
    """
    first_line = content.split("\n", 1)[0].strip()
    if not first_line:
        return "untitled"
    if len(first_line) <= max_length:
        return first_line
    return first_line[:max_length].rstrip() + "…"


def build_memory_frontmatter(chunk: "MemoryChunk") -> dict[str, Any]:
    """Canonical frontmatter for a ``MemoryChunk``.

    应用层写入（MemoryManagementService.create / update）和基础设施层重建
    （FsReconciler orphan DB 行重写）共享此构造——保证两条路径序列化结果一致，
    不至于"应用层写出来的文件格式"和"reconciler 重建出来的文件格式"互相漂移。

    设计 L63-73 / L424：title 降级为内容首行（``derive_title``），tags 取自
    ``metadata.tags``，auto_promoted_at（可选扩展字段）在有值时以 ISO 串附加。
    """
    fm: dict[str, Any] = {
        "id": chunk.id,
        "title": derive_title(chunk.content),
        "category": chunk.category,
        "source": chunk.source,
        "created_at": chunk.created_at.isoformat(),
        "updated_at": chunk.updated_at.isoformat(),
        "tags": chunk.metadata.get("tags", []) if chunk.metadata else [],
        "pinned": chunk.pinned,
    }
    if chunk.auto_promoted_at is not None:
        fm["auto_promoted_at"] = chunk.auto_promoted_at.isoformat()
    return fm
