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

``serialize_memory_file`` wraps frontmatter + body with ``---`` fences, matches
the SKILL.md exporter pattern but enforces a stable key order (``sort_keys=False``
relies on Python dict insertion order, so we rebuild the dict in canonical order
before dumping — callers can't accidentally reorder by feeding an arbitrarily
ordered mapping).

Parsing / roundtrip is deferred to PR-5B (``FsReconciler``).
"""
from __future__ import annotations

from typing import Any, Mapping

import yaml

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
