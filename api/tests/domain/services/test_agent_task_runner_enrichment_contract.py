"""R1 CS1 — pin the enrichable vs identity-only category partition.

Runs on every CI build to keep `_handle_tool_event` in sync with the
KNOWN_CATEGORIES contract. If someone adds a new category, this test
fails until they decide which side of the partition it lives on.
"""
from __future__ import annotations

import pytest

from app.domain.services.tools.tool_source_resolver import KNOWN_CATEGORIES


ENRICHABLE_CATEGORIES: frozenset[str] = frozenset({
    "browser", "search", "shell", "file",
    "mcp", "a2a", "skill", "skill creator",
})
"""Categories where _handle_tool_event triggers UI enrichment."""

IDENTITY_ONLY_CATEGORIES: frozenset[str] = frozenset({
    "message", "memory", "mcp discovery", "skill guide",
    # R2 CS2: sentinel for LLM-hallucinated tool names. Listed here so
    # the partition invariant keeps holding — no _handle_tool_event
    # branch matches "unknown", so the raw error message surfaces
    # instead of triggering shell / browser / file side effects.
    "unknown",
})
"""Categories that exist in the contract but intentionally skip enrichment.

These are not bugs — their tool outputs are plain text that needs no
structured UI representation. R1 documents this as design, not accident.
"""


class TestCategoryPartition:
    def test_partition_covers_all_known_categories(self):
        """Union of enrichable + identity-only == KNOWN_CATEGORIES."""
        assert (ENRICHABLE_CATEGORIES | IDENTITY_ONLY_CATEGORIES) == KNOWN_CATEGORIES

    def test_partition_is_disjoint(self):
        """A category is either enrichable or identity-only, not both."""
        assert not (ENRICHABLE_CATEGORIES & IDENTITY_ONLY_CATEGORIES)

    def test_skill_creator_uses_space_not_underscore(self):
        """Regression: pre-R1 agent_task_runner used 'skill_creation' (underscore).
        R1 canonical is 'skill creator' (space). KNOWN_CATEGORIES must agree."""
        assert "skill creator" in KNOWN_CATEGORIES
        assert "skill_creation" not in KNOWN_CATEGORIES
        assert "skill creator" in ENRICHABLE_CATEGORIES
