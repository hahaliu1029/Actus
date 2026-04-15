"""Test-only reset helper for the ToolSource registry.

Deliberately kept OUT of the resolver module itself so prod import path
cannot reach this function. If a prod code path ever called reset_tool_source_registry(),
all dynamically-registered MCP wrappers and dynamic skill tools would vanish
mid-process and subsequent lookups would fail.

This file lives under api/tests/ so it's excluded from the production package.
"""
from __future__ import annotations

from app.domain.services.tools.tool_source_resolver import (
    _REGISTRY,
    _bootstrap_registry,
)


def reset_tool_source_registry() -> None:
    """Clear _REGISTRY and re-seed the canonical static identities.

    Call this from test fixtures (autouse recommended) to ensure every test
    starts from a production-equivalent state: bootstrap-seeded canonical
    names present, any dynamic registrations from previous tests cleared.
    """
    _REGISTRY.clear()
    _bootstrap_registry()
