"""Tests for the B9 extension attribution registry (Task 18, P-8/P-9).

Covers:
- P-8 register_extension_tool / resolve_extension roundtrip + idempotency + caps.
- Negative anchor: attribution is driven by EXPLICIT bindings, NOT by name
  parsing (this file must contain NO ``split(`` on tool names).
- Registration-point wiring for langchain_mcp (consumes tool_server_bindings)
  and langchain_dynamic_skill_tools (consumes _tool_bindings).
- conftest autouse isolation between tests (no manual reset in this file).
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.services.tools.extension_attribution import (
    MAX_REGISTRY_SIZE,
    MAX_TOOLS_PER_EXTENSION,
    register_extension_tool,
    resolve_extension,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------- #
# P-8 core registry behavior
# --------------------------------------------------------------------------- #


def test_register_and_resolve_roundtrip():
    register_extension_tool("mcp_notion_search", "mcp", "notion")
    assert resolve_extension("mcp_notion_search") == ("mcp", "notion")


def test_resolve_unknown_returns_none():
    # A native tool name and an A2A tool name — neither is ever registered
    # (native tools have no extension; A2A never registers per D9).
    assert resolve_extension("file_read") is None
    assert resolve_extension("call_remote_agent") is None


def test_idempotent_overwrite():
    register_extension_tool("skill_demo_run", "skill", "skill-123")
    # Re-register same name + SAME owner (idempotent overwrite): must NOT double-count
    # the per-extension quota (zero-cost refresh), and resolve returns the latest value.
    from app.domain.services.tools.extension_attribution import _PER_EXTENSION_COUNTS

    assert _PER_EXTENSION_COUNTS[("skill", "skill-123")] == 1
    register_extension_tool("skill_demo_run", "skill", "skill-123")
    register_extension_tool("skill_demo_run", "skill", "skill-123")
    assert resolve_extension("skill_demo_run") == ("skill", "skill-123")
    # Same-owner re-register stays zero-cost — count is still exactly 1.
    assert _PER_EXTENSION_COUNTS[("skill", "skill-123")] == 1

    # Register MAX-1 more tools for the SAME extension — the idempotent
    # re-registrations above must not have consumed any quota, so all fit.
    for i in range(MAX_TOOLS_PER_EXTENSION - 1):
        register_extension_tool(f"skill_demo_extra_{i}", "skill", "skill-123")
    # The very last one that would exceed the cap is dropped.
    register_extension_tool("skill_demo_overflow", "skill", "skill-123")
    assert resolve_extension("skill_demo_overflow") is None
    # But the ones within the cap all resolved.
    assert resolve_extension(f"skill_demo_extra_{MAX_TOOLS_PER_EXTENSION - 2}") == (
        "skill",
        "skill-123",
    )


# --------------------------------------------------------------------------- #
# P3 fix — same tool_name re-registers under a DIFFERENT owner
# --------------------------------------------------------------------------- #


def _count_for(kind: str, ext_id: str) -> int:
    """Peek the per-extension count for an owner (0 if absent). Test-only introspection."""
    from app.domain.services.tools.extension_attribution import _PER_EXTENSION_COUNTS

    return _PER_EXTENSION_COUNTS.get((kind, ext_id), 0)


def test_owner_change_moves_count_old_pruned_new_incremented():
    """Same tool_name re-registers under a new (kind, ext_id): old owner's count is
    decremented (pruned at 0), new owner is counted, resolve points at new owner."""
    register_extension_tool("skill_slug_run", "skill", "skill-old-id")
    assert resolve_extension("skill_slug_run") == ("skill", "skill-old-id")
    assert _count_for("skill", "skill-old-id") == 1
    assert _count_for("skill", "skill-new-id") == 0

    # Skill reinstalled: same slug-derived tool name, new skill.id.
    register_extension_tool("skill_slug_run", "skill", "skill-new-id")

    # resolve now points at the new owner.
    assert resolve_extension("skill_slug_run") == ("skill", "skill-new-id")
    # Old owner's count decremented to 0 → key pruned; new owner counted.
    assert _count_for("skill", "skill-old-id") == 0
    assert _count_for("skill", "skill-new-id") == 1


def test_owner_change_old_owner_keeps_remaining_tools():
    """Old owner had two tools; moving one to a new owner leaves the old count at 1."""
    register_extension_tool("skill_a_run", "skill", "skill-old-id")
    register_extension_tool("skill_b_run", "skill", "skill-old-id")
    assert _count_for("skill", "skill-old-id") == 2

    # Move only skill_a_run to a new owner.
    register_extension_tool("skill_a_run", "skill", "skill-new-id")

    assert resolve_extension("skill_a_run") == ("skill", "skill-new-id")
    assert resolve_extension("skill_b_run") == ("skill", "skill-old-id")
    assert _count_for("skill", "skill-old-id") == 1
    assert _count_for("skill", "skill-new-id") == 1


def test_owner_change_into_capped_new_owner_refused_and_entry_removed(
    caplog: pytest.LogCaptureFixture,
):
    """Re-binding an existing tool_name to a new owner that is AT its cap is refused;
    the stale registry entry is removed (no dangling mis-attribution) and the old
    owner's count was already decremented."""
    caplog.set_level(logging.WARNING)

    # Fill new owner to exactly its cap with distinct tools.
    for i in range(MAX_TOOLS_PER_EXTENSION):
        register_extension_tool(f"mcp_full_{i}", "mcp", "srv-full")
    assert _count_for("mcp", "srv-full") == MAX_TOOLS_PER_EXTENSION

    # A separate tool owned by a different extension.
    register_extension_tool("mcp_movable", "mcp", "srv-src")
    assert resolve_extension("mcp_movable") == ("mcp", "srv-src")
    assert _count_for("mcp", "srv-src") == 1

    # Try to re-bind mcp_movable to the already-capped srv-full → refused.
    register_extension_tool("mcp_movable", "mcp", "srv-full")

    # Registry entry removed (refuse → no dangling mis-attribution to old owner).
    assert resolve_extension("mcp_movable") is None
    # Old owner's count was decremented (and pruned at 0).
    assert _count_for("mcp", "srv-src") == 0
    # New owner did NOT gain a slot (still exactly at cap).
    assert _count_for("mcp", "srv-full") == MAX_TOOLS_PER_EXTENSION
    # A per-extension cap warning was emitted for the refused re-bind.
    assert any(
        "per-extension" in r.message or "扩展" in r.message for r in caplog.records
    )


def test_owner_change_clears_cap_warned_so_future_recap_rewarns(
    caplog: pytest.LogCaptureFixture,
):
    """When an owner sitting at its cap loses a tool (count drops), its cap-warned
    flag is cleared so a future re-cap warns again (rising-edge)."""
    from app.domain.services.tools.extension_attribution import _PER_EXTENSION_CAP_WARNED

    caplog.set_level(logging.WARNING)
    # Fill owner to cap, then trigger the cap once (257th tool dropped + warned).
    for i in range(MAX_TOOLS_PER_EXTENSION):
        register_extension_tool(f"mcp_cap_{i}", "mcp", "srv-cap")
    register_extension_tool("mcp_cap_overflow", "mcp", "srv-cap")
    assert ("mcp", "srv-cap") in _PER_EXTENSION_CAP_WARNED

    # Move one of srv-cap's tools to another owner → count drops below cap,
    # cap-warned flag for srv-cap is cleared.
    register_extension_tool("mcp_cap_0", "mcp", "srv-other")
    assert ("mcp", "srv-cap") not in _PER_EXTENSION_CAP_WARNED
    assert _count_for("mcp", "srv-cap") == MAX_TOOLS_PER_EXTENSION - 1


def test_per_extension_cap_256_drops_and_warns(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    for i in range(MAX_TOOLS_PER_EXTENSION):
        register_extension_tool(f"mcp_srv_tool_{i}", "mcp", "srv")
    # The 257th distinct tool for the same extension is dropped.
    register_extension_tool("mcp_srv_tool_overflow", "mcp", "srv")
    assert resolve_extension("mcp_srv_tool_overflow") is None
    assert resolve_extension("mcp_srv_tool_0") == ("mcp", "srv")

    # Warn exactly once per extension (not once per dropped tool).
    register_extension_tool("mcp_srv_tool_overflow2", "mcp", "srv")
    per_ext_warnings = [
        r for r in caplog.records if "per-extension" in r.message or "扩展工具数" in r.message
    ]
    assert len(per_ext_warnings) == 1


def test_registry_total_cap_4096(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    # Spread across many extensions so the per-extension cap never trips first;
    # each extension gets a small batch until the global registry cap is hit.
    per_ext = 16
    n_ext = MAX_REGISTRY_SIZE // per_ext  # 4096 / 16 = 256 extensions
    for e in range(n_ext):
        for t in range(per_ext):
            register_extension_tool(f"mcp_e{e}_t{t}", "mcp", f"ext-{e}")
    # Registry is now full at MAX_REGISTRY_SIZE entries. One more distinct tool
    # is dropped.
    register_extension_tool("mcp_overflow_tool", "mcp", "ext-new")
    assert resolve_extension("mcp_overflow_tool") is None
    # A tool registered before the cap still resolves.
    assert resolve_extension("mcp_e0_t0") == ("mcp", "ext-0")


def test_attribution_negative_no_name_parsing():
    """Attribution comes from explicit bindings, never from parsing the name.

    Anchor 1: an underscore-heavy server name registered via explicit binding
    resolves back to the WHOLE server name — a name-splitting implementation
    would mis-attribute ``my`` or ``my_multi`` instead of ``my_multi_under_score``.

    Anchor 2: a sha1-truncated skill tool name resolves back to the REAL skill
    id, which is nowhere in the tool name — impossible via name parsing.
    """
    register_extension_tool(
        "mcp_my_multi_under_score_toolx", "mcp", "my_multi_under_score"
    )
    assert resolve_extension("mcp_my_multi_under_score_toolx") == (
        "mcp",
        "my_multi_under_score",
    )

    # sha1-truncated skill tool name → real skill id (not derivable from name).
    register_extension_tool("skill_ab12cd34_run", "skill", "real-skill-uuid-9999")
    assert resolve_extension("skill_ab12cd34_run") == (
        "skill",
        "real-skill-uuid-9999",
    )


# --------------------------------------------------------------------------- #
# conftest autouse isolation (R3#7) — NO manual reset in this file
# --------------------------------------------------------------------------- #


def test_isolation_step_one_registers():
    register_extension_tool("mcp_isolation_probe", "mcp", "iso-srv")
    assert resolve_extension("mcp_isolation_probe") == ("mcp", "iso-srv")


def test_conftest_autouse_isolates():
    # If the autouse reset fixture is wired, the entry registered by the
    # previous test must be gone here (tests run in file order).
    assert resolve_extension("mcp_isolation_probe") is None


# --------------------------------------------------------------------------- #
# Registration-point wiring
# --------------------------------------------------------------------------- #


class _FakeMCPTool:
    """Minimal MCPTool stand-in: get_tools() schemas + tool_server_bindings()."""

    def __init__(self, schemas: list[dict[str, Any]], bindings: dict[str, str]):
        self._schemas = schemas
        self._bindings = bindings
        self.invoke = AsyncMock()

    def get_tools(self) -> list[dict[str, Any]]:
        return list(self._schemas)

    def tool_server_bindings(self) -> dict[str, str]:
        return dict(self._bindings)


def _mcp_schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "q"}},
                "required": ["query"],
            },
        },
    }


def test_mcp_langchain_registration_wires_bindings():
    from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools

    fake = _FakeMCPTool(
        schemas=[_mcp_schema("mcp_notion_notion_search")],
        bindings={"mcp_notion_notion_search": "notion"},
    )
    tools = create_mcp_langchain_tools(fake)
    assert [t.name for t in tools] == ["mcp_notion_notion_search"]
    assert resolve_extension("mcp_notion_notion_search") == ("mcp", "notion")


def test_mcp_registration_fail_open(caplog: pytest.LogCaptureFixture):
    """tool_server_bindings blowing up must NOT break tool creation (fail-open)."""
    caplog.set_level(logging.WARNING)

    fake = _FakeMCPTool(
        schemas=[_mcp_schema("mcp_x_tool")],
        bindings={},
    )

    def _boom() -> dict[str, str]:
        raise RuntimeError("bindings unavailable")

    fake.tool_server_bindings = _boom  # type: ignore[assignment]

    from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools

    tools = create_mcp_langchain_tools(fake)
    # Tool still created despite attribution failure.
    assert [t.name for t in tools] == ["mcp_x_tool"]
    # Attribution absent (fail-open) but logged.
    assert resolve_extension("mcp_x_tool") is None
    assert any("归因" in r.message or "attribution" in r.message.lower() for r in caplog.records)


def _openai_tool_schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"skill tool {name}",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string", "description": "x"}},
                "required": ["x"],
            },
        },
    }


def _make_skill_tool_mock(
    tools_schemas: list[dict[str, Any]], bindings: dict[str, dict[str, Any]]
) -> MagicMock:
    mock = MagicMock()
    mock.get_tools.return_value = tools_schemas
    mock.invoke = AsyncMock()
    mock._tool_bindings = bindings
    return mock


def test_dynamic_skill_registration():
    from app.domain.services.tools.langchain_dynamic_skill_tools import (
        create_dynamic_skill_langchain_tools,
    )

    skill_obj = MagicMock()
    skill_obj.id = "skill-uuid-42"
    mock = _make_skill_tool_mock(
        [_openai_tool_schema("skill_demo_hello")],
        bindings={"skill_demo_hello": {"skill": skill_obj}},
    )
    tools = create_dynamic_skill_langchain_tools(mock)
    assert [t.name for t in tools] == ["skill_demo_hello"]
    assert resolve_extension("skill_demo_hello") == ("skill", "skill-uuid-42")


def test_dynamic_skill_registration_no_skill_obj_is_skipped():
    """Binding without a ``skill`` object registers nothing (fail-open, no crash)."""
    from app.domain.services.tools.langchain_dynamic_skill_tools import (
        create_dynamic_skill_langchain_tools,
    )

    mock = _make_skill_tool_mock(
        [_openai_tool_schema("skill_no_binding")],
        bindings={},  # no entry for the tool
    )
    tools = create_dynamic_skill_langchain_tools(mock)
    assert [t.name for t in tools] == ["skill_no_binding"]
    assert resolve_extension("skill_no_binding") is None
