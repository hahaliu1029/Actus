"""PE-2 §7: MCP server-qualified identity — cross-server same-name tools must
not share a permission key (key = call.tool_name). Names are server-prefixed
in production (mcp.py:337-341); this locks that against regression."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.domain.models.app_config import MCPConfig, MCPServerConfig
from app.domain.services.tools.mcp import (
    MCPClientManager,
    _mcp_namespaces_conflict,
    _mcp_tool_namespace,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _server(enabled: bool = True) -> MCPServerConfig:
    """Minimal valid MCPServerConfig: default transport is STREAMABLE_HTTP,
    whose model_validator requires a url. enabled defaults True."""
    return MCPServerConfig(url="http://localhost", enabled=enabled)


async def test_same_bare_name_across_servers_yields_distinct_tool_names():
    mgr = MCPClientManager(mcp_config=None)
    # MCP SDK Tool objects expose .name / .description / .inputSchema — stub them.
    tool_a = SimpleNamespace(name="search", description="A",
                             inputSchema={"type": "object", "properties": {}})
    tool_b = SimpleNamespace(name="search", description="B",
                             inputSchema={"type": "object", "properties": {}})
    mgr._tools = {"srvA": [tool_a], "srvB": [tool_b]}

    schemas = await mgr.get_all_tools()
    names = {s["function"]["name"] for s in schemas}

    assert names == {"mcp_srvA_search", "mcp_srvB_search"}, (
        "cross-server same bare-name MCP tools must get distinct server-qualified "
        f"names (permission key = tool_name); got {names}"
    )


def test_mcp_tool_namespace_helper():
    # A bare name gets the mcp_ prefix; a name already mcp_-prefixed is its own
    # namespace — so 'foo' and 'mcp_foo' collapse to the SAME namespace.
    assert _mcp_tool_namespace("foo") == "mcp_foo"
    assert _mcp_tool_namespace("mcp_foo") == "mcp_foo"
    assert _mcp_tool_namespace("srvA") == "mcp_srvA"


def test_prefix_collision_across_servers_rejected_at_load():
    # 'foo' -> namespace 'mcp_foo'; 'mcp_foo' -> namespace 'mcp_foo' => collision.
    config = MCPConfig(
        mcpServers={"foo": _server(), "mcp_foo": _server()}
    )
    mgr = MCPClientManager(mcp_config=config)

    with pytest.raises(ValueError) as excinfo:
        mgr._validate_no_tool_namespace_collisions()

    msg = str(excinfo.value)
    assert "foo" in msg and "mcp_foo" in msg, (
        f"error must name both colliding servers + shared namespace; got: {msg}"
    )


def test_distinct_servers_no_collision():
    config = MCPConfig(
        mcpServers={"srvA": _server(), "srvB": _server()}
    )
    mgr = MCPClientManager(mcp_config=config)

    # Distinct namespaces (mcp_srvA / mcp_srvB) → must NOT raise.
    mgr._validate_no_tool_namespace_collisions()


def test_collision_ignored_when_one_disabled():
    # Backward-safe: a currently-working config where only one of the colliding
    # pair is enabled must keep loading unchanged.
    config = MCPConfig(
        mcpServers={"foo": _server(enabled=True), "mcp_foo": _server(enabled=False)}
    )
    mgr = MCPClientManager(mcp_config=config)

    mgr._validate_no_tool_namespace_collisions()


def test_collision_validation_skipped_when_config_none():
    # No config → nothing to validate, must not raise.
    mgr = MCPClientManager(mcp_config=None)
    mgr._validate_no_tool_namespace_collisions()


def test_mcp_namespaces_conflict_truth_table():
    # Equality conflicts.
    assert _mcp_namespaces_conflict("mcp_foo", "mcp_foo") is True
    # '_'-boundary prefix overlap conflicts (both directions).
    assert _mcp_namespaces_conflict("mcp_foo", "mcp_foo_bar") is True
    assert _mcp_namespaces_conflict("mcp_foo_bar", "mcp_foo") is True
    # Substring WITHOUT a '_' boundary must NOT conflict (precision).
    assert _mcp_namespaces_conflict("mcp_foo", "mcp_foobar") is False
    assert _mcp_namespaces_conflict("mcp_foobar", "mcp_foo") is False
    # Sibling extensions sharing a stem but neither a prefix of the other.
    assert (
        _mcp_namespaces_conflict("mcp_github_cloud", "mcp_github_enterprise") is False
    )
    # Wholly distinct.
    assert _mcp_namespaces_conflict("mcp_srvA", "mcp_srvB") is False


def test_prefix_overlap_collision_rejected():
    # 'foo' -> ns 'mcp_foo'; 'foo_bar' -> ns 'mcp_foo_bar'; distinct strings but
    # invoke() would match 'mcp_foo_bar_*'.startswith('mcp_foo_') -> mis-dispatch.
    config = MCPConfig(
        mcpServers={"foo": _server(), "foo_bar": _server()}
    )
    mgr = MCPClientManager(mcp_config=config)

    with pytest.raises(ValueError) as excinfo:
        mgr._validate_no_tool_namespace_collisions()

    msg = str(excinfo.value)
    assert "foo" in msg and "foo_bar" in msg, (
        f"error must name both colliding servers; got: {msg}"
    )


def test_mcp_prefixed_overlap_rejected():
    # Already-prefixed names: 'mcp_foo' vs 'mcp_foo_bar' -> same overlap class.
    config = MCPConfig(
        mcpServers={"mcp_foo": _server(), "mcp_foo_bar": _server()}
    )
    mgr = MCPClientManager(mcp_config=config)

    with pytest.raises(ValueError) as excinfo:
        mgr._validate_no_tool_namespace_collisions()

    msg = str(excinfo.value)
    assert "mcp_foo" in msg and "mcp_foo_bar" in msg, (
        f"error must name both colliding servers; got: {msg}"
    )


def test_substring_without_underscore_boundary_allowed():
    # 'foo' (ns mcp_foo) + 'foobar' (ns mcp_foobar): NOT a '_'-boundary prefix
    # (char after 'mcp_foo' is 'b', not '_') -> must be ALLOWED.
    config = MCPConfig(
        mcpServers={"foo": _server(), "foobar": _server()}
    )
    mgr = MCPClientManager(mcp_config=config)

    mgr._validate_no_tool_namespace_collisions()


def test_sibling_extension_names_allowed():
    # 'github_cloud' + 'github_enterprise': neither is a '_'-boundary prefix of
    # the other -> must be ALLOWED.
    config = MCPConfig(
        mcpServers={"github_cloud": _server(), "github_enterprise": _server()}
    )
    mgr = MCPClientManager(mcp_config=config)

    mgr._validate_no_tool_namespace_collisions()


def test_overlap_ignored_when_one_disabled():
    # Backward-safe: prefix-overlap pair where one is disabled must keep loading.
    config = MCPConfig(
        mcpServers={
            "foo": _server(enabled=True),
            "foo_bar": _server(enabled=False),
        }
    )
    mgr = MCPClientManager(mcp_config=config)

    mgr._validate_no_tool_namespace_collisions()


async def test_initialize_invokes_collision_validation_before_connect():
    """Regression guard for the R4/R5 fix WIRING: MCPClientManager.initialize()
    must call the namespace-collision validation (before connecting), so a
    colliding config fails fast at load. Locks the callsite against silent
    removal (dropping it would re-open the tool-name/permission-key collision)."""
    cfg = MCPConfig(mcpServers={
        "foo": _server(enabled=True),
        "mcp_foo": _server(enabled=True),
    })
    mgr = MCPClientManager(mcp_config=cfg)
    with pytest.raises(ValueError) as exc:
        await mgr.initialize()
    # distinctive validator message (NOT a connection error) proves the
    # validation ran inside initialize() before any connect attempt.
    assert "namespace" in str(exc.value).lower()
