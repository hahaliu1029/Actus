"""Unit tests for the ToolSource resolver module."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from app.domain.services.tools.tool_source_resolver import (
    KNOWN_CATEGORIES,
    ToolSource,
    ToolSourceUnknownError,
)


class TestToolSourceModel:
    def test_tool_source_is_frozen(self):
        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        with pytest.raises(ValidationError):
            ts.source = "mcp"  # type: ignore[misc]

    def test_tool_source_can_be_embedded_in_pydantic_model(self):
        """Roundtrip check — R4 will embed ToolSource into ToolEvent (BaseModel)."""
        class Wrapper(BaseModel):
            tool_source: ToolSource

        ts = ToolSource(source="mcp", category="mcp", canonical_name="mcp_gh_issue")
        wrapper = Wrapper(tool_source=ts)
        dumped = wrapper.model_dump_json()
        restored = Wrapper.model_validate_json(dumped)
        assert restored.tool_source == ts

    def test_tool_source_equal_when_fields_equal(self):
        a = ToolSource(source="native", category="file", canonical_name="file_read")
        b = ToolSource(source="native", category="file", canonical_name="file_read")
        assert a == b

    def test_tool_source_post_init_warns_on_unknown_category(self, caplog):
        with caplog.at_level("WARNING"):
            ToolSource(source="native", category="made_up", canonical_name="x")
        assert "Unknown category" in caplog.text

    def test_tool_source_rejects_unknown_field(self):
        """extra='forbid' seals the contract surface so nested payloads
        (e.g. R2 ToolArtifact.tool_source) fail loudly on schema drift."""
        with pytest.raises(ValidationError):
            ToolSource.model_validate(
                {
                    "source": "native",
                    "category": "shell",
                    "canonical_name": "shell_execute",
                    "unexpected_nested": 1,
                }
            )

    def test_known_categories_has_13_values(self):
        """12 canonical categories + 1 ``unknown`` sentinel (R2 CS2)."""
        assert len(KNOWN_CATEGORIES) == 13
        # Spot-check canonical values from spec
        assert "shell" in KNOWN_CATEGORIES
        assert "skill creator" in KNOWN_CATEGORIES  # with space
        assert "mcp discovery" in KNOWN_CATEGORIES
        # R2 CS2: "unknown" is a sentinel category downstream emitters
        # construct for LLM-hallucinated tool names. The resolver itself
        # still NEVER returns it — see ToolSourceUnknownError docstring.
        assert "unknown" in KNOWN_CATEGORIES

    def test_unknown_category_does_not_warn(self, caplog):
        """Constructing ``ToolSource(category='unknown', ...)`` is a
        legitimate sentinel path (see ``react_graph`` unknown-tool
        branch). It must not emit an ``Unknown category`` warning each
        time an LLM hallucinates a tool name."""
        with caplog.at_level("WARNING"):
            ToolSource(
                source="native",
                category="unknown",
                canonical_name="hallucinated_ghost",
            )
        assert "Unknown category" not in caplog.text


class TestToolSourceUnknownError:
    def test_tool_source_unknown_error_exists(self):
        with pytest.raises(ToolSourceUnknownError):
            raise ToolSourceUnknownError("fake_tool")


from app.domain.services.tools.tool_source_resolver import (
    _CANONICAL_TOOL_IDENTITIES,
    _REGISTRY,
    _bootstrap_registry,
)


class TestBootstrap:
    def test_canonical_identities_has_38_entries(self):
        # 30 native (file 7 + shell 5 + browser 12 + message 2 + search 1 +
        # memory 3) + 2 a2a + 2 mcp discovery + 3 skill creator + 1 skill guide
        assert len(_CANONICAL_TOOL_IDENTITIES) == 38

    def test_canonical_identities_categories_subset_of_known(self):
        """Pin the invariant: every category in bootstrap is in KNOWN_CATEGORIES.

        Without this test, changing KNOWN_CATEGORIES while bootstrap drifts
        silently produces warnings (field_validator does not raise).
        """
        for name, (source, category) in _CANONICAL_TOOL_IDENTITIES.items():
            assert category in KNOWN_CATEGORIES, (
                f"bootstrap name {name!r} category {category!r} not in KNOWN_CATEGORIES"
            )

    def test_canonical_identities_never_use_unknown_sentinel(self):
        """R2 CS2: ``unknown`` is a downstream-emitter-only sentinel. No
        real tool is ever classified as ``unknown`` — if the bootstrap
        ever seeds one, that's a contract bug."""
        for name, (_, category) in _CANONICAL_TOOL_IDENTITIES.items():
            assert category != "unknown", (
                f"bootstrap name {name!r} was seeded with the unknown "
                f"sentinel — only react_graph's error emitters may use it"
            )

    def test_canonical_identities_covers_known_native_names(self):
        """Spot-check 11 native names that were missing in spec v3.0 (codex P1)."""
        missing_in_v3 = [
            "browser_input", "browser_move_mouse", "browser_press_key",
            "browser_select_option", "browser_restart",
            "shell_wait_process", "shell_write_input", "shell_kill_process",
            "file_find_in_content", "file_find_by_name", "file_list",
        ]
        for name in missing_in_v3:
            assert name in _CANONICAL_TOOL_IDENTITIES, f"{name} missing from bootstrap"
            assert _CANONICAL_TOOL_IDENTITIES[name][0] == "native"

    def test_bootstrap_populates_registry_on_import(self):
        """Import triggers _bootstrap_registry() which fills _REGISTRY."""
        # Bootstrap already ran at module import. Verify side effect.
        for name, (source, category) in _CANONICAL_TOOL_IDENTITIES.items():
            assert name in _REGISTRY, f"{name} missing from _REGISTRY after bootstrap"
            ts = _REGISTRY[name]
            assert ts.source == source
            assert ts.category == category
            assert ts.canonical_name == name


from langchain_core.tools import StructuredTool

from app.domain.services.tools.tool_source_resolver import (
    RegistryConflictError,
    annotate_and_register_tool_source,
)


def _make_fixture_tool(name: str) -> StructuredTool:
    async def _noop(**kwargs):
        return ""
    return StructuredTool.from_function(
        coroutine=_noop, name=name, description=f"fixture {name}",
    )


class TestAnnotateAndRegister:
    def test_writes_metadata_and_registry(self):
        tool = _make_fixture_tool("test_tool_xyz")
        annotate_and_register_tool_source(tool, source="native", category="shell")
        assert tool.metadata is not None
        ts = tool.metadata["_actus_source"]
        assert ts.source == "native"
        assert ts.category == "shell"
        assert ts.canonical_name == "test_tool_xyz"
        assert _REGISTRY["test_tool_xyz"] == ts

    def test_returns_tool_for_chaining(self):
        tool = _make_fixture_tool("test_chain_xyz")
        result = annotate_and_register_tool_source(tool, source="native", category="shell")
        assert result is tool

    def test_same_value_idempotent(self):
        tool1 = _make_fixture_tool("test_idem_xyz")
        annotate_and_register_tool_source(tool1, source="native", category="shell")
        tool2 = _make_fixture_tool("test_idem_xyz")
        annotate_and_register_tool_source(tool2, source="native", category="shell")
        # Second call is a no-op, no raise

    def test_bootstrap_name_idempotent(self):
        """Re-registering a bootstrap-covered name with matching value is a no-op."""
        tool = _make_fixture_tool("shell_execute")
        annotate_and_register_tool_source(tool, source="native", category="shell")
        # No raise; _REGISTRY unchanged

    def test_conflict_raises(self):
        tool1 = _make_fixture_tool("test_conflict_xyz")
        annotate_and_register_tool_source(tool1, source="native", category="shell")
        tool2 = _make_fixture_tool("test_conflict_xyz")
        with pytest.raises(RegistryConflictError) as exc_info:
            annotate_and_register_tool_source(tool2, source="native", category="file")
        assert "test_conflict_xyz" in str(exc_info.value)

    def test_drift_from_bootstrap_raises(self):
        """Trying to register shell_execute as (skill, skill) raises."""
        tool = _make_fixture_tool("shell_execute")
        with pytest.raises(RegistryConflictError):
            annotate_and_register_tool_source(tool, source="skill", category="skill")

    def test_conflict_does_not_pollute_metadata_or_registry(self):
        """On RegistryConflictError, neither the conflicting tool's metadata
        nor the _REGISTRY entry is mutated. Prevents downstream
        resolve_tool_source_from_tool from returning the rejected value."""
        tool1 = _make_fixture_tool("test_nopollute_xyz")
        annotate_and_register_tool_source(tool1, source="native", category="shell")
        original_ts_in_registry = _REGISTRY["test_nopollute_xyz"]

        tool2 = _make_fixture_tool("test_nopollute_xyz")
        # Force fresh metadata so we can detect any illegal write.
        tool2.metadata = None
        with pytest.raises(RegistryConflictError):
            annotate_and_register_tool_source(tool2, source="native", category="file")

        # tool2's metadata must not hold a rejected ToolSource
        assert tool2.metadata is None or "_actus_source" not in tool2.metadata
        # Registry still holds the original value
        assert _REGISTRY["test_nopollute_xyz"] == original_ts_in_registry
        assert _REGISTRY["test_nopollute_xyz"].category == "shell"


from app.domain.services.tools.tool_source_resolver import (
    resolve_tool_source,
    resolve_tool_source_from_tool,
)


class TestResolveBootstrapHit:
    """Names covered by bootstrap resolve without heuristic."""

    def test_browser_navigate(self):
        ts = resolve_tool_source("browser_navigate")
        assert ts.source == "native"
        assert ts.category == "browser"
        assert ts.canonical_name == "browser_navigate"

    def test_all_canonical_names_resolvable(self):
        for name, (source, category) in _CANONICAL_TOOL_IDENTITIES.items():
            ts = resolve_tool_source(name)
            assert ts.source == source
            assert ts.category == category
            assert ts.canonical_name == name

    def test_file_find_in_content_resolvable(self):
        """Regression guard: 11 names the spec v3 missed are all in bootstrap."""
        ts = resolve_tool_source("file_find_in_content")
        assert ts == ToolSource(
            source="native", category="file", canonical_name="file_find_in_content",
        )

    def test_identity_is_availability_independent(self):
        """Resolver does not care whether tool is in bound_tool_names."""
        ts = resolve_tool_source("browser_navigate")
        assert ts is not None
        # No bound_tool_names argument, no availability check


class TestResolveFactoryRegisterHit:
    def test_dynamic_mcp_after_register(self):
        tool = _make_fixture_tool("mcp_github_create_issue")
        annotate_and_register_tool_source(tool, source="mcp", category="mcp")
        ts = resolve_tool_source("mcp_github_create_issue")
        assert ts.source == "mcp"
        assert ts.category == "mcp"


class TestResolveHeuristic:
    """Dynamic names not yet registered fall through to heuristic."""

    def test_dynamic_mcp_single_underscore(self):
        """Regression test: pre-R1 startswith('mcp__') never matched; heuristic must match mcp_ single underscore."""
        # Use a name NOT yet registered to force heuristic
        ts = resolve_tool_source("mcp_notregistered_xxx")
        assert ts.source == "mcp"
        assert ts.category == "mcp"
        assert ts.canonical_name == "mcp_notregistered_xxx"

    def test_dynamic_skill_prefix(self):
        ts = resolve_tool_source("skill_notregistered_bar")
        assert ts.source == "skill"
        assert ts.category == "skill"


class TestResolveRaise:
    """Fail-closed: anything not in bootstrap + not matching heuristic raises."""

    def test_fake_native_prefix_raises(self):
        """No native prefix fallback — should be in bootstrap or it raises."""
        with pytest.raises(ToolSourceUnknownError):
            resolve_tool_source("browser_fakefake_xyz")

    def test_fake_shell_prefix_raises(self):
        with pytest.raises(ToolSourceUnknownError):
            resolve_tool_source("shell_fakefake_xyz")

    def test_completely_unknown_raises(self):
        with pytest.raises(ToolSourceUnknownError):
            resolve_tool_source("zzz_definitely_not_a_tool")


class TestResolveFromTool:
    def test_reads_metadata(self):
        tool = _make_fixture_tool("test_fromtool_xyz")
        annotate_and_register_tool_source(tool, source="native", category="shell")
        ts = resolve_tool_source_from_tool(tool)
        assert ts.category == "shell"

    def test_missing_metadata_raises(self):
        tool = _make_fixture_tool("test_fromtool_bare")
        # No annotate_and_register call; metadata never set
        with pytest.raises(ToolSourceUnknownError):
            resolve_tool_source_from_tool(tool)


class TestFixtureCtxResolvable:
    def test_fixture_ctx_bound_tool_names_all_resolve(self):
        """Guard: _FIXTURE_CTX.bound_tool_names must all resolve without raise.

        This is the direct guard for 'AgentTaskRunner.__init__ triggers
        SectionRegistry validation which calls resolve on every _FIXTURE_CTX
        name' — the startup fail-fast contract.
        """
        from app.domain.services.prompts.section import _FIXTURE_CTX

        for name in _FIXTURE_CTX.bound_tool_names:
            try:
                ts = resolve_tool_source(name)
                assert ts is not None
            except ToolSourceUnknownError as e:
                pytest.fail(f"_FIXTURE_CTX name {name!r} failed to resolve: {e}")

