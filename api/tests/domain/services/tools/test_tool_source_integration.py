"""R1 CS1 integration tests — factory contract + regression gates.

This gate exercises THREE factory groups:

1. Native factories (langchain_tools.py + memory_tools.py) — every returned
   tool must be in _CANONICAL_TOOL_IDENTITIES with source="native".

2. Static extension factories (langchain_a2a, langchain_mcp_discovery,
   langchain_skill_tools creator, langchain_skill_tools guide) — every
   returned tool must be in _CANONICAL_TOOL_IDENTITIES with non-native source.

3. Dynamic extension factories (langchain_mcp, langchain_dynamic_skill_tools)
   — produce runtime-named wrappers. Gate only verifies metadata is set with
   the correct source/category; name is NOT expected in _CANONICAL_TOOL_IDENTITIES.

Calling all three groups from Task 14 means classifier replacement tasks
(15-19) cannot start until every factory in the codebase has been fixed in
Tasks 7-13. The gate is the hard precondition for the classifier cutover —
if ANY factory's annotate call is missing or wrong, this test fails before
react_graph / render_context / risk_assessor are touched.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.tools.tool_source_resolver import (
    _CANONICAL_TOOL_IDENTITIES,
    KNOWN_CATEGORIES,
    ToolSource,
    resolve_tool_source,
    resolve_tool_source_from_tool,
)


# ---- Mock helpers ------------------------------------------------------ #


def _mock_sandbox():
    sandbox = MagicMock()
    sandbox.execute = AsyncMock(return_value="")
    return sandbox


def _mock_browser():
    browser = MagicMock()
    browser.navigate = AsyncMock(return_value="")
    return browser


def _mock_search_engine():
    se = MagicMock()
    se.search = AsyncMock(return_value=[])
    return se


# ---- Group 1: Native factories ----------------------------------------- #


def _enumerate_native_tools():
    """Native factories: langchain_tools 6 sub-factories + memory_tools.

    All names produced must be native + in _CANONICAL_TOOL_IDENTITIES.
    """
    from app.domain.services.tools import langchain_tools
    from app.domain.services.tools.memory_tools import create_memory_tools

    sandbox = _mock_sandbox()
    browser = _mock_browser()
    search_engine = _mock_search_engine()

    tools = []
    tools.extend(langchain_tools._make_message_tools())
    tools.extend(langchain_tools._make_file_tools(EagerSandboxAccessor(sandbox)))
    tools.extend(langchain_tools._make_shell_tools(EagerSandboxAccessor(sandbox)))
    tools.extend(langchain_tools._make_browser_tools(EagerBrowserAccessor(browser)))
    tools.extend(langchain_tools._make_search_tools(search_engine))
    tools.extend(
        langchain_tools._make_file_view_tools(
            EagerSandboxAccessor(sandbox),
            file_processor_lookup=MagicMock(),
            supports_vision=True,
            supports_pdf_input=True,
        )
    )
    # memory_tools real signature — 4 required positional args + 2 defaults.
    # Factory does not invoke the deps at construction time, so MagicMock is fine.
    tools.extend(
        create_memory_tools(
            embedding_provider=MagicMock(),
            session_factory=MagicMock(),
            repo_factory=MagicMock(),
            user_id="fixture-user",
        )
    )
    return tools


# ---- Group 2: Static extension factories ------------------------------- #


def _enumerate_static_extension_tools():
    """Static extension factories: a2a, mcp_discovery, skill creator, skill guide.

    Every name produced must be in _CANONICAL_TOOL_IDENTITIES with
    source != 'native'. These factories take configuration deps but the
    tool names they produce are hard-coded, so the set is deterministic
    regardless of what's in the mocked deps.
    """
    from app.domain.services.tools.langchain_a2a import create_a2a_langchain_tools
    from app.domain.services.tools.langchain_mcp_discovery import (
        create_mcp_discovery_tools,
    )
    from app.domain.services.tools.langchain_skill_tools import (
        create_skill_guide_tool,
        create_skill_langchain_tools,
    )

    tools = []

    # A2A (2 tools): get_remote_agent_cards, call_remote_agent
    a2a_tool = MagicMock()
    tools.extend(create_a2a_langchain_tools(a2a_tool))

    # MCP discovery (2 tools): list_mcp_tools, get_mcp_tool
    mcp_tool_mock = MagicMock()
    mcp_tool_mock.get_tools = MagicMock(return_value=[])
    tools.extend(
        create_mcp_discovery_tools(
            mcp_tool_ref=lambda: mcp_tool_mock,
            activated_tools_ref=lambda: set(),
        )
    )

    # Skill creator (3 tools when BOTH deps non-None): brainstorm_skill,
    # generate_skill, install_skill. Pass MagicMock for both BaseTool args
    # so the factory enters both branches.
    brainstorm_dep = MagicMock()
    creator_dep = MagicMock()
    tools.extend(
        create_skill_langchain_tools(
            brainstorm_skill_tool=brainstorm_dep,
            create_skill_tool=creator_dep,
        )
    )

    # Skill guide (1 tool): get_skill_guide. create_skill_guide_tool returns
    # a SINGLE StructuredTool, not a list — wrap it explicitly.
    guide_tool = create_skill_guide_tool(
        skill_pool_ref=lambda: [],
        file_listings_ref=None,
    )
    tools.append(guide_tool)

    return tools


# ---- Group 3: Dynamic extension factories ------------------------------ #


def _enumerate_dynamic_extension_tools():
    """Dynamic extension factories: mcp wrapper + dynamic skill wrapper.

    Both factories iterate schemas returned by their outer service. We feed
    each with ONE fake schema so the gate can verify the factory annotates
    whatever it constructs. The name is NOT expected in canonical identities
    (dynamic names are runtime-only).
    """
    from app.domain.services.tools.langchain_dynamic_skill_tools import (
        create_dynamic_skill_langchain_tools,
    )
    from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools

    tools = []

    # MCP wrapper with 1 fake schema
    mcp_tool_mock = MagicMock()
    mcp_tool_mock.get_tools = MagicMock(
        return_value=[
            {
                "function": {
                    "name": "mcp_fake_server_fake_tool",
                    "description": "fake test tool for gate",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }
            }
        ]
    )
    tools.extend(create_mcp_langchain_tools(mcp_tool_mock))

    # Dynamic skill wrapper with 1 fake schema
    skill_tool_mock = MagicMock()
    skill_tool_mock.get_tools = MagicMock(
        return_value=[
            {
                "function": {
                    "name": "skill_fakeslug_fakefn",
                    "description": "fake dynamic skill tool for gate",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }
            }
        ]
    )
    tools.extend(create_dynamic_skill_langchain_tools(skill_tool_mock))

    return tools


# ---- Gate tests -------------------------------------------------------- #


class TestNativeFactoryAnnotation:
    """Group 1 gate — every native factory must annotate AND register in _REGISTRY."""

    def test_all_native_tools_annotated_with_metadata(self):
        tools = _enumerate_native_tools()
        assert len(tools) > 0
        for tool in tools:
            assert tool.metadata is not None, f"{tool.name} missing metadata dict"
            ts = tool.metadata.get("_actus_source")
            assert isinstance(ts, ToolSource), (
                f"{tool.name} metadata _actus_source is {ts!r}, expected ToolSource"
            )
            assert ts.source == "native"
            assert ts.canonical_name == tool.name

    def test_all_native_tools_registry_consistent_with_metadata(self):
        """Hard contract: factory must go through annotate_and_register_tool_source,
        which writes BOTH metadata AND _REGISTRY. A factory that sneaks in a
        direct `tool.metadata = {...}` write without registry update would pass
        the metadata-only test above. This test closes that loophole.
        """
        tools = _enumerate_native_tools()
        for tool in tools:
            metadata_ts = tool.metadata["_actus_source"]
            registry_ts = resolve_tool_source(tool.name)
            assert registry_ts == metadata_ts, (
                f"{tool.name}: metadata={metadata_ts!r} but "
                f"resolve_tool_source={registry_ts!r}. Factory did not go "
                f"through annotate_and_register_tool_source helper."
            )

    def test_all_native_tools_in_canonical_identities(self):
        tools = _enumerate_native_tools()
        native_names = {t.name for t in tools}
        canonical_native_names = {
            name for name, (source, _) in _CANONICAL_TOOL_IDENTITIES.items()
            if source == "native"
        }
        missing = native_names - canonical_native_names
        assert not missing, (
            f"Native tools not in _CANONICAL_TOOL_IDENTITIES: {missing}. "
            f"Add them to tool_source_resolver.py:_CANONICAL_TOOL_IDENTITIES."
        )


class TestStaticExtensionFactoryAnnotation:
    """Group 2 gate — every static extension factory must annotate, register
    in _REGISTRY, AND every name must be in _CANONICAL_TOOL_IDENTITIES."""

    def test_all_static_extension_tools_annotated(self):
        tools = _enumerate_static_extension_tools()
        assert len(tools) > 0
        for tool in tools:
            ts = tool.metadata.get("_actus_source")
            assert isinstance(ts, ToolSource), (
                f"{tool.name} missing metadata — {tool.metadata!r}"
            )
            assert ts.canonical_name == tool.name
            assert ts.source != "native"

    def test_all_static_extension_registry_consistent_with_metadata(self):
        """Static extension factory must call the single helper. This closes
        the 'metadata only, no registry' loophole for static extensions.
        """
        tools = _enumerate_static_extension_tools()
        for tool in tools:
            metadata_ts = tool.metadata["_actus_source"]
            registry_ts = resolve_tool_source(tool.name)
            assert registry_ts == metadata_ts, (
                f"{tool.name}: metadata={metadata_ts!r} but "
                f"resolve_tool_source={registry_ts!r}. Factory did not go "
                f"through annotate_and_register_tool_source helper."
            )

    def test_all_static_extension_tools_in_canonical_identities(self):
        tools = _enumerate_static_extension_tools()
        for tool in tools:
            assert tool.name in _CANONICAL_TOOL_IDENTITIES, (
                f"Static extension tool {tool.name!r} is not in "
                f"_CANONICAL_TOOL_IDENTITIES. Either the bootstrap seed is "
                f"missing this name, or the factory produced a name the "
                f"spec did not anticipate."
            )
            canonical_source, canonical_category = _CANONICAL_TOOL_IDENTITIES[tool.name]
            ts = tool.metadata["_actus_source"]
            assert ts.source == canonical_source, (
                f"{tool.name} factory annotated source={ts.source!r} but "
                f"bootstrap says {canonical_source!r}"
            )
            assert ts.category == canonical_category, (
                f"{tool.name} factory annotated category={ts.category!r} but "
                f"bootstrap says {canonical_category!r}"
            )

    def test_static_extension_covers_expected_names(self):
        """Sanity: the 8 hardcoded extension names show up."""
        tools = _enumerate_static_extension_tools()
        names = {t.name for t in tools}
        expected = {
            # A2A
            "get_remote_agent_cards", "call_remote_agent",
            # MCP discovery
            "list_mcp_tools", "get_mcp_tool",
            # Skill creator
            "brainstorm_skill", "generate_skill", "install_skill",
            # Skill guide
            "get_skill_guide",
        }
        missing = expected - names
        assert not missing, (
            f"Static extension factories did not produce {missing}. "
            f"Check the corresponding factory and its required deps."
        )


class TestDynamicExtensionFactoryAnnotation:
    """Group 3 gate — dynamic factories must annotate whatever they construct,
    register in _REGISTRY, with correct source/category. Name is dynamic,
    not in canonical."""

    def test_mcp_dynamic_wrapper_annotated_as_mcp(self):
        tools = _enumerate_dynamic_extension_tools()
        mcp_tools = [t for t in tools if t.name == "mcp_fake_server_fake_tool"]
        assert len(mcp_tools) == 1, f"MCP dynamic factory did not produce fake tool"
        ts = mcp_tools[0].metadata["_actus_source"]
        assert ts.source == "mcp"
        assert ts.category == "mcp"
        # Dynamic names are NOT in canonical — confirm explicitly
        assert "mcp_fake_server_fake_tool" not in _CANONICAL_TOOL_IDENTITIES

    def test_mcp_dynamic_wrapper_registry_consistent(self):
        """After factory runs, _REGISTRY must contain the dynamic name mapped
        to the same ToolSource as metadata. Without this assertion, a factory
        that only writes metadata but skips _REGISTRY would resolve to a stale
        value via heuristic (which also returns mcp/mcp for mcp_* prefix) and
        silently pass the metadata-only test.
        """
        tools = _enumerate_dynamic_extension_tools()
        mcp_tool = next(t for t in tools if t.name == "mcp_fake_server_fake_tool")
        metadata_ts = mcp_tool.metadata["_actus_source"]
        registry_ts = resolve_tool_source("mcp_fake_server_fake_tool")
        assert registry_ts == metadata_ts
        # Also verify the name actually landed in _REGISTRY (not just routed
        # through heuristic — heuristic returns a fresh ToolSource every call
        # so equality is value-based, not identity-based)
        from app.domain.services.tools.tool_source_resolver import _REGISTRY
        assert "mcp_fake_server_fake_tool" in _REGISTRY, (
            "Factory did not register dynamic name in _REGISTRY. "
            "The metadata-only pass would silently rely on heuristic."
        )

    def test_dynamic_skill_wrapper_annotated_as_skill(self):
        tools = _enumerate_dynamic_extension_tools()
        skill_tools = [t for t in tools if t.name == "skill_fakeslug_fakefn"]
        assert len(skill_tools) == 1, (
            "Dynamic skill factory did not produce fake tool"
        )
        ts = skill_tools[0].metadata["_actus_source"]
        assert ts.source == "skill"
        assert ts.category == "skill"
        assert "skill_fakeslug_fakefn" not in _CANONICAL_TOOL_IDENTITIES

    def test_dynamic_skill_wrapper_registry_consistent(self):
        """Same 'no registry bypass' guard for dynamic skill factory."""
        tools = _enumerate_dynamic_extension_tools()
        skill_tool = next(t for t in tools if t.name == "skill_fakeslug_fakefn")
        metadata_ts = skill_tool.metadata["_actus_source"]
        registry_ts = resolve_tool_source("skill_fakeslug_fakefn")
        assert registry_ts == metadata_ts
        from app.domain.services.tools.tool_source_resolver import _REGISTRY
        assert "skill_fakeslug_fakefn" in _REGISTRY


class TestAllFactoryCategoriesSubsetOfKnown:
    """Spec §Concurrency + KNOWN_CATEGORIES invariant."""

    def test_all_native_categories_in_known(self):
        for tool in _enumerate_native_tools():
            ts = tool.metadata["_actus_source"]
            assert ts.category in KNOWN_CATEGORIES

    def test_all_static_extension_categories_in_known(self):
        for tool in _enumerate_static_extension_tools():
            ts = tool.metadata["_actus_source"]
            assert ts.category in KNOWN_CATEGORIES

    def test_all_dynamic_extension_categories_in_known(self):
        for tool in _enumerate_dynamic_extension_tools():
            ts = tool.metadata["_actus_source"]
            assert ts.category in KNOWN_CATEGORIES


from app.domain.services.risk_assessor import RiskAssessor, RiskLevel


class TestRiskAssessorRegression:
    def test_mcp_wrapper_single_underscore_becomes_medium(self):
        """mcp_github_create_issue (single underscore wrapper) reaches MEDIUM."""
        from app.domain.services.tools.tool_source_resolver import (
            annotate_and_register_tool_source,
        )
        from langchain_core.tools import StructuredTool

        async def _noop(**kwargs):
            return ""

        tool = StructuredTool.from_function(
            coroutine=_noop, name="mcp_github_create_issue", description="fx",
        )
        annotate_and_register_tool_source(tool, source="mcp", category="mcp")

        assessment = RiskAssessor().assess("mcp_github_create_issue", {})
        assert assessment.static_level >= RiskLevel.MEDIUM

    def test_mcp_discovery_stays_none(self):
        """list_mcp_tools / get_mcp_tool (discovery meta-tools) stay at NONE.

        Regression guard against using source == 'mcp' instead of
        category == 'mcp' — discovery tools would then incorrectly escalate.
        """
        for name in ("list_mcp_tools", "get_mcp_tool"):
            assessment = RiskAssessor().assess(name, {})
            assert assessment.static_level == RiskLevel.NONE, (
                f"{name} should stay NONE (identity-only discovery meta-tool)"
            )


class TestRenderContextBooleanFlags:
    def test_mcp_active_false_for_discovery_only_step(self):
        """Pre-R1 regression guard: mcp_active must be False when only discovery
        meta-tools are bound (list_mcp_tools / get_mcp_tool). If implementer
        used source == 'mcp' instead of category == 'mcp', this test fails.
        """
        from app.domain.services.prompts.render_context import _categorize_tools

        bound = frozenset({"list_mcp_tools", "get_mcp_tool"})
        cats = _categorize_tools(bound)
        assert "mcp discovery" in cats
        assert "mcp" not in cats  # pure mcp wrapper category absent

    def test_mcp_active_true_with_real_wrapper(self):
        from app.domain.services.tools.tool_source_resolver import (
            annotate_and_register_tool_source,
        )
        from langchain_core.tools import StructuredTool

        async def _noop(**kwargs):
            return ""

        tool = StructuredTool.from_function(
            coroutine=_noop, name="mcp_foo_bar", description="fx",
        )
        annotate_and_register_tool_source(tool, source="mcp", category="mcp")

        from app.domain.services.prompts.render_context import _categorize_tools
        bound = frozenset({"mcp_foo_bar", "list_mcp_tools"})
        cats = _categorize_tools(bound)
        assert "mcp" in cats
        assert "mcp discovery" in cats


class TestDynamicSkillLogFilter:
    def test_dynamic_skill_filter_excludes_creator_and_guide(self):
        """Pre-R1 startswith('skill_') only matched dynamic skill_{slug}_{tool}
        wrappers; creator / guide names (brainstorm_skill / get_skill_guide)
        didn't start with 'skill_'. Post-R1, category == 'skill' preserves that
        exclusion."""
        from app.domain.services.tools.tool_source_resolver import (
            annotate_and_register_tool_source,
            resolve_tool_source_from_tool,
        )
        from langchain_core.tools import StructuredTool

        async def _noop(**kwargs):
            return ""

        dynamic = StructuredTool.from_function(
            coroutine=_noop, name="skill_foo_bar", description="fx",
        )
        annotate_and_register_tool_source(dynamic, source="skill", category="skill")

        brainstorm = StructuredTool.from_function(
            coroutine=_noop, name="brainstorm_skill", description="fx",
        )
        annotate_and_register_tool_source(brainstorm, source="skill", category="skill creator")

        guide = StructuredTool.from_function(
            coroutine=_noop, name="get_skill_guide", description="fx",
        )
        annotate_and_register_tool_source(guide, source="skill", category="skill guide")

        lc_tools = [dynamic, brainstorm, guide]
        # Replicates agent_task_runner:1783 logic
        filtered = [
            t.name for t in lc_tools
            if resolve_tool_source_from_tool(t).category == "skill"
        ]
        assert filtered == ["skill_foo_bar"]


class TestReactGraphToolEventCategory:
    """R1 intentional behavior changes for ToolEvent.tool_name:
    - Native tools: keep existing canonical category (stable, no regression)
    - Extension tools: transition from raw function name to canonical category
    - Memory: transition from raw function name to "memory"
    """

    def test_native_category_stable(self):
        """Native tools pre-R1 already had canonical categories via prefix match."""
        for name, expected_cat in [
            ("shell_execute", "shell"),
            ("file_read", "file"),
            ("browser_navigate", "browser"),
            ("search_web", "search"),
            ("message_notify_user", "message"),
        ]:
            assert resolve_tool_source(name).category == expected_cat

    def test_extension_category_normalized(self):
        """Extension tools pre-R1 passed function name through; R1 normalizes
        to canonical category."""
        for name, expected_cat in [
            ("list_mcp_tools", "mcp discovery"),
            ("get_mcp_tool", "mcp discovery"),
            ("get_skill_guide", "skill guide"),
            ("brainstorm_skill", "skill creator"),
            ("generate_skill", "skill creator"),
            ("install_skill", "skill creator"),
            ("call_remote_agent", "a2a"),
            ("get_remote_agent_cards", "a2a"),
        ]:
            assert resolve_tool_source(name).category == expected_cat, (
                f"{name}: expected {expected_cat!r}"
            )

    def test_memory_category_normalized(self):
        """Memory tools pre-R1 had no prefix match in react_graph; R1 assigns
        canonical 'memory' category."""
        assert resolve_tool_source("memory_search").category == "memory"
        assert resolve_tool_source("memory_get").category == "memory"
