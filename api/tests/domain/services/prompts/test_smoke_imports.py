"""B5 C1: smoke import test — verify all C1 modules import without error.

This test catches circular import bugs and missing transitive dependencies
before C5a tries to wire them into main_graph.
"""
from __future__ import annotations


def test_section_module_imports() -> None:
    from app.domain.services.prompts.section import (  # noqa: F401
        MINIMAL_MODE_ALLOWLIST,
        PromptBundle,
        PromptMode,
        RenderContext,
        Section,
        SectionOutput,
        SectionRegistry,
        _FIXTURE_CTX,
    )


def test_assembler_module_imports() -> None:
    from app.domain.services.prompts.assembler import (  # noqa: F401
        CRITICAL_PRIORITY_MIN,
        AssembleResult,
        PromptAssembler,
    )


def test_budget_module_imports() -> None:
    from app.domain.services.prompts.budget import (  # noqa: F401
        SystemPromptBudget,
        compute_effective_window,
    )


def test_render_context_module_imports() -> None:
    from app.domain.services.prompts.render_context import (  # noqa: F401
        _categorize_tools,
        _infer_provider,
        build_render_context,
    )


def test_invariants_module_imports() -> None:
    from app.domain.services.prompts.invariants import (  # noqa: F401
        _SKILL_TOOL_PATTERN,
        _assert_no_dangling_skill_tool_refs,
    )


def test_telemetry_port_imports() -> None:
    from app.domain.external.telemetry import PromptTelemetryPort  # noqa: F401


def test_jsonl_telemetry_imports() -> None:
    from app.infrastructure.telemetry.prompt_telemetry import (  # noqa: F401
        JsonlPromptTelemetry,
    )


def test_domain_exceptions_added() -> None:
    """Domain-layer exceptions live in prompts.errors (not application)."""
    from app.domain.services.prompts.errors import (  # noqa: F401
        PromptAssemblyError,
        SectionValidationError,
        ToolBindingInvariantError,
    )


def test_application_layer_has_http_wrapper() -> None:
    """Application layer wraps domain errors as AppException for HTTP responses."""
    from app.application.errors.exceptions import PromptAssemblyHTTPError  # noqa: F401


def test_full_assembly_chain_works() -> None:
    """End-to-end smoke: build assembler + tiny registry → assemble succeeds."""
    from app.domain.services.graphs.token_estimator import TokenEstimator
    from app.domain.services.prompts.assembler import PromptAssembler
    from app.domain.services.prompts.budget import SystemPromptBudget
    from app.domain.services.prompts.section import (
        PromptMode,
        RenderContext,
        Section,
        SectionOutput,
        SectionRegistry,
    )

    section = Section(
        id="identity",
        priority=10,
        cacheable=True,
        dynamic=False,
        render=lambda ctx: SectionOutput(text="You are an agent."),
    )
    registry = SectionRegistry(sections=[section], name="zh_executor")
    assembler = PromptAssembler(
        budget=SystemPromptBudget(max_tokens=3500),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )
    ctx = RenderContext(lang="zh")
    result = assembler.assemble(registry, ctx, PromptMode.FULL)
    assert result.text == "You are an agent."
    assert result.sections_included == ["identity"]
    assert result.tokens_used > 0
    assert len(result.version_hash) == 16


def test_render_context_helper_runs() -> None:
    """Smoke: build_render_context with synthetic state/config doesn't crash."""
    from app.domain.services.prompts.render_context import build_render_context

    class _FakeAgentConfig:
        supports_vision = True
        supports_pdf_input = False

    class _FakeLLM:
        provider_name = "openai"

    state = {
        "language": "en",
        "message": "test",
        "skill_context": "## Skills\n- foo",
        "skill_names_in_context": ["foo"],
        "conversation_summaries": [],
    }
    config = {
        "configurable": {
            "llm": _FakeLLM(),
            "bound_tool_names": frozenset({"file_view", "skill_foo_bar"}),
        }
    }
    ctx = build_render_context(state, config, _FakeAgentConfig())
    assert ctx.lang == "en"
    assert ctx.provider == "openai"
    assert "skill_foo_bar" in ctx.bound_tool_names
    assert ctx.has_file_view is True
    assert ctx.has_pdf is False
