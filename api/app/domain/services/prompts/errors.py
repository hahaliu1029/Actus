"""Domain-layer exceptions for prompt assembly.

B5 C1: defined in domain (not application) per Clean Architecture.
The application layer's ``AppException`` subclasses can re-export or wrap
these for HTTP error handling — see ``application/errors/exceptions.py``.

Both errors are plain ``Exception`` subclasses with no HTTP-status coupling.
"""
from __future__ import annotations


class PromptAssemblyError(Exception):
    """Base class for prompt assembly errors raised by the prompts module."""


class ToolBindingInvariantError(PromptAssemblyError):
    """System prompt mentions a skill tool that is not in bound_tool_names.

    Raised by ``invariants._assert_no_dangling_skill_tool_refs`` when a
    section's rendered output references a ``skill_*`` tool name not present
    in the LLM's actually bound tool list. See B5 design doc Risk #3.
    """


class SectionValidationError(PromptAssemblyError):
    """Section.render() failed during SectionRegistry startup validation.

    Raised by ``SectionRegistry.__post_init__`` when a section's render()
    callable raises on the canonical fixture RenderContext, or when its
    output contains a dangling ``skill_*`` reference. Failing at startup
    catches section authoring bugs before runtime.
    """
