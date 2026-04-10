"""skill_context section — pass-through of pre-built skill markdown blob.

B5 C3: dumps the skill-guide portion of ``ctx.skill_context``. The blob
is built upstream by ``agent_task_runner._build_runtime_system_context``
and contains two concatenated parts: per-skill guides (active skill
names, parameter hints, usage examples) AND the ``## Available Tool
Summary`` block.

**Codex audit HIGH #2 fix (post-B5)**: the executor registry also
renders ``tools_guide_dynamic_section``, which generates the authoritative
``## Available Tool Summary`` block from ``ctx.bound_tool_names``. If
we emit the whole upstream blob here verbatim, the final prompt
contains two copies of ``## Available Tool Summary``. To keep the
``planner_tool_summary_legacy_section`` path working (planner reads the
marker substring via the same blob), we do NOT change
``_build_runtime_system_context``. Instead this section strips the
``## Available Tool Summary`` block (and everything after it, since the
summary is always the trailing section of the blob) before rendering.

**Coupling note**: this stripping assumes the tool summary is the LAST
section in the blob and is prefixed by the exact literal header
``## Available Tool Summary``. Any upstream change to the blob layout
that puts another section after the tool summary, or that changes the
header text, will cause this section to emit content it shouldn't.
``planner_tool_summary_legacy_section`` has the same coupling.

**Note on invariants**: the ``_assert_no_dangling_skill_tool_refs`` scan
runs at registry construction time against ``_FIXTURE_CTX`` (see
``assembler.py`` docstring), NOT at assemble-time. This section passes
through runtime ``ctx.skill_context`` (with tool summary stripped), so
any ``skill_*`` tokens in the skill-guide portion are emitted unchecked.
That is an explicit tradeoff: the startup check catches
section-authoring errors, while runtime drift between
``state.skill_context`` and ``bound_tool_names`` remains the upstream
builder's responsibility.

Returns ``SectionOutput(text=None)`` when ``ctx.skill_context`` is empty
or consists entirely of the tool summary block (nothing to emit after
stripping).
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


_TOOL_SUMMARY_MARKER = "## Available Tool Summary"


def _strip_tool_summary(blob: str) -> str:
    """Remove the ``## Available Tool Summary`` block from a skill_context blob.

    The tool summary is always the trailing section in the blob (see
    ``_build_runtime_system_context`` in agent_task_runner.py). We find
    the marker and return everything BEFORE it, stripped of trailing
    whitespace. Returns the input unchanged if the marker is absent.
    """
    idx = blob.find(_TOOL_SUMMARY_MARKER)
    if idx == -1:
        return blob
    return blob[:idx].rstrip()


def _render(ctx: RenderContext) -> SectionOutput:
    """Render the skill-guide portion of ``ctx.skill_context``.

    Strips the ``## Available Tool Summary`` block (emitted
    authoritatively by ``tools_guide_dynamic_section`` instead).
    """
    if not ctx.skill_context or not ctx.skill_context.strip():
        return SectionOutput(text=None)
    skill_only = _strip_tool_summary(ctx.skill_context).strip()
    if not skill_only:
        return SectionOutput(text=None)
    return SectionOutput(
        text=skill_only,
        metadata={"skill_ids_used": list(ctx.skill_names_in_context)},
    )


skill_context_section = Section(
    id="skill_context",
    priority=7,
    cacheable=False,
    dynamic=True,
    render=_render,
)
