"""sandbox_state section — STUB for future sandbox runtime context injection.

B5 C3: Actus currently does NOT inject sandbox runtime state (cwd, env vars,
disk usage, recent commands, etc.) into the system prompt. This stub
reserves the section id and registry slot so future work can populate it
without restructuring the bundles.

Always returns ``SectionOutput(text=None)`` — the assembler skips it.

When this section gets a real implementation, ``RenderContext`` will need
new fields (``sandbox_cwd``, ``sandbox_env``, etc.) and ``build_render_context``
will populate them from ``state``.
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


def _render(ctx: RenderContext) -> SectionOutput:
    """Reserved stub. Returns no output."""
    return SectionOutput(text=None)


# dynamic=True ensures this section is never placed in the cached prefix
# once the real implementation emits sandbox state (cwd/env/etc. change
# per step, so caching would be incorrect).
sandbox_state_section = Section(
    id="sandbox_state",
    priority=5,
    cacheable=False,
    dynamic=True,
    render=_render,
)
