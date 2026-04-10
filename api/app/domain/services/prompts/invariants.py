"""Defensive invariant scans over assembled prompt text.

B5 C1: ``_assert_no_dangling_skill_tool_refs`` is called by
``SectionRegistry.__post_init__`` (startup-time validation) on each section's
output rendered against a fixture RenderContext.

The invariant: any token matching the skill-tool naming pattern
(``skill_{slug}_{tool}``) that appears in the rendered text MUST exist in
``bound_tool_names``. Otherwise the prompt promises a tool that the LLM
isn't actually bound to, which causes ``UNKNOWN_TOOL`` errors at runtime.

The scan is **startup-time only** in B5. It does not run on every assemble
in production hot path. See B5 design doc Risk #3.
"""
from __future__ import annotations

import re

from app.domain.services.prompts.errors import ToolBindingInvariantError

# Match \bskill_<one or more alphanumeric/underscore segments ending in alphanumeric>\b
#
# Permissive enough to cover the double-underscore edge case from
# `_normalize_function_part` not compressing repeated `_` (see skill.py:273-280):
# - skill_foo_bar_tool      → matches
# - skill_foo__bar_tool     → matches (double underscore allowed in middle)
# - skill_foo               → matches (single segment)
# - skill_                  → does NOT match (must end in alphanumeric)
# - skill_foo_              → does NOT match (trailing underscore disallowed)
_SKILL_TOOL_PATTERN = re.compile(r"\bskill_[a-z0-9_]*[a-z0-9]\b")


def _assert_no_dangling_skill_tool_refs(
    rendered_text: str,
    declared_bound_tool_names: frozenset[str],
    section_id: str = "<unknown>",
) -> None:
    """Startup-time validator: rendered text must not reference unbound skill tools.

    NOT in runtime hot path. Called by ``SectionRegistry.__post_init__`` once
    per section against a canonical fixture context. If a section's render
    output produces a ``skill_*`` token not in the fixture's
    ``bound_tool_names``, raise ``ToolBindingInvariantError`` to fail fast at
    application startup.

    Section authors must:
    - Read tool names from ``ctx.bound_tool_names``, never hard-code
    - Use placeholders like ``<skill_name>`` in example/prose contexts to
      avoid false positives matching the pattern
    """
    referenced = set(_SKILL_TOOL_PATTERN.findall(rendered_text))
    missing = referenced - declared_bound_tool_names
    if missing:
        raise ToolBindingInvariantError(
            f"Section '{section_id}' references skill tools not in declared "
            f"bound_tool_names fixture: {sorted(missing)}. "
            f"Either the section hard-codes a tool name (bad — should read from "
            f"ctx.bound_tool_names), or the fixture ctx is missing those names."
        )
