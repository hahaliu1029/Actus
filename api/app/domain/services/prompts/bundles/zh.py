"""ZH prompt bundle — Chinese SectionRegistry instances for C2 + C3 sections.

B5 C4: wires the 8 executor sections from C2 (identity / behavior_core /
output_format) and C3 (tools_guide_stable / tools_guide_dynamic /
skill_context / conversation_summaries / sandbox_state) into the ZH
executor registry.

Planner and updater registries are intentionally empty at C4 — C6 will
populate them when switching ``planner_node`` and ``updater_node`` over
to the assembler path. Empty ``SectionRegistry`` is valid: the
``__post_init__`` validation loop iterates zero sections.

**Declaration order = assembly order**: the list below is read top-to-bottom
by ``PromptAssembler.assemble``. Priority only governs budget-driven
dropping, not output ordering.
"""
from __future__ import annotations

from app.domain.services.prompts.section import PromptBundle, SectionRegistry
from app.domain.services.prompts.sections.agent_team_teaching import (
    agent_team_teaching_section,
)
from app.domain.services.prompts.sections.behavior_core import behavior_core_section
from app.domain.services.prompts.sections.conversation_summaries import (
    conversation_summaries_section,
)
from app.domain.services.prompts.sections.identity import identity_section
from app.domain.services.prompts.sections.memory_fact_index import (
    memory_fact_index_section,
)
from app.domain.services.prompts.sections.memory_rules import memory_rules_section
from app.domain.services.prompts.sections.memory_user_profile import (
    memory_user_profile_section,
)
from app.domain.services.prompts.sections.output_format import output_format_section
from app.domain.services.prompts.sections.parallel_work_units_teaching import (
    parallel_work_units_teaching_section,
)
from app.domain.services.prompts.sections.planner_identity import (
    planner_identity_section,
)
from app.domain.services.prompts.sections.planner_tool_summary_legacy import (
    planner_tool_summary_legacy_section,
)
from app.domain.services.prompts.sections.sandbox_state import sandbox_state_section
from app.domain.services.prompts.sections.skill_context import skill_context_section
from app.domain.services.prompts.sections.tools_guide_dynamic import (
    tools_guide_dynamic_section,
)
from app.domain.services.prompts.sections.tools_guide_stable import (
    tools_guide_stable_section,
)


# ---- Executor registry (ReAct system prompt) --------------------------- #

# **Intentional ordering delta vs legacy** (noted for C5b A/B verification):
# the legacy ``main_graph.executor_node`` builds the system prompt as
# ``REACT_SYSTEM_PROMPT + FILE_VIEW_HINT + MEMORY_TOOLS_HINT + skill_context +
# conversation_summaries``, where ``skill_context`` contains the
# ``## Available Tool Summary`` embedded at the bottom (via
# ``agent_task_runner._build_runtime_system_context``). The new bundle
# declares ``tools_guide_dynamic`` BEFORE ``skill_context`` so the tool
# summary appears as a standalone top-level block immediately after the
# capability hints — before the per-skill guides. C5a will dedupe the
# embedded copy inside ``state.skill_context``; until then, expect one
# "## Available Tool Summary" to appear twice when the feature flag is ON.
ZH_EXECUTOR_REGISTRY = SectionRegistry(
    sections=(
        # C2 sections (priority 10, 10, 9) — always render
        identity_section,
        behavior_core_section,
        output_format_section,
        # C3 sections (priority 8, 7, 7, 6, 5) — conditional render.
        # M2-PR3 memory sections interleave by priority bucket:
        #   memory_rules (prio 8) after tools_guide_stable (prio 8) so
        #     hard behavior rules reach the agent before it engages tools;
        #   memory_user_profile (prio 7) before skill_context (prio 7) so
        #     preferences inform skill selection;
        #   memory_fact_index (prio 5) before sandbox_state (prio 5) as a
        #     last-block reference the agent can consult by id.
        tools_guide_stable_section,
        tools_guide_dynamic_section,
        memory_rules_section,
        memory_user_profile_section,
        skill_context_section,
        conversation_summaries_section,
        memory_fact_index_section,
        sandbox_state_section,
    ),
    name="zh_executor",
)


# ---- Planner registry (B5 C6) ------------------------------------------ #

# B5 C6: planner_node consumes this registry in the PromptAssembler path.
# The 3 sections together reproduce the legacy concatenation:
#   PLANNER_SYSTEM_PROMPT + (tool summary from state.skill_context)
#   + conversation summaries.
# planner_identity (prio 10) is stable text. planner_tool_summary_legacy
# (prio 7) is the pass-through that reads ``state.skill_context`` and
# extracts the ``## Available Tool Summary`` block. conversation_summaries
# (prio 6) is reused from the executor registry — section instances are
# stateless and can be shared across registries.
ZH_PLANNER_REGISTRY = SectionRegistry(
    sections=(
        planner_identity_section,
        parallel_work_units_teaching_section,
        agent_team_teaching_section,
        planner_tool_summary_legacy_section,
        conversation_summaries_section,
    ),
    name="zh_planner",
)


# ---- Updater registry (B5 C6) ------------------------------------------ #

# B5 C6: updater_node reuses the same 3 sections as the planner. The
# updater's system prompt in legacy code is identical to the planner's —
# both use ``PLANNER_SYSTEM_PROMPT`` + tool summary + conversation summaries.
# Keeping distinct registry instances (same section contents, different
# ``name`` field) preserves ``PromptBundle.planner != PromptBundle.updater``
# as distinct fields so future divergence doesn't require a refactor.
ZH_UPDATER_REGISTRY = SectionRegistry(
    sections=(
        planner_identity_section,
        parallel_work_units_teaching_section,
        agent_team_teaching_section,
        planner_tool_summary_legacy_section,
        conversation_summaries_section,
    ),
    name="zh_updater",
)


# ---- Bundle ------------------------------------------------------------ #

ZH_BUNDLE = PromptBundle(
    lang="zh",
    executor=ZH_EXECUTOR_REGISTRY,
    planner=ZH_PLANNER_REGISTRY,
    updater=ZH_UPDATER_REGISTRY,
)
