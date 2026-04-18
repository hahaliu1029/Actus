"""EN prompt bundle — English SectionRegistry instances for C2 + C3 sections.

B5 C4: mirror of ``bundles/zh.py`` for English output. Sections dispatch
language via ``ctx.lang == "en"``; the registry structure and section
ordering are identical to the ZH bundle. Using the same section instances
is intentional — per-language text is encoded inside each section's
``_render`` function, not at the registry level.

**Keep this file in sync with ``bundles/zh.py``** — section lists, ordering,
and registry names must stay parallel. The test
``test_executor_registries_declare_8_sections_in_canonical_order`` enforces
this. See ``bundles/zh.py`` for detailed commentary on priorities,
declaration order, the intentional ordering delta vs legacy, and
planner/updater placeholders.
"""
from __future__ import annotations

from app.domain.services.prompts.section import PromptBundle, SectionRegistry
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

# Mirror of ZH_EXECUTOR_REGISTRY — see bundles/zh.py for the rationale on
# the M2-PR3 memory section placement (rules near tool guides,
# user_profile before skill_context, fact_index as a bottom reference).
EN_EXECUTOR_REGISTRY = SectionRegistry(
    sections=(
        identity_section,
        behavior_core_section,
        output_format_section,
        tools_guide_stable_section,
        tools_guide_dynamic_section,
        memory_rules_section,
        memory_user_profile_section,
        skill_context_section,
        conversation_summaries_section,
        memory_fact_index_section,
        sandbox_state_section,
    ),
    name="en_executor",
)


# ---- Planner registry (B5 C6) ------------------------------------------ #

# Mirror of ZH_PLANNER_REGISTRY. See bundles/zh.py for commentary.
EN_PLANNER_REGISTRY = SectionRegistry(
    sections=(
        planner_identity_section,
        planner_tool_summary_legacy_section,
        conversation_summaries_section,
    ),
    name="en_planner",
)


# ---- Updater registry (B5 C6) ------------------------------------------ #

# Mirror of ZH_UPDATER_REGISTRY. See bundles/zh.py for commentary.
EN_UPDATER_REGISTRY = SectionRegistry(
    sections=(
        planner_identity_section,
        planner_tool_summary_legacy_section,
        conversation_summaries_section,
    ),
    name="en_updater",
)


# ---- Bundle ------------------------------------------------------------ #

EN_BUNDLE = PromptBundle(
    lang="en",
    executor=EN_EXECUTOR_REGISTRY,
    planner=EN_PLANNER_REGISTRY,
    updater=EN_UPDATER_REGISTRY,
)
