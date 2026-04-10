"""Prompt bundle package — per-language SectionRegistry bundles.

B5 C4: each language has a ``PromptBundle`` containing three
``SectionRegistry`` instances — one per graph node (executor / planner /
updater). C4 only fleshes out the executor registry (8 sections from C2 + C3);
planner and updater get empty placeholder registries that C6 will populate.

Consumers: the top-level ``prompts.__init__`` exports
``get_prompt_section_bundle(lang)`` which returns the appropriate
``PromptBundle``. C5b wires this into ``main_graph.py``'s
``executor_node``; C6 extends the wiring to ``planner_node`` and
``updater_node``.

**Import cost**: importing this package eagerly loads both ZH and EN
bundles, which triggers ``SectionRegistry.__post_init__`` for all six
registries (2 × 8 executor sections + 4 empty registries). This is a
deliberate fail-fast check at startup — any section authoring error
surfaces at import time rather than at first user request.
"""
from __future__ import annotations

from app.domain.services.prompts.bundles.en import EN_BUNDLE
from app.domain.services.prompts.bundles.zh import ZH_BUNDLE

__all__ = ["EN_BUNDLE", "ZH_BUNDLE"]
