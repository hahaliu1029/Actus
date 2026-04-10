"""Prompt section implementations.

B5 C2-C3: each module in this package defines exactly one ``Section`` and
its render function. Sections are imported by ``prompts/bundles/`` (C4) to
build the per-language ``PromptBundle``.

Section files are organized by id (identity.py, behavior_core.py, etc.) and
hold the ZH/EN text constants as private module-level strings. The render
function dispatches by ``ctx.lang``.
"""
