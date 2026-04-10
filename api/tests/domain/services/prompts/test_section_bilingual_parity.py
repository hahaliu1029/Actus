"""B5 C2: ZH/EN bilingual parity tests across all C2 sections.

For every section, verify that:
1. Both languages produce non-empty output
2. Length ratio (EN / ZH chars) is within 0.5 - 3.0 (CJK is denser than ASCII)

This catches a future commit accidentally adding an English-only or
Chinese-only paragraph to one side without the other.
"""
from __future__ import annotations

import pytest

from app.domain.services.prompts.section import RenderContext, Section
from app.domain.services.prompts.sections.behavior_core import behavior_core_section
from app.domain.services.prompts.sections.identity import identity_section
from app.domain.services.prompts.sections.output_format import output_format_section


C2_SECTIONS: list[Section] = [
    identity_section,
    behavior_core_section,
    output_format_section,
]


@pytest.mark.parametrize("section", C2_SECTIONS, ids=lambda s: s.id)
def test_section_renders_in_zh(section: Section) -> None:
    output = section.render(RenderContext(lang="zh"))
    assert output.text is not None
    assert output.text.strip()


@pytest.mark.parametrize("section", C2_SECTIONS, ids=lambda s: s.id)
def test_section_renders_in_en(section: Section) -> None:
    output = section.render(RenderContext(lang="en"))
    assert output.text is not None
    assert output.text.strip()


@pytest.mark.parametrize("section", C2_SECTIONS, ids=lambda s: s.id)
def test_section_zh_en_length_parity(section: Section) -> None:
    """ZH/EN length ratio must be within 0.8x - 3.0x.

    English text is generally 1.2-2.5x the length of Chinese (which uses CJK
    characters that pack more meaning per char). A ratio outside this band
    suggests one language is missing content the other has.

    Band kept consistent with the older C0a/C0b parity test in
    ``test_prompt_language_dispatch.py``.
    """
    zh_text = section.render(RenderContext(lang="zh")).text or ""
    en_text = section.render(RenderContext(lang="en")).text or ""
    ratio = len(en_text) / max(len(zh_text), 1)
    assert 0.8 < ratio < 3.0, (
        f"{section.id}: EN/ZH length ratio {ratio:.2f} out of parity band "
        f"(zh={len(zh_text)} chars, en={len(en_text)} chars)"
    )


@pytest.mark.parametrize("section", C2_SECTIONS, ids=lambda s: s.id)
def test_section_zh_contains_chinese(section: Section) -> None:
    """Sanity: ZH output should contain CJK characters."""
    text = section.render(RenderContext(lang="zh")).text or ""
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    assert has_cjk, f"{section.id}: ZH output has no CJK characters"


@pytest.mark.parametrize("section", C2_SECTIONS, ids=lambda s: s.id)
def test_section_en_first_line_is_ascii(section: Section) -> None:
    """Sanity: EN output's first non-empty line should be pure ASCII (no leaked CJK)."""
    text = section.render(RenderContext(lang="en")).text or ""
    first_line = next((line for line in text.split("\n") if line.strip()), "")
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in first_line)
    assert not has_cjk, (
        f"{section.id}: EN output's first line contains CJK characters: {first_line!r}"
    )
