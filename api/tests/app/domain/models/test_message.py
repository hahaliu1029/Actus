"""Tests for the ``Message`` domain model (#29 bootstrap language field)."""
from __future__ import annotations

from app.domain.models.message import Message


class TestMessageLanguageField:
    """#29 — verify the new ``language`` bootstrap hint field."""

    def test_language_defaults_to_zh(self) -> None:
        """Constructing a Message without ``language`` yields the 'zh' default."""
        msg = Message(message="hi")
        assert msg.language == "zh"

    def test_language_explicit_en(self) -> None:
        """Explicit ``language='en'`` is preserved through construction."""
        msg = Message(message="hello", language="en")
        assert msg.language == "en"
