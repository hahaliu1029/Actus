"""B5 PR-S1-3 acceptance: pattern ordering — most specific first.

Several v1 patterns share a prefix:

- ``sk-ant-...`` (Anthropic) is a strict subset of ``sk-...`` (OpenAI)
- ``Authorization: Bearer ...`` is a strict subset of
  ``Authorization: ...``
- ``ghp_`` / ``gho_`` / ``ghu_`` / ``ghs_`` (GitHub variants) all
  share the ``gh<one-char>_`` prefix

If the table walks generic-first, the generic regex claims the longer
match and the specific regex never sees the input. The result is the
same string getting redacted twice: once with the generic rule (e.g.,
losing the ``Bearer`` scheme) and once with the specific rule
(no-op because nothing left to match).

This test class locks the order by feeding inputs that ONLY behave
correctly when the specific pattern fires first.
"""
from __future__ import annotations

import logging

from app.infrastructure.logging.redaction import RedactingFormatter


def _format(message: str) -> str:
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=message,
        args=None,
        exc_info=None,
    )
    return formatter.format(record)


class TestAnthropicBeforeOpenai:
    def test_anthropic_key_keeps_ant_prefix(self):
        rendered = _format("creds sk-ant-aaaaaaaaaaaaaaaaaaaaaaaa1234XXXX")
        assert "sk-ant" in rendered
        assert "[REDACTED]" in rendered
        assert "aaaaaaaaaaaa" not in rendered

    def test_openai_key_does_not_have_ant_prefix(self):
        rendered = _format("openai sk-aaaaaaaaaaaaaaaaaaaa1234XX")
        assert "sk-aaa" in rendered
        assert "[REDACTED]" in rendered


class TestBearerBeforeGenericAuth:
    def test_bearer_keeps_scheme_word(self):
        rendered = _format(
            "Authorization: Bearer abc123def456ghi789jkl0XYZ"
        )
        assert "Bearer" in rendered
        assert "[REDACTED]" in rendered
        assert "def456ghi" not in rendered

    def test_basic_auth_redacts_full_blob(self):
        rendered = _format(
            "Authorization: Basic dXNlcjpwYXNzd29yZGZvb2JhcgABCD"
        )
        assert "Authorization:" in rendered
        assert "[REDACTED]" in rendered
        assert "cjpwYXNzd29yZGZvb" not in rendered

    def test_bearer_in_basic_lookahead_does_not_break_basic(self):
        # Edge case: a value that starts with ``Basic`` but contains
        # the substring ``Bearer`` later in the line — the lookahead
        # only checks at the position immediately after
        # ``Authorization:\s*``, so this should still redact via the
        # generic auth pattern.
        rendered = _format(
            "Authorization: Basic dXNlcjpwYXNzZm9vYmFyQmVhcmVy"
        )
        assert "[REDACTED]" in rendered
        assert "cjpwYXNzZm9vYmFyQmVhcmVy" not in rendered


class TestGithubVariants:
    def test_each_github_prefix_independently_redacted(self):
        msg = (
            "PAT ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaXXXX "
            "OAuth gho_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbYYYY "
            "App user ghu_ccccccccccccccccccccccccccccccccZZZZ "
            "App server ghs_ddddddddddddddddddddddddddddddddWWWW"
        )
        rendered = _format(msg)
        assert "ghp_aa" in rendered
        assert "gho_bb" in rendered
        assert "ghu_cc" in rendered
        assert "ghs_dd" in rendered
        assert "[REDACTED]" in rendered
        for raw in (
            "aaaaaaaaaaaa",
            "bbbbbbbbbbbb",
            "cccccccccccc",
            "dddddddddddd",
        ):
            assert raw not in rendered
