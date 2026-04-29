"""B5 PR-S1-3 acceptance: AWS secret pattern doesn't eat S3 ETags.

Pattern #10 (AWS secret access key) is shaped as 40 chars of
``[A-Za-z0-9/+=]``. Without context-aware anchoring, that regex would
also match S3 object ETags (32-hex MD5 digests, often padded to 40
chars with base64 wrappers), git SHA1+padding, base64-encoded
checksums, and similar non-secret blobs.

The fix in PR-S1-3 anchors pattern #10 to require an adjacent
``aws_secret_access_key`` / ``secret_access_key`` field name. This
test class locks that anchoring: standalone 40-char hex/base64 strings
(NO key field name nearby) MUST flow through unchanged.
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


class TestEtagSurvival:
    def test_s3_etag_md5_passes_through(self):
        etag = "d41d8cd98f00b204e9800998ecf8427e"
        rendered = _format(f'ETag: "{etag}"')
        assert etag in rendered

    def test_s3_etag_base64_padded_passes_through(self):
        blob = "abcdef0123456789ABCDEF/+abcdef0123456789AB"
        rendered = _format(f"checksum: {blob}")
        assert blob in rendered

    def test_git_sha1_padded_passes_through(self):
        sha = "1234567890abcdef1234567890abcdef12345678"
        rendered = _format(f"commit {sha}")
        assert sha in rendered

    def test_aws_secret_in_context_still_redacted(self):
        rendered = _format(
            'creds aws_secret_access_key="abcdef0123456789ABCDEF/+abcdef0123456789AB"'
        )
        assert "[REDACTED]" in rendered
        assert "0123456789ABCDEF" not in rendered

    def test_etag_alongside_real_secret_still_redacts_secret(self):
        rendered = _format(
            'response ETag="d41d8cd98f00b204e9800998ecf8427e" '
            "Authorization: Bearer sk-aaaaaaaaaaaaaaaaaaaa1234"
        )
        assert "d41d8cd98f00b204e9800998ecf8427e" in rendered
        assert "[REDACTED]" in rendered
