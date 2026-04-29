"""B5 PR-S1-3 acceptance: uvicorn.access query-string redaction.

Uvicorn installs its own ``StreamHandler`` on the ``uvicorn.access``
logger at server boot (``propagate=False`` by default). Without
clearing those handlers, an access-log line like
``GET /api/foo?token=abc123def456ghi789jkl0 HTTP/1.1`` flows directly
to stdout and bypasses the root ``RedactingFormatter``.

``setup_logging`` calls ``clear_propagate_only_loggers`` which strips
the local handler and forces ``propagate=True`` on uvicorn.access (and
the other 7 known noisy libs). Once propagating, every record routes
through root and the URL pattern (#30) and Bearer pattern (#14)
redact secrets in the access log.
"""
from __future__ import annotations

import io
import logging

import pytest

from app.infrastructure.logging.redaction import (
    RedactingFormatter,
    clear_propagate_only_loggers,
)


@pytest.fixture
def isolated_uvicorn_access_logger():
    """Snapshot uvicorn.access state and the root handlers."""
    name = "uvicorn.access"
    logger = logging.getLogger(name)
    snapshot = (logger.handlers[:], logger.propagate, logger.level)

    root = logging.getLogger()
    root_snapshot = (root.handlers[:], root.level)

    yield name

    logger.handlers, logger.propagate, logger.level = snapshot
    root.handlers, root.level = root_snapshot


def _install_root_redacting_formatter() -> io.StringIO:
    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    return captured


class TestUvicornAccessRedaction:
    def test_clear_propagate_only_loggers_strips_uvicorn_access(
        self, isolated_uvicorn_access_logger
    ):
        # Pre-condition: uvicorn already attached its own handler and
        # disabled propagation (the actual uvicorn boot behavior).
        uv = logging.getLogger(isolated_uvicorn_access_logger)
        rogue_stream = io.StringIO()
        rogue_handler = logging.StreamHandler(rogue_stream)
        uv.addHandler(rogue_handler)
        uv.propagate = False

        clear_propagate_only_loggers([isolated_uvicorn_access_logger])

        assert rogue_handler not in uv.handlers
        assert uv.handlers == []
        assert uv.propagate is True

    def test_query_token_redacted_in_access_log(
        self, isolated_uvicorn_access_logger
    ):
        captured = _install_root_redacting_formatter()
        clear_propagate_only_loggers([isolated_uvicorn_access_logger])

        uv = logging.getLogger(isolated_uvicorn_access_logger)
        uv.setLevel(logging.INFO)
        uv.info(
            "127.0.0.1 - - [02/Jan/2025] "
            '"GET /api/foo?token=abc123def456ghi789jkl0 HTTP/1.1" 200 -'
        )

        output = captured.getvalue()
        assert "[REDACTED]" in output
        assert "def456ghi" not in output
        assert "GET /api/foo" in output
        assert "200" in output
        assert "token=" in output

    def test_query_api_key_redacted_in_access_log(
        self, isolated_uvicorn_access_logger
    ):
        captured = _install_root_redacting_formatter()
        clear_propagate_only_loggers([isolated_uvicorn_access_logger])

        uv = logging.getLogger(isolated_uvicorn_access_logger)
        uv.setLevel(logging.INFO)
        uv.info(
            '"GET /v1/skills?api_key=longapikeybytesover18chars HTTP/1.1" 200'
        )

        output = captured.getvalue()
        assert "[REDACTED]" in output
        assert "longapikeybytes" not in output
        assert "api_key=" in output

    def test_uuid_in_query_path_passes_through(
        self, isolated_uvicorn_access_logger
    ):
        # Sanity check: a UUIDv4 in the path is a canonical attribute
        # (likely a session_id / event_id) and must not be eaten.
        captured = _install_root_redacting_formatter()
        clear_propagate_only_loggers([isolated_uvicorn_access_logger])

        uv = logging.getLogger(isolated_uvicorn_access_logger)
        uv.setLevel(logging.INFO)
        uv.info(
            '"GET /sessions/00000000-0000-4000-8000-000000000003 HTTP/1.1" 200'
        )

        output = captured.getvalue()
        assert "00000000-0000-4000-8000-000000000003" in output
