"""PE-1: tests for 3 new exception classes — UnsupportedSource,
PEInfrastructureUnavailable, PermissionConfigurationError.

Spec ref: §3.2 errors.py + §5.1 + §5.2 HTTP mapping.
"""

from __future__ import annotations

import pytest

from app.domain.services.permission.errors import (
    PermissionConfigurationError,
    PermissionError,
    PEInfrastructureUnavailable,
    UnsupportedSource,
)


def test_unsupported_source_carries_source_attr():
    exc = UnsupportedSource("mystery")
    assert exc.source == "mystery"
    assert "mystery" in str(exc)
    assert isinstance(exc, PermissionError)


def test_unsupported_source_unsupported_str_form():
    exc = UnsupportedSource("mcp")
    assert "tool_source" in str(exc).lower()
    assert "mcp" in str(exc)


def test_pe_infrastructure_unavailable_carries_reason_attr():
    exc = PEInfrastructureUnavailable("redis_get_fail_key: timeout")
    assert exc.reason == "redis_get_fail_key: timeout"
    assert "pe_infrastructure_unavailable" in str(exc)
    assert "redis_get_fail_key" in str(exc)
    assert isinstance(exc, PermissionError)


def test_pe_infrastructure_unavailable_chained_from_origin():
    origin = TimeoutError("redis read timeout")
    try:
        try:
            raise origin
        except Exception as exc:
            raise PEInfrastructureUnavailable(f"redis_get: {exc}") from exc
    except PEInfrastructureUnavailable as wrapped:
        assert wrapped.__cause__ is origin


def test_permission_configuration_error_carries_message():
    msg = "PE_SUPPORTED_SOURCES claims {'skill'} but DI registered only {'native'}."
    exc = PermissionConfigurationError(msg)
    assert msg in str(exc)
    assert isinstance(exc, PermissionError)


def test_new_exceptions_distinct_from_each_other():
    """Subclassing PermissionError but not each other (avoid catching one to mask another)."""
    assert not issubclass(UnsupportedSource, PEInfrastructureUnavailable)
    assert not issubclass(PEInfrastructureUnavailable, UnsupportedSource)
    assert not issubclass(PermissionConfigurationError, UnsupportedSource)
    assert not issubclass(PermissionConfigurationError, PEInfrastructureUnavailable)
