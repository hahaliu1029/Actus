"""PE-1 §2.5 + §3.2 — agent_service callsite migration to is_pe_enabled_for_source.

Focus areas:
- _create_task `_flag_pe_active_at_create` becomes source-aware
  (native off + skill on must still build PE/SSM)
- broad except in build_permission_engine path lets
  PermissionConfigurationError escape
- preflight_resume hardcoded tool_source="native" removed
"""

from __future__ import annotations

import inspect

import pytest


def _agent_service_src() -> str:
    import app.application.services.agent_service as m
    return inspect.getsource(m)


class TestCreateTaskGateIsSourceAware:
    def test_create_task_no_longer_uses_native_only_flag_only(self):
        """spec §2.5: _flag_pe_active_at_create must no longer compute
        as ``tc.enabled and permission_engine_native_enabled``. It should
        use ``is_pe_enabled_for_source`` for at least 'native' and 'skill'."""
        src = _agent_service_src()
        assert "is_pe_enabled_for_source" in src
        # We accept both: explicit any(...) loop OR per-source checks; but
        # the legacy boolean alias '_flag_pe_active_at_create = _flag_tc_enabled and _flag_pe_native'
        # must be gone.
        assert "_flag_pe_active_at_create = _flag_tc_enabled and _flag_pe_native" not in src


class TestPermissionConfigurationErrorEscapesBroadExcept:
    def test_pe_config_error_handled_before_broad_except(self):
        src = _agent_service_src()
        # The contract: in the _create_task PE build, ``PermissionConfigurationError``
        # is re-raised explicitly BEFORE the generic ``except Exception``.
        assert "except PermissionConfigurationError" in src
        # Order check: find positions
        pos_re_raise = src.find("except PermissionConfigurationError")
        # We don't pin the exact next line, but a plain Exception clause must
        # come at a strictly later character offset within the function body.
        following_block_pos = src.find("except Exception", pos_re_raise)
        assert following_block_pos > pos_re_raise


class TestHTTPPreflightHardcodeRemoved:
    def test_no_hardcoded_native_tool_source(self):
        """spec Round 1 P2 — remove `tool_source="native"` literal at the
        HTTP preflight build site (formerly line 1111)."""
        src = _agent_service_src()
        # Only flagged occurrence is the legacy preflight literal; anything
        # else (in tests, comments) must not exist in source.
        # We strict-match the literal arg form to avoid noise from variables.
        assert 'tool_source="native"' not in src
