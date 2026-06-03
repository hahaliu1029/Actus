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


class TestCreateTaskGateIsMasterOnly:
    def test_create_gate_collapsed_to_master_switch(self):
        """PE-4c: per-source flags retired. The create gate degenerates to
        'build PE iff no tool_confirmation config OR master enabled'. The
        per-source ``any(is_pe_enabled_for_source(...))`` comprehension and
        its ``PE_SUPPORTED_SOURCES`` import are gone."""
        src = _agent_service_src()
        # Per-source loop is gone from the create/resume gates.
        assert "any(\n                is_pe_enabled_for_source" not in src
        assert "for src in PE_SUPPORTED_SOURCES" not in src
        # The collapsed master-only form is present (create + resume).
        assert "_tc_at_create is None" in src
        assert "all per-source PE flags off" not in src

    def test_create_gate_still_builds_pe_when_no_config(self):
        """Invariant (a): a no-config session (tc is None) still activates PE."""
        src = _agent_service_src()
        # The no-config default-on branch must survive the collapse.
        assert "_tc_at_create is None" in src


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


class TestMcpSourceRegisteredAtBothPEBuildSites:
    def test_mcp_source_registered_at_both_pe_build_sites(self):
        """PE-2 regression — McpSource must be registered at BOTH
        PermissionEngine build sites in agent_service:

        - ``_pe_sources`` in ``_create_task`` (initial turn)
        - ``_pe_sources_r`` in ``_build_pe_ssm_for_resume`` (resume turn)

        WHY: if a future refactor silently drops the resume-path
        registration, a fresh MCP tool call made during a *resumed* turn
        would hit a PermissionEngine that has no 'mcp' source registered and
        raise ``UnsupportedSource`` — re-introducing the exact behavior gap
        PE-2 closed. Asserting the literal appears exactly twice makes the
        guard genuinely fail (count drops to 1) if either site is removed.
        """
        src = _agent_service_src()
        # Both PE build sites register the mcp source object.
        assert src.count('"mcp": McpSource()') == 2, (
            "McpSource must be registered at BOTH PE build sites "
            "(_pe_sources in _create_task + _pe_sources_r in "
            "_build_pe_ssm_for_resume); dropping one re-introduces "
            "UnsupportedSource for a fresh MCP call on the resume path"
        )
        # And McpSource is imported (from the permission sources package).
        assert "McpSource" in src


class TestA2aSourceRegisteredAtBothPEBuildSites:
    def test_a2a_source_registered_at_both_pe_build_sites(self):
        """PE-3 regression — A2aSource must be registered at BOTH
        PermissionEngine build sites in agent_service:

        - ``_pe_sources`` in ``_create_task`` (initial turn)
        - ``_pe_sources_r`` in ``_build_pe_ssm_for_resume`` (resume turn)

        WHY: if a future refactor silently drops the resume-path registration,
        a fresh A2A tool call made during a *resumed* turn would hit a
        PermissionEngine that has no 'a2a' source registered and raise
        ``UnsupportedSource``. Asserting the literal appears exactly twice makes
        the guard fail (count drops to 1) if either site is removed.
        """
        src = _agent_service_src()
        assert src.count('"a2a": A2aSource()') == 2, (
            "A2aSource must be registered at BOTH PE build sites "
            "(_pe_sources in _create_task + _pe_sources_r in "
            "_build_pe_ssm_for_resume); dropping one re-introduces "
            "UnsupportedSource for a fresh A2A call on the resume path"
        )
        assert "A2aSource" in src


class TestCreateGateInvariantsPreservedAfterCollapse:
    """PE-4c: the master-only gate collapse must preserve two invariants."""

    def test_invariant_a_no_config_session_builds_pe(self):
        """(a) tc is None → PE is still built (default-on)."""
        src = _agent_service_src()
        # The create gate's no-config branch (default-on) is present.
        assert "_tc_at_create is None or getattr(_tc_at_create" in src

    def test_invariant_b_master_off_returns_none_none_on_resume(self):
        """(b) master enabled=False on the resume path → (None, None)."""
        src = _agent_service_src()
        # The resume gate returns (None, None) on master-off.
        assert 'return None, None  # confirmation master switch off' in src
        # And the per-source 'all flags off' early-return is gone.
        assert "all per-source PE flags off" not in src


class TestCreateGateFailsClosedWhenPECannotBuild:
    """PE-4c P1 (codex-found): with the legacy native risk gate deleted, PE is the
    SOLE confirmation path. When the master switch is on (or default-on) but the PE
    cannot be built — missing writer/reader/confirmation_queue, OR the build throws
    — the create path must fail CLOSED (raise), NOT silently leave
    permission_engine=None (which would let risk-bearing native tools like
    shell_execute run UNCONFIRMED via tool_node's direct-execute passthrough)."""

    def test_silent_fail_open_fallback_is_removed(self):
        """The pre-fix silent fallback (set PE=None, run legacy) must be GONE — its
        promised 'legacy confirmation path' was deleted in PE-4c."""
        src = _agent_service_src()
        assert (
            "legacy confirmation path will be used for this session" not in src
        ), "PE-4c must not silently fall back to the (deleted) legacy path"

    def test_missing_pe_dependency_fails_closed(self):
        """master-on + a missing PE dependency → raise (not silent PE=None)."""
        src = _agent_service_src()
        # The dep-missing guard raises with the PE-4c fail-closed message.
        assert (
            "without the confirmation boundary (PE-4c fail-closed)" in src
        ), "missing-dependency path must raise PermissionConfigurationError"
        assert "missing: {_missing_pe_deps}" in src

    def test_pe_build_exception_fails_closed(self):
        """master-on + the PE build raising → re-raise (not silent PE=None)."""
        src = _agent_service_src()
        # The build except re-raises instead of swallowing to PE=None.
        assert "failing closed (PE-4c)." in src
        assert (
            "Failed to build the PermissionEngine while tool_confirmation" in src
        )

    def test_explicit_master_off_escape_hatch_documented(self):
        """The only way to run without confirmation is the explicit master-off
        escape hatch — never an implicit infra-degradation fallthrough."""
        src = _agent_service_src()
        # NB: assert within a single string literal (the source splits "... Set "
        # and "tool_confirmation.enabled=False ..." across two adjacent literals).
        assert "tool_confirmation.enabled=False to run without confirmation" in src
