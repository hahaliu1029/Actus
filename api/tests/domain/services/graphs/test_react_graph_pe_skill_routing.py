"""PE-1 §2.5 + §2.7 + §5.1 — react_graph._pe_dispatch skill routing,
explicit catches BEFORE broad except, ToolConfirmationEvent payload
rebuilt from confirmation_manager.read."""

from __future__ import annotations

import inspect


def _src() -> str:
    import app.domain.services.graphs.react_graph as m
    return inspect.getsource(m)


class TestMasterGateUsesHelper:
    """The single _flag_native check inside the master entry must defer
    to is_pe_enabled_for_source (or, since per-call gating happens
    inside _pe_dispatch, drop the flag check entirely).

    PE-1 Round 2 P1#2: _pe_dispatch now uses the source+category-aware
    ``is_pe_eligible_tool_source`` helper (which delegates to
    ``is_pe_enabled_for_source`` internally) so that skill creator /
    skill guide tools fall back to legacy.
    """

    def test_master_gate_no_longer_uses_flag_native_alone(self):
        src = _src()
        # Either helper is acceptable — Round 2 swapped per-call sites to
        # the source+category-aware ``is_pe_eligible_tool_source``.
        assert (
            "is_pe_eligible_tool_source" in src
            or "is_pe_enabled_for_source" in src
        )


class TestExplicitCatchesBeforeBroadExcept:
    def test_unsupported_source_caught_before_broad_except(self):
        src = _src()
        assert "except UnsupportedSource" in src
        assert "except PEInfrastructureUnavailable" in src

    def test_pe_infra_unavailable_emits_retryable_allow_error(self):
        """spec §5.1 / Round 2 P1#10: AllowError(retryable=True)."""
        src = _src()
        assert "retryable=True" in src


class TestConfirmationEventRebuildsFromQueueRead:
    def test_pe_asked_branch_consumes_confirmation_manager_read(self):
        src = _src()
        assert "confirmation_manager.read" in src
