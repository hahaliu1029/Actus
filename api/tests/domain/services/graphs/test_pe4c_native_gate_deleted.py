"""PE-4c: the legacy native risk gate is deleted from react_graph.

After PE-4c, real native tools route through PE (_pe_dispatch); a native tool
that falls to legacy via a mixed batch is caught by the PE-4b native
fail-closed guard. The old inline risk gate (risk_assessor.assess →
ApprovalStateReader.check → inline SmartApprove → interrupt) is removed, along
with the now-dead ``risk_level_meta`` assignment. The meta-tool/unknown
direct-execute passthrough MUST survive.
"""

from __future__ import annotations

import inspect

import app.domain.services.graphs.react_graph as rg


def _src() -> str:
    return inspect.getsource(rg)


def test_native_risk_gate_assessment_call_removed():
    src = _src()
    # NOTE: ``assessment = _risk_assessor.assess(tool_name, args)`` is NOT a
    # unique marker — that exact line ALSO exists at the LIVE KEEP PE-path
    # site (react_graph.py:1539), so asserting it `not in src` would stay
    # RED even after the gate is correctly deleted. Assert on a string that
    # exists ONLY in the deleted legacy native gate: its distinctive
    # ``# Original native tool gate`` comment header (verified unique @
    # react_graph.py:2661; the PE-path site has no such comment).
    assert "# Original native tool gate" not in src


def test_risk_level_meta_assignment_removed():
    src = _src()
    # The dead assignment that fed the gate (and tripped INV-6) is gone.
    assert "risk_level_meta = (getattr(tool_fn" not in src
    # Defensive: no functional read of risk_level_meta remains (comments OK
    # are stripped by inspect.getsource? No — getsource keeps comments. The
    # INV-6 AST scan ignores comments; here we only assert the assignment +
    # the gate condition are gone).
    assert 'risk_level_meta in ("high", "medium")' not in src


def test_meta_tool_direct_execute_passthrough_survives():
    src = _src()
    # The permanent passthrough comment + invoke must remain.
    assert "No risk metadata (or bypass via pre-approved) → execute directly" in src


def test_native_fail_closed_guard_present_from_pe4b():
    """Sanity: PE-4b's native guard is the thing that closes the hole the
    deleted gate used to cover. Its distinctive zh message must be present."""
    src = _src()
    assert "native_mixed_batch_fail_closed" in src
    assert "原生工具" in src
