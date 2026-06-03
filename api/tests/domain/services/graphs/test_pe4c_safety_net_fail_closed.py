"""PE-4c: the _pe_dispatch per-call safety-net is fail-CLOSED.

Previously a non-PE-eligible call that reached the PE loop (documented
unreachable) executed via _invoke_wrapper (fail-OPEN). PE-4c converts it to a
Denied so a misclassified call never runs unconfirmed — and removes the lone
_invoke_wrapper callsite that needed the INV-5 safety-net whitelist.
"""

from __future__ import annotations

import inspect

import app.domain.services.graphs.react_graph as rg


def test_safety_net_no_longer_invoke_wrapper():
    """The per-call safety-net must NOT call _invoke_wrapper (the fail-open
    execution). Its former callsite (the INV-5 :1486 whitelist anchor) is gone."""
    src = inspect.getsource(rg)
    assert "_non_pe_result = await _invoke_wrapper(" not in src


def test_safety_net_uses_fail_closed_reason_code():
    src = inspect.getsource(rg)
    assert "pe_eligibility_invariant_violation" in src
