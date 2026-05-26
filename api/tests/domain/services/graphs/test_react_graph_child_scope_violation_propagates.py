"""C2 PR-2 §5.4 — ChildScopeViolation must propagate past react_graph's PE catch-all.

Without this, PR-4's CoordinatorChildRunner finalizer cannot convert the
violation to RESULT_READY(needs_authorization).
"""
import inspect
import re


def test_react_graph_reraises_child_scope_violation():
    """AST-grep: react_graph.py contains `except ChildScopeViolation` re-raise.

    Read-only structural assertion; running the full graph would require
    extensive fixture wiring. The intent is to lock in that the catch-all
    `except Exception` block does NOT swallow ChildScopeViolation.
    """
    from app.domain.services.graphs import react_graph

    src = inspect.getsource(react_graph)
    # Find the pe_evaluate_crash catch-all and confirm a ChildScopeViolation
    # re-raise precedes it lexically (same try/except group).
    crash_idx = src.find("pe_evaluate_crash")
    assert crash_idx != -1, "pe_evaluate_crash branch must exist in react_graph"
    preceding = src[:crash_idx]
    # The closest re-raise BEFORE the catch-all
    last_reraise = preceding.rfind("except ChildScopeViolation")
    assert last_reraise != -1, (
        "ChildScopeViolation must be re-raised BEFORE the broad 'except Exception' "
        "that builds pe_evaluate_crash; otherwise PR-4 finalizer never sees it."
    )
    # Confirm the re-raise block contains `raise` (not just a pass).
    # Widen window to 600 chars to fit multi-line justification comment.
    block_tail = src[last_reraise: last_reraise + 600]
    assert re.search(r"except ChildScopeViolation[^:]*:\s*(?:#[^\n]*\n\s*)*raise\b", block_tail), (
        f"ChildScopeViolation except block must `raise`; got: {block_tail!r}"
    )


def test_child_scope_violation_imported_in_react_graph():
    """react_graph must have ChildScopeViolation bound (not just source-string presence)."""
    from app.domain.services.graphs import react_graph
    from app.domain.services.permission.child_scope_violation import ChildScopeViolation

    # Actual binding check — source-string presence is NOT enough; if the import
    # is removed but the `except ChildScopeViolation:` clause remains, source-grep
    # would still pass but runtime would NameError.
    assert hasattr(react_graph, "ChildScopeViolation"), (
        "react_graph must bind ChildScopeViolation at module scope (import line removed?)"
    )
    assert react_graph.ChildScopeViolation is ChildScopeViolation, (
        "react_graph.ChildScopeViolation must reference the same class as the canonical import"
    )
