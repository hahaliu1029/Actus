"""[PR-9b-D INV-D1] Structural guard — LOCK the honest-skip state of the three
coordinator E2E tests.

The 3 coordinator E2E tests under ``api/tests/integration/`` are intentionally
``@pytest.mark.skip``-ped because the coordinator's child-dispatch PRODUCTION
path is unfinished cold code: ``service_dependencies.py:1126`` injects the bare
``AgentTaskRunner`` class, so a flag-on child spawn raises ``TypeError``.
Re-enabling them is deferred to the dedicated "C2 coordinator finish" follow-up
epic. Each of those files carries a ``@pytest.mark.skip(reason=...)`` whose
reason begins with the machine-checkable tag ``[C2-E2E-DEFERRED]`` and cites the
real production blocker.

This guard locks that state against two anti-vanity regressions:

1. **Silent un-skip** — someone removing the ``@pytest.mark.skip`` decorator
   would make CI *error* (not just fail) by exercising cold production code that
   raises ``TypeError`` on a flag-on child spawn. Test #1 catches a deleted
   decorator.
2. **Reason degradation** — the skip ``reason`` decaying to a vague placeholder
   (e.g. "flaky" / "TODO") would hide that genuine production work is still
   pending. Test #2 asserts the reason keeps both the ``[C2-E2E-DEFERRED]`` tag
   AND at least one concrete blocker keyword, so a tag-only stub cannot pass.

REPLACEMENT NOTICE (INV-D1 lineage): this guard is the *inverse* of the
eventual unskip-guard. When the "C2 coordinator finish" follow-up epic lands and
re-enables the 3 tests (production child-runner wiring + fake-LLM injection seam
+ the ``setup_responses`` / ``X-Test-User-Id`` / ``atomic_write_file`` harness),
THIS file MUST be replaced by its inverse — an "assert NO skip" un-skip guard
that fails if any of the 3 tests is still skip-ped. Do not delete this file
silently; swap it for its inverse so the honest-state invariant stays pinned in
both directions.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.structure

ENUMERATED_FILES = (
    "api/tests/integration/test_coordinator_e2e_apply_rollback.py",
    "api/tests/integration/test_coordinator_e2e_3_work_units.py",
    "api/tests/integration/test_coordinator_e2e_sibling_cancel.py",
)
DEFERRAL_TAG = "[C2-E2E-DEFERRED]"

# At least one of these concrete production-blocker tokens must appear in the
# skip reason alongside the tag — proving the reason is the genuine honest
# deferral and not a tag-only stub.
_BLOCKER_KEYWORDS = ("child-runner", "service_dependencies.py:1126", "TypeError")


def _repo_root() -> Path:
    """Repo root. This file lives at ``api/tests/structure/test_X.py`` so
    ``parents[3]`` (structure -> tests -> api -> <repo>) is the repo root."""
    return Path(__file__).resolve().parents[3]


def _resolve_enumerated_path(rel: str) -> Path:
    """Resolve an enumerated rel-path against the repo root, asserting it exists
    so a renamed/deleted file fails loudly here rather than silently passing."""
    path = _repo_root() / rel
    assert path.exists(), (
        f"Enumerated coordinator E2E test file not found: {path}. "
        f"If it was renamed or deleted, update ENUMERATED_FILES (and reassess "
        f"whether the INV-D1 honest-skip invariant still holds)."
    )
    return path


def _test_function_nodes(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Module-level ``test_*`` functions (sync or async)."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def _decorator_attr_chain(node: ast.expr) -> str | None:
    """Render a decorator's attribute chain as a dotted string, e.g.
    ``pytest.mark.skip`` for ``@pytest.mark.skip(...)``. Returns None if the
    decorator is not a plain attribute/name chain."""
    parts: list[str] = []
    cur: ast.expr | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _find_skip_call(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call | None:
    """Return the ``@pytest.mark.skip(...)`` decorator ``ast.Call`` on ``func``,
    or None if it is not skip-decorated in call form."""
    for dec in func.decorator_list:
        if isinstance(dec, ast.Call):
            chain = _decorator_attr_chain(dec.func)
            if chain is not None and chain.endswith("pytest.mark.skip"):
                return dec
    return None


def _extract_skip_reason(skip_call: ast.Call, src: str) -> str | None:
    """Extract the ``reason`` text from a ``pytest.mark.skip(reason=...)`` call.

    Robust extraction order:
      1. ``ast.Constant`` — adjacent string-literal concatenation folds to ONE
         Constant at parse time, so ``.value`` is the full reason. (This is the
         path Task A's files actually take.)
      2. ``ast.get_source_segment`` — fall back to the raw source slice of the
         value node for any non-constant expression form.
    """
    reason_node: ast.expr | None = None
    for kw in skip_call.keywords:
        if kw.arg == "reason":
            reason_node = kw.value
            break
    # pytest.mark.skip(reason) is valid POSITIONALLY too (skip("msg")); if no
    # reason= keyword was given, use the first positional arg so the positional
    # form cannot dodge the reason-content check below.
    if reason_node is None and skip_call.args:
        reason_node = skip_call.args[0]
    if reason_node is None:
        return None
    if isinstance(reason_node, ast.Constant) and isinstance(reason_node.value, str):
        return reason_node.value
    segment = ast.get_source_segment(src, reason_node)
    return segment


def test_enumerated_e2e_tests_carry_deferred_skip() -> None:
    """Every enumerated file: EVERY module-level ``test_*`` function must carry a
    ``@pytest.mark.skip(...)`` decorator — NOT merely "at least one". A newly
    added or split-out un-skipped E2E test in these files would exercise cold
    production code that raises ``TypeError`` on a flag-on child spawn, so a
    PARTIAL un-skip must be caught too (INV-D1)."""
    failures: list[str] = []
    for rel in ENUMERATED_FILES:
        path = _resolve_enumerated_path(rel)
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        test_funcs = _test_function_nodes(tree)
        if not test_funcs:
            failures.append(
                f"{rel}: no module-level test_* function found at all "
                f"(expected at least one skip-decorated E2E test)."
            )
            continue
        for fn in test_funcs:
            if _find_skip_call(fn) is None:
                failures.append(
                    f"{rel}::{fn.name}: test function is NOT "
                    f"@pytest.mark.skip-decorated — the honest-skip state was "
                    f"(partially) removed (INV-D1). Re-enabling the coordinator "
                    f"E2E tests is deferred to the 'C2 coordinator finish' epic; "
                    f"do not un-skip here."
                )
    assert failures == [], "Honest-skip decorator missing:\n" + "\n".join(failures)


def test_skip_reason_cites_production_blocker() -> None:
    """EVERY skip-decorated ``test_*`` in each enumerated file must keep the
    ``[C2-E2E-DEFERRED]`` tag AND at least one concrete production-blocker keyword
    in its OWN reason — checked per-function (not just the first), so a second
    skipped test with a degraded reason cannot hide behind a compliant first one."""
    failures: list[str] = []
    for rel in ENUMERATED_FILES:
        path = _resolve_enumerated_path(rel)
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)

        for fn in _test_function_nodes(tree):
            skip_call = _find_skip_call(fn)
            if skip_call is None:
                continue  # presence of >=1 skip is enforced by the other test
            reason = _extract_skip_reason(skip_call, src)
            # No whole-file fallback: checks run ONLY against THIS function's
            # extracted reason string. A non-extractable reason is itself a
            # violation — the docstrings also contain the blocker keywords, so a
            # whole-file search would let a degraded reason pass vacuously
            # (codex PR-9b-D R1 P1).
            if reason is None:
                failures.append(
                    f"{rel}::{fn.name}: could not extract a string literal skip "
                    f"reason from the @pytest.mark.skip decorator — the honest-skip "
                    f"contract requires an explicit string reason citing the "
                    f"production blocker."
                )
                continue
            if DEFERRAL_TAG not in reason:
                failures.append(
                    f"{rel}::{fn.name}: skip reason does not contain the deferral "
                    f"tag {DEFERRAL_TAG!r} (reason={reason!r})."
                )
                continue
            if not any(kw in reason for kw in _BLOCKER_KEYWORDS):
                failures.append(
                    f"{rel}::{fn.name}: skip reason carries the {DEFERRAL_TAG} tag "
                    f"but cites NONE of the production blockers {_BLOCKER_KEYWORDS} "
                    f"— it has degraded to a tag-only stub that hides the pending "
                    f"production work (INV-D1). (reason={reason!r})."
                )
    assert failures == [], "Skip reason degraded:\n" + "\n".join(failures)


def test_enumerated_file_list_matches_expected() -> None:
    """Defensive: pin the enumerated basenames against a literal expected set so
    a copy-paste typo or silent list drift in ENUMERATED_FILES is caught."""
    actual = {Path(f).name for f in ENUMERATED_FILES}
    expected = {
        "test_coordinator_e2e_apply_rollback.py",
        "test_coordinator_e2e_3_work_units.py",
        "test_coordinator_e2e_sibling_cancel.py",
    }
    assert actual == expected, (
        f"ENUMERATED_FILES basenames drifted from the expected set. "
        f"actual={actual!r} expected={expected!r}"
    )
