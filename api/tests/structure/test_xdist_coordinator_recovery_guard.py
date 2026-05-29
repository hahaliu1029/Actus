"""[PR-9b-C Task C8 / INV-C5] Structural guard — ``-m coordinator_recovery`` must
NOT run under xdist (``-n>1``).

The coordinator E2E recovery harness shares mutable global state across tests
(Redis DB 15 flushed before/after each test + a ``TRUNCATE ... CASCADE`` reset
in ``coordinator_truncation``).  There is no per-worker DB/Redis isolation, so
``-n>1`` lets concurrent workers clobber each other's rows.  The real guard
lives in ``tests/integration/conftest.py`` as a module-level
``pytest_collection_modifyitems`` that raises ``RuntimeError`` at collection
time when the worker count exceeds 1 and any ``coordinator_recovery``-marked
item is collected.

This test pins that contract two ways:

1. **Real-wiring AST assertion** — parse the genuine
   ``tests/integration/conftest.py`` and assert it defines a module-level
   ``pytest_collection_modifyitems`` whose body references BOTH
   ``PYTEST_XDIST_WORKER_COUNT`` and ``coordinator_recovery`` AND contains a
   ``raise RuntimeError`` (or ``pytest.UsageError``).  The synthetic subprocess
   below only proves the *pattern* works, not that the real conftest is wired —
   so this AST check is the load-bearing half.

2. **Synthetic subprocess** — materialise a throwaway pytest project carrying
   the guard, a ``coordinator_recovery``-marked dummy test, and a marker
   declaration, then run ``pytest -n 2 -m coordinator_recovery --collect-only``
   and assert it exits non-zero with the xdist message; and ``-n 0`` collects
   cleanly.

NOTE (implementation reality): pytest-xdist does NOT populate
``PYTEST_XDIST_WORKER_COUNT`` during ``pytest_collection_modifyitems`` in the
pinned version — it is exported per-worker for test *execution*.  At collection
time the reliable signal is ``config.option.numprocesses`` (the ``-n`` value).
The real guard reads BOTH (max of the two); the synthetic conftest below must
mirror that, otherwise the ``-n 2`` subprocess would silently collect 0 errors
and this test would be vacuous.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.structure

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REAL_CONFTEST = (
    REPO_ROOT / "api" / "tests" / "integration" / "conftest.py"
)

# Guard body shared by the real conftest's contract AND the synthetic subprocess.
# Reads BOTH signals (env + numprocesses) and raises at collection time.
_SYNTHETIC_GUARD = '''import os

import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    env_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0") or "0")
    opt_count = getattr(config.option, "numprocesses", None) or 0
    worker_count = max(env_count, int(opt_count))
    if worker_count > 1:
        for item in items:
            if "coordinator_recovery" in item.keywords:
                raise RuntimeError(
                    "xdist parallelism is NOT supported with the "
                    "-m coordinator_recovery marker; the fixture harness uses "
                    "shared Redis DB 15 + truncation which collide under -n>1. "
                    "Run single-worker (-n0) instead."
                )
'''

# Two tests: one recovery-marked (guard target) + one plain (must survive a
# ``-m "not coordinator_recovery"`` selection under -n2 without tripping the
# guard — codex R3-F2 regression). The ``trylast=True`` guard above runs AFTER
# pytest's built-in markexpr deselection, so under ``-m "not
# coordinator_recovery"`` the recovery item is already gone from ``items`` and
# only ``test_plain`` remains → no raise.
_SYNTHETIC_DUMMY_TEST = '''import pytest


@pytest.mark.coordinator_recovery
def test_x():
    pass


def test_plain():
    pass
'''

_SYNTHETIC_PYPROJECT = '''[tool.pytest.ini_options]
markers = [
    "coordinator_recovery: synthetic marker for the xdist guard structure test",
]
'''

# Substring that must surface in the failing run's combined stdout+stderr.
_XDIST_MSG_NEEDLE = "xdist parallelism is NOT supported"


def _find_real_guard() -> ast.FunctionDef:
    """Return the module-level ``pytest_collection_modifyitems`` AST node from the
    genuine integration conftest, or fail loudly if it isn't wired."""
    assert REAL_CONFTEST.exists(), (
        f"Real integration conftest not found at {REAL_CONFTEST}; "
        f"the xdist guard cannot be verified."
    )
    tree = ast.parse(REAL_CONFTEST.read_text(encoding="utf-8"))
    for node in tree.body:  # module level only — NOT nested inside a class/fn
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "pytest_collection_modifyitems"
        ):
            return node
    pytest.fail(
        "tests/integration/conftest.py does NOT define a module-level "
        "pytest_collection_modifyitems hook — the INV-C5 xdist guard is not "
        "wired. The synthetic subprocess below only proves the pattern, not the "
        "real wiring."
    )


def test_xdist_with_coordinator_recovery_fails_at_collection(tmp_path):
    """Real guard is wired (AST) AND the pattern fails ``-n 2`` collection (subprocess)."""
    # ── Part 1: AST-assert the REAL conftest guard is wired. ──────────────────
    guard = _find_real_guard()
    guard_src = ast.unparse(guard)

    assert "PYTEST_XDIST_WORKER_COUNT" in guard_src, (
        "Real pytest_collection_modifyitems guard must reference "
        "PYTEST_XDIST_WORKER_COUNT (the documented xdist worker-count signal)."
    )
    assert "coordinator_recovery" in guard_src, (
        "Real pytest_collection_modifyitems guard must gate on the "
        "'coordinator_recovery' marker keyword."
    )
    raises_runtimeerror = "raise RuntimeError" in guard_src
    raises_usageerror = (
        "UsageError" in guard_src and "raise" in guard_src
    )
    assert raises_runtimeerror or raises_usageerror, (
        "Real pytest_collection_modifyitems guard must `raise RuntimeError` "
        "(or pytest.UsageError) when xdist is detected with coordinator_recovery."
    )

    # codex R3-F2 (HIGH): the real guard MUST be decorated
    # ``@pytest.hookimpl(trylast=True)`` so pytest's built-in markexpr
    # deselection runs first — otherwise ``pytest -n2 -m "not
    # coordinator_recovery"`` falsely trips the guard on items that are about to
    # be deselected. Assert the decorator is wired on the real hook.
    trylast_decorated = any(
        "trylast" in ast.unparse(dec) and "True" in ast.unparse(dec)
        for dec in guard.decorator_list
    )
    assert trylast_decorated, (
        "Real pytest_collection_modifyitems guard must be decorated "
        "@pytest.hookimpl(trylast=True) so it runs AFTER pytest's -m "
        "deselection; otherwise a legitimate parallel `-n2 -m \"not "
        "coordinator_recovery\"` run falsely trips the guard (codex R3-F2)."
    )

    # ── Part 2: Synthetic subprocess — the pattern actually fails under -n 2. ─
    (tmp_path / "conftest.py").write_text(_SYNTHETIC_GUARD, encoding="utf-8")
    (tmp_path / "test_dummy.py").write_text(_SYNTHETIC_DUMMY_TEST, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(_SYNTHETIC_PYPROJECT, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "2",
            "-m",
            "coordinator_recovery",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            str(tmp_path),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"Expected non-zero exit when collecting -m coordinator_recovery under "
        f"-n 2, got returncode={result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert _XDIST_MSG_NEEDLE in combined, (
        f"Expected the xdist-incompatibility message "
        f"({_XDIST_MSG_NEEDLE!r}) in the failing run's output, but it was "
        f"absent.\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_single_worker_passes_collection(tmp_path):
    """Same synthetic guard, but ``-n 0`` collects cleanly (no RuntimeError)."""
    (tmp_path / "conftest.py").write_text(_SYNTHETIC_GUARD, encoding="utf-8")
    (tmp_path / "test_dummy.py").write_text(_SYNTHETIC_DUMMY_TEST, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(_SYNTHETIC_PYPROJECT, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-m",
            "coordinator_recovery",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            str(tmp_path),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    # -n 0 disables xdist parallelism → guard's worker_count <= 1 → no raise.
    # pytest exit codes: 0 = collected+selected, 5 = none collected/selected.
    assert result.returncode in (0, 5), (
        f"Expected clean collection under -n 0 (returncode 0 or 5), got "
        f"{result.returncode}.\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert _XDIST_MSG_NEEDLE not in combined, (
        f"The xdist-incompatibility guard must NOT fire under -n 0, but its "
        f"message appeared.\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_n2_not_coordinator_recovery_does_not_trip_guard(tmp_path):
    """codex R3-F2 (HIGH) regression — ``-n2 -m "not coordinator_recovery"`` is a
    legitimate parallel non-recovery run and must NOT trip the guard.

    The recovery-marked ``test_x`` is deselected by ``-m "not
    coordinator_recovery"``; only ``test_plain`` survives. Because the guard is
    decorated ``@pytest.hookimpl(trylast=True)`` it runs AFTER pytest's built-in
    markexpr deselection, so by the time it inspects ``items`` the recovery item
    is gone and the guard does not fire. Before the fix the guard ran with the
    recovery item still present and raised ``RuntimeError`` here (an
    INTERNALERROR / non-zero exit with ``_XDIST_MSG_NEEDLE``).
    """
    (tmp_path / "conftest.py").write_text(_SYNTHETIC_GUARD, encoding="utf-8")
    (tmp_path / "test_dummy.py").write_text(_SYNTHETIC_DUMMY_TEST, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(_SYNTHETIC_PYPROJECT, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "2",
            "-m",
            "not coordinator_recovery",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            str(tmp_path),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    # Recovery item deselected → only test_plain collected+selected → exit 0.
    # (5 = none-selected would also be acceptable, but test_plain DOES select.)
    assert result.returncode in (0, 5), (
        f"Expected clean collection for -n2 -m 'not coordinator_recovery' "
        f"(returncode 0 or 5), got {result.returncode} — the guard falsely "
        f"tripped on a deselected recovery item.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert _XDIST_MSG_NEEDLE not in combined, (
        f"The xdist guard must NOT fire under -n2 when coordinator_recovery is "
        f"DESELECTED (the surviving items will never run the shared-state "
        f"harness), but its message appeared.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
