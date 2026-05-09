"""B3-core PR-0: Anchor count gate.

Asserts pytest collection finds exactly 34 anchor xfail tests across the
B3-core supervisor anchor suite.  PR-1..PR-4 must NOT change this count;
they only flip ``xfail`` → ``xpass``.

Spec basis: docs/superpowers/specs/2026-05-07-b3-core-design.md §8.1
Plan basis: docs/superpowers/plans/2026-05-07-b3-core-pr0-plan.md (Task 2).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


B3_CORE_ANCHOR_FILES = [
    # Round-3 audit P1#1 fix: db-bound anchors moved to integration tree
    # (per spec §8.4 — fixtures depending on db_session live in
    # api/tests/integration/conftest.py, so anchors that use them must
    # live alongside, not in the app tree).
    "api/tests/integration/test_supervisor_contract.py",
    "api/tests/integration/test_supervisor_pg_invariants.py",
    "api/tests/integration/test_supervisor_wire_contract.py",
]


def _find_repo_root() -> Path:
    """Walk up until the current checkout root is reached (P3-1 fix).

    Prefer the git marker so nested worktrees under another checkout do not
    accidentally resolve to the outer repository's CLAUDE.md.
    """
    cur = Path(__file__).resolve()
    while (
        cur.parent != cur
        and not (cur / ".git").exists()
        and not (cur / "CLAUDE.md").exists()
    ):
        cur = cur.parent
    return cur


def test_b3_core_anchor_count_is_34():
    """Pytest collection across the 3 anchor files must find exactly 34 xfail tests."""
    repo_root = _find_repo_root()
    cmd = [
        "uv", "run", "pytest",
        *B3_CORE_ANCHOR_FILES,
        "--collect-only", "-q",
        "--no-header",
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 5):
        pytest.fail(f"pytest collection failed: stderr={result.stderr}")
    # pytest 8.x --collect-only tree output uses `<Function test_...>` format,
    # not flat `path::test_name`.  Parse the trailing "N tests collected" summary line.
    import re
    match = re.search(r"(\d+)\s+tests?\s+collected", result.stdout)
    assert match is not None, (
        f"Could not parse 'N tests collected' from pytest stdout.\n"
        f"Collection output:\n{result.stdout}"
    )
    collected = int(match.group(1))
    assert collected == 34, (
        f"Expected exactly 34 anchor tests; got {collected}.\n"
        f"Anchor file list may have drifted from spec §8.1.\n"
        f"Collection output tail:\n{result.stdout[-500:]}"
    )


def test_anchor_files_have_expected_xfail_decorator_count():
    """Round-3 audit P1#3 fix: count `@pytest.mark.xfail` decorators across the
    3 anchor files; assert the per-PR expected count.

    Prior gate only verified ``"pytest.mark.xfail" in content`` (substring match
    once per file), which would PASS even if 33 of 34 xfails were silently
    deleted.  This stricter count locks the contract: any future PR that flips
    a single xfail → xpass MUST remove that decorator AND simultaneously update
    the EXPECTED_XFAIL_COUNT below — keeping the drift consciously visible.

    Per-PR ledger (anchors flipped from xfail → unmarked):
      - PR-0 ship: 34 xfail decorators, 0 anchors flipped.
      - PR-1 ship: 31 xfail decorators, 3 anchors flipped
        (C-Wire-1, C-Wire-4, C-Redis-1).
      - PR-2 ship: 13 xfail decorators, 21 anchors flipped
        (C-PG-1..5, C-FSM-1..3, C-Admission-1..2, C-Lua-*,
         C-Restart-1/2/NEW, C-Repo-*, C-Redis-2).
      - PR-3a ship: 11 xfail decorators, 23 anchors flipped
        (C-FINISHING-1, C-Callback-Compose).
      - PR-3b ship: 11 xfail decorators, 23 anchors flipped (no anchor flip).
      - PR-3c ship: 8 xfail decorators, 26 anchors flipped
        (C-Cancel-1, C-Auth-1, C-MultiTab-1).
      - PR-4 ship: TBD

    Decorator forms covered:
      - ``@pytest.mark.xfail(strict=False, reason=...)`` (canonical)
      - ``@pytest.mark.xfail(...)``  (any form starting with ``@pytest.mark.xfail``)
    """
    import re

    # PR-3c ship: 8 = 34 - 26 flipped anchors.
    EXPECTED_XFAIL_COUNT = 8

    repo_root = _find_repo_root()
    total = 0
    per_file: dict[str, int] = {}
    for relpath in B3_CORE_ANCHOR_FILES:
        path = repo_root / relpath
        assert path.exists(), f"Anchor file missing: {relpath}"
        content = path.read_text()
        # Match `@pytest.mark.xfail` at the start of a line (allowing leading whitespace).
        count = len(re.findall(r"^\s*@pytest\.mark\.xfail\b", content, re.MULTILINE))
        per_file[relpath] = count
        total += count

    assert total == EXPECTED_XFAIL_COUNT, (
        f"Expected exactly {EXPECTED_XFAIL_COUNT} @pytest.mark.xfail decorators across anchor files; "
        f"got {total}.\n"
        f"Per-file breakdown: {per_file}\n"
        f"If your PR flipped an anchor, update EXPECTED_XFAIL_COUNT and document in the per-PR ledger.\n"
        f"Spec §8.1 anchor groups (collection count must stay at 34): "
        f"5 PG + 3 FSM + 2 Admission + 2 Lua + 3 Restart + 2 Repo + 4 Wire + 2 Redis + 1 FINISHING + "
        f"1 Callback + 2 Inflight + 1 Cancel + 1 Auth + 1 MultiTab + 4 Notif."
    )
