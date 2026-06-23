"""[S2 PR-6] Fast guard: `.gitignore` must NARROW the blanket `docs/` ignore so
the checked-in shell-mode contributor doc (`docs/shell-mode/c2-shell-mode.md`)
is tracked while every other `docs/` path stays ignored.

This proves the intended ignore rules are in place WITHOUT invoking git (the
plan forbids any git command as an executable step). It reads `.gitignore` as
TEXT and asserts the four narrowing lines plus the removal of the blanket
`docs/` (alone). The git-ignore *negation semantics* this encodes:
git cannot re-include a file whose parent directory was pruned by a blanket
`docs/`, so we ignore directory CONTENTS (`docs/*`), then re-include the subtree
in order — `!docs/shell-mode/` (descend), `docs/shell-mode/*` (re-ignore inside),
`!docs/shell-mode/c2-shell-mode.md` (re-include the one tracked doc).

Colocated under `api/tests/structure/` so it reuses that dir's `parents[3]`
repo-root idiom (matching `test_coordinator_e2e_unskipped.py:19` and
`test_tool_node_calls_ast_validator_for_shell.py:18`): from
`api/tests/structure/<this file>` → `parents[3]` is the repo root, then
`.gitignore` at the root. Runs in the default `cd api && uv run pytest`
(no DB/Redis/sandbox) — it only parses text.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.structure]

# api/tests/structure/<file> -> .../structure -> .../tests -> api -> repo root
_GITIGNORE = Path(__file__).resolve().parents[3] / ".gitignore"

# The four narrowing lines the .gitignore edit must add (Step 3a), in order.
_REQUIRED_IN_ORDER = (
    "docs/*",
    "!docs/shell-mode/",
    "docs/shell-mode/*",
    "!docs/shell-mode/c2-shell-mode.md",
)


def test_shell_mode_doc_gitignore_narrowing_present() -> None:
    assert _GITIGNORE.exists(), f".gitignore missing at {_GITIGNORE}"
    raw = _GITIGNORE.read_text(encoding="utf-8")
    # Compare on stripped, non-comment, non-blank lines (ignore-rule semantics
    # are whitespace/comment-insensitive).
    lines = [
        ln.strip()
        for ln in raw.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]

    # 1) Every narrowing line must be present.
    for needed in _REQUIRED_IN_ORDER:
        assert needed in lines, (
            f"`.gitignore` is missing the narrowing rule {needed!r}; the "
            "checked-in shell-mode doc would be silently dropped from the PR. "
            "See Task 6.5 Step 3a."
        )

    # 2) They must appear in the right RELATIVE order — git applies last-match-
    #    wins, and a file under a pruned dir cannot be re-included, so the dir
    #    re-include MUST precede its inner re-ignore which MUST precede the file
    #    re-include.
    positions = [lines.index(n) for n in _REQUIRED_IN_ORDER]
    assert positions == sorted(positions), (
        "the `.gitignore` narrowing rules are out of order; required order is "
        f"{_REQUIRED_IN_ORDER!r}. git cannot re-include a file whose parent "
        "directory is still excluded, so `!docs/shell-mode/` must come before "
        "`docs/shell-mode/*` which must come before "
        "`!docs/shell-mode/c2-shell-mode.md`."
    )

    # 3) The blanket `docs/` (alone) must be GONE — if it survives, git prunes
    #    the whole tree and the nested re-include is dead.
    assert "docs/" not in lines, (
        "the blanket `docs/` ignore must be REPLACED by `docs/*` + the nested "
        "re-include block; while a bare `docs/` line remains, git never "
        "descends into docs/shell-mode/ and the doc stays ignored."
    )
