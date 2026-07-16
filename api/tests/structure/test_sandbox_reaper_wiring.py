"""C2 cancel Part B — main.py wiring/order controller-verify (T8).

The full sweep is exercised in CI (live pg+sandbox). This structural test pins
the lifespan ordering: the leaked-sandbox reaper runs AFTER the C2b child-row
reaper and BEFORE the background skill scan (spec §3.5.3).
"""
from __future__ import annotations

import pathlib


def _main_src() -> str:
    return (
        pathlib.Path(__file__).resolve().parents[2]
        .joinpath("app", "main.py")
        .read_text()
    )


def test_sandbox_reaper_wired_after_child_row_reaper_before_skill_scan():
    src = _main_src()
    i_child = src.index("sweep_running_mailbox_children")
    # SPM Task 28 extracted the reaper into the module-scope helper
    # ``_run_terminal_sandbox_reaper_if_enabled``, so the sweep symbol now first
    # appears textually in that helper's import (BEFORE the lifespan body). Anchor
    # the ordering on the lifespan CALL SITE instead — the ``await``-prefix is
    # unique to the call site (the ``async def`` line has no ``await``).
    i_sandbox = src.index("await _run_terminal_sandbox_reaper_if_enabled(")
    i_skill = src.index("_background_skill_scan")
    assert i_child < i_sandbox < i_skill, (
        "sandbox_terminal_reaper must run after the C2b child-row reaper and "
        "before the background skill scan"
    )


def test_sandbox_reaper_block_is_best_effort_swallowed():
    src = _main_src()
    # Robust to comment-length changes: the swallow log lives inside the reaper
    # block, which sits between its first symbol and the background skill scan.
    i_sandbox = src.index("sweep_terminal_coordinator_active_sandboxes")
    i_skill = src.index("_background_skill_scan")
    block = src[i_sandbox:i_skill]
    assert "sandbox_reaper: sweep failed (swallowed)" in block


def test_sandbox_reaper_sweep_is_time_bounded():
    # codex final-audit P2: the sweep must be wrapped in asyncio.wait_for to bound
    # its async-cancellable portion (the total sweep budget). NOTE: this does NOT
    # fully prevent a hung Docker daemon from blocking startup — DockerSandbox's
    # Docker SDK calls are synchronous-on-loop; the complete fix (async-safe
    # DockerSandbox) is a deferred follow-up (see the main.py comment). This pins
    # only that the budget exists.
    src = _main_src()
    start = src.index("sweep_terminal_coordinator_active_sandboxes")  # import anchor
    end = src.index("sandbox_reaper: sweep failed (swallowed)")
    block = src[start:end]
    assert "asyncio.wait_for(" in block, (
        "the startup leaked-sandbox sweep must be wrapped in asyncio.wait_for to "
        "bound its async-cancellable portion (total sweep budget)"
    )
