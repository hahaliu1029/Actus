"""M2 PR-5: annotator drift detection via weekly control-set re-labelling.

Design (rubric §111-127):

- The solo author carves **30 stable samples** off the private real
  dataset into ``${ACTUS_EVAL_DATA_DIR}/memory_gate/real/control_set.jsonl``.
- Once a week, the author re-labels those 30 samples **from fresh
  judgement** (no reference to the previous week's labels).
- The week-over-week Cohen's kappa is the drift signal:
  - kappa ≥ 0.7 → rubric is stable
  - 0.4 ≤ kappa < 0.7 → tighten the worst-drifting sample's rubric line
  - kappa < 0.4 → rubric is ambiguous; split categories or rewrite

Weekly re-labels live in a sibling directory
``${ACTUS_EVAL_DATA_DIR}/memory_gate/real/control_set_history/`` with
one file per pass named by ISO date (``2026-04-20.jsonl``, etc.). The
newest two files define "this week vs last week" for the kappa gate.

This module ships three test functions, each with its own marking so
the slow/fast split is explicit (the whole module used to be slow-marked,
but that deselected even the pure unit test for the filename filter).

Contributors / CI machines without the private data populated see
pytest skips, not failures. Think of the slow tests as the "first run
after a relabel" the author runs weekly.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pytest

from tests.eval.memory_gate.paths import (
    control_set_available,
    control_set_path,
)
from tests.eval.memory_gate.stats import cohens_kappa


_HISTORY_DIR_NAME = "control_set_history"
"""Sibling directory next to ``control_set.jsonl`` holding weekly
re-labelled passes. One file per pass, named by ISO date so ``sorted()``
gives chronological order. See module docstring for rationale."""


_HISTORY_FILE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")
"""Filename regex enforced by ``_list_history_files``. Any ``.jsonl``
file in the history directory must match this pattern — enforcement
prevents two distinct failure modes:

1. Wrong-pair selection (original motivation): ``week-12.jsonl`` could
   sort before or after ``2026-04-27.jsonl`` depending on lexical rules,
   so a naive ``sorted()[-2:]`` might pick the wrong pair.
2. Stale-pass regression (round-2 codex finding): if THIS week's file
   is accidentally named ``week-18.jsonl``, a silent filter would drop
   it and kappa would compare last-week vs two-weeks-ago — an obviously
   stale result masquerading as "drift unchanged". Raising loudly
   surfaces the misname before the stale kappa can be reported.

Non-``.jsonl`` files (README.md, .txt drafts) are fine and silently
excluded — the glob pattern is the first filter, so only ``.jsonl``
files are candidates for ISO-name enforcement."""


class HistoryFilenameError(ValueError):
    """Raised when the history dir contains ``.jsonl`` files with
    non-ISO names. See ``_HISTORY_FILE_PATTERN`` docstring for rationale.

    Lives as a named subclass of ``ValueError`` so call sites can
    ``pytest.raises(HistoryFilenameError)`` without tangling with
    other ``ValueError``-raising code paths.
    """


def _control_set_history_dir() -> Path:
    """Return the history directory path without creating it.

    Lives next to the canonical control_set.jsonl because both files
    move together when the author relocates ``ACTUS_EVAL_DATA_DIR``.
    """
    return control_set_path().parent / _HISTORY_DIR_NAME


def _list_history_files() -> list[Path]:
    """Return ISO-date-named history files sorted lexicographically.

    Any ``.jsonl`` file in the directory MUST match
    ``YYYY-MM-DD.jsonl``. Non-matching ``.jsonl`` files raise
    ``HistoryFilenameError`` so a mis-named current-week pass gets
    surfaced immediately — silent filtering would produce a stale
    kappa reading from older correctly-named files.

    Non-``.jsonl`` files (README.md, text drafts) are silently excluded
    by the glob pattern — they're clearly not history.

    Returns empty list if the directory does not exist. Callers decide
    whether to skip.
    """
    d = _control_set_history_dir()
    if not d.is_dir():
        return []
    all_jsonl = sorted(p for p in d.glob("*.jsonl") if p.is_file())
    invalid = [p for p in all_jsonl if not _HISTORY_FILE_PATTERN.match(p.name)]
    if invalid:
        raise HistoryFilenameError(
            f"control_set_history/ contains {len(invalid)} .jsonl file(s) "
            f"with non-ISO names: {[p.name for p in invalid]}. Expected "
            f"YYYY-MM-DD.jsonl (e.g., 2026-04-20.jsonl). Rename or move "
            f"these files — silent filtering would compare stale historic "
            f"pairs and hide the latest drift."
        )
    return all_jsonl


def _load_labels_by_id(path: Path) -> dict[str, str]:
    """Return ``{sample_id: expected_verdict}`` parsed from one pass file.

    Kappa is computed on the verdict label (keep/drop) — not category —
    because that is the agent-facing decision that drives gate behavior.
    Category drift can be meaningful but is a second-order signal; if
    the verdict is stable, the category can be tightened separately.
    """
    labels: dict[str, str] = {}
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = str(obj["id"])
            verdict = str(obj["expected_verdict"])
            labels[sample_id] = verdict
    return labels


def test_list_history_files_accepts_iso_dates_and_skips_non_jsonl(
    monkeypatch, tmp_path: Path
) -> None:
    """Non-``.jsonl`` files (README.md, .txt drafts) are silently
    excluded — the glob is the first filter.

    Setup:
    - ``2026-04-20.jsonl`` — valid ISO date
    - ``2026-04-27.jsonl`` — valid ISO date
    - ``2026-05-03.txt`` — wrong extension
    - ``README.md`` — unrelated file

    Result: exactly two ISO-date files, sorted ascending. Non-jsonl
    files don't exist in the candidate set.
    """
    monkeypatch.setenv("ACTUS_EVAL_DATA_DIR", str(tmp_path))
    d = tmp_path / "memory_gate" / "real" / "control_set_history"
    d.mkdir(parents=True)
    (d / "2026-04-20.jsonl").touch()
    (d / "2026-04-27.jsonl").touch()
    (d / "2026-05-03.txt").touch()
    (d / "README.md").touch()

    files = _list_history_files()
    assert [p.name for p in files] == [
        "2026-04-20.jsonl",
        "2026-04-27.jsonl",
    ]


def test_list_history_files_raises_on_non_iso_jsonl(
    monkeypatch, tmp_path: Path
) -> None:
    """Round-2 codex finding: if current-week file is misnamed (e.g.
    ``week-18.jsonl``), a silent filter would drop it and leave the
    kappa computation picking older correctly-named files — a stale
    "drift unchanged" result that hides today's actual drift.

    Guarantee: any ``.jsonl`` file that doesn't match ``YYYY-MM-DD.jsonl``
    triggers ``HistoryFilenameError``, with the offending filename in
    the message so the author can fix it immediately.
    """
    monkeypatch.setenv("ACTUS_EVAL_DATA_DIR", str(tmp_path))
    d = tmp_path / "memory_gate" / "real" / "control_set_history"
    d.mkdir(parents=True)
    (d / "2026-04-20.jsonl").touch()
    (d / "2026-04-27.jsonl").touch()
    (d / "week-18.jsonl").touch()  # misnamed current-week file

    with pytest.raises(HistoryFilenameError, match=r"week-18\.jsonl"):
        _list_history_files()


def test_list_history_files_empty_when_dir_missing(
    monkeypatch, tmp_path: Path
) -> None:
    """Contract: caller can rely on empty-list when history dir isn't
    carved out yet. Driven by the common path — first-week author has
    no prior passes to compare against."""
    monkeypatch.setenv("ACTUS_EVAL_DATA_DIR", str(tmp_path / "not-there"))
    assert _list_history_files() == []


@pytest.mark.slow
def test_control_set_present() -> None:
    """Infrastructure smoke test — documents the expected file layout.

    Skips (not fails) if the control set isn't populated, so CI /
    contributor machines see a clear signal rather than a cryptic
    FileNotFoundError.
    """
    if not control_set_available():
        pytest.skip(
            f"control set not populated at {control_set_path()}; "
            f"see rubric §111-127 for the carve-off workflow."
        )
    labels = _load_labels_by_id(control_set_path())
    assert len(labels) >= 20, (
        f"control set shrunk unexpectedly: {len(labels)} rows. "
        f"Design target is 30; below 20 makes kappa noisy."
    )


@pytest.mark.slow
def test_weekly_kappa_clears_drift_gate() -> None:
    """Run ONLY when 2+ history passes exist — asserts kappa >= 0.4.

    Rubric §115-127 three-archiv gate:

    - kappa ≥ 0.7 → rubric stable, test silently passes
    - 0.4 ≤ kappa < 0.7 → emits a ``UserWarning`` so pytest surfaces it
      in the ``-W`` / ``-r`` summary; test still passes because this is
      "action item, not ship blocker"
    - kappa < 0.4 → assertion fails; rubric is too ambiguous to defend

    Using ``warnings.warn`` instead of plain ``print`` because pytest's
    default stdout capture hides prints outside of failures; UserWarning
    shows up in the standard warnings summary and is greppable in
    CI logs.
    """
    history = _list_history_files()
    if len(history) < 2:
        pytest.skip(
            f"need ≥ 2 weekly passes under {_control_set_history_dir()} "
            f"to compute drift kappa; found {len(history)}."
        )

    prev_path, curr_path = history[-2], history[-1]
    prev = _load_labels_by_id(prev_path)
    curr = _load_labels_by_id(curr_path)

    # Align by id so re-ordering (e.g., author appended new samples at
    # end) doesn't break the kappa computation. Only use ids present in
    # BOTH passes — added samples in the newer pass will show up first
    # in next week's kappa once the current pass becomes the "previous".
    shared_ids = sorted(set(prev) & set(curr))
    if len(shared_ids) < 15:
        pytest.skip(
            f"only {len(shared_ids)} ids shared between "
            f"{prev_path.name} and {curr_path.name}; kappa needs more "
            f"overlap to be meaningful."
        )

    prev_labels = [prev[sid] for sid in shared_ids]
    curr_labels = [curr[sid] for sid in shared_ids]
    kappa = cohens_kappa(prev_labels, curr_labels)

    print("\n" + "=" * 60)
    print(f"annotator drift: {prev_path.name} → {curr_path.name}")
    print(f"  shared samples = {len(shared_ids)}")
    print(f"  Cohen's kappa  = {kappa:.3f}")
    print("  gates:")
    print("    kappa ≥ 0.70  — rubric stable")
    print("    0.40-0.70     — UserWarning emitted; tighten rubric")
    print("    kappa < 0.40  — rewrite rubric (FAILS)")
    print("=" * 60)

    if 0.4 <= kappa < 0.7:
        # Borderline: surface via warnings.warn so pytest's warnings
        # summary captures it. Plain print gets swallowed by default
        # stdout capture unless the test fails.
        warnings.warn(
            f"annotator drift kappa = {kappa:.3f} is borderline "
            f"(0.4-0.7) between {prev_path.name} and {curr_path.name}. "
            f"Inspect disagreements and tighten the weakest rubric line "
            f"before the next re-label pass.",
            UserWarning,
            stacklevel=2,
        )

    assert kappa >= 0.4, (
        f"annotator drift kappa = {kappa:.3f} < 0.40 — rubric is too "
        f"ambiguous to defend. Compare labels between {prev_path.name} "
        f"and {curr_path.name} to find the categories you flipped on, "
        f"then rewrite the ambiguous section of labelling_rubric.md."
    )
