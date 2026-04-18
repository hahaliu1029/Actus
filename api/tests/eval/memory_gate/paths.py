"""M2-PR1: path helpers for the memory gate eval harness.

Defines where the two datasets live:

- **Public (synthetic)** — ``api/tests/eval/memory_gate/synthetic/`` —
  checked into the repo. 25 samples today; M2 scales to 100 with
  adversarial suites.
- **Private (real)** — ``${ACTUS_EVAL_DATA_DIR}/memory_gate/real/`` with
  a ``~/.actus/eval/memory_gate/real/`` default. NOT checked in. Solo
  author scrapes their own dev sessions, redacts with ``detect-secrets``
  + manual pass, stores locally.

Reason for separating: synthetic data can go in a public repo but isn't
representative of real user dialog; private data is real but contains
PII / project secrets. The M2 design requires running eval on the union
so point precision reflects real distribution, while CI (and other
contributors) can still run the public subset.

All paths are resolved lazily via functions — no Path constants at
import time — so tests that set env vars via ``monkeypatch`` see the
right values.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---- Constants --------------------------------------------------------- #


_MODULE_DIR = Path(__file__).resolve().parent
"""``api/tests/eval/memory_gate/`` — the package that owns this file."""

_DEFAULT_PRIVATE_ROOT = Path.home() / ".actus" / "eval"
"""Default private-dataset root if ``ACTUS_EVAL_DATA_DIR`` is unset.

Mirrors the ``MEMORY_ROOT_HOST=~/.actus/memory`` convention established
by M1 — data the user-facing solo deployment accumulates sits under
``~/.actus/`` by default.
"""

_ENV_VAR = "ACTUS_EVAL_DATA_DIR"
"""Env var that overrides the private-dataset root. Set by the solo
author running eval locally; left unset in CI and on other contributors'
machines (which still run the public synthetic suite)."""


# ---- Public dataset ---------------------------------------------------- #


def synthetic_dir() -> Path:
    """Return ``api/tests/eval/memory_gate/synthetic/``.

    Always exists (checked into the repo). The first dataset file lives
    at ``synthetic/dataset.jsonl``.
    """
    return _MODULE_DIR / "synthetic"


def synthetic_dataset_path() -> Path:
    """Return the canonical synthetic dataset path.

    M1 ships 25 samples; M2 PR-5 expanded to 45 core (original 25 + 20
    paraphrases). Adversarial suites live in sibling files; see
    ``synthetic_adversarial_path``.
    """
    return synthetic_dir() / "dataset.jsonl"


def synthetic_adversarial_path() -> Path:
    """Return the canonical adversarial dataset path.

    M2 PR-5 adds this file to probe gate robustness against patterns
    that *look* memory-worthy but the rubric says to drop (or vice-versa
    on a small borderline-keep subset). Scoring is reported separately
    from the core dataset so a precision dip in adversarial doesn't
    swamp the Wilson CI on core.

    Kept under ``synthetic/`` because the content is hand-crafted and
    safe to publish; private real adversarial samples (if any) would
    live under the private root using the same filename convention.
    """
    return synthetic_dir() / "adversarial.jsonl"


# ---- Private dataset --------------------------------------------------- #


def resolve_private_root() -> Path:
    """Return the private-dataset root directory.

    Resolution order:
    1. ``${ACTUS_EVAL_DATA_DIR}`` env var if set and non-empty
    2. ``~/.actus/eval/`` (default)

    Does NOT create the directory. Callers should check ``exists()``
    and skip the private suite when absent.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_PRIVATE_ROOT.expanduser().resolve()


def private_dir() -> Path:
    """Return ``<private_root>/memory_gate/real/``.

    The M2 design groups by eval suite under the private root so future
    suites (retrieval reranker, auto-promote tuning) can live alongside
    without colliding.
    """
    return resolve_private_root() / "memory_gate" / "real"


def private_dataset_path() -> Path:
    """Return the canonical private dataset file path.

    Callers must check ``exists()`` before opening: in CI / on
    contributor machines this file will be absent.
    """
    return private_dir() / "dataset.jsonl"


def control_set_path() -> Path:
    """Return the path to the 30-sample annotator-drift control set.

    Per M2 design: 30 samples pulled from the private set, re-labeled
    weekly by the solo author, Cohen's kappa monitored for drift. If
    the kappa drops below 0.7 the rubric needs to be tightened.

    Lives under the private root (annotator labels are PII-adjacent).
    """
    return private_dir() / "control_set.jsonl"


# ---- Discovery --------------------------------------------------------- #


def private_available() -> bool:
    """True if the private dataset file exists on the local machine.

    Cheap check: the eval harness calls this to decide whether to run
    the union suite (public + private) or the public-only suite. Same
    value on subsequent calls unless the file is created/removed.
    """
    return private_dataset_path().is_file()


def synthetic_adversarial_available() -> bool:
    """True if the synthetic adversarial dataset file exists.

    Currently always True (checked into the repo), but callers that
    want to gracefully degrade if someone deletes the file locally
    (e.g. during bisection) can use this check instead of a bare
    ``is_file()``.
    """
    return synthetic_adversarial_path().is_file()


def control_set_available() -> bool:
    """True if the control set file exists on the local machine."""
    return control_set_path().is_file()
