"""B5 PR-S1-7a acceptance: ``setup_cli_logging`` package surface.

Locks the public-API contract of
``app.infrastructure.logging.setup_cli_logging`` so a future refactor
that forgets to re-export it from the package ``__init__`` cannot
silently break the two CLI entrypoints. The reviewer's P1 caught
exactly that regression — the helper lived in the inner module but
``__init__`` only exported ``setup_logging``, so
``from app.infrastructure.logging import setup_cli_logging`` raised
``ImportError`` whenever the CLIs reached their lazy-import block
(i.e., on every non-``--help`` invocation).

Pinned invariants:

- ``setup_cli_logging`` is reachable from the package root.
- It is the same callable object as
  ``app.infrastructure.logging.logging.setup_cli_logging`` (no shim
  wrapping that could drift).
- It appears in the package ``__all__`` (so ``from … import *``
  picks it up too).
- After invocation, the root logger has exactly one StreamHandler
  whose stream is ``sys.stderr`` — the whole point of the helper
  vs. ``setup_logging``.
- Re-invoking is idempotent (no handler stacking).
- The CLI modules' lazy-import statements actually resolve.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


def test_setup_cli_logging_reachable_from_package_root() -> None:
    """``from app.infrastructure.logging import setup_cli_logging`` works.

    Direct package-level import — exactly the form the CLI lazy
    blocks use. Pre-fix this raised ``ImportError`` because the
    package ``__init__`` only re-exported ``setup_logging``.
    """
    from app.infrastructure.logging import setup_cli_logging  # noqa: F401

    assert callable(setup_cli_logging)


def test_package_export_matches_inner_module() -> None:
    """The package-level symbol is the same object as the inner module's."""
    from app.infrastructure.logging import setup_cli_logging as pkg_export
    from app.infrastructure.logging.logging import (
        setup_cli_logging as inner_export,
    )

    assert pkg_export is inner_export


def test_package_all_lists_setup_cli_logging() -> None:
    """``__all__`` advertises the helper for ``from … import *`` consumers."""
    import app.infrastructure.logging as pkg

    assert "setup_cli_logging" in pkg.__all__
    # Sanity: ``setup_logging`` still exported (no regression on the
    # FastAPI-side helper).
    assert "setup_logging" in pkg.__all__


def test_setup_cli_logging_routes_to_stderr(
    isolated_root_logger: logging.Logger,
) -> None:
    """Reviewer P2 invariant: console handler stream is ``sys.stderr``."""
    from app.infrastructure.logging import setup_cli_logging

    setup_cli_logging()

    stream_handlers = [
        h
        for h in isolated_root_logger.handlers
        if isinstance(h, logging.StreamHandler)
    ]
    assert stream_handlers, "no StreamHandler installed"
    assert all(
        h.stream is sys.stderr for h in stream_handlers
    ), (
        "CLI logging routes to stdout — would pollute "
        "operator-facing JSON output"
    )


def test_setup_cli_logging_no_file_handlers(
    isolated_root_logger: logging.Logger,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI mode is short-lived; no rotating file handlers needed."""
    from app.infrastructure.logging import setup_cli_logging
    from concurrent_log_handler import ConcurrentRotatingFileHandler
    from core import config as config_mod
    from core.config import Settings

    # Override log_dir to a writable tmp_path so even if a misbehaving
    # implementation tried to install file handlers we'd see them
    # cleanly here (instead of failing on ``/app/data/logs``).
    original = config_mod.get_settings()
    overridden = Settings(
        log_dir=str(tmp_path),
        jwt_secret_key=original.jwt_secret_key,
        env=original.env,
        log_level=original.log_level,
    )
    monkeypatch.setattr(config_mod, "get_settings", lambda: overridden)
    monkeypatch.setattr(
        "app.infrastructure.logging.logging.get_settings",
        lambda: overridden,
    )

    setup_cli_logging()

    file_handlers = [
        h
        for h in isolated_root_logger.handlers
        if isinstance(h, ConcurrentRotatingFileHandler)
    ]
    assert not file_handlers, (
        "setup_cli_logging installed file handlers — should be stderr-only"
    )


def test_setup_cli_logging_is_idempotent(
    isolated_root_logger: logging.Logger,
) -> None:
    """Re-running the helper does not stack handlers."""
    from app.infrastructure.logging import setup_cli_logging

    setup_cli_logging()
    first = len(isolated_root_logger.handlers)
    setup_cli_logging()
    second = len(isolated_root_logger.handlers)
    setup_cli_logging()
    third = len(isolated_root_logger.handlers)

    assert first == second == third == 1, (
        f"handler count drifted across re-invocations: "
        f"{first} → {second} → {third}"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "app.cli.memory_reconcile",
    ],
)
def test_cli_modules_can_resolve_setup_cli_logging(module_name: str) -> None:
    """The lazy-import statement inside each CLI's ``main`` resolves.

    Pre-fix, this would have raised ``ImportError`` on any non-help
    invocation — argparse's ``--help`` exits before reaching the
    lazy import, so the package-export bug shipped to commit
    without breaking CI. This test exercises the import path the
    real run takes.
    """
    import importlib

    mod = importlib.import_module(module_name)
    # Directly resolve the same import the CLI's ``main`` does.
    from app.infrastructure.logging import setup_cli_logging

    assert callable(setup_cli_logging)
    # Sanity: the CLI module imports cleanly (would surface secondary
    # issues like circular imports introduced by the fix).
    assert hasattr(mod, "main")


def test_backfill_cli_not_in_setup_cli_logging_parametrize() -> None:
    """PE-4d3: the backfill CLI is deleted; it must not be import-resolved
    by the parametrize list (which calls importlib.import_module on each
    entry). A stale entry would turn the resolve test into ModuleNotFoundError.
    """
    import inspect

    src = inspect.getsource(test_cli_modules_can_resolve_setup_cli_logging)
    assert "app.cli.backfill_approval_grants" not in src, (
        "backfill_approval_grants CLI was deleted in PE-4d3; remove it from "
        "the module_name parametrize list in test_cli_modules_can_resolve_setup_cli_logging"
    )
