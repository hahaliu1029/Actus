"""B5 PR-S1-7b: forbid ``logging.basicConfig(...)`` in the backend.

Rationale (from design doc Sprint 1 §"Lint Gates"):

- ``logging.basicConfig`` configures the root logger ONCE on first
  call and is otherwise a NoOp; using it in a service entrypoint
  silently wins the race against ``setup_logging`` and bypasses the
  ``RedactingFormatter`` + LogRecord factory + Q2 self-heal.
- PR-S1-4 + PR-S1-7a migrated every existing ``basicConfig`` to
  ``setup_logging`` / ``setup_cli_logging``. This gate prevents
  regressions from sneaking the pattern back via a future PR.

CLI usage::

    python api/tools/lint/no_logging_basicconfig.py [PATHS...]

When no ``PATHS`` are given, scans the canonical backend tree
(``api/app``, ``api/scripts``, ``api/tools``). Exits ``0`` on
clean, ``1`` on violation; emits ``file:line:col`` diagnostics to
``stderr``.
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_DEFAULT_TARGETS: tuple[str, ...] = (
    "api/app",
    "api/scripts",
    "api/tools",
)


@dataclass(frozen=True)
class Violation:
    """A single ``logging.basicConfig`` call site."""

    path: Path
    line: int
    col: int

    def format_diagnostic(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: logging.basicConfig is forbidden"


class _BasicConfigVisitor(ast.NodeVisitor):
    """Track every shape of ``logging.basicConfig`` call in a single file.

    Review-found P2: a Name-based AST match that hard-codes the
    callee as ``logging.basicConfig`` misses every aliased form a
    future PR could use to bypass the gate. This visitor instead
    tracks the per-file import bindings:

    - ``logging_aliases`` collects every local name bound to the
      ``logging`` module — ``import logging`` (binds ``logging``),
      ``import logging as X`` (binds ``X``), and the parent-binding
      side-effect of ``import logging.config`` (binds ``logging``
      again, so ``logging.basicConfig()`` stays catchable).
      ``import logging.config as lc`` is intentionally NOT tracked:
      ``lc`` aliases the submodule, and ``lc.basicConfig()`` would
      not be a real Python expression.
    - ``basicconfig_names`` collects every name bound directly to
      ``logging.basicConfig`` —
      ``from logging import basicConfig`` (binds ``basicConfig``),
      ``from logging import basicConfig as bc`` (binds ``bc``).

    Calls are flagged when:

    - they are ``Attribute(Name(<x>), "basicConfig")`` and ``<x>``
      is in ``logging_aliases``; OR
    - they are ``Name(<x>)`` and ``<x>`` is in
      ``basicconfig_names``.

    Limitations: variable assignment of the form
    ``bc = logging.basicConfig`` slips past (would require dataflow
    tracking), and runtime ``getattr`` reflection is unreachable
    via static AST. Both are escape hatches; reviewer's findings
    target the common alias forms.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.logging_aliases: set[str] = set()
        self.basicconfig_names: set[str] = set()
        self.violations: list[Violation] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "logging":
                # ``import logging``        → binds ``logging``
                # ``import logging as X``  → binds ``X``
                self.logging_aliases.add(alias.asname or "logging")
            elif alias.name.startswith("logging.") and alias.asname is None:
                # ``import logging.config`` (no asname) ALSO binds
                # ``logging`` at the local scope, so
                # ``logging.basicConfig()`` is reachable.
                self.logging_aliases.add("logging")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "logging":
            for alias in node.names:
                if alias.name == "basicConfig":
                    self.basicconfig_names.add(alias.asname or "basicConfig")
                elif alias.name == "*":
                    # Review-found P2: ``from logging import *`` also
                    # binds ``basicConfig`` at module scope. Without
                    # this branch the star-import form would slip past
                    # the gate (and Actus has no F403 wildcard-import
                    # ban as a fallback). The asname is always None on
                    # a star import, so we register the canonical
                    # name directly.
                    self.basicconfig_names.add("basicConfig")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        flagged = False
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "basicConfig":
            value = func.value
            if isinstance(value, ast.Name) and value.id in self.logging_aliases:
                flagged = True
        elif isinstance(func, ast.Name) and func.id in self.basicconfig_names:
            flagged = True
        if flagged:
            self.violations.append(
                Violation(path=self.path, line=node.lineno, col=node.col_offset)
            )
        self.generic_visit(node)


def find_violations_in_file(path: Path) -> list[Violation]:
    """Parse ``path`` and return every ``logging.basicConfig`` call site.

    Catches all of:

    - ``import logging`` + ``logging.basicConfig(...)``
    - ``import logging as log`` + ``log.basicConfig(...)``
    - ``import logging.config`` + ``logging.basicConfig(...)``
    - ``from logging import basicConfig`` + ``basicConfig(...)``
    - ``from logging import basicConfig as bc`` + ``bc(...)``
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # A syntax error is its own problem; not this gate's job to
        # report. Return clean so we don't double-report.
        return []
    visitor = _BasicConfigVisitor(path=path)
    visitor.visit(tree)
    return visitor.violations


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield every ``*.py`` file under ``roots`` (recursive, sorted)."""
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for path in sorted(root.rglob("*.py")):
            # Skip caches and venvs that may have been bind-mounted in.
            parts = set(path.parts)
            if "__pycache__" in parts or ".venv" in parts or "venv" in parts:
                continue
            yield path


def find_violations(roots: Iterable[Path]) -> list[Violation]:
    """Aggregate violations across every ``.py`` file under ``roots``."""
    out: list[Violation] = []
    for path in _iter_python_files(roots):
        out.extend(find_violations_in_file(path))
    return out


def _resolve_targets(paths: list[str], repo_root: Path) -> list[Path]:
    """Resolve CLI path arguments (or defaults) into absolute paths."""
    if not paths:
        return [repo_root / target for target in _DEFAULT_TARGETS]
    return [Path(p).resolve() for p in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject logging.basicConfig() in the Actus backend.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Files or directories to scan. Defaults to api/app, "
            "api/scripts, api/tools (resolved from the script's "
            "working directory)."
        ),
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    targets = _resolve_targets(args.paths, repo_root)
    violations = find_violations(targets)

    if not violations:
        return 0

    sys.stderr.write(
        f"no_logging_basicconfig: {len(violations)} violation(s) found:\n"
    )
    for v in violations:
        sys.stderr.write("  " + v.format_diagnostic() + "\n")
    sys.stderr.write(
        "\nUse setup_logging() (FastAPI) or setup_cli_logging() (CLI) "
        "from app.infrastructure.logging instead.\n"
    )
    sys.stderr.flush()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
