"""B5 PR-S1-7b: forbid bare ``print(...)`` in the backend.

Rationale (from design doc Sprint 1 §"Lint Gates" + Q4):

- ``print()`` writes uncontrolled bytes to stdout, bypassing the
  ``RedactingFormatter`` and the LogRecord factory; secrets in
  prod logs are real harm. Backend code must use ``logging`` for
  events and ``sys.stdout.write`` (or equivalent explicit API) for
  CLI tools whose stdout is the user interface.
- PR-S1-7a migrated every existing ``print()`` in ``api/app`` and
  ``api/scripts`` to either ``logger.info`` (category a) or
  ``sys.stdout.write`` (category b). This gate prevents
  regressions.

Allowlist policy (per spec §"Lint Gates"):

- Files under ``tests/`` or ``api/tools/`` are out of scope (the
  gate's own scaffolding + test suite are not subject to
  production constraints).
- Any file containing the marker ``# noqa: NO-PRINT`` (anywhere)
  is exempted at the file level — for surfaces where the
  operator-facing intent is unambiguous and using
  ``sys.stdout.write`` would obscure the call site (rare;
  preferred path is migration).
- The whitelist intentionally does **not** cover ``api/scripts/``
  as a directory blanket — every scripts/ file must be migrated
  individually so a future regression cannot ride in via "throw
  it in scripts/".

CLI usage::

    python api/tools/lint/no_print_in_backend.py [PATHS...]

When no ``PATHS`` are given, scans ``api/app`` + ``api/scripts``.
Exits ``0`` on clean, ``1`` on violation; emits ``file:line:col``
diagnostics to ``stderr``.
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
)
_ALLOWLIST_DIRS: tuple[str, ...] = (
    "tests",
    "api/tools",
)
_FILE_LEVEL_MARKER: str = "# noqa: NO-PRINT"


@dataclass(frozen=True)
class Violation:
    """A single ``print(...)`` call site."""

    path: Path
    line: int
    col: int

    def format_diagnostic(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: print() is forbidden in backend"


def _is_print_call(node: ast.Call) -> bool:
    """Return ``True`` if ``node`` is ``print(...)`` (bare name).

    Only matches the canonical bare-name form. ``foo.print(x)`` /
    ``builtins.print(x)`` slip past — they're rare in this codebase
    and a future hardening can extend the predicate.
    """
    func = node.func
    return isinstance(func, ast.Name) and func.id == "print"


def _is_file_allowlisted(source: str) -> bool:
    """Return ``True`` if ``source`` carries the file-level marker."""
    return _FILE_LEVEL_MARKER in source


def _is_path_allowlisted(path: Path) -> bool:
    """Return ``True`` if ``path`` is under an always-skipped directory."""
    parts = path.parts
    for skip in _ALLOWLIST_DIRS:
        skip_parts = tuple(Path(skip).parts)
        if not skip_parts:
            continue
        # Match anywhere in the path so ``tests/`` matches both
        # ``api/tests/...`` and a top-level ``tests/...``.
        for i in range(len(parts) - len(skip_parts) + 1):
            if parts[i : i + len(skip_parts)] == skip_parts:
                return True
    return False


def find_violations_in_file(path: Path) -> list[Violation]:
    """Parse ``path`` and return every bare ``print(...)`` call site.

    Returns empty list if the file is path-allowlisted, carries the
    file-level marker, fails to read, or fails to parse. Strings
    that *contain* the literal text ``"print("`` (e.g., the
    ``file_processors`` sandbox-script templates) are not
    matched — AST never traverses string contents.
    """
    if _is_path_allowlisted(path):
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if _is_file_allowlisted(source):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_print_call(node):
            violations.append(
                Violation(path=path, line=node.lineno, col=node.col_offset)
            )
    return violations


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield every ``*.py`` file under ``roots`` (recursive, sorted)."""
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for path in sorted(root.rglob("*.py")):
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
    if not paths:
        return [repo_root / target for target in _DEFAULT_TARGETS]
    return [Path(p).resolve() for p in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject bare print() in the Actus backend.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Files or directories to scan. Defaults to api/app + "
            "api/scripts. tests/ and api/tools/ are always skipped."
        ),
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    targets = _resolve_targets(args.paths, repo_root)
    violations = find_violations(targets)

    if not violations:
        return 0

    sys.stderr.write(
        f"no_print_in_backend: {len(violations)} violation(s) found:\n"
    )
    for v in violations:
        sys.stderr.write("  " + v.format_diagnostic() + "\n")
    sys.stderr.write(
        "\nUse logger.info(...) for runtime events or "
        "sys.stdout.write(...) for explicit operator-facing CLI output.\n"
        "If you have a legitimate operator stdout case, add the file-level\n"
        f"marker {_FILE_LEVEL_MARKER!r} (still NOT a license to bypass\n"
        "the lint gate in api/scripts/ — migrate to sys.stdout.write).\n"
    )
    sys.stderr.flush()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
