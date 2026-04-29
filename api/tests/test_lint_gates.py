"""B5 PR-S1-7b acceptance: AST lint gates behave correctly.

Two gates:

- ``no_logging_basicconfig`` rejects every ``logging.basicConfig(
  ...)`` call site in the canonical backend tree.
- ``no_print_in_backend`` rejects every bare ``print(...)`` call
  site in ``api/app`` + ``api/scripts``, with allowlists for
  ``tests/`` paths, ``api/tools/`` paths, and any file containing
  the marker ``# noqa: NO-PRINT``.

This suite drives each scanner against synthetic fixtures (so the
AST logic itself is trustworthy) AND against the real backend tree
(so a regression that lands a fresh ``print()`` / ``basicConfig``
in committed code fails CI loudly).
"""
from __future__ import annotations

from pathlib import Path

from tools.lint import (
    no_logging_basicconfig,
    no_print_in_backend,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# no_logging_basicconfig
# ---------------------------------------------------------------------------
class TestNoLoggingBasicConfig:
    def test_violation_is_caught(self, tmp_path: Path) -> None:
        offender = tmp_path / "offender.py"
        offender.write_text(
            "import logging\n\n"
            "def main() -> None:\n"
            "    logging.basicConfig(level=logging.INFO)\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        v = violations[0]
        assert v.path == offender
        assert v.line == 4

    def test_clean_file_no_false_positives(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.py"
        clean.write_text(
            "import logging\n\n"
            "logger = logging.getLogger(__name__)\n"
            "def f() -> None:\n"
            "    logger.info('ok')\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_setup_logging_call_not_matched(self, tmp_path: Path) -> None:
        """``setup_logging()`` (the canonical replacement) is not flagged."""
        clean = tmp_path / "uses_setup.py"
        clean.write_text(
            "from app.infrastructure.logging import setup_logging\n\n"
            "setup_logging()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_main_returns_zero_when_clean(self, tmp_path: Path) -> None:
        """CLI exit code is 0 when no violations."""
        clean = tmp_path / "clean.py"
        clean.write_text("x = 1\n", encoding="utf-8")

        rc = no_logging_basicconfig.main([str(tmp_path)])

        assert rc == 0

    def test_main_returns_one_on_violation(self, tmp_path: Path) -> None:
        """CLI exit code is 1 when at least one violation."""
        offender = tmp_path / "offender.py"
        offender.write_text(
            "import logging\nlogging.basicConfig()\n", encoding="utf-8"
        )

        rc = no_logging_basicconfig.main([str(tmp_path)])

        assert rc == 1

    def test_import_logging_as_alias_call_caught(self, tmp_path: Path) -> None:
        """Review-found P2: ``import logging as log; log.basicConfig()``."""
        offender = tmp_path / "aliased_module.py"
        offender.write_text(
            "import logging as log\n\n"
            "def main() -> None:\n"
            "    log.basicConfig(level=log.INFO)\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_from_logging_import_basicconfig_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2: ``from logging import basicConfig; basicConfig()``."""
        offender = tmp_path / "from_import.py"
        offender.write_text(
            "from logging import basicConfig\n\n"
            "def main() -> None:\n"
            "    basicConfig(level=20)\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_from_logging_import_basicconfig_as_alias_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2: ``from logging import basicConfig as bc; bc()``."""
        offender = tmp_path / "from_aliased.py"
        offender.write_text(
            "from logging import basicConfig as bc\n\n"
            "def main() -> None:\n"
            "    bc(level=20)\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_from_logging_star_import_caught(self, tmp_path: Path) -> None:
        """Review-found P2: ``from logging import *; basicConfig()`` slips past.

        ``from logging import *`` binds every public name from the
        ``logging`` module at module scope, including
        ``basicConfig``. Without star-aware handling, this form
        would silently bypass the gate even though it has the same
        runtime effect as the explicit-name import.
        """
        offender = tmp_path / "star_import.py"
        offender.write_text(
            "from logging import *\n\n"
            "def main() -> None:\n"
            "    basicConfig(level=20)\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_import_logging_dot_config_parent_binding_caught(
        self, tmp_path: Path
    ) -> None:
        """``import logging.config`` parent-binds ``logging`` for free.

        Plain ``import logging.config`` (no asname) makes
        ``logging.basicConfig()`` a real Python expression because
        the ``logging`` package itself is bound at the local scope
        as a side effect of the dotted import.
        """
        offender = tmp_path / "dotted_import.py"
        offender.write_text(
            "import logging.config\n\n"
            "def main() -> None:\n"
            "    logging.basicConfig(level=20)\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].line == 4

    def test_submodule_alias_not_misattributed_to_logging(
        self, tmp_path: Path
    ) -> None:
        """``import logging.config as lc`` does NOT bind ``logging``.

        The asname binds ``lc`` to the submodule
        ``logging.config``, which does not own ``basicConfig``;
        ``lc.basicConfig()`` is not a real Python expression and
        should not be flagged.
        """
        clean = tmp_path / "submodule_alias.py"
        clean.write_text(
            "import logging.config as lc\n\n"
            "def main() -> None:\n"
            "    lc.basicConfig(level=20)\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_other_module_basic_config_not_matched(
        self, tmp_path: Path
    ) -> None:
        """``from foo import basicConfig`` is NOT a logging.basicConfig call."""
        clean = tmp_path / "other_module.py"
        clean.write_text(
            "from foo import basicConfig\n\n"
            "def main() -> None:\n"
            "    basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_unrelated_attribute_method_not_matched(
        self, tmp_path: Path
    ) -> None:
        """``cfg.basicConfig()`` on a non-logging object is not a violation."""
        clean = tmp_path / "object_method.py"
        clean.write_text(
            "class Cfg:\n"
            "    def basicConfig(self) -> None:\n"
            "        pass\n\n"
            "Cfg().basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_real_backend_tree_is_clean(self) -> None:
        """Anti-regression: real ``api/app + api/scripts + api/tools`` is clean.

        Locks the post-PR-S1-7a contract; landing a fresh
        ``logging.basicConfig`` in the codebase fails this test.
        """
        targets = [
            _REPO_ROOT / "api/app",
            _REPO_ROOT / "api/scripts",
            _REPO_ROOT / "api/tools",
        ]
        violations = no_logging_basicconfig.find_violations(targets)
        assert violations == [], (
            "logging.basicConfig regression in committed code:\n  "
            + "\n  ".join(v.format_diagnostic() for v in violations)
        )


# ---------------------------------------------------------------------------
# no_print_in_backend
# ---------------------------------------------------------------------------
class TestNoPrintInBackend:
    def test_violation_is_caught(self, tmp_path: Path) -> None:
        offender = tmp_path / "offender.py"
        offender.write_text(
            "def main() -> None:\n    print('hi')\n",
            encoding="utf-8",
        )

        violations = no_print_in_backend.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 2

    def test_sys_stdout_write_not_matched(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.py"
        clean.write_text(
            "import sys\nsys.stdout.write('hi\\n')\n",
            encoding="utf-8",
        )

        violations = no_print_in_backend.find_violations([tmp_path])

        assert violations == []

    def test_logger_info_not_matched(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.py"
        clean.write_text(
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
            "logger.info('hi')\n",
            encoding="utf-8",
        )

        violations = no_print_in_backend.find_violations([tmp_path])

        assert violations == []

    def test_file_level_marker_exempts_file(self, tmp_path: Path) -> None:
        """A file with ``# noqa: NO-PRINT`` is fully allowlisted."""
        exempt = tmp_path / "operator.py"
        exempt.write_text(
            "# noqa: NO-PRINT — intentional operator stdout\n"
            "def main() -> None:\n"
            "    print('operator status')\n",
            encoding="utf-8",
        )

        violations = no_print_in_backend.find_violations([tmp_path])

        assert violations == []

    def test_marker_in_docstring_still_exempts(self, tmp_path: Path) -> None:
        """The marker is matched anywhere in the file (docstring counts)."""
        exempt = tmp_path / "with_doc.py"
        exempt.write_text(
            '"""Docs about the file.\n\n# noqa: NO-PRINT — see bottom\n"""\n\n'
            "print('hi')\n",
            encoding="utf-8",
        )

        violations = no_print_in_backend.find_violations([tmp_path])

        assert violations == []

    def test_string_contents_with_print_not_matched(
        self, tmp_path: Path
    ) -> None:
        """Sandbox-script string templates that contain ``print(...)`` text are not flagged.

        Pins the AST-level matching so embedded sandbox scripts
        (``api/app/infrastructure/external/file_processors/pdf.py``
        et al.) keep working without per-file allowlists.
        """
        clean = tmp_path / "embedded.py"
        clean.write_text(
            'TEMPLATE = """\n'
            "import json\n"
            'print(json.dumps({"ok": True}))\n'
            '"""\n',
            encoding="utf-8",
        )

        violations = no_print_in_backend.find_violations([tmp_path])

        assert violations == []

    def test_path_under_tests_is_allowlisted(self, tmp_path: Path) -> None:
        nested = tmp_path / "tests" / "test_x.py"
        nested.parent.mkdir(parents=True)
        nested.write_text("print('debug from a test')\n", encoding="utf-8")

        violations = no_print_in_backend.find_violations([tmp_path])

        assert violations == []

    def test_path_under_api_tools_is_allowlisted(self, tmp_path: Path) -> None:
        nested = tmp_path / "api" / "tools" / "diag.py"
        nested.parent.mkdir(parents=True)
        nested.write_text("print('lint diagnostic')\n", encoding="utf-8")

        violations = no_print_in_backend.find_violations([tmp_path])

        assert violations == []

    def test_main_returns_zero_when_clean(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.py"
        clean.write_text("x = 1\n", encoding="utf-8")

        rc = no_print_in_backend.main([str(tmp_path)])

        assert rc == 0

    def test_main_returns_one_on_violation(self, tmp_path: Path) -> None:
        offender = tmp_path / "offender.py"
        offender.write_text("print('hi')\n", encoding="utf-8")

        rc = no_print_in_backend.main([str(tmp_path)])

        assert rc == 1

    def test_real_backend_tree_is_clean(self) -> None:
        """Anti-regression: real ``api/app + api/scripts`` is clean.

        Locks the post-PR-S1-7a contract; landing a fresh ``print()``
        in committed code fails this test (unless it's wrapped in
        the file-level marker, which is what category (b) edge
        cases are supposed to do).
        """
        targets = [
            _REPO_ROOT / "api/app",
            _REPO_ROOT / "api/scripts",
        ]
        violations = no_print_in_backend.find_violations(targets)
        assert violations == [], (
            "print() regression in committed code:\n  "
            + "\n  ".join(v.format_diagnostic() for v in violations)
        )
