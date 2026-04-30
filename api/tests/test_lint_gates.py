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

    def test_import_after_function_canonical_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2: ``def f(): logging.basicConfig() … import logging``.

        Walks-as-you-go visitors hit the ``FunctionDef`` first, so
        the inner ``logging.basicConfig()`` is checked before the
        bottom-of-file ``import logging`` populates the alias set —
        and the violation slips past. The two-phase scanner
        pre-collects every import binding before walking calls,
        eliminating the source-order dependency.
        """
        offender = tmp_path / "import_after_func.py"
        offender.write_text(
            "def configure() -> None:\n"
            "    logging.basicConfig(level=20)\n"
            "\n"
            "import logging\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 2

    def test_import_after_function_module_alias_caught(
        self, tmp_path: Path
    ) -> None:
        """``def f(): log.basicConfig() … import logging as log``."""
        offender = tmp_path / "import_after_func_alias.py"
        offender.write_text(
            "def configure() -> None:\n"
            "    log.basicConfig(level=log.INFO)\n"
            "\n"
            "import logging as log\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].line == 2

    def test_import_after_function_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """``def f(): basicConfig() … from logging import basicConfig``."""
        offender = tmp_path / "from_after_func.py"
        offender.write_text(
            "def configure() -> None:\n"
            "    basicConfig(level=20)\n"
            "\n"
            "from logging import basicConfig\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].line == 2

    def test_import_after_function_star_import_caught(
        self, tmp_path: Path
    ) -> None:
        """``def f(): basicConfig() … from logging import *``."""
        offender = tmp_path / "star_after_func.py"
        offender.write_text(
            "def configure() -> None:\n"
            "    basicConfig(level=20)\n"
            "\n"
            "from logging import *\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].line == 2

    def test_function_local_import_does_not_leak_to_sibling(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 3): function-local import must not pollute siblings.

        ``setup`` has a parameter named ``cfg``; ``helper`` happens to
        do ``import logging as cfg``. Pre-fix, the global walk added
        ``cfg`` to the module-wide alias set, and ``setup``'s
        ``cfg.basicConfig()`` got flagged — false positive on a
        legitimate parameter call.

        Post-fix (scope-aware visitor): ``helper``'s local import
        stays in ``helper``'s scope and never enters ``setup``'s
        resolution chain. ``setup``'s param ``cfg`` resolves to
        ``"other"`` (shadow), and the call is not flagged.
        """
        clean = tmp_path / "sibling_local_import.py"
        clean.write_text(
            "def setup(cfg):\n"
            "    cfg.basicConfig()\n"
            "\n"
            "def helper():\n"
            "    import logging as cfg\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_parameter_shadows_module_level_logging(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 3): parameter named ``logging`` shadows the module.

        Module-level ``import logging`` makes ``logging`` resolve to
        the module at module scope, but ``def setup(logging)`` rebinds
        the name as a parameter inside the function body. Pre-fix,
        the scanner ignored the parameter shadow and flagged the
        method-style call; post-fix, the parameter binding takes
        precedence inside the function.
        """
        clean = tmp_path / "param_shadow.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "def setup(logging):\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_module_level_import_still_flags_function_call(
        self, tmp_path: Path
    ) -> None:
        """Sanity boundary: shadowing only kicks in when the name is rebound.

        Pins the contract that the scope-aware refactor still
        catches the canonical positive case — a function with no
        parameter / local rebind that uses the module-level
        ``logging`` alias.
        """
        offender = tmp_path / "function_uses_module_logging.py"
        offender.write_text(
            "import logging\n"
            "\n"
            "def setup() -> None:\n"
            "    logging.basicConfig(level=20)\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_class_body_import_not_visible_to_method(
        self, tmp_path: Path
    ) -> None:
        """Class scope is not lexically visible to methods (Python rule).

        ``class Foo: import logging`` binds ``logging`` in ``Foo``'s
        namespace, accessible as ``Foo.logging`` from outside but
        NOT visible inside ``Foo.method`` bodies. The scanner must
        respect this — flagging ``method``'s ``logging.basicConfig()``
        based on the class-body import would be a false positive
        because that call would actually ``NameError`` at runtime.
        """
        clean = tmp_path / "class_body_import.py"
        clean.write_text(
            "class Foo:\n"
            "    import logging\n"
            "\n"
            "    def method(self):\n"
            "        logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_assign_after_call_does_not_shadow_module_call(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 4): later rebind must not retroactively un-flag.

        ::

            import logging          # binds logging → module
            logging.basicConfig()   # ← real violation at this line
            logging = object()      # later rebind to "other"

        Pre-fix, the collapsed-final-binding map ended with
        ``logging → "other"`` and the call was silently dropped.
        Post-fix (line-aware lookup), the call at line 2 sees the
        most recent event up to its own line — the import at line
        1 — and is correctly flagged.
        """
        offender = tmp_path / "call_then_assign_module.py"
        offender.write_text(
            "import logging\n"
            "logging.basicConfig()\n"
            "logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 2

    def test_assign_after_call_does_not_shadow_direct_import(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 4): direct-bind variant of the assign-after rebind.

        ::

            from logging import basicConfig  # binds basicConfig → callable
            basicConfig()                     # real violation
            basicConfig = object()            # later rebind
        """
        offender = tmp_path / "call_then_assign_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "basicConfig()\n"
            "basicConfig = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 2

    def test_assign_before_call_does_shadow(self, tmp_path: Path) -> None:
        """Boundary anchor: shadow that genuinely precedes the call still wins.

        Pins the symmetric case of the P2 (round 4) fix — when
        the assign happens BEFORE the call in source order, the
        shadow IS effective and the call is NOT flagged.
        """
        clean = tmp_path / "assign_before_call.py"
        clean.write_text(
            "import logging\n"
            "logging = object()\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_same_line_call_then_assign_module_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 5): same-line rebind after a real call.

        ::

            import logging
            logging.basicConfig(); logging = object()

        At runtime the call evaluates first, then the assign rebinds
        ``logging``. Pre-fix, the line-only event timeline put the
        assign at ``line=2`` (same as the call), so the lookup
        treated the assign as "≤ call_line=2" and silently dropped
        the violation. Post-fix the events carry
        ``(end_line, end_col)`` and the call is checked at its
        START position — the assign's end col is greater than the
        call's start col on the same line, so it's correctly
        excluded.
        """
        offender = tmp_path / "same_line_call_then_assign_module.py"
        offender.write_text(
            "import logging\n"
            "logging.basicConfig(); logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 2
        assert violations[0].col == 0  # call starts at column 0

    def test_same_line_call_then_assign_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 5): same-line direct-bind variant.

        ::

            from logging import basicConfig
            basicConfig(); basicConfig = object()
        """
        offender = tmp_path / "same_line_call_then_assign_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "basicConfig(); basicConfig = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 2
        assert violations[0].col == 0

    def test_same_line_assign_then_call_does_shadow(
        self, tmp_path: Path
    ) -> None:
        """Boundary anchor: same-line assign BEFORE call still shadows.

        Mirror of the round 5 fix — when the assign genuinely
        precedes the call (same line), its end position is
        ``< call's start position`` so the shadow IS effective and
        the call is NOT flagged.
        """
        clean = tmp_path / "same_line_assign_then_call.py"
        clean.write_text(
            "import logging\n"
            "logging = object(); logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_annassign_shadows_module_logging(self, tmp_path: Path) -> None:
        """Review-found P2 (round 6): ``logging: int = 1`` shadows module."""
        clean = tmp_path / "annassign_shadow.py"
        clean.write_text(
            "import logging\n"
            "logging: int = 1\n"
            "logging.basicConfig()\n",  # ``logging`` is now an int
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_function_def_shadows_module_logging(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 6): ``def logging(): ...`` shadows module."""
        clean = tmp_path / "function_def_shadow.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "def logging():\n"
            "    pass\n"
            "\n"
            "logging.basicConfig()\n",  # ``logging`` is now a function
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_class_def_shadows_module_logging(self, tmp_path: Path) -> None:
        """Review-found P2 (round 6): ``class logging: ...`` shadows module."""
        clean = tmp_path / "class_def_shadow.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "class logging:\n"
            "    pass\n"
            "\n"
            "logging.basicConfig()\n",  # ``logging`` is now a class
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_for_target_shadows_module_logging(self, tmp_path: Path) -> None:
        """Review-found P2 (round 6): ``for logging in items:`` shadows INSIDE body.

        Round-6's intent was that the for-target IS the loop
        variable from the body's POV — a method-style call on it
        is NOT a call to the imported module. Inside the body
        branch scope (round-12 P2-a + round-13 P1), the target
        is recorded as an unconditional ``"other"`` event so
        body code resolves to the loop variable.

        The post-loop case (where zero iter leaves the target
        unbound) is covered by
        ``test_for_target_zero_iter_post_loop_caught``.
        """
        clean = tmp_path / "for_shadow_inside.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "for logging in [1, 2, 3]:\n"
            "    logging.basicConfig()\n",  # ``logging`` is the loop var
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_for_target_zero_iter_post_loop_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-13 P1: zero-iter for-loop leaves target unbound, post-loop call caught.

        ::

            import logging
            for logging in []:        # never iterates → target unbound
                pass
            logging.basicConfig()    # ← still the stdlib module → flag

        Pre-fix the for-target was recorded as unconditional
        ``"other"`` in the parent scope (inheriting the caller's
        flag), silently shadowing the import. Post-fix the parent
        target event is conditional, so the conditional-other
        filter drops it and the import wins resolution.
        """
        offender = tmp_path / "for_zero_iter_post_loop.py"
        offender.write_text(
            "import logging\n"
            "for logging in []:\n"
            "    pass\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_for_target_zero_iter_post_loop_caught_direct(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of the round-13 P1 zero-iter probe."""
        offender = tmp_path / "for_zero_iter_post_loop_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "for basicConfig in []:\n"
            "    pass\n"
            "basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_async_for_target_shadow_inside_body_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-13 P1 ``async for`` variant: target shadows inside body.

        ``async for`` only appears inside ``async def``, where the
        compiler makes the target a function-local for the ENTIRE
        function body — references both inside and after the loop
        bind to that local, never falling through to the module-
        level ``import logging``. A method call on the loop var
        inside the body is NOT a call to stdlib ``logging``.

        This is the function-scope counterpart of the module-level
        zero-iter probe — same branch-scope extra-events seeding
        the loop var as unconditionally bound inside the body.
        """
        clean = tmp_path / "async_for_inside_body.py"
        clean.write_text(
            "import logging\n"
            "async def f():\n"
            "    async for logging in aiter([1, 2]):\n"
            "        logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_with_as_shadows_module_logging(self, tmp_path: Path) -> None:
        """Review-found P2 (round 6): ``with X() as logging:`` shadows module."""
        clean = tmp_path / "with_shadow.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "class Cm:\n"
            "    def __enter__(self): return 1\n"
            "    def __exit__(self, *a): return False\n"
            "\n"
            "with Cm() as logging:\n"
            "    logging.basicConfig()\n",  # ``logging`` is 1 here
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_except_as_shadows_module_logging(self, tmp_path: Path) -> None:
        """Review-found P2 (round 6): ``except E as logging:`` shadows module."""
        clean = tmp_path / "except_shadow.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "try:\n"
            "    pass\n"
            "except Exception as logging:\n"
            "    logging.basicConfig()\n",  # ``logging`` is the exception
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_function_def_shadows_basicconfig_direct_import(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant: ``def basicConfig(): ...`` shadows the imported."""
        clean = tmp_path / "function_def_shadow_direct.py"
        clean.write_text(
            "from logging import basicConfig\n"
            "\n"
            "def basicConfig():\n"
            "    pass\n"
            "\n"
            "basicConfig()\n",  # local def, not the imported
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_for_target_shadows_basicconfig_direct_import(
        self, tmp_path: Path
    ) -> None:
        """Reviewer's exact P2 (round 6) example: for-target rebinds basicConfig.

        ::

            from logging import basicConfig
            for basicConfig in items:
                basicConfig()

        ``basicConfig`` inside the loop is the loop variable
        (an item from ``items``), not the imported callable. The
        ``basicConfig()`` call is on whatever value the iterator
        yields, not ``logging.basicConfig`` — must not flag.
        """
        clean = tmp_path / "for_shadow_direct.py"
        clean.write_text(
            "from logging import basicConfig\n"
            "\n"
            "items = [object()]\n"
            "for basicConfig in items:\n"
            "    basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_decorator_before_def_shadow_caught(self, tmp_path: Path) -> None:
        """Review-found P2 (round 7): decorator call IS a real violation.

        Decorators evaluate in the ENCLOSING scope at def-time —
        BEFORE the function's own name binding completes. The
        decorator's call to ``logging.basicConfig()`` runs against
        the module-level ``logging`` import, not against the
        not-yet-bound ``def logging``::

            import logging

            def deco(x): return lambda f: f

            @deco(logging.basicConfig())   # ← real call at this line
            def logging():
                pass
        """
        offender = tmp_path / "decorator_before_def_shadow.py"
        offender.write_text(
            "import logging\n"
            "\n"
            "def deco(x): return lambda f: f\n"
            "\n"
            "@deco(logging.basicConfig())\n"
            "def logging():\n"
            "    pass\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_default_before_later_assign_caught(self, tmp_path: Path) -> None:
        """Review-found P2 (round 7): default-value call evaluates in enclosing scope.

        Default arguments are evaluated when the ``def`` statement
        runs, BEFORE any later module-level rebind. The
        ``logging.basicConfig()`` in the default IS a real call
        against the module from line 1 — line-aware lookup must
        flag it even though a later assign rebinds ``logging``::

            import logging

            def f(x=logging.basicConfig()):  # ← real call here
                pass

            logging = object()
        """
        offender = tmp_path / "default_before_later_assign.py"
        offender.write_text(
            "import logging\n"
            "\n"
            "def f(x=logging.basicConfig()):\n"
            "    pass\n"
            "\n"
            "logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_class_base_before_class_shadow_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 7): class-base call IS a real violation.

        Class bases / keyword args evaluate in the ENCLOSING scope
        at class-definition time — BEFORE the class's name binding
        completes. ``logging.basicConfig()`` inside a base spec
        runs against the module ``logging``, not the class about to
        rebind ``logging``::

            import logging

            def passthrough(_):
                return type('AnyBase', (object,), {})

            class logging(passthrough(logging.basicConfig())):  # ← real
                pass
        """
        offender = tmp_path / "class_base_before_class_shadow.py"
        offender.write_text(
            "import logging\n"
            "\n"
            "def passthrough(_):\n"
            "    return type('AnyBase', (object,), {})\n"
            "\n"
            "class logging(passthrough(logging.basicConfig())):\n"
            "    pass\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 6

    def test_decorator_before_def_shadow_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant: decorator call against ``basicConfig`` direct binding."""
        offender = tmp_path / "decorator_before_def_shadow_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "\n"
            "def deco(x): return lambda f: f\n"
            "\n"
            "@deco(basicConfig())\n"
            "def basicConfig():\n"
            "    pass\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_default_before_later_assign_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant: default-value call against ``basicConfig`` direct binding."""
        offender = tmp_path / "default_before_later_assign_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "\n"
            "def f(x=basicConfig()):\n"
            "    pass\n"
            "\n"
            "basicConfig = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_class_base_before_class_shadow_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant: class-base call against ``basicConfig`` direct binding."""
        offender = tmp_path / "class_base_before_class_shadow_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "\n"
            "def passthrough(_):\n"
            "    return type('AnyBase', (object,), {})\n"
            "\n"
            "class basicConfig(passthrough(basicConfig())):\n"
            "    pass\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 6

    def test_match_case_body_caught(self, tmp_path: Path) -> None:
        """Review-found P1 (round 8): ``match … case`` body must be scanned.

        ::

            def f(x):
                match x:
                    case _:
                        import logging
                        logging.basicConfig()  # ← real call

        Pre-fix the case body was opaque to ``_iter_scope_stmts``,
        so the import didn't enter the function scope's timeline
        and ``logging`` resolved upstream as unbound — no flag.
        Post-fix the case body is descended like any other
        control-flow body.
        """
        offender = tmp_path / "match_import_module.py"
        offender.write_text(
            "def f(x):\n"
            "    match x:\n"
            "        case _:\n"
            "            import logging\n"
            "            logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_match_case_body_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of the round 8 ``match`` regression."""
        offender = tmp_path / "match_import_direct.py"
        offender.write_text(
            "def f(x):\n"
            "    match x:\n"
            "        case _:\n"
            "            from logging import basicConfig\n"
            "            basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_except_handler_does_not_persist_after_try(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 8): ``except as N`` binds only inside handler body.

        Python 3 deletes the exception name at end of handler, so
        code AFTER the ``try`` sees the original outer binding.
        Pre-fix the handler.name was written into the enclosing
        scope's permanent timeline as ``"other"``, silently
        suppressing legitimate violations after the ``try``::

            import logging

            try:
                pass
            except Exception as logging:
                pass

            logging.basicConfig()  # ← still the module — should flag
        """
        offender = tmp_path / "except_after_try.py"
        offender.write_text(
            "import logging\n"
            "\n"
            "try:\n"
            "    pass\n"
            "except Exception as logging:\n"
            "    pass\n"
            "\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 8

    def test_import_as_logging_shadows_module_logging(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 8): ``import math as logging`` rebinds the name.

        ::

            import logging
            import math as logging
            logging.basicConfig()   # ← logging is now math, not the module

        Pre-fix only ``import logging`` produced an event for
        ``logging``; the second import rebinding the same name to
        ``math`` was ignored, so the call was falsely flagged.
        Post-fix every import-bound name with mismatched target is
        recorded as ``"other"`` shadow.
        """
        clean = tmp_path / "import_as_shadow.py"
        clean.write_text(
            "import logging\n"
            "import math as logging\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_from_other_import_shadows_basicconfig(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 8): ``from other import basicConfig`` rebinds.

        ::

            from logging import basicConfig
            from other import basicConfig
            basicConfig()   # ← bound to other.basicConfig now
        """
        clean = tmp_path / "from_other_import_shadow.py"
        clean.write_text(
            "from logging import basicConfig\n"
            "from other_pkg import basicConfig\n"
            "basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_except_handler_body_import_overrides_header_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P1 (round 9): handler-body import beats header shadow.

        ::

            try:
                raise Exception()
            except Exception as logging:
                import logging          # rebinds to module
                logging.basicConfig()   # real call — must flag

        The ``as logging`` header makes ``logging`` a local
        exception variable, but the very next ``import logging``
        rebinds the same local name to the actual ``logging``
        module. The subsequent call IS a real
        ``logging.basicConfig`` invocation.
        """
        offender = tmp_path / "except_handler_import_module.py"
        offender.write_text(
            "try:\n"
            "    raise Exception()\n"
            "except Exception as logging:\n"
            "    import logging\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_except_handler_body_import_overrides_header_direct_caught(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of round-9 handler body override."""
        offender = tmp_path / "except_handler_import_direct.py"
        offender.write_text(
            "try:\n"
            "    raise Exception()\n"
            "except Exception as basicConfig:\n"
            "    from logging import basicConfig\n"
            "    basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_function_scope_except_as_is_local_capture(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 9): function ``except as N`` is local-capture.

        Inside a function, Python's compiler treats the ``as N``
        target as a local for the **entire** function body — not
        just the handler. References to ``N`` after the ``try``
        therefore resolve to "local-but-unbound" (UnboundLocalError
        at runtime), NOT to any outer binding. The lint gate must
        not pierce through to module-level ``logging`` for that
        broken-at-runtime reference::

            import logging
            def f():
                try:
                    pass
                except Exception as logging:
                    pass
                logging.basicConfig()  # UnboundLocalError, not module
        """
        clean = tmp_path / "function_except_local_capture.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except Exception as logging:\n"
            "        pass\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_match_pattern_capture_shadows_module_logging(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 9): ``case logging:`` captures the subject.

        ``case logging:`` is a ``MatchAs(name="logging")`` pattern
        that captures the match subject as ``logging`` for the
        case body / guard. A subsequent
        ``logging.basicConfig()`` references the captured value,
        not the module — must not flag::

            import logging
            def f(x):
                match x:
                    case logging:
                        logging.basicConfig()
        """
        clean = tmp_path / "match_capture_shadow.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "def f(x):\n"
            "    match x:\n"
            "        case logging:\n"
            "            logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_match_pattern_as_capture_shadows(self, tmp_path: Path) -> None:
        """Review-found P2 (round 9): ``case _ as logging`` capture variant.

        Same semantics as ``case logging:`` — the ``as`` clause
        captures the matched subject under the given name.
        """
        clean = tmp_path / "match_as_capture_shadow.py"
        clean.write_text(
            "import logging\n"
            "\n"
            "def f(x):\n"
            "    match x:\n"
            "        case _ as logging:\n"
            "            logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_nested_function_call_before_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P1 (round 10): nested function called BEFORE rebind.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()  # ← real call
                inner()
                logging = object()

        Pre-fix the outer-scope lookup unconditionally took the
        last event (``"other"`` from the assign), suppressing the
        violation. Post-fix the conservative rule reports any
        ``logging`` event in the outer timeline — the deferred
        nested function may run while ``logging`` is still the
        module.
        """
        offender = tmp_path / "nested_before_rebind.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    inner()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_nested_function_call_before_rebind_direct_caught(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of round-10 nested-call regression."""
        offender = tmp_path / "nested_before_rebind_direct.py"
        offender.write_text(
            "def outer():\n"
            "    from logging import basicConfig\n"
            "    def inner():\n"
            "        basicConfig()\n"
            "    inner()\n"
            "    basicConfig = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_global_declaration_keeps_module_resolution_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 10): ``global logging`` is not local.

        ::

            import logging
            def f():
                global logging
                logging.basicConfig()  # ← module-level logging
                logging = object()

        With ``global``, every binding statement for ``logging``
        in ``f`` writes to the module-level name; the call
        resolves to the module ``logging`` (not a function-local
        about to be assigned). Pre-fix the assign was treated as
        a function-local shadow, suppressing the call.
        """
        offender = tmp_path / "global_logging.py"
        offender.write_text(
            "import logging\n"
            "def f():\n"
            "    global logging\n"
            "    logging.basicConfig()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_global_declaration_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of round-10 ``global`` regression."""
        offender = tmp_path / "global_basicconfig.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "def f():\n"
            "    global basicConfig\n"
            "    basicConfig()\n"
            "    basicConfig = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_module_level_match_other_case_call_caught(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 10): module-level match capture is per-case.

        ::

            import logging
            x = 0
            match x:
                case 0:
                    pass
                case logging:
                    pass
            logging.basicConfig()  # ← x=0 path: logging unchanged

        When ``case 0`` matches, ``logging`` keeps its imported
        binding for the rest of the module. Pre-fix the
        ``case logging:`` capture was added as a permanent
        scope-wide shadow, masking the post-match call.
        Post-fix captures live only in their own case body.
        """
        offender = tmp_path / "module_match_other_case.py"
        offender.write_text(
            "import logging\n"
            "x = 0\n"
            "match x:\n"
            "    case 0:\n"
            "        pass\n"
            "    case logging:\n"
            "        pass\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 8

    def test_module_level_match_other_case_call_direct_caught(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of round-10 module-level match regression."""
        offender = tmp_path / "module_match_other_case_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "x = 0\n"
            "match x:\n"
            "    case 0:\n"
            "        pass\n"
            "    case basicConfig:\n"
            "        pass\n"
            "basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 8

    def test_conditional_branch_rebind_does_not_shadow(
        self, tmp_path: Path
    ) -> None:
        """Review-found P1 (round 11): ``if False:`` body never runs.

        ::

            import logging
            if False:
                logging = object()
            logging.basicConfig()

        Pre-fix the assign inside ``if False:`` was treated as
        unconditional, suppressing the violation. Post-fix
        conditional ``"other"`` rebinds are filtered out of
        innermost line-aware lookup.
        """
        offender = tmp_path / "if_false_rebind.py"
        offender.write_text(
            "import logging\n"
            "if False:\n"
            "    logging = object()\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_try_handler_rebind_does_not_shadow_after_try(
        self, tmp_path: Path
    ) -> None:
        """Round-11: rebind inside except handler is conditional.

        Same logic as ``if`` — only the matching path runs.
        """
        offender = tmp_path / "try_handler_rebind.py"
        offender.write_text(
            "import logging\n"
            "try:\n"
            "    pass\n"
            "except Exception:\n"
            "    logging = object()\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 6

    def test_match_case_rebind_does_not_shadow_after_match(
        self, tmp_path: Path
    ) -> None:
        """Round-11: rebind inside a non-matching case is conditional."""
        offender = tmp_path / "match_case_rebind.py"
        offender.write_text(
            "import logging\n"
            "x = 0\n"
            "match x:\n"
            "    case 99:\n"
            "        logging = object()\n"
            "    case _:\n"
            "        pass\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 8

    def test_global_rebind_before_call_in_function_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 11): ``global`` rebind BEFORE call.

        ::

            import logging
            def f():
                global logging
                logging = object()
                logging.basicConfig()  # uses object — AttributeError, not module

        At the call line, ``logging`` has been rebound to
        ``object()`` via the global. The call would raise
        ``AttributeError``, not invoke ``logging.basicConfig`` —
        so no flag.
        """
        clean = tmp_path / "global_rebind_before_call.py"
        clean.write_text(
            "import logging\n"
            "def f():\n"
            "    global logging\n"
            "    logging = object()\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_nested_function_after_rebind_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Review-found P2 (round 11): nested function defined AFTER rebind.

        ::

            def outer():
                import logging
                logging = object()
                def inner():
                    logging.basicConfig()
                inner()

        Inner is defined AFTER the rebind, so its closure capture
        of ``logging`` references the rebound value (``object``).
        Conservative-flag from round 10 over-reported this; the
        round-11 snapshot rule (use inner's def_line as the
        outer-scope cutoff) correctly returns ``"other"``.
        """
        clean = tmp_path / "nested_after_rebind.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    logging = object()\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_nested_function_after_rebind_direct_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of the round-11 nested-after-rebind anchor."""
        clean = tmp_path / "nested_after_rebind_direct.py"
        clean.write_text(
            "def outer():\n"
            "    from logging import basicConfig\n"
            "    basicConfig = object()\n"
            "    def inner():\n"
            "        basicConfig()\n"
            "    inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_try_body_raise_does_not_suppress_after_try(
        self, tmp_path: Path
    ) -> None:
        """Round-12 P1: try-body rebind is conditional (mid-body raise may skip).

        ::

            import logging
            try:
                raise Exception()
                logging = object()      # never reached
            except Exception:
                pass
            logging.basicConfig()       # ← logging still the module → flag

        Pre-fix the try-body inherited the parent's
        ``is_conditional`` flag, so the rebind at line 4 was
        treated as unconditional and silently suppressed the
        post-try call. Post-fix the try body is conditional like
        the handlers / orelse, so the rebind is filtered out of
        innermost lookup and the import wins.
        """
        offender = tmp_path / "try_body_raise_module.py"
        offender.write_text(
            "import logging\n"
            "try:\n"
            "    raise Exception()\n"
            "    logging = object()\n"
            "except Exception:\n"
            "    pass\n"
            "logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 7

    def test_try_body_raise_does_not_suppress_after_try_direct(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of the round-12 P1 try-body probe."""
        offender = tmp_path / "try_body_raise_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "try:\n"
            "    raise Exception()\n"
            "    basicConfig = object()\n"
            "except Exception:\n"
            "    pass\n"
            "basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 7

    def test_branch_internal_real_shadow_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-12 P2-a: same-branch real shadow suppresses.

        ::

            if cond:
                import logging
                logging = object()
                logging.basicConfig()    # ``logging`` is object — no flag

        Pre-fix the conditional ``"other"`` rebind on line 3 was
        filtered globally, so the call at line 4 still saw the
        conditional ``"logging"`` import on line 2 and flagged.
        Post-fix the branch-local timeline treats both events as
        unconditional from inside the branch — the assign correctly
        shadows the import.
        """
        clean = tmp_path / "branch_internal_shadow.py"
        clean.write_text(
            "cond = True\n"
            "if cond:\n"
            "    import logging\n"
            "    logging = object()\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_branch_internal_real_shadow_no_flag_for_body(
        self, tmp_path: Path
    ) -> None:
        """Round-12 P2-a for-body variant of the branch-local timeline rule."""
        clean = tmp_path / "branch_internal_shadow_for.py"
        clean.write_text(
            "for _ in [1]:\n"
            "    import logging\n"
            "    logging = object()\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_branch_internal_real_shadow_no_flag_while_body(
        self, tmp_path: Path
    ) -> None:
        """Round-12 P2-a while-body variant of the branch-local timeline rule."""
        clean = tmp_path / "branch_internal_shadow_while.py"
        clean.write_text(
            "i = 0\n"
            "while i < 1:\n"
            "    import logging\n"
            "    logging = object()\n"
            "    logging.basicConfig()\n"
            "    i += 1\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_branch_internal_call_without_shadow_still_flags(
        self, tmp_path: Path
    ) -> None:
        """Boundary anchor: branch-local timeline still flags without intra-branch rebind.

        Pins the symmetric round-12 P2-a fix — when there is no
        rebind inside the branch, the conditional ``"logging"``
        import IS kept and the call is correctly flagged. Stops
        the branch-scope addition from regressing into a global
        no-flag.
        """
        offender = tmp_path / "branch_internal_no_shadow.py"
        offender.write_text(
            "cond = True\n"
            "if cond:\n"
            "    import logging\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_closure_after_rebind_inner_called_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-12 P2-b: closure cell semantics for inner called after rebind.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                logging = object()      # rebind first
                inner()                  # ← closure sees object()

        Pre-fix the def-line snapshot rule pinned the outer-scope
        lookup at ``inner.def_line`` and saw the pre-rebind import
        — false positive. Post-fix the lookup uses the EARLIEST
        line ``inner`` is referenced in outer (the call site after
        the rebind), so the snapshot reflects the post-rebind
        ``"other"`` and the call is not flagged.
        """
        clean = tmp_path / "closure_after_rebind.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    logging = object()\n"
            "    inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_closure_after_rebind_inner_called_no_flag_direct(
        self, tmp_path: Path
    ) -> None:
        """Direct-import variant of the round-12 P2-b closure probe."""
        clean = tmp_path / "closure_after_rebind_direct.py"
        clean.write_text(
            "def outer():\n"
            "    from logging import basicConfig\n"
            "    def inner():\n"
            "        basicConfig()\n"
            "    basicConfig = object()\n"
            "    inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_relative_logging_module_is_not_stdlib(
        self, tmp_path: Path
    ) -> None:
        """Round-13 P2: ``from .logging import basicConfig`` is NOT stdlib.

        ::

            from .logging import basicConfig    # in-package helper
            basicConfig()                        # local API, not stdlib

        Pre-fix the scanner only checked ``stmt.module == "logging"``
        and ignored ``stmt.level``, so any in-package ``logging``
        helper at level >= 1 was mis-attributed to stdlib —
        flagging legitimate calls and breaking CI for packages
        that name a helper ``logging``. Post-fix the scanner
        requires ``level == 0`` (absolute import).
        """
        clean = tmp_path / "package" / "module.py"
        clean.parent.mkdir(parents=True)
        (tmp_path / "package" / "__init__.py").write_text(
            "", encoding="utf-8"
        )
        (tmp_path / "package" / "logging.py").write_text(
            "def basicConfig():\n    return None\n", encoding="utf-8"
        )
        clean.write_text(
            "from .logging import basicConfig\n"
            "\n"
            "def main() -> None:\n"
            "    basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_relative_logging_module_dotted_not_stdlib(
        self, tmp_path: Path
    ) -> None:
        """Round-13 P2 dotted variant: ``from ..pkg.logging import basicConfig``."""
        clean = tmp_path / "deep_relative.py"
        clean.write_text(
            "from ..pkg.logging import basicConfig\n"
            "\n"
            "def main() -> None:\n"
            "    basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_match_no_capture_case_branch_local_shadow(
        self, tmp_path: Path
    ) -> None:
        """Round-13 P2: ``case _:`` (no capture) gets a case scope too.

        ::

            x = 0
            match x:
                case _:
                    import logging
                    logging = object()
                    logging.basicConfig()    # logging is object → no flag

        Pre-fix the match visitor only pushed a case scope when
        the pattern bound a name. Wildcards (``case _:``), value
        patterns (``case 0:``), etc., ran inside the parent scope's
        timeline — and ``_iter_scope_stmts`` marks case bodies as
        conditional, so the in-case rebind became a conditional
        ``"other"`` that the resolver filtered out. The earlier
        conditional ``"logging"`` import survived and the in-case
        call falsely flagged. Post-fix every case body gets its
        own case scope with branch-local (unconditional) events.
        """
        clean = tmp_path / "match_no_capture_shadow.py"
        clean.write_text(
            "x = 0\n"
            "match x:\n"
            "    case _:\n"
            "        import logging\n"
            "        logging = object()\n"
            "        logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_closure_escape_non_call_ref_does_not_pin_snapshot(
        self, tmp_path: Path
    ) -> None:
        """Round-13 P2: non-call ``Name(Load)`` ref to inner does NOT fire the closure.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                alias = inner       # ← escape (not a call!)
                logging = object()  # rebind
                inner()              # actual call AFTER rebind

        Pre-fix ``_earliest_load_ref_line`` recorded
        ``alias = inner`` (line 5) as the "earliest reference"
        and snapshot pinned at line 5 saw the pre-rebind
        ``logging`` import → false positive. Post-fix the
        ref-finder splits "direct call" from "escape" — only the
        actual ``inner()`` at line 7 (after rebind) counts as a
        firing site, so the snapshot returns the post-rebind
        ``"other"`` and no flag.
        """
        clean = tmp_path / "closure_escape_then_call.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    logging = object()\n"
            "    inner()\n"
            "    return alias\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_closure_escape_only_still_flags(
        self, tmp_path: Path
    ) -> None:
        """Boundary anchor: pure escape (no direct call) still flags conservatively.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                return inner             # escapes — caller may fire it

        With no direct ``inner()`` call site visible, the closure
        could fire at any time — including before any rebind.
        Conservative flag via events[-1] permissive: when the
        latest non-conditional event is ``"logging"``, flag.
        Pins the round-13 P2 escape semantic so a future change
        that drops escape→flag would regress this case.
        """
        offender = tmp_path / "closure_escape_only.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    return inner\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_alias_via_assign_from_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-14 P1: ``bc = basicConfig; bc()`` propagates the alias.

        ::

            from logging import basicConfig
            bc = basicConfig
            bc()                 # ← real call via alias → flag

        Pre-fix every ``Assign`` LHS was recorded as
        ``_BINDING_OTHER``, so the alias chain ``basicConfig``
        → ``bc`` lost the kind information and the call slipped
        past the gate. Post-fix same-scope alias propagation
        evaluates the RHS ``Name`` against the partial events
        dict and inherits the kind onto the LHS.
        """
        offender = tmp_path / "alias_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "bc = basicConfig\n"
            "bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_alias_via_assign_from_logging_attribute_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-14 P1: ``bc = logging.basicConfig; bc()`` propagates via Attribute.

        ::

            import logging
            bc = logging.basicConfig
            bc()                       # ← real call via alias → flag

        Pre-fix the Attribute access on the RHS was opaque to
        the Assign event recorder. Post-fix
        ``Attribute(value=Name(logging), attr="basicConfig")``
        with logging resolved to ``_BINDING_LOGGING`` makes the
        LHS a ``_BINDING_BASICCONFIG`` alias.
        """
        offender = tmp_path / "alias_attribute.py"
        offender.write_text(
            "import logging\n"
            "bc = logging.basicConfig\n"
            "bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_alias_chain_propagates_through_multiple_assigns(
        self, tmp_path: Path
    ) -> None:
        """Boundary anchor: round-14 P1 alias propagation chains.

        ``a = basicConfig; b = a; b()`` — each link in the chain
        resolves through the previous Assign's recorded kind, so
        ``b`` ends up with ``_BINDING_BASICCONFIG`` and the call
        flags. Pins the chain behavior so a future change that
        only handles "first-hop" aliases regresses.
        """
        offender = tmp_path / "alias_chain.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "a = basicConfig\n"
            "b = a\n"
            "b()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_function_default_logging_attribute_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-14 P1 probe 3: ``def f(fn=logging.basicConfig): fn()``.

        Function default expressions evaluate in the ENCLOSING
        scope at def time. Pre-fix the parameter ``fn`` was
        always seeded with ``_BINDING_OTHER``, so calls through
        it inside the body never flagged. Post-fix
        ``_compute_param_kinds`` resolves each default's kind via
        the scope chain (``logging`` → module-level
        ``_BINDING_LOGGING``, ``logging.basicConfig`` →
        ``_BINDING_BASICCONFIG``) and seeds the parameter event
        with that kind.
        """
        offender = tmp_path / "default_logging_attr.py"
        offender.write_text(
            "import logging\n"
            "\n"
            "def f(fn=logging.basicConfig):\n"
            "    fn()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_function_default_basicconfig_direct_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-14 P1 probe 3 direct-import variant.

        ::

            from logging import basicConfig

            def f(fn=basicConfig):
                fn()
        """
        offender = tmp_path / "default_basicconfig_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "\n"
            "def f(fn=basicConfig):\n"
            "    fn()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_lambda_default_basicconfig_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-14 P1 probe 3 lambda variant.

        Lambda defaults follow the same enclosing-scope evaluation
        as ``def`` defaults, so the param-kinds machinery applies
        identically. ::

            from logging import basicConfig
            f = lambda fn=basicConfig: fn()
            f()
        """
        offender = tmp_path / "lambda_default.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "f = lambda fn=basicConfig: fn()\n"
            "f()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        # ``fn()`` is at column ~32 inside the lambda body on line 2.
        assert violations[0].line == 2

    def test_match_capture_in_function_is_local_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-14 P2: function-scope match capture is compile-time local.

        ::

            import logging
            def f(x):
                match x:
                    case logging:
                        pass
                logging.basicConfig()    # function-local logging → no flag

        The ``case logging`` capture makes ``logging`` a
        function-local of ``f`` for the entire body — references
        AFTER the match block no longer fall through to the
        module-level ``logging`` import. At runtime the post-
        match call would either invoke ``.basicConfig`` on the
        captured pattern value (when the case matched) or raise
        ``UnboundLocalError`` (when it didn't) — neither path
        invokes stdlib ``logging.basicConfig``.

        Module-level matches retain their existing behavior
        (see ``test_module_level_match_other_case_call_caught``).
        """
        clean = tmp_path / "match_capture_local.py"
        clean.write_text(
            "import logging\n"
            "def f(x):\n"
            "    match x:\n"
            "        case logging:\n"
            "            pass\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_match_capture_in_lambda_is_local_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-14 P2 lambda-scope variant.

        Lambdas don't directly support ``match`` (single-expression
        body), but a ``def`` nested inside a lambda's body can —
        and that nested ``def`` has its own function-scope rule.
        Anchor the function-scope semantic so lambdas with nested
        match captures don't regress to flagging.
        """
        clean = tmp_path / "lambda_match_local.py"
        clean.write_text(
            "import logging\n"
            "def f(x):\n"
            "    match x:\n"
            "        case _ as logging:\n"
            "            pass\n"
            "    logging.basicConfig()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_function_body_alias_resolves_outer_logging_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P1: function-body alias of OUTER ``logging`` flags.

        ::

            import logging
            def f():
                bc = logging.basicConfig    # outer-scope ref
                bc()

        Pre-fix the same-scope-only RHS analysis didn't see the
        module-level ``import logging`` from inside ``f``, so
        ``bc`` was recorded as ``"other"`` and ``bc()`` slipped
        past. Post-fix the visitor's ``_propagate_aliases`` pass
        re-evaluates the RHS via the full scope chain after the
        function scope is pushed, promoting ``bc`` to
        ``_BINDING_BASICCONFIG``.
        """
        offender = tmp_path / "func_outer_logging_alias.py"
        offender.write_text(
            "import logging\n"
            "def f():\n"
            "    bc = logging.basicConfig\n"
            "    bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_function_body_alias_from_outer_basicconfig_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P1 direct-import variant of cross-scope alias.

        ::

            from logging import basicConfig
            def f():
                bc = basicConfig
                bc()
        """
        offender = tmp_path / "func_outer_basicconfig_alias.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "def f():\n"
            "    bc = basicConfig\n"
            "    bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_branch_body_alias_resolves_outer_logging_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P1 branch variant of cross-scope alias.

        ::

            import logging
            cond = True
            if cond:
                bc = logging.basicConfig
                bc()

        The Assign sits inside an ``if`` branch but the RHS still
        needs to walk out to the module's ``logging``. With
        ``_propagate_aliases`` running inside the branch scope,
        the chain lookup pierces out correctly and the call flags.
        """
        offender = tmp_path / "branch_outer_logging_alias.py"
        offender.write_text(
            "import logging\n"
            "cond = True\n"
            "if cond:\n"
            "    bc = logging.basicConfig\n"
            "    bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_cross_scope_alias_chain_propagates(
        self, tmp_path: Path
    ) -> None:
        """Boundary anchor: cross-scope alias chain inside function.

        ::

            import logging
            def f():
                bc = logging.basicConfig    # cross-scope, line 3
                cc = bc                       # same-scope chain, line 4
                cc()

        Pins the source-order propagation: line 3's update lands
        first, so when line 4 evaluates ``bc`` via the chain
        ``_resolve`` returns ``_BINDING_BASICCONFIG``.
        """
        offender = tmp_path / "cross_chain.py"
        offender.write_text(
            "import logging\n"
            "def f():\n"
            "    bc = logging.basicConfig\n"
            "    cc = bc\n"
            "    cc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_annassign_alias_from_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P1: annotated ``bc: object = basicConfig`` flags.

        ::

            from logging import basicConfig
            bc: object = basicConfig
            bc()

        Pre-fix every ``AnnAssign`` was recorded as
        ``_BINDING_OTHER`` regardless of RHS, so a single type
        annotation laundered a real basicConfig alias into
        nothing the gate could catch. Post-fix
        ``_scope_bindings`` AnnAssign branch reuses the same
        ``_kind_of_assign_rhs`` logic when ``stmt.value`` is
        present.
        """
        offender = tmp_path / "annassign_alias_direct.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "bc: object = basicConfig\n"
            "bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_annassign_alias_from_logging_attribute_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P1 AnnAssign attribute variant.

        ::

            import logging
            bc: object = logging.basicConfig
            bc()
        """
        offender = tmp_path / "annassign_alias_attribute.py"
        offender.write_text(
            "import logging\n"
            "bc: object = logging.basicConfig\n"
            "bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_annassign_cross_scope_alias_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P1 AnnAssign cross-scope variant.

        ``def f(): bc: object = logging.basicConfig; bc()`` —
        annotated assign inside a function body whose RHS
        resolves to outer ``logging``. Same propagation path
        as plain ``Assign``.
        """
        offender = tmp_path / "annassign_cross_scope.py"
        offender.write_text(
            "import logging\n"
            "def f():\n"
            "    bc: object = logging.basicConfig\n"
            "    bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_tuple_unpack_propagates_basicconfig_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P2: ``bc, _ = basicConfig, None; bc()`` flags.

        Element-wise tuple unpacking pairs each LHS name with
        the corresponding RHS element. Pre-fix the whole RHS
        ``Tuple`` was treated as a single non-alias expression,
        so ``bc`` was recorded as ``"other"`` and the call slipped
        past. Post-fix ``_kind_of_value_for_target`` recurses on
        matching ``Tuple`` shapes so the basicConfig kind lands
        on the right LHS leaf.
        """
        offender = tmp_path / "tuple_unpack_basicconfig.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "bc, _ = basicConfig, None\n"
            "bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_tuple_unpack_propagates_logging_attribute_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P2 attribute variant.

        ::

            import logging
            bc, dd = logging.basicConfig, None
            bc()
        """
        offender = tmp_path / "tuple_unpack_attribute.py"
        offender.write_text(
            "import logging\n"
            "bc, dd = logging.basicConfig, None\n"
            "bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_list_unpack_propagates_basicconfig_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-15 P2 list-target variant — same rule applies.

        ::

            from logging import basicConfig
            [bc, _] = [basicConfig, None]
            bc()
        """
        offender = tmp_path / "list_unpack.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "[bc, _] = [basicConfig, None]\n"
            "bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_starred_unpack_does_not_misattribute(
        self, tmp_path: Path
    ) -> None:
        """Boundary anchor: starred unpack falls back to ``"other"``.

        ::

            from logging import basicConfig
            bc, *rest = basicConfig, None
            bc()

        With a ``Starred`` element the LHS shape doesn't pair
        1:1 with the RHS — Python takes the catch-all as a
        list. To stay sound, the structural matcher refuses to
        propagate when any target is starred and falls back to
        ``_BINDING_OTHER`` for every name. ``bc()`` thus does
        NOT flag (we simply can't prove the alias). Pins the
        sound-when-uncertain semantics.
        """
        clean = tmp_path / "starred_unpack.py"
        clean.write_text(
            "from logging import basicConfig\n"
            "bc, *rest = basicConfig, None\n"
            "bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_class_body_pre_rebind_call_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-16 P1: class body ``LOAD_NAME`` falls through to globals.

        ::

            import logging
            class C:
                logging.basicConfig()    # ← runs at class-def time, sees stdlib
                logging = object()        # ← rebinds class-local AFTER call

        Pre-fix the scanner treated class scope like a function-
        local (``LOAD_FAST``-style), so a forward reference
        before the local rebind returned ``_BINDING_OTHER`` and
        the call slipped past. Post-fix the early-return for
        "name bound later in scope" is restricted to function /
        lambda scopes; class / module / branch / handler / case
        fall through to outer scope chain — Python's
        ``LOAD_NAME`` semantics for class bodies hit globals
        when the local namespace doesn't yet have the name.
        """
        offender = tmp_path / "class_pre_rebind_call.py"
        offender.write_text(
            "import logging\n"
            "class C:\n"
            "    logging.basicConfig()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_class_body_alias_before_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-16 P1 class-body alias variant.

        ::

            import logging
            class C:
                bc = logging.basicConfig
                logging = object()
                bc()

        ``bc = logging.basicConfig`` evaluates the RHS while
        class-local ``logging`` is not yet bound, so it walks
        out to the module's import. Post-fix the cross-scope
        alias propagation succeeds and ``bc()`` flags.
        """
        offender = tmp_path / "class_body_alias_before_rebind.py"
        offender.write_text(
            "import logging\n"
            "class C:\n"
            "    bc = logging.basicConfig\n"
            "    logging = object()\n"
            "    bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_closure_alias_call_before_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-16 P2: ``alias = inner; alias()`` fires the closure.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                alias = inner       # direct alias
                alias()              # ← actual firing site BEFORE rebind
                logging = object()

        Pre-fix ``_LoadRefFinder`` only counted ``inner(...)``
        as a call site, treating ``alias = inner`` as pure escape
        and snapshotting at "after all events" → no flag. Post-
        fix ``_LoadRefFinder.visit_Assign`` records the simple
        ``alias = inner`` direct alias and ``visit_Call`` then
        treats ``alias()`` as a call site at its line, so the
        snapshot pins at the pre-rebind ``logging`` import.
        """
        offender = tmp_path / "closure_alias_call.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_walrus_callee_module_level_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-16 P2: ``(bc := logging.basicConfig)()`` flags.

        ::

            import logging
            (bc := logging.basicConfig)()

        Pre-fix ``visit_Call`` only matched ``Attribute`` /
        ``Name`` callees and walrus-as-callee slipped past.
        Post-fix the callee is unwrapped through any ``NamedExpr``
        and the inner ``logging.basicConfig`` is matched as
        usual — ``bc`` is also recorded as a basicConfig alias
        so any later reference flags too.
        """
        offender = tmp_path / "walrus_callee.py"
        offender.write_text(
            "import logging\n"
            "(bc := logging.basicConfig)()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 2

    def test_walrus_in_if_test_then_call_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-16 P2: walrus in ``if`` test binds for body call.

        ::

            import logging
            if (bc := logging.basicConfig):
                bc()

        ``bc`` is bound by the walrus in the if-test (executed
        regardless of branch outcome), then called inside the
        if-body. Post-fix the scanner records ``bc`` as a
        basicConfig alias at the walrus's end position, and the
        body's ``bc()`` resolves through that event.
        """
        offender = tmp_path / "walrus_if_test.py"
        offender.write_text(
            "import logging\n"
            "if (bc := logging.basicConfig):\n"
            "    bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 3

    def test_walrus_in_function_body_cross_scope_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-16 P2 cross-scope walrus inside function body.

        ::

            import logging
            def f():
                if (bc := logging.basicConfig):
                    bc()

        The walrus value resolves ``logging`` via the scope
        chain (function → module). Post-fix
        ``_propagate_aliases`` walks the function-body walrus
        targets in addition to plain Assigns / AnnAssigns.
        """
        offender = tmp_path / "walrus_func_cross_scope.py"
        offender.write_text(
            "import logging\n"
            "def f():\n"
            "    if (bc := logging.basicConfig):\n"
            "        bc()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_walrus_direct_import_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-16 P2 walrus with direct import.

        ::

            from logging import basicConfig
            (bc := basicConfig)()
        """
        offender = tmp_path / "walrus_direct_import.py"
        offender.write_text(
            "from logging import basicConfig\n"
            "(bc := basicConfig)()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 2

    def test_closure_annassign_alias_call_before_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-17 P2: ``alias: object = inner; alias()`` fires the closure.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                alias: object = inner
                alias()
                logging = object()

        Pre-fix ``_LoadRefFinder.visit_Assign`` only handled
        plain ``Assign`` direct aliases — the AnnAssign form
        was treated as pure escape, so the snapshot landed
        after the rebind and the call slipped past. Post-fix
        ``visit_AnnAssign`` records the alias the same way.
        """
        offender = tmp_path / "closure_annassign_alias.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias: object = inner\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_closure_tuple_unpack_alias_call_before_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-17 P2: ``alias, _ = inner, None; alias()`` fires the closure.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                alias, _ = inner, None
                alias()
                logging = object()

        Pre-fix the tuple-unpack ``Assign`` shape was opaque to
        the alias detector. Post-fix the visitor pairs LHS
        elements with RHS elements 1:1 (when shapes match and
        no ``Starred`` is present) and records each alias.
        """
        offender = tmp_path / "closure_tuple_unpack_alias.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias, _ = inner, None\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_closure_walrus_alias_call_before_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-17 P2: ``if (alias := inner): alias()`` fires the closure.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                if (alias := inner):
                    alias()
                logging = object()

        Pre-fix the walrus binding shape was treated as escape.
        Post-fix ``_LoadRefFinder.visit_NamedExpr`` records the
        alias the same way ``visit_Assign`` does.
        """
        offender = tmp_path / "closure_walrus_alias.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    if (alias := inner):\n"
            "        alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_closure_transitive_alias_chain_call_before_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-17 P2: ``a = inner; b = a; b()`` fires the closure.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                a = inner
                b = a
                b()
                logging = object()

        Pre-fix the alias detector only matched a single hop
        ``Name = target`` — a 2-hop chain ``a = target; b = a;
        b()`` left ``b`` as escape. Post-fix the detector also
        matches when the RHS Name is itself an already-known
        alias, propagating forward in source order.
        """
        offender = tmp_path / "closure_alias_chain.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    a = inner\n"
            "    b = a\n"
            "    b()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_closure_same_line_call_then_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-18 P2: same-line ``inner(); logging = object()`` flags.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                inner(); logging = object()

        Pre-fix the closure snapshot used a line-only compare
        (``evt_line <= snapshot_line``) so the post-call rebind
        on the same line was treated as already in effect, and
        the call slipped past. Post-fix
        ``_find_target_refs`` returns ``(call_line, call_col)``
        and the resolver compares ``(evt_line, evt_col) <=
        (call_line, call_col)`` — the rebind's end column is
        greater than the call's start column on the same line,
        so it's correctly excluded.
        """
        offender = tmp_path / "closure_same_line_call_rebind.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    inner(); logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_closure_same_line_alias_call_then_rebind_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-18 P2: same-line ``alias(); logging = object()`` flags.

        Variant of the column-aware fix using the alias path —
        ``alias`` was bound to ``inner`` on a prior line and
        called on the same line as the rebind.
        """
        offender = tmp_path / "closure_same_line_alias.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    alias(); logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_alias_killed_by_rebind_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-18 P2: ``alias = inner; alias = noop; alias()`` does not flag.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                def noop():
                    pass
                alias = inner
                alias = noop      # ← kills the alias
                alias()           # ← actually invokes noop, NOT inner
                logging = object()

        Pre-fix ``direct_aliases`` was a monotone set — once
        ``alias`` was added, a later ``alias = noop`` did NOT
        remove it, so the scanner treated ``alias()`` as a
        call to ``inner``. Post-fix ``alias_events`` is a
        position-aware list of ``(end_line, end_col, is_alias)``
        tuples, so ``_is_alias_at`` returns ``False`` once the
        rebind kicks in. The call to ``noop`` is then
        categorised as a non-target call (not an inner-fire
        site), pure-escape semantics take over for ``inner``,
        and the snapshot lands at events[-1] = ``"other"``
        from the rebind — no flag.
        """
        clean = tmp_path / "alias_killed.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def noop():\n"
            "        pass\n"
            "    alias = inner\n"
            "    alias = noop\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_alias_revived_after_kill_caught(
        self, tmp_path: Path
    ) -> None:
        """Boundary anchor: alias rebound to inner after a kill still flags.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                def noop():
                    pass
                alias = inner   # alias → True
                alias = noop    # alias → False
                alias = inner   # alias → True again
                alias()          # ← real call site for inner
                logging = object()

        Pins the position-aware alias state — at ``alias()``,
        the latest event is ``alias = inner`` so the call IS
        categorised as an inner fire site. Stops a future
        change that "remembers a kill forever" from regressing
        the legitimate-flag direction.
        """
        offender = tmp_path / "alias_revived.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def noop():\n"
            "        pass\n"
            "    alias = inner\n"
            "    alias = noop\n"
            "    alias = inner\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_walrus_callee_in_closure_caught(
        self, tmp_path: Path
    ) -> None:
        """Round-19 P2: ``(alias := inner)()`` walrus-as-callee fires the closure.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                (alias := inner)(); logging = object()

        Pre-fix ``_LoadRefFinder.visit_Call`` only matched
        ``Name`` callees, so the ``NamedExpr`` wrapper turned
        the call into pure escape and the snapshot landed
        after the rebind. Post-fix the callee is unwrapped
        through ``_unwrap_named_expr`` (mirroring the main
        ``_ScopeAwareVisitor.visit_Call``) and the inner Name
        is categorised as the call site.
        """
        offender = tmp_path / "walrus_closure_callee.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    (alias := inner)(); logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_conditional_alias_kill_still_flags(
        self, tmp_path: Path
    ) -> None:
        """Round-19 P2: kill inside ``if`` body is conditional, alias still possibly live.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                def noop():
                    pass
                alias = inner
                if flag:
                    alias = noop      # ← conditional kill
                alias()
                logging = object()

        Pre-fix any ``alias = noop`` past the inner anchor was
        treated as an unconditional kill, so when ``flag=False``
        at runtime the call would fire ``inner`` (real
        violation) but the scanner missed it. Post-fix the
        kill event carries ``is_conditional=True`` and
        ``_is_alias_at`` filters those out, keeping the prior
        ``alias=inner`` True event live → conservative flag.
        """
        offender = tmp_path / "alias_killed_conditional.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def noop():\n"
            "        pass\n"
            "    flag = True\n"
            "    alias = inner\n"
            "    if flag:\n"
            "        alias = noop\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_for_body_alias_kill_still_flags(
        self, tmp_path: Path
    ) -> None:
        """Round-19 P2 for-body variant of conditional kill.

        ``for _ in []: alias = noop`` — empty iter never runs,
        so alias is still inner at the post-loop call. The
        for-body kill must be conditional from outside.
        """
        offender = tmp_path / "alias_killed_for_body.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def noop():\n"
            "        pass\n"
            "    alias = inner\n"
            "    for _ in []:\n"
            "        alias = noop\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_def_kills_alias_no_flag(self, tmp_path: Path) -> None:
        """Round-19 P2: ``def alias(): pass`` rebinds alias — no flag.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                alias = inner
                def alias():     # ← rebinds alias
                    pass
                alias()           # ← invokes the new function, NOT inner
                logging = object()

        Pre-fix the def shape didn't emit a kill event for
        ``alias``, so the prior ``alias = inner`` stayed live
        and the call falsely flagged. Post-fix
        ``_LoadRefFinder.visit_FunctionDef`` records a kill
        event for the def's name in the enclosing scope
        (skipping the target function's own def).
        """
        clean = tmp_path / "def_kills_alias.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    def alias():\n"
            "        pass\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_class_kills_alias_no_flag(self, tmp_path: Path) -> None:
        """Round-19 P2: ``class alias: pass`` rebinds alias — no flag."""
        clean = tmp_path / "class_kills_alias.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    class alias:\n"
            "        pass\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_import_as_kills_alias_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-19 P2: ``import math as alias`` rebinds alias — no flag."""
        clean = tmp_path / "import_kills_alias.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    import math as alias\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_from_import_kills_alias_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-19 P2: ``from math import sqrt as alias`` rebinds alias — no flag."""
        clean = tmp_path / "from_import_kills_alias.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    from math import sqrt as alias\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_for_target_kill_is_conditional_still_flags(
        self, tmp_path: Path
    ) -> None:
        """Round-19 P2: for-target rebind is conditional — conservative flag.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                alias = inner
                for alias in []:    # ← zero-iter, alias still inner
                    pass
                alias()
                logging = object()

        For-target rebind only takes effect IF the loop
        iterates. Static analysis can't decide statically
        (and ``for alias in []`` literally won't iterate).
        Post-fix ``visit_For`` records the for-target kill
        with ``is_conditional=True`` so the conservative
        filter keeps the prior ``alias=inner`` True event
        and the call flags.
        """
        offender = tmp_path / "for_target_kill_conditional.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    for alias in []:\n"
            "        pass\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_with_as_kills_alias_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-19 P2: ``with EXPR as alias`` rebinds alias — no flag.

        ``with`` statement runs ``__enter__`` first and then
        binds the as-target before the body. From outside the
        with, the binding latches. Treat as unconditional kill.
        """
        clean = tmp_path / "with_kills_alias.py"
        clean.write_text(
            "import contextlib\n"
            "@contextlib.contextmanager\n"
            "def cm():\n"
            "    yield None\n"
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    with cm() as alias:\n"
            "        pass\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_except_as_kill_is_conditional_still_flags(
        self, tmp_path: Path
    ) -> None:
        """Round-19 P2: ``except E as alias`` is conditional — conservative flag.

        Handler binding only takes effect if the exception is
        actually caught. Treat as conditional kill so the
        prior ``alias=inner`` is preserved for the post-try
        call. Mirrors the binding resolver's
        conservative-flag direction.
        """
        offender = tmp_path / "except_alias_conditional.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    alias = inner\n"
            "    try:\n"
            "        pass\n"
            "    except Exception as alias:\n"
            "        pass\n"
            "    alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 4

    def test_in_branch_alias_kill_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-20 P1: same-branch kill suppresses subsequent in-branch call.

        ::

            def outer():
                import logging
                def inner():
                    logging.basicConfig()
                def noop():
                    pass
                alias = inner
                if flag:
                    alias = noop
                    alias()        # ← in-branch — alias is noop here
                logging = object()

        Pre-fix the conditional kill ``alias = noop`` was
        filtered globally by ``_is_alias_at``, so the
        in-branch ``alias()`` falsely resolved through the
        prior True ``alias = inner`` event and the scanner
        flagged ``inner``'s ``logging.basicConfig()``. Post-fix
        the alias-state filter checks branch lineage: the kill
        event's branch path is a prefix of the in-branch
        query's path, so the kill IS applied within its own
        branch. Outside the branch (post-if), the kill is
        filtered as before — see
        ``test_conditional_alias_kill_still_flags`` for the
        complementary post-branch anchor.
        """
        clean = tmp_path / "in_branch_alias_kill.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def noop():\n"
            "        pass\n"
            "    flag = True\n"
            "    alias = inner\n"
            "    if flag:\n"
            "        alias = noop\n"
            "        alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_in_for_body_alias_kill_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-20 P1 for-body variant — same-branch kill suppresses."""
        clean = tmp_path / "in_for_body_alias_kill.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def noop():\n"
            "        pass\n"
            "    alias = inner\n"
            "    for _ in [1]:\n"
            "        alias = noop\n"
            "        alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_in_handler_alias_kill_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-20 P1 except-handler variant — handler-local kill suppresses."""
        clean = tmp_path / "in_handler_alias_kill.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def noop():\n"
            "        pass\n"
            "    alias = inner\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        alias = noop\n"
            "        alias()\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_in_match_case_alias_kill_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-20 P1 match-case variant — case-local kill suppresses."""
        clean = tmp_path / "in_match_case_alias_kill.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def noop():\n"
            "        pass\n"
            "    alias = inner\n"
            "    match True:\n"
            "        case True:\n"
            "            alias = noop\n"
            "            alias()\n"
            "        case _:\n"
            "            pass\n"
            "    logging = object()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_same_name_def_shadows_inner_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-20 P2: same-name redef shadows the original function.

        ::

            import logging
            def inner():
                logging.basicConfig()
            def inner():
                pass
            inner()

        Pre-fix the scanner attributed the ``inner()`` call to
        the FIRST ``inner`` (whose body has the forbidden
        call), so the lint gate flagged dead code that runtime
        never executes. Post-fix ``_LoadRefFinder`` seeds an
        alias True event at the target function's
        ``def_end_pos`` and emits a kill for any same-name
        redef thereafter. With no live ref to inner_1 in the
        outer scope AND ``target_killed_in_scope=True``, the
        ``_resolve`` walker skips the def_line snapshot
        fallback for that scope and the closure-cell pre-def
        ``logging`` import no longer flags.
        """
        clean = tmp_path / "same_name_def_shadow.py"
        clean.write_text(
            "import logging\n"
            "def inner():\n"
            "    logging.basicConfig()\n"
            "def inner():\n"
            "    pass\n"
            "inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_same_name_def_followed_by_call_in_outer_scope_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Round-20 P2 nested variant — inner shadowed inside outer."""
        clean = tmp_path / "same_name_def_nested.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def inner():\n"
            "        pass\n"
            "    inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert violations == []

    def test_pre_def_assign_does_not_suppress_target_call(
        self, tmp_path: Path
    ) -> None:
        """Round-21 P2: ``inner = object()`` BEFORE ``def inner()`` doesn't suppress.

        ::

            def outer():
                import logging
                inner = object()        # ← pre-def kill (line 3)
                def inner():            # ← target def (line 4-5)
                    logging.basicConfig()
                inner()                  # ← runtime calls TARGET → flag

        Pre-fix the seed True event for ``inner`` was written
        in ``__init__`` BEFORE the walker started, so the
        line-3 kill ended up appended AFTER the seed in the
        events list. ``_is_alias_at`` then iterated in
        insertion order and treated the line-3 kill as the
        latest event, returning False at the call site →
        scanner skipped the call → no flag. Post-fix
        ``_is_alias_at`` scans ALL events and picks the one
        with maximal ``(line, col) ≤ query_pos``, regardless of
        insertion order — so the line-4 seed wins over the
        line-3 kill at the call line. ``target_killed`` is also
        position-aware so a pre-def kill no longer trips the
        dead-code skip in ``_resolve``.
        """
        offender = tmp_path / "pre_def_assign_kill.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    inner = object()\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_pre_def_class_does_not_suppress_target_call(
        self, tmp_path: Path
    ) -> None:
        """Round-21 P2 class variant: ``class inner: pass; def inner(): ...; inner()``."""
        offender = tmp_path / "pre_def_class_kill.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    class inner:\n"
            "        pass\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 6

    def test_pre_def_import_does_not_suppress_target_call(
        self, tmp_path: Path
    ) -> None:
        """Round-21 P2 import variant: ``import math as inner; def inner(): ...; inner()``."""
        offender = tmp_path / "pre_def_import_kill.py"
        offender.write_text(
            "def outer():\n"
            "    import logging\n"
            "    import math as inner\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    inner()\n",
            encoding="utf-8",
        )

        violations = no_logging_basicconfig.find_violations([tmp_path])

        assert len(violations) == 1
        assert violations[0].path == offender
        assert violations[0].line == 5

    def test_pre_def_kill_then_post_def_redef_no_flag(
        self, tmp_path: Path
    ) -> None:
        """Boundary anchor: pre-def kill + post-def redef still shadows.

        ::

            def outer():
                import logging
                inner = object()        # pre-def kill (irrelevant)
                def inner():            # target def
                    logging.basicConfig()
                def inner():            # post-def redef (real shadow)
                    pass
                inner()

        At runtime the third inner is invoked. With round-21's
        position-aware target_killed (counting ONLY kills after
        target_def_end_pos), the post-def redef at line 6 IS
        counted → ``target_killed_in_scope=True`` → resolver
        skips the def_line fallback → no flag. Pre-def kill at
        line 3 is excluded by the position filter so it doesn't
        falsely trip the dead-code path on its own.
        """
        clean = tmp_path / "pre_def_kill_post_def_redef.py"
        clean.write_text(
            "def outer():\n"
            "    import logging\n"
            "    inner = object()\n"
            "    def inner():\n"
            "        logging.basicConfig()\n"
            "    def inner():\n"
            "        pass\n"
            "    inner()\n",
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
