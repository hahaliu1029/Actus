"""B5 PR-S1-7b CI lint gates (AST-based).

Each module in this package is a stand-alone CLI script:

    python api/tools/lint/no_logging_basicconfig.py [PATHS...]
    python api/tools/lint/no_print_in_backend.py [PATHS...]

When run with no PATHS, each gate uses sensible defaults pinned to
the Actus backend layout (``api/app`` + ``api/scripts`` for
``no_print_in_backend``; same plus ``api/tools`` for
``no_logging_basicconfig``). Exit 0 = clean, exit 1 = violation
(plus ``file:line`` diagnostics on stderr).

The gates run in CI before the pytest step so a regression cannot
ride into ``main`` even if no Python test imports the offending
module. ``test_lint_gates.py`` exercises each gate against a
synthetic temp directory so the AST logic itself stays trustworthy.
"""
