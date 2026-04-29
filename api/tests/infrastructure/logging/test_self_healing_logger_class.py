"""B5 PR-S1-3 acceptance: Q2 self-healing Logger subclass.

After ``setup_logging`` calls ``install_self_healing_logger_class``,
every NEW ``logging.getLogger(name)`` returns a
``_RedactingPropagateOnlyLogger`` instance:

- ``propagate`` is ``True`` so records reach the root
  ``RedactingFormatter``.
- ``addHandler`` is a no-op so a late-imported third-party library
  cannot attach its own ``StreamHandler`` and bypass root.

This test fixture saves and restores ``logging.getLoggerClass()`` so
the process-wide setLoggerClass mutation does not leak into other
tests' state.
"""
from __future__ import annotations

import io
import logging

import pytest

from app.infrastructure.logging.redaction import (
    RedactingFormatter,
    _RedactingPropagateOnlyLogger,
    clear_propagate_only_loggers,
    install_self_healing_logger_class,
)


@pytest.fixture
def restore_logger_class():
    """Snapshot and restore the process-wide default Logger class."""
    original = logging.getLoggerClass()
    yield original
    logging.setLoggerClass(original)


@pytest.fixture
def restore_named_loggers():
    """Snapshot and restore named loggers we mutate during the test."""
    names = (
        "test_late_imported_lib_xyz",
        "test_lazy_openai_client_alpha",
        "test_lazy_httpx_client_beta",
    )
    snapshots: dict[str, tuple[list[logging.Handler], bool]] = {}
    for name in names:
        logger = logging.getLogger(name)
        snapshots[name] = (logger.handlers[:], logger.propagate)
    yield
    for name, (handlers, propagate) in snapshots.items():
        logger = logging.getLogger(name)
        logger.handlers = handlers
        logger.propagate = propagate


class TestSelfHealingLoggerClass:
    def test_install_changes_default_logger_class(
        self, restore_logger_class
    ):
        install_self_healing_logger_class()
        assert logging.getLoggerClass() is _RedactingPropagateOnlyLogger

    def test_new_logger_after_install_is_subclass(
        self, restore_logger_class, restore_named_loggers
    ):
        install_self_healing_logger_class()

        late_lib = logging.getLogger("test_late_imported_lib_xyz")
        assert isinstance(late_lib, _RedactingPropagateOnlyLogger)
        assert late_lib.propagate is True

    def test_late_lib_addhandler_is_noop(
        self, restore_logger_class, restore_named_loggers
    ):
        install_self_healing_logger_class()

        late_lib = logging.getLogger("test_lazy_openai_client_alpha")
        rogue_handler = logging.StreamHandler(io.StringIO())
        late_lib.addHandler(rogue_handler)

        assert rogue_handler not in late_lib.handlers
        assert late_lib.handlers == []

    def test_late_lib_log_propagates_through_root_redacting_formatter(
        self, restore_logger_class, restore_named_loggers
    ):
        install_self_healing_logger_class()

        captured = io.StringIO()
        root_handler = logging.StreamHandler(captured)
        root_handler.setFormatter(RedactingFormatter("%(message)s"))
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        root.handlers = [root_handler]
        root.setLevel(logging.INFO)

        try:
            late_lib = logging.getLogger("test_lazy_httpx_client_beta")
            late_lib.addHandler(logging.StreamHandler(io.StringIO()))
            late_lib.setLevel(logging.INFO)
            late_lib.info(
                "downstream call sk-aaaaaaaaaaaaaaaaaaaa1234bbbb"
            )

            output = captured.getvalue()
            assert "[REDACTED]" in output
            assert "aaaaaaaaaaaa" not in output
        finally:
            root.handlers = original_handlers
            root.setLevel(original_level)

    def test_clear_propagate_only_loggers_drops_existing_handlers(
        self, restore_logger_class, restore_named_loggers
    ):
        before = logging.getLogger("test_late_imported_lib_xyz")
        sentinel_handler = logging.StreamHandler(io.StringIO())
        before.addHandler(sentinel_handler)
        before.propagate = False

        clear_propagate_only_loggers(["test_late_imported_lib_xyz"])

        assert sentinel_handler not in before.handlers
        assert before.handlers == []
        assert before.propagate is True

    def test_root_logger_is_unaffected_by_setloggerclass(
        self, restore_logger_class
    ):
        # The root logger is created once at module import and keeps
        # its real logging.RootLogger class. setLoggerClass only
        # affects future named loggers — root must still accept
        # addHandler so setup_logging can install the RedactingFormatter
        # handler.
        install_self_healing_logger_class()

        root = logging.getLogger()
        assert not isinstance(root, _RedactingPropagateOnlyLogger)
        probe_handler = logging.StreamHandler(io.StringIO())
        try:
            root.addHandler(probe_handler)
            assert probe_handler in root.handlers
        finally:
            root.removeHandler(probe_handler)
