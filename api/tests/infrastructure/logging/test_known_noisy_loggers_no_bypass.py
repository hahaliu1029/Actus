"""B5 PR-S1-3 regression: no bypass on the default noisy-logger list.

Reviewer caught an ordering bug — ``clear_propagate_only_loggers``
ran BEFORE ``install_self_healing_logger_class``, so each name in
``_KNOWN_NOISY_LOGGERS`` was instantiated as plain ``logging.Logger``.
``setLoggerClass`` only affects FUTURE ``getLogger(name)`` calls —
existing instances keep their class. Result: ``openai.addHandler(...)``
silently attached a rogue handler that bypassed the root
``RedactingFormatter``.

Fix landed in two places:

- ``setup_logging`` now calls ``install_self_healing_logger_class``
  BEFORE ``clear_propagate_only_loggers``.
- ``clear_propagate_only_loggers`` mutates ``logger.__class__`` in
  place to upgrade existing instances to the subclass — closing the
  bypass for SDKs that were imported before ``setup_logging`` ran.

This test runs after conftest has imported ``app.main`` (which fires
``setup_logging`` at module load), so by the time these assertions
execute the production setup is in effect. The test then probes
EVERY name in the default noisy list and confirms ``addHandler`` is
a no-op.
"""
from __future__ import annotations

import io
import logging

import pytest

from app.infrastructure.logging.redaction import (
    _KNOWN_NOISY_LOGGERS,
    _RedactingPropagateOnlyLogger,
)


@pytest.fixture(autouse=True)
def _ensure_setup_logging_has_fired():
    """``app.main`` import (done by conftest) calls ``setup_logging``
    at module load. Re-import is a no-op (Python module cache); we
    just confirm the side effect via this fixture so the test never
    runs before the production setup."""
    import app.main  # noqa: F401  (side effect: setup_logging())


class TestKnownNoisyLoggersNoBypass:
    @pytest.mark.parametrize("logger_name", list(_KNOWN_NOISY_LOGGERS))
    def test_logger_is_propagate_only_subclass(self, logger_name):
        logger = logging.getLogger(logger_name)
        assert isinstance(logger, _RedactingPropagateOnlyLogger), (
            f"{logger_name} is {type(logger).__name__}; expected "
            "_RedactingPropagateOnlyLogger so addHandler bypass is impossible"
        )

    @pytest.mark.parametrize("logger_name", list(_KNOWN_NOISY_LOGGERS))
    def test_logger_addhandler_is_noop(self, logger_name):
        logger = logging.getLogger(logger_name)
        before = list(logger.handlers)
        sentinel = logging.StreamHandler(io.StringIO())
        try:
            logger.addHandler(sentinel)
            assert sentinel not in logger.handlers, (
                f"{logger_name}.addHandler attached {sentinel!r} — "
                "self-heal failed; rogue handler can bypass root "
                "RedactingFormatter"
            )
            assert logger.handlers == before, (
                f"{logger_name}.handlers mutated by addHandler call"
            )
        finally:
            logger.handlers = before

    @pytest.mark.parametrize("logger_name", list(_KNOWN_NOISY_LOGGERS))
    def test_logger_propagate_is_true(self, logger_name):
        # Without propagate=True, records emitted by these loggers
        # never reach root and never run through RedactingFormatter.
        logger = logging.getLogger(logger_name)
        assert logger.propagate is True, (
            f"{logger_name}.propagate is False — records skip root"
        )


class TestNoisyListHasUvicornAccess:
    """Lock the uvicorn.access membership separately so a rename or
    accidental drop from the default list fails fast — this is the
    one entry whose removal would re-open the access-log query-string
    leak that PR-S1-3 specifically targets."""

    def test_uvicorn_access_in_default_list(self):
        assert "uvicorn.access" in _KNOWN_NOISY_LOGGERS

    def test_huggingface_hub_in_default_list(self):
        # Reviewer probe found huggingface_hub leaking raw secrets via
        # its local StreamHandler (transitive dep through
        # langchain / sentence-transformers).
        assert "huggingface_hub" in _KNOWN_NOISY_LOGGERS

    def test_transformers_in_default_list(self):
        # transformers ships with propagate=False, so even with handlers
        # cleared its records would never reach root without the
        # registry-wide sweep.
        assert "transformers" in _KNOWN_NOISY_LOGGERS

    def test_default_list_size(self):
        # 10 entries: 7 SDK loggers + uvicorn.access + 2 transitive
        # deps (huggingface_hub, transformers). If this changes, the
        # spec needs to follow.
        assert len(_KNOWN_NOISY_LOGGERS) == 10


class TestNoLoggerBypassesRoot:
    """Registry-wide sweep regression.

    Beyond the explicit ``_KNOWN_NOISY_LOGGERS`` list, the production
    setup must run ``isolate_all_non_root_loggers`` to catch transitive
    dependencies the spec does not enumerate. These two tests walk the
    global ``logging.Logger.manager.loggerDict`` and confirm:

    - No non-root logger retains a non-NullHandler local handler
      (which would emit raw records that bypass the root
      ``RedactingFormatter``).
    - No non-root logger has ``propagate=False`` (which would prevent
      its records from ever reaching root).
    """

    def test_no_non_root_logger_has_non_null_handler(self):
        leaks: list[tuple[str, list[str]]] = []
        for name, item in logging.Logger.manager.loggerDict.items():
            if not isinstance(item, logging.Logger):
                continue
            non_null = [
                h
                for h in item.handlers
                if not isinstance(h, logging.NullHandler)
            ]
            if non_null:
                leaks.append((name, [type(h).__name__ for h in non_null]))

        assert not leaks, (
            "Non-root loggers retain non-NullHandler local handlers "
            f"(records bypass root RedactingFormatter): {leaks}"
        )

    def test_no_non_root_logger_has_propagate_false(self):
        broken: list[str] = []
        for name, item in logging.Logger.manager.loggerDict.items():
            if not isinstance(item, logging.Logger):
                continue
            if not item.propagate:
                broken.append(name)

        assert not broken, (
            "Non-root loggers with propagate=False (records skip root "
            f"RedactingFormatter): {broken}"
        )
