"""B5 PR-S1-4 acceptance: per-helper unit + composer integration tests (Q1).

The original 35-line ``setup_logging()`` is split into five composable
helpers — each is exercised here in isolation, then the composer is
exercised as a whole. The split is the main Q1 deliverable: shipping
the helpers without unit tests would erase the testability win that
motivated the refactor.

Test surface:

- ``_install_redacting_formatter`` clears existing handlers and
  installs exactly one ``StreamHandler`` whose formatter is a
  ``RedactingFormatter``.
- ``_install_file_handlers`` adds two
  ``ConcurrentRotatingFileHandler`` instances pointed at
  ``agent.log`` (5 MB × 5) and ``errors.log`` (2 MB × 3), or degrades
  silently to a stdout warning if the directory cannot be written.
- ``_install_third_party_isolation`` swaps known noisy loggers'
  ``__class__`` to the propagate-only subclass.
- ``_install_component_filter`` attaches ``_ComponentFilter`` to each
  root handler exactly once (idempotent on re-invocation).
- ``setup_logging()`` end-to-end produces the expected handler stack:
  one stdout + two file (or only stdout under degraded path) + every
  handler carries the component filter.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from concurrent_log_handler import ConcurrentRotatingFileHandler

from app.infrastructure.logging.logging import (
    _AGENT_LOG_BACKUPS,
    _AGENT_LOG_FILENAME,
    _AGENT_LOG_MAX_BYTES,
    _ERROR_LOG_BACKUPS,
    _ERROR_LOG_FILENAME,
    _ERROR_LOG_MAX_BYTES,
    _ComponentFilter,
    _install_component_filter,
    _install_file_handlers,
    _install_redacting_formatter,
    _install_third_party_isolation,
    setup_logging,
)
from app.infrastructure.logging.redaction import (
    RedactingFormatter,
    _RedactingPropagateOnlyLogger,
)


class TestInstallRedactingFormatter:
    def test_replaces_existing_handlers_with_single_redacting_stdout(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        prior = logging.NullHandler()
        isolated_root_logger.addHandler(prior)

        _install_redacting_formatter(isolated_root_logger, logging.INFO)

        assert len(isolated_root_logger.handlers) == 1
        installed = isolated_root_logger.handlers[0]
        assert isinstance(installed, logging.StreamHandler)
        assert isinstance(installed.formatter, RedactingFormatter)
        assert installed.level == logging.INFO

    def test_idempotent_repeated_install(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        _install_redacting_formatter(isolated_root_logger, logging.INFO)
        _install_redacting_formatter(isolated_root_logger, logging.INFO)

        assert len(isolated_root_logger.handlers) == 1

    def test_removed_handlers_get_closed(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        """Review-found P2: removed handlers must be ``.close()``-ed.

        ``ConcurrentRotatingFileHandler`` holds an OS file lock + open
        file descriptor; without ``close()`` on remove, repeated
        ``setup_logging()`` invocations leak both. We verify with a
        sentinel handler that the helper closes whatever it removes.
        """

        class _Sentinel(logging.NullHandler):
            def __init__(self) -> None:
                super().__init__()
                self.was_closed = False

            def close(self) -> None:  # type: ignore[override]
                self.was_closed = True
                super().close()

        sentinel = _Sentinel()
        isolated_root_logger.addHandler(sentinel)

        _install_redacting_formatter(isolated_root_logger, logging.INFO)

        assert sentinel.was_closed is True
        assert sentinel not in isolated_root_logger.handlers

    def test_close_failure_does_not_break_install(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        """A misbehaving handler's ``close()`` must not break the install."""

        class _Misbehaving(logging.NullHandler):
            def close(self) -> None:  # type: ignore[override]
                raise RuntimeError("third-party close exploded")

        isolated_root_logger.addHandler(_Misbehaving())

        _install_redacting_formatter(isolated_root_logger, logging.INFO)

        # Install completed; canonical stdout handler is in place.
        assert len(isolated_root_logger.handlers) == 1
        assert isinstance(isolated_root_logger.handlers[0], logging.StreamHandler)


class TestInstallFileHandlers:
    def test_adds_agent_and_error_handlers_with_spec_limits(
        self,
        isolated_root_logger: logging.Logger,
        tmp_path: Path,
    ) -> None:
        for h in list(isolated_root_logger.handlers):
            isolated_root_logger.removeHandler(h)

        _install_file_handlers(
            isolated_root_logger, str(tmp_path), logging.INFO
        )

        file_handlers = [
            h
            for h in isolated_root_logger.handlers
            if isinstance(h, ConcurrentRotatingFileHandler)
        ]
        assert len(file_handlers) == 2

        agent_h = next(
            h
            for h in file_handlers
            if h.baseFilename.endswith(_AGENT_LOG_FILENAME)
        )
        error_h = next(
            h
            for h in file_handlers
            if h.baseFilename.endswith(_ERROR_LOG_FILENAME)
        )

        assert agent_h.maxBytes == _AGENT_LOG_MAX_BYTES
        assert agent_h.backupCount == _AGENT_LOG_BACKUPS
        assert agent_h.level == logging.INFO
        assert isinstance(agent_h.formatter, RedactingFormatter)

        assert error_h.maxBytes == _ERROR_LOG_MAX_BYTES
        assert error_h.backupCount == _ERROR_LOG_BACKUPS
        assert error_h.level == logging.WARNING
        assert isinstance(error_h.formatter, RedactingFormatter)

    def test_creates_log_dir_if_missing(
        self,
        isolated_root_logger: logging.Logger,
        tmp_path: Path,
    ) -> None:
        nested = tmp_path / "nested" / "logs"
        assert not nested.exists()

        _install_file_handlers(
            isolated_root_logger, str(nested), logging.INFO
        )

        assert nested.is_dir()

    def test_degrades_to_stdout_on_oserror(
        self,
        isolated_root_logger: logging.Logger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Permission denied (or any OSError) → warning, no crash."""

        def _raise_oserror(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "mkdir", _raise_oserror)

        _install_redacting_formatter(isolated_root_logger, logging.WARNING)
        before = len(isolated_root_logger.handlers)

        _install_file_handlers(
            isolated_root_logger, str(tmp_path), logging.INFO
        )

        after = len(isolated_root_logger.handlers)
        assert after == before
        assert all(
            not isinstance(h, ConcurrentRotatingFileHandler)
            for h in isolated_root_logger.handlers
        )


class TestInstallThirdPartyIsolation:
    def test_known_noisy_loggers_are_propagate_only_after_install(
        self,
    ) -> None:
        # Pre-condition: ``setup_logging`` ran in conftest's ``app.main``
        # import, so isolation is already in effect — calling again is the
        # idempotent path.
        _install_third_party_isolation()

        for name in ("openai", "httpx", "langchain", "uvicorn.access"):
            lg = logging.getLogger(name)
            assert isinstance(lg, _RedactingPropagateOnlyLogger), (
                f"{name} not class-swapped; got {type(lg).__name__}"
            )
            assert lg.propagate is True


class TestInstallComponentFilter:
    def test_attaches_filter_to_every_root_handler(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        _install_redacting_formatter(isolated_root_logger, logging.INFO)
        for h in isolated_root_logger.handlers:
            for f in list(h.filters):
                h.removeFilter(f)

        _install_component_filter(isolated_root_logger)

        for handler in isolated_root_logger.handlers:
            assert any(
                isinstance(f, _ComponentFilter) for f in handler.filters
            ), "no _ComponentFilter on handler"

    def test_idempotent_does_not_stack_duplicate_filters(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        _install_redacting_formatter(isolated_root_logger, logging.INFO)

        _install_component_filter(isolated_root_logger)
        _install_component_filter(isolated_root_logger)
        _install_component_filter(isolated_root_logger)

        for handler in isolated_root_logger.handlers:
            n_filters = sum(
                1 for f in handler.filters if isinstance(f, _ComponentFilter)
            )
            assert n_filters == 1, (
                f"handler {handler} has {n_filters} _ComponentFilters; "
                "expected exactly 1"
            )


class TestComponentFilterBehaviour:
    @pytest.mark.parametrize(
        "logger_name",
        ["httpx", "httpx.client", "httpcore", "asyncio", "openai", "langchain.X"],
    )
    def test_drops_info_from_noisy_loggers(self, logger_name: str) -> None:
        component = _ComponentFilter()
        record = logging.LogRecord(
            name=logger_name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="noise",
            args=None,
            exc_info=None,
        )
        assert component.filter(record) is False

    @pytest.mark.parametrize(
        "logger_name",
        ["app.main", "my.module", "tests"],
    )
    def test_keeps_info_from_app_loggers(self, logger_name: str) -> None:
        component = _ComponentFilter()
        record = logging.LogRecord(
            name=logger_name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="signal",
            args=None,
            exc_info=None,
        )
        assert component.filter(record) is True

    def test_keeps_warning_from_noisy_loggers(self) -> None:
        component = _ComponentFilter()
        record = logging.LogRecord(
            name="httpx",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="real problem",
            args=None,
            exc_info=None,
        )
        assert component.filter(record) is True


class TestSetupLoggingIntegration:
    def test_setup_logging_produces_full_handler_stack(
        self,
        isolated_root_logger: logging.Logger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core import config as config_mod
        from core.config import Settings

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

        setup_logging()

        stream_handlers = [
            h
            for h in isolated_root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, ConcurrentRotatingFileHandler)
        ]
        file_handlers = [
            h
            for h in isolated_root_logger.handlers
            if isinstance(h, ConcurrentRotatingFileHandler)
        ]
        assert len(stream_handlers) == 1
        assert len(file_handlers) == 2

        for handler in isolated_root_logger.handlers:
            assert any(
                isinstance(f, _ComponentFilter) for f in handler.filters
            )

        agent_log_path = tmp_path / _AGENT_LOG_FILENAME
        assert agent_log_path.is_file()

    def test_setup_logging_is_idempotent_no_handler_stacking(
        self,
        isolated_root_logger: logging.Logger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from core import config as config_mod
        from core.config import Settings

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

        setup_logging()
        first = len(isolated_root_logger.handlers)
        setup_logging()
        second = len(isolated_root_logger.handlers)
        setup_logging()
        third = len(isolated_root_logger.handlers)

        assert first == second == third

    def test_setup_logging_reinvocation_closes_old_file_handlers(
        self,
        isolated_root_logger: logging.Logger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-running ``setup_logging`` must close prior file handlers.

        Review-found P2 regression: without ``close()`` on remove,
        the prior ``ConcurrentRotatingFileHandler`` keeps its file
        lock + stream open, so a uvicorn lifespan reload (or any
        idempotent re-install) silently leaks file descriptors. We
        wrap ``ConcurrentRotatingFileHandler`` with a tracking
        subclass that flips ``was_closed`` when ``close()`` runs and
        assert every first-round handler is closed after the second
        ``setup_logging()`` call (proves the ``close()`` contract,
        independent of the lazy stream-open timing of
        ``ConcurrentRotatingFileHandler``).
        """
        from core import config as config_mod
        from core.config import Settings

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

        instances: list[ConcurrentRotatingFileHandler] = []

        class _TrackingHandler(ConcurrentRotatingFileHandler):
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                super().__init__(*args, **kwargs)
                self.was_closed = False
                instances.append(self)

            def close(self) -> None:  # type: ignore[override]
                self.was_closed = True
                super().close()

        monkeypatch.setattr(
            "app.infrastructure.logging.logging.ConcurrentRotatingFileHandler",
            _TrackingHandler,
        )

        setup_logging()
        first_round = list(instances)
        assert len(first_round) == 2, (
            f"expected 2 file handlers, got {len(first_round)}"
        )
        for h in first_round:
            assert h.was_closed is False, (
                f"{h.baseFilename} closed prematurely"
            )

        setup_logging()

        for h in first_round:
            assert h.was_closed is True, (
                f"{h.baseFilename} not closed on re-invocation — file lock leak"
            )
