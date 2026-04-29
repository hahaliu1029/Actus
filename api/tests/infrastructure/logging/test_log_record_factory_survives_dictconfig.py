"""B5 PR-S1-4 acceptance: LogRecord factory survives ``dictConfig``.

Uvicorn / gunicorn typically call ``logging.config.dictConfig`` on
startup (or on ``--log-config`` reload) to install their own logger
hierarchy. ``dictConfig`` rebuilds Logger / Handler / Formatter
instances but does not reset the module-level
``logging.setLogRecordFactory`` slot. We pin that contract here:
after ``dictConfig`` runs, the actus factory is still the registered
factory and continues to inject ``trace_id`` / ``request_id`` /
``session_id`` on every emitted record.

Regression target: a future refactor that wraps the factory in a
context manager or moves it inside ``setup_logging`` only would
break uvicorn integrations and surface as silent attribute errors
in production formatters. This test catches that drift.
"""
from __future__ import annotations

import logging
import logging.config

from app.infrastructure.logging.logging import (
    _actus_log_record_factory,
    _install_log_record_factory,
)


def test_factory_persists_after_dictconfig() -> None:
    """``dictConfig`` rebuilds loggers but leaves the factory slot intact."""
    _install_log_record_factory()
    assert logging.getLogRecordFactory() is _actus_log_record_factory

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "plain": {"format": "%(levelname)s %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "plain",
                }
            },
            "root": {"level": "INFO", "handlers": ["console"]},
        }
    )

    assert logging.getLogRecordFactory() is _actus_log_record_factory


def test_record_emitted_after_dictconfig_still_has_actus_attrs() -> None:
    """Records produced post-``dictConfig`` carry the Q6 attrs."""
    _install_log_record_factory()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "console": {"class": "logging.StreamHandler", "level": "INFO"}
            },
            "root": {"level": "INFO", "handlers": ["console"]},
        }
    )

    record = logging.getLogRecordFactory()(
        "test_post_dictconfig",
        logging.INFO,
        "/path/file.py",
        1,
        "hello",
        None,
        None,
    )

    assert hasattr(record, "trace_id")
    assert hasattr(record, "request_id")
    assert hasattr(record, "session_id")
    # Outside any request scope the values fall back to "-".
    assert record.trace_id == "-"
    assert record.request_id == "-"
    assert record.session_id == "-"


def test_install_factory_is_idempotent() -> None:
    """Re-invoking the installer does not double-wrap the factory."""
    _install_log_record_factory()
    first = logging.getLogRecordFactory()

    _install_log_record_factory()
    second = logging.getLogRecordFactory()

    assert first is second is _actus_log_record_factory
