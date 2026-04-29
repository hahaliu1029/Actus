"""Application logging composer.

B5 PR-S1-4 splits the original 35-line ``setup_logging()`` into a
composer + 5 isolated helpers (Q1) and adds three responsibilities on
top of the PR-S1-3 redacting console:

- ``_actus_log_record_factory`` (Q6) wraps the stdlib ``LogRecord``
  factory to inject ``trace_id`` / ``request_id`` / ``session_id``
  fields from the per-request ``TraceContext`` carrier. The factory is
  defensive: any exception (contextvar uninitialised, boot-time
  circular import, attribute missing) falls through the ``except``
  arm and pins all three fields to ``"-"`` so downstream formatters
  using ``%(trace_id)s`` never blow up with ``AttributeError``.
- ``ConcurrentRotatingFileHandler`` is wired for ``agent.log`` (INFO+,
  5 MB × 5 backups) and ``errors.log`` (WARNING+, 2 MB × 3 backups).
  The concurrent variant is rotation-safe under multi-worker uvicorn —
  plain ``RotatingFileHandler`` races on ``os.rename``.
- ``_ComponentFilter`` suppresses INFO/DEBUG noise from
  ``httpx`` / ``httpcore`` / ``asyncio`` / ``openai`` / ``langchain``.
  It is attached to root *handlers* (not the root logger) because
  records propagated up from child loggers bypass parent
  ``Logger.filter`` — only handler filters run on the propagation path.

Idempotency
-----------
Every helper is safe to call repeatedly; ``setup_logging()`` itself can
be re-invoked (uvicorn lifespan reload, test fixtures) without stacking
duplicate handlers / filters / factories.

File-handler degradation
------------------------
If the configured ``log_dir`` is unwritable (typical for local dev
running tests outside the docker bind mount), ``_install_file_handlers``
catches ``OSError`` and emits a warning to the already-installed stdout
handler. Stdout logging stays operational; the run is not aborted.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

from concurrent_log_handler import ConcurrentRotatingFileHandler

from app.infrastructure.logging.redaction import (
    RedactingFormatter,
    clear_propagate_only_loggers,
    install_self_healing_logger_class,
    isolate_all_non_root_loggers,
)
from core.config import get_settings

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Q6 LogRecord 字段名 + 缺省值 ``"-"``。值不能是 ``None``，否则 formatter
# 用 ``%(trace_id)s`` 时会输出字面量 ``"None"`` 污染 grep。
_LOG_RECORD_FIELDS: tuple[str, ...] = ("trace_id", "request_id", "session_id")
_LOG_RECORD_DEFAULT: str = "-"

# 噪声第三方 logger，``_ComponentFilter`` 抑制其 INFO / DEBUG 流。
_NOISY_COMPONENT_PREFIXES: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "asyncio",
    "openai",
    "langchain",
)
_NOISY_COMPONENT_LEVEL: int = logging.WARNING

# Rotating file handler 参数。spec 见 design doc Sprint 1 §"PR-S1-4"。
_AGENT_LOG_FILENAME = "agent.log"
_AGENT_LOG_MAX_BYTES = 5 * 1024 * 1024
_AGENT_LOG_BACKUPS = 5

_ERROR_LOG_FILENAME = "errors.log"
_ERROR_LOG_MAX_BYTES = 2 * 1024 * 1024
_ERROR_LOG_BACKUPS = 3


# ---------------------------------------------------------------------------
# LogRecord factory (Q6) — defensive try/except, never raises
# ---------------------------------------------------------------------------
# 首次 ``_install_log_record_factory`` 时捕获当时的 factory，作为 wrap
# base；这样我们可以与早于 setup_logging 安装 factory 的库（OTel SDK 等）
# 共存——把它们的 record 作为输入，再追加 trace/session/request 字段。
#
# **once-only capture（review-found）**：``_FACTORY_INSTALLED`` 一旦置 True
# 后，再次调用 ``_install_log_record_factory()`` 只会重设 factory 指针、
# 不重新捕获 ``_ORIG_LOG_RECORD_FACTORY``。否则典型递归场景会触发：
#   1. setup_logging() 首次安装；orig=stdlib LogRecord。
#   2. 外部 lib wrap 我们：``setLogRecordFactory(wrapper)``，wrapper 内部
#      ``actus_factory(*args)``。
#   3. setup_logging() 第二次调用——若没有 flag，``current`` 是 wrapper、
#      被捕获为 ``_ORIG``。setLogRecordFactory 又装回 ``actus_factory``。
#      此后 ``actus_factory → _ORIG(=wrapper) → actus_factory → ...``
#      ``RecursionError: maximum recursion depth exceeded``。
# Flag 切断该路径：第二次 install 不重新捕获、保留首次的 stdlib base，
# wrapper 被换下台是预期成本。
_ORIG_LOG_RECORD_FACTORY: Callable[..., logging.LogRecord] | None = None
_FACTORY_INSTALLED: bool = False


def _set_default_attrs(record: logging.LogRecord) -> None:
    """Pin ``trace_id`` / ``request_id`` / ``session_id`` to ``"-"`` on ``record``."""
    for field in _LOG_RECORD_FIELDS:
        setattr(record, field, _LOG_RECORD_DEFAULT)


def _actus_log_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """Build a ``LogRecord`` augmented with trace/request/session attrs.

    Wraps either the stdlib ``LogRecord`` constructor or any
    pre-existing factory captured at install time. **Both the base
    call AND the trace-injection block are wrapped in ``try/except``
    per Q6** — review-found regression: prior implementation left
    ``base(*args, **kwargs)`` outside the try, so a captured base
    factory that raised would propagate the exception out of
    ``_actus_log_record_factory`` and crash the very logging path it
    was supposed to protect.

    Failure ladder:
    1. Try the captured base factory. If it raises, fall back to the
       stdlib ``logging.LogRecord`` constructor.
    2. If stdlib construction itself fails, re-raise — that is a
       genuine programmer error (bad ``args``/``kwargs`` shape) and
       no fallback record could be valid.
    3. If trace-injection raises (contextvar broken, circular import,
       missing attr), pin all three Q6 fields to ``"-"``.
    """
    record: logging.LogRecord
    try:
        base = _ORIG_LOG_RECORD_FACTORY or logging.LogRecord
        record = base(*args, **kwargs)
    except Exception:
        # Captured base factory exploded — fall back to stdlib direct.
        # If THIS also raises, let it propagate; the args themselves
        # are malformed and no LogRecord shape is recoverable.
        record = logging.LogRecord(*args, **kwargs)
    try:
        # Local import: ``app.infrastructure.observability.context``
        # imports ``app.domain.external.observability``; both modules
        # may be in the middle of their own import chain when the very
        # first LogRecord is created. Defensive ImportError handling
        # lives in the outer ``except``.
        from app.infrastructure.observability.context import get_trace_context

        ctx = get_trace_context()
        if ctx is None:
            _set_default_attrs(record)
        else:
            record.trace_id = (
                getattr(ctx, "trace_id", None) or _LOG_RECORD_DEFAULT
            )
            record.request_id = (
                getattr(ctx, "request_id", None) or _LOG_RECORD_DEFAULT
            )
            record.session_id = (
                getattr(ctx, "session_id", None) or _LOG_RECORD_DEFAULT
            )
    except Exception:
        _set_default_attrs(record)
    return record


def _install_log_record_factory() -> None:
    """Idempotently install the actus LogRecord factory.

    First call captures ``logging.getLogRecordFactory()`` as the wrap
    base and sets ``_FACTORY_INSTALLED = True``. Subsequent calls
    only re-set the factory pointer to ``_actus_log_record_factory``
    — they do NOT re-capture, because a third-party wrapper installed
    between the first and the n-th call would otherwise be captured
    as the new base, and the wrapper internally calling our factory
    would recurse infinitely.

    ``logging.config.dictConfig`` rebuilds Logger / Handler /
    Formatter instances but does not reset the module-level factory
    slot, so the wrapper survives uvicorn / gunicorn config reloads
    — the dedicated ``test_log_record_factory_survives_dictconfig.py``
    regression pins this guarantee.
    """
    global _ORIG_LOG_RECORD_FACTORY, _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        # Once-only capture: re-set our factory but do NOT re-capture
        # the current factory — that would be a wrapper around our
        # factory and capturing it leads to infinite recursion.
        if logging.getLogRecordFactory() is not _actus_log_record_factory:
            logging.setLogRecordFactory(_actus_log_record_factory)
        return
    _ORIG_LOG_RECORD_FACTORY = logging.getLogRecordFactory()
    logging.setLogRecordFactory(_actus_log_record_factory)
    _FACTORY_INSTALLED = True


# ---------------------------------------------------------------------------
# Component filter (noise suppression, Hermes-style)
# ---------------------------------------------------------------------------
class _ComponentFilter(logging.Filter):
    """Drop INFO/DEBUG records from noisy third-party loggers.

    Attached to every root *handler* — not the root logger — because
    records propagated from a child logger (``httpx`` etc.) reach root
    handlers via ``Logger.callHandlers`` which calls ``handler.handle``
    (handler-side filter) but bypasses the root logger's own
    ``Logger.filter``.
    """

    PREFIXES: tuple[str, ...] = _NOISY_COMPONENT_PREFIXES
    LEVEL: int = _NOISY_COMPONENT_LEVEL

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.levelno >= self.LEVEL:
            return True
        for prefix in self.PREFIXES:
            if record.name == prefix or record.name.startswith(prefix + "."):
                return False
        return True


# ---------------------------------------------------------------------------
# Composable helpers (Q1)
# ---------------------------------------------------------------------------
def _build_formatter() -> RedactingFormatter:
    """Construct a fresh ``RedactingFormatter``.

    Each handler gets its own instance because ``logging.Formatter``
    instances are not thread-safe in their internal ``_style`` state
    when used concurrently across handlers.
    """
    return RedactingFormatter(_FORMAT, datefmt=_DATEFMT)


def _install_redacting_formatter(
    root_logger: logging.Logger, log_level: int
) -> None:
    """Replace root handlers with a single redacting stdout handler.

    Idempotent: removes every preexisting handler before installing
    the canonical stdout one. ``_install_file_handlers`` runs after
    this and re-adds rotating file handlers; both share the same
    formatter shape (and both rely on ``RedactingFormatter`` so
    secrets in rotated backups are also masked).

    **Resource hygiene (review-found)**: each removed handler is
    ``.close()``-ed before being dropped. The PR-S1-4 file handlers
    (``ConcurrentRotatingFileHandler``) hold an OS file lock + open
    file descriptor; without ``close()``, repeated ``setup_logging()``
    invocations (uvicorn reload, test fixtures) would leak both. The
    ``close()`` call is wrapped in ``try/except`` so a misbehaving
    third-party handler does not break the install path.
    """
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            # Closing a third-party handler must not abort the
            # install — the handler is already detached from root.
            pass
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_build_formatter())
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)


def _install_file_handlers(
    root_logger: logging.Logger, log_dir: str, log_level: int
) -> None:
    """Add ``agent.log`` + ``errors.log`` rotating file handlers.

    ``ConcurrentRotatingFileHandler`` from ``concurrent-log-handler``
    serializes rotation across processes via an OS file lock, which
    is required under multi-worker uvicorn (plain stdlib
    ``RotatingFileHandler`` races on rotation and can lose lines or
    truncate the active file).

    Failure mode: if ``log_dir`` cannot be created or written
    (permission denied, immutable mount, etc.), this helper logs a
    WARNING through the stdout handler installed by
    ``_install_redacting_formatter`` and returns silently. The app
    continues with stdout-only logging instead of crashing — local
    dev / CI without ``/app/data/logs`` typically takes this path.
    """
    log_dir_path = Path(log_dir)
    try:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        agent_handler = ConcurrentRotatingFileHandler(
            filename=str(log_dir_path / _AGENT_LOG_FILENAME),
            maxBytes=_AGENT_LOG_MAX_BYTES,
            backupCount=_AGENT_LOG_BACKUPS,
            encoding="utf-8",
        )
        agent_handler.setFormatter(_build_formatter())
        agent_handler.setLevel(max(log_level, logging.INFO))
        root_logger.addHandler(agent_handler)

        error_handler = ConcurrentRotatingFileHandler(
            filename=str(log_dir_path / _ERROR_LOG_FILENAME),
            maxBytes=_ERROR_LOG_MAX_BYTES,
            backupCount=_ERROR_LOG_BACKUPS,
            encoding="utf-8",
        )
        error_handler.setFormatter(_build_formatter())
        error_handler.setLevel(logging.WARNING)
        root_logger.addHandler(error_handler)
    except OSError as exc:
        root_logger.warning(
            "B5 PR-S1-4: rotating file handlers could not be installed at %s "
            "(%s); falling back to stdout-only logging.",
            log_dir,
            exc,
        )


def _install_third_party_isolation() -> None:
    """Three-stage Q2 self-heal — order is load-bearing.

    1. ``install_self_healing_logger_class`` first so any logger
       constructed AFTER this call comes up as the propagate-only
       subclass with NoOp ``addHandler`` (covers names this composer
       does not know about and any late-imported library).
    2. ``clear_propagate_only_loggers`` next; for each known noisy
       name it both clears the existing handler list AND swaps
       ``__class__`` to the propagate-only subclass — closing the
       bypass for SDKs (openai / httpx / anthropic / langchain /
       uvicorn.access) imported before ``setup_logging`` ran.
    3. ``isolate_all_non_root_loggers`` last — registry-wide sweep
       catches transitive deps not on the explicit list (notably
       ``huggingface_hub`` and ``transformers`` which were observed
       leaking via local ``StreamHandler`` instances and severing
       root-propagation entirely).
    """
    install_self_healing_logger_class()
    clear_propagate_only_loggers()
    isolate_all_non_root_loggers()


def _install_component_filter(root_logger: logging.Logger) -> None:
    """Attach ``_ComponentFilter`` to every root handler exactly once.

    Idempotent: if a handler already carries a ``_ComponentFilter``,
    it is left alone. This matters for re-invocation from test
    fixtures so repeated ``setup_logging()`` calls don't stack
    multiple identical filters on the same handler.
    """
    component_filter = _ComponentFilter()
    for handler in root_logger.handlers:
        if any(isinstance(f, _ComponentFilter) for f in handler.filters):
            continue
        handler.addFilter(component_filter)


# ---------------------------------------------------------------------------
# Public composer
# ---------------------------------------------------------------------------
def setup_cli_logging() -> None:
    """Minimal logging setup for CLI scripts that need clean stdout.

    Variant of ``setup_logging`` for one-shot CLI tools whose stdout
    is the operator-facing channel (JSON output, status reports,
    pipe-friendly text). Differences from the FastAPI-targeted
    ``setup_logging``:

    - Console handler writes to **stderr** instead of stdout. Stdout
      stays reserved for the script's own ``sys.stdout.write``
      output, so a downstream ``| jq`` or ``> result.json`` keeps
      working — review-found P2 against ``app.cli.memory_reconcile``
      whose JSON summary print was getting interleaved with
      ``setup_logging``'s ``"日志记录器已初始化"`` bootstrap line and
      the ``/app/data/logs`` write-failure warning.
    - **No file handlers** — CLI scripts are short-lived; the
      rotating-file handlers add IO + cleanup overhead with no
      operational benefit.
    - **No bootstrap log line** — keep CLI startup silent unless
      the script itself wants to emit something.

    Still installs the LogRecord factory (Q6 trace/request/session
    fields), the RedactingFormatter (so any third-party log emitted
    during the run still has secrets masked), the third-party
    isolation chain (Q2 self-heal), and the component filter (noise
    suppression) so anything the CLI run emits via ``logging``
    behaves like the production stack — just routed to stderr.
    """
    settings = get_settings()
    root_logger = logging.getLogger()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    _install_log_record_factory()

    # Clear-and-reinstall single stderr handler. Idempotent: re-running
    # ``setup_cli_logging`` (or running it after a prior ``setup_logging``)
    # closes the old handlers cleanly before swapping the stream.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_build_formatter())
    stderr_handler.setLevel(log_level)
    root_logger.addHandler(stderr_handler)

    _install_third_party_isolation()
    _install_component_filter(root_logger)


def setup_logging() -> None:
    """Initialise application-wide logging.

    Call order (each step depends on the previous step's state):

    1. ``_install_log_record_factory`` — install BEFORE any new
       handler emits its first record so the bootstrap log line below
       already carries trace/request/session attrs (defaulting to
       ``"-"`` because the bootstrap path runs outside any request
       scope).
    2. ``_install_redacting_formatter`` — clear stale root handlers
       (idempotent re-invoke) and install the canonical redacting
       stdout handler.
    3. ``_install_file_handlers`` — append rotating file handlers;
       degrades to stdout-only on ``OSError``.
    4. ``_install_third_party_isolation`` — three-stage Q2 self-heal
       so third-party logger handlers cannot bypass the redacting
       formatter.
    5. ``_install_component_filter`` — suppress INFO/DEBUG noise from
       known third-party prefixes on every root handler.

    The function is fully idempotent — uvicorn lifespan reload and
    pytest fixtures may call it repeatedly without leaking handlers
    or stacking filters.
    """
    settings = get_settings()
    root_logger = logging.getLogger()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    _install_log_record_factory()
    _install_redacting_formatter(root_logger, log_level)
    _install_file_handlers(root_logger, settings.log_dir, log_level)
    _install_third_party_isolation()
    _install_component_filter(root_logger)

    root_logger.info("日志记录器已初始化，日志级别: %s", settings.log_level)
