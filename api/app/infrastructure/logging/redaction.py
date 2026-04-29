"""B5 PR-S1-3: secret-redacting log formatter.

Strips known secret formats from rendered ``LogRecord`` output before
they reach stdout, files, or third-party log shippers. Runs as the LAST
step of ``logging.Formatter.format`` so the rendered string includes
``formatException``-rendered traceback content (where most accidental
secret leaks hide — e.g., a ``ValueError("failed to auth with key
sk-abcdef...")`` shows the secret in the traceback frame, not in
``record.getMessage()``).

Hermes provenance
-----------------
The pattern set is anchored on Hermes's ``agent/redact.py`` plus
production telemetry observed across Anthropic / OpenAI / GitHub /
Slack / AWS / Stripe / Telegram tokens, JWTs, generic Authorization
headers, JSON-shaped credential blobs, env-var assignments, PEM
blocks, database URLs, and URL query tokens.

Key implementation choices (from B5 design doc)
-----------------------------------------------
- **Closure capture (Q5)**: ``_REDACT_ENABLED`` is read once at
  ``RedactingFormatter.__init__`` and stashed on the instance. Tests
  monkey-patching the module global later cannot disable redaction for
  formatters that already exist; this defends against accidental late
  imports that flip the flag.
- **Token truncation**: short tokens (<18 chars) become the full
  ``[REDACTED]`` mask; long tokens (≥18 chars) keep the first 6 and
  last 4 chars so support engineers can correlate against external
  systems without seeing the secret material.
- **UUIDv4 exception**: a captured value that fullmatches the canonical
  UUIDv4 pattern is emitted verbatim. ``trace_id`` / ``session_id`` /
  ``event_id`` flow through logs as canonical attributes and are not
  secrets; collapsing them would break trace correlation.
- **Pattern ordering**: most specific first (``sk-ant-`` before ``sk-``,
  Bearer-specific Authorization before generic Authorization). Earlier
  patterns claim the longer match so later patterns don't bite the
  prefix and over-redact.
- **ReDoS guard**: every pattern is hand-checked for catastrophic
  backtracking (no nested quantifiers, no alternation overlap inside
  repetitions). The acceptance test
  ``test_redaction_redos_guard.py`` confirms 100KB of pathological
  input formats in well under 50ms.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable

# Import-time lock. Set ``False`` only in deployments that explicitly
# opt out of redaction (e.g., a sandboxed dev container with synthetic
# data); production callers should keep this ``True``. Per Q5 the value
# is captured by the formatter at ``__init__`` time, so monkey-patching
# this attribute later is intentionally a no-op for live formatters.
_REDACT_ENABLED: bool = True

# UUIDv4 (lowercase, with dashes). Captured values that fullmatch this
# regex are emitted verbatim — they are canonical attributes
# (trace_id / session_id / event_id) flowing through logs, not secrets.
_UUID4_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


def _truncate(value: str) -> str:
    """Apply the v1 token-truncation rules to a single captured value."""
    if _UUID4_RE.fullmatch(value):
        return value
    if len(value) < 18:
        return "[REDACTED]"
    return f"{value[:6]}[REDACTED]{value[-4:]}"


def _redact_full(match: re.Match[str]) -> str:
    """Replacer: redact the entire matched text with the truncate rule."""
    return _truncate(match.group(0))


def _redact_kv(match: re.Match[str]) -> str:
    """Replacer for two-group patterns ``(prefix)(secret)``.

    Keeps group 1 verbatim and applies truncation to group 2. Used for
    headers, JSON ``"key":"value"`` shapes, and similar contexts where
    we want to preserve the field name for log readability.
    """
    return f"{match.group(1)}{_truncate(match.group(2))}"


def _redact_kvw(match: re.Match[str]) -> str:
    """Replacer for three-group patterns ``(prefix)(secret)(suffix)``.

    Keeps groups 1 and 3 (e.g., open and close quotes) verbatim and
    truncates group 2 (the secret value).
    """
    return f"{match.group(1)}{_truncate(match.group(2))}{match.group(3)}"


# Pattern table (compiled at import time). Order matters — most-specific
# first. Each entry pairs a compiled regex with a replacer function that
# decides what survives.
_PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    # 1. Anthropic API key (must come before generic OpenAI sk- pattern)
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), _redact_full),
    # 2. OpenAI API key
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), _redact_full),
    # 3-6. GitHub tokens (PAT / OAuth / App user / App server)
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), _redact_full),
    (re.compile(r"gho_[A-Za-z0-9]{36,}"), _redact_full),
    (re.compile(r"ghu_[A-Za-z0-9]{36,}"), _redact_full),
    (re.compile(r"ghs_[A-Za-z0-9]{36,}"), _redact_full),
    # 7. Slack tokens
    (re.compile(r"xox[abprs]-[A-Za-z0-9-]+"), _redact_full),
    # 8. Google API key
    (re.compile(r"AIza[A-Za-z0-9_-]{35,}"), _redact_full),
    # 9. AWS access key id
    (re.compile(r"AKIA[A-Z0-9]{16}"), _redact_full),
    # 10. AWS secret key (context-aware: only when adjacent to a key
    # name, otherwise a 40-char base64 string is too generic and would
    # eat S3 ETags / git SHA1+padding / etc.).
    (
        re.compile(
            r"((?:aws_secret(?:_access)?_key|secret_access_key)\s*[=:]\s*['\"]?)"
            r"([A-Za-z0-9/+=]{40})"
            r"(['\"]?)",
            re.IGNORECASE,
        ),
        _redact_kvw,
    ),
    # 11. Stripe key (live or test)
    (re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{24,}"), _redact_full),
    # 12. Telegram bot token
    (re.compile(r"\d{9,10}:[A-Za-z0-9_-]{35}"), _redact_full),
    # 13. JWT (three base64url parts separated by dots, header starts
    # with ``eyJ`` which is the base64 prefix for ``{"``). The body
    # repetitions are bounded — typical JWTs are <2KB; the upper bound
    # protects against pathological input where many ``eyJ`` prefixes
    # appear without inner dots, which would otherwise force the engine
    # to backtrack across the whole remaining string per attempt.
    (
        re.compile(
            r"eyJ[A-Za-z0-9_-]{1,2048}"
            r"\.eyJ[A-Za-z0-9_-]{1,2048}"
            r"\.[A-Za-z0-9_-]{1,4096}"
        ),
        _redact_full,
    ),
    # 14. Bearer token (Authorization header, specific). Must come
    # before #15 generic Authorization so the Bearer scheme stays in
    # the rendered output.
    (
        re.compile(r"(Authorization:\s*Bearer\s+)(\S+)", re.IGNORECASE),
        _redact_kv,
    ),
    # 15. Generic Authorization header (anything that isn't Bearer).
    # Negative lookahead ``(?!Bearer\s)`` skips this pattern when the
    # value starts with ``Bearer`` so #14's redacted output (which keeps
    # the ``Bearer`` scheme prefix) is not re-truncated. ``(\S[^\r\n]*)``
    # forces group 2 to start with a non-whitespace char — without that,
    # the engine could backtrack ``\s*`` to zero whitespace, making the
    # Bearer-lookahead pass at a position where the next char is a
    # space (and then group 2 would swallow ``" Bearer ..."`` whole).
    # Captures the rest of the line so ``Basic <base64>`` /
    # ``Digest <whatever>`` / ``Token <opaque>`` etc. all redact the
    # full credential blob, not just the scheme word.
    (
        re.compile(
            r"(Authorization:\s*)(?!Bearer\s)(\S[^\r\n]*)",
            re.IGNORECASE,
        ),
        _redact_kv,
    ),
    # 16. Cookie session value. The inner repetition uses ``[^\n;]``
    # (don't cross either a newline or the cookie-attribute separator)
    # AND a bounded ``{0,200}`` so each match attempt fails fast on a
    # ``Set-Cookie:`` prefix that doesn't actually carry a ``session=``
    # attribute. Without these bounds, ``[^\n]*?session=`` would scan
    # the rest of the line per attempt — pathological inputs with many
    # ``Set-Cookie:`` prefixes (no session) hit O(N²) total time.
    (
        re.compile(
            r"(Set-Cookie:[^\n;]{0,200}?session=)([^;\s]+)",
            re.IGNORECASE,
        ),
        _redact_kv,
    ),
    # 17. JSON apiKey / api_key / access_token
    (
        re.compile(
            r'("(?:apiKey|api_key|access_token)"\s*:\s*")([^"]+)(")'
        ),
        _redact_kvw,
    ),
    # 18. JSON password
    (
        re.compile(r'("password"\s*:\s*")([^"]+)(")', re.IGNORECASE),
        _redact_kvw,
    ),
    # 19. JSON token / refresh_token
    (
        re.compile(r'("(?:token|refresh_token)"\s*:\s*")([^"]+)(")'),
        _redact_kvw,
    ),
    # 20. JSON secret / client_secret
    (
        re.compile(r'("(?:secret|client_secret)"\s*:\s*")([^"]+)(")'),
        _redact_kvw,
    ),
    # 21. JSON authorization (case-insensitive)
    (
        re.compile(r'("authorization"\s*:\s*")([^"]+)(")', re.IGNORECASE),
        _redact_kvw,
    ),
    # 22. ENV var sensitive assignment (e.g., from a traceback rendering
    # ``os.environ`` via ``ValueError(f"...{settings.SECRET_KEY=}")``).
    (
        re.compile(
            r"((?:SECRET_KEY|API_KEY|TOKEN|PASSWORD|AUTH(?:_\w+)?|"
            r"PRIVATE_KEY|CLIENT_SECRET|ACCESS_KEY)\s*=\s*['\"]?)"
            r"([^\s'\"]+)"
            r"(['\"]?)"
        ),
        _redact_kvw,
    ),
    # 23. PEM block (RSA / OPENSSH / generic). One regex covers all
    # PEM-style begin/end pairs. ``[\s\S]+?`` is the non-greedy any-
    # including-newline variant; safe under ReDoS because the literal
    # ``-----END`` anchors the inner repetition.
    (
        re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]+?-----END [A-Z ]+-----"),
        _redact_full,
    ),
    # 26. Postgres URL credentials. The driver suffix is open-ended
    # (``+asyncpg`` / ``+psycopg2`` / ``+psycopg`` / ``+pg8000`` / etc.)
    # because Actus' startup path rewrites the asyncpg DSN into the
    # ``postgresql+psycopg2://`` form for Alembic; without the broad
    # match, an exception or log line carrying the sync DSN would
    # leak the DB password unredacted.
    (
        re.compile(
            r"(postgres(?:ql)?(?:\+[A-Za-z0-9_]+)?://[^:/\s]+:)([^@\s]+)(@)"
        ),
        _redact_kvw,
    ),
    # 27. MySQL URL credentials
    (
        re.compile(r"(mysql://[^:/\s]+:)([^@\s]+)(@)"),
        _redact_kvw,
    ),
    # 28. MongoDB URL credentials
    (
        re.compile(r"(mongodb://[^:/\s]+:)([^@\s]+)(@)"),
        _redact_kvw,
    ),
    # 29. Redis URL credentials (``redis://:password@host`` or
    # ``redis://user:password@host``).
    (
        re.compile(r"(redis://(?:[^:/\s]*:))([^@\s]+)(@)"),
        _redact_kvw,
    ),
    # 30. URL query parameter (api_key / token / access_token)
    (
        re.compile(r"([?&](?:api_key|token|access_token)=)([^&\s]+)"),
        _redact_kv,
    ),
    # 31. E.164 phone number (Signal / WhatsApp scenarios)
    (re.compile(r"\+\d{8,15}\b"), _redact_full),
]


class RedactingFormatter(logging.Formatter):
    """``logging.Formatter`` that strips secrets from rendered output.

    Override sequence::

        rendered = super().format(record)   # includes traceback text
        if self._enabled:
            rendered = self._redact(rendered)
        return rendered

    Redacting only ``record.getMessage()`` would miss secrets that
    surface from ``formatException`` (the most common leak path). The
    full-string approach guarantees the rendered output cannot contain
    a secret that any of the v1 patterns would have matched.
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: str = "%",
        validate: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        # Q5: closure-capture the redaction-enabled flag at construction.
        # Subsequent monkey-patches of ``_REDACT_ENABLED`` cannot disable
        # this instance — the test ``test_redact_lock_cannot_be_disabled``
        # locks that invariant.
        self._enabled: bool = _REDACT_ENABLED

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self._enabled:
            return rendered
        return self._redact(rendered)

    @staticmethod
    def _redact(text: str) -> str:
        for pattern, replacer in _PATTERNS:
            text = pattern.sub(replacer, text)
        return text


# Third-party loggers that ship their own ``StreamHandler`` and would
# otherwise bypass the root ``RedactingFormatter``. ``uvicorn.access``
# is included because uvicorn installs its own handler on
# ``uvicorn.access`` at startup; without clearing, the access log
# line ``GET /x?token=foo`` reaches stdout unredacted. Production
# adds anything matched here at setup_logging time.
_KNOWN_NOISY_LOGGERS: tuple[str, ...] = (
    "openai",
    "httpx",
    "httpcore",
    "anthropic",
    "langchain",
    "langgraph",
    "langsmith",
    "uvicorn.access",
    # huggingface_hub ships a top-level StreamHandler with
    # propagate=True; the local handler emits raw secrets in addition
    # to the root copy. transformers ships a StreamHandler with
    # propagate=False, so its records skip root entirely.
    # ``isolate_all_non_root_loggers`` would catch both via the
    # registry sweep, but listing them explicitly here keeps the
    # known-noisy-list test surface anchored to specific names that
    # we have observed leaking in practice.
    "huggingface_hub",
    "transformers",
)


class _RedactingPropagateOnlyLogger(logging.Logger):
    """``logging.Logger`` subclass that refuses local handlers.

    Q2 self-heal mechanism: ``install_self_healing_logger_class()``
    sets this as the default ``Logger`` subclass via
    ``logging.setLoggerClass``. After that point every NEW
    ``logging.getLogger(name)`` call (i.e., names not previously
    instantiated, including names a future late-imported library will
    create) returns an instance of this subclass.

    Two invariants the subclass enforces:

    - ``propagate`` is ``True`` after ``__init__`` so the logger's
      records reach the root logger (which carries ``RedactingFormatter``).
    - ``addHandler`` is a no-op so a third-party library cannot attach
      its own ``StreamHandler`` and bypass root.

    Limits: a library that directly assigns ``logger.propagate = False``
    or appends to ``logger.handlers`` cannot be stopped at runtime
    (those are attribute mutations, not method calls). The known-noisy
    library list above is also processed up-front by
    ``clear_propagate_only_loggers`` to cover libs that were imported
    before ``setup_logging`` ran.
    """

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)
        self.propagate = True

    def addHandler(self, hdlr: logging.Handler) -> None:
        # NoOp. Logging code paths must NEVER raise from inside
        # third-party caller frames; silently dropping is the
        # documented Q2 mode.
        return


def clear_propagate_only_loggers(
    names: Iterable[str] = _KNOWN_NOISY_LOGGERS,
) -> None:
    """Strip handlers + force propagate on each named logger.

    Handles loggers that already exist before ``setup_logging`` runs
    (typical case: ``uvicorn.access`` is created by uvicorn during
    server boot; ``openai`` / ``httpx`` / ``anthropic`` / ``langchain``
    are created the first time those SDKs are imported, which usually
    happens earlier in ``app.main`` than the ``setup_logging`` call
    itself).

    For each named logger this function:

    1. Removes every local handler (so logs don't double-emit through
       both the lib's handler and the root ``RedactingFormatter``).
    2. Forces ``propagate=True`` so records reach root.
    3. **Swaps ``__class__`` to ``_RedactingPropagateOnlyLogger``** for
       any instance not already in that subclass. ``setLoggerClass``
       only affects FUTURE ``getLogger(name)`` calls — existing
       instances keep the class they were constructed with. Without
       the in-place swap, an SDK that was imported before
       ``setup_logging`` (and therefore created its top-level logger
       as plain ``logging.Logger``) could still call ``addHandler``
       later and bypass the root ``RedactingFormatter``. Class swap
       is layout-compatible because ``_RedactingPropagateOnlyLogger``
       inherits from ``logging.Logger`` and adds no instance state.
    """
    for name in names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.propagate = True
        if not isinstance(logger, _RedactingPropagateOnlyLogger):
            logger.__class__ = _RedactingPropagateOnlyLogger


def install_self_healing_logger_class() -> None:
    """Install ``_RedactingPropagateOnlyLogger`` as the default Logger
    subclass.

    Affects FUTURE ``logging.getLogger(name)`` calls only. Existing
    loggers (already instantiated by libraries imported before this
    function runs) keep their stock ``logging.Logger`` class — those
    are handled by ``clear_propagate_only_loggers``.

    Idempotent: repeated calls re-set the class to the same subclass.
    """
    logging.setLoggerClass(_RedactingPropagateOnlyLogger)


def isolate_all_non_root_loggers() -> None:
    """Registry sweep: isolate every existing non-root logger.

    The named list approach (``clear_propagate_only_loggers``) only
    covers loggers we explicitly know about. Transitive dependencies
    (``huggingface_hub`` is pulled in by langchain / sentence-transformers,
    ``transformers`` likewise) ship their own handlers at import time
    and would slip through. A reviewer probe found
    ``transformers.propagate=False`` plus a local ``StreamHandler`` —
    a complete bypass of the root ``RedactingFormatter``.

    This function walks ``logging.Logger.manager.loggerDict`` and:

    1. Skips ``PlaceHolder`` entries (internal hierarchy nodes, not
       real loggers).
    2. Preserves ``NullHandler`` instances (the harmless "no logging
       configured" defensive pattern that many libraries use).
    3. Removes every non-NullHandler from the logger.
    4. Forces ``propagate=True`` so records reach root.
    5. Swaps ``__class__`` to ``_RedactingPropagateOnlyLogger`` so any
       subsequent ``addHandler`` call is a no-op.

    Idempotent: subsequent calls find every non-root logger already in
    the propagate-only state (``setup_logging`` re-runs are a no-op
    after the first).
    """
    manager = logging.Logger.manager
    for _name, item in list(manager.loggerDict.items()):
        if not isinstance(item, logging.Logger):
            continue  # PlaceHolder
        for handler in list(item.handlers):
            if not isinstance(handler, logging.NullHandler):
                item.removeHandler(handler)
        item.propagate = True
        if not isinstance(item, _RedactingPropagateOnlyLogger):
            item.__class__ = _RedactingPropagateOnlyLogger
