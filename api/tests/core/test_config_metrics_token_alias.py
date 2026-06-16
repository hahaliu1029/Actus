"""Settings regression: ``metrics_endpoint_token`` must resolve from EITHER env
name.

``Settings`` has no ``env_prefix``, so the canonical env var is the unprefixed
``METRICS_ENDPOINT_TOKEN`` (this is what the live endpoint + observability tests
already use). But the C2 coordinator runbooks
(``docs/runbooks/c2-coordinator-live-acceptance.md`` /
``c2-coordinator-canary-rollback.md``) and the perf CLI
(``app/cli/coordinator_perf_sample.py``) prescribe the ``ACTUS_``-prefixed name
``ACTUS_METRICS_ENDPOINT_TOKEN`` — consistent with ``ACTUS_C2_COORDINATOR_ENABLED``
and the ``ACTUS_COORDINATOR_*`` caps.

Before the ``AliasChoices`` fix only the unprefixed name was read, so an operator
following the runbook verbatim set ``ACTUS_METRICS_ENDPOINT_TOKEN`` → Settings
left the token empty → ``/api/v1/metrics`` 404s → the metrics scrape silently
returns nothing and the perf CLI crashes on ``resp.raise_for_status()``. This
test pins that BOTH names populate the field so neither convention breaks.
"""
from __future__ import annotations

import pytest

from core.config import Settings

_TOKEN_ENV_NAMES = ("METRICS_ENDPOINT_TOKEN", "ACTUS_METRICS_ENDPOINT_TOKEN")


def _required_env() -> dict[str, str]:
    """Minimal kwargs so ``Settings()`` doesn't crash on other required fields.

    Mirrors ``tests/core/test_config_memory_gate_rollout_at.py``. These are passed
    as init kwargs (highest-priority source); ``metrics_endpoint_token`` is left
    OUT so it resolves through the env-var alias path under test.
    """
    return {
        "postgres_password": "ci-dummy",
        "jwt_secret_key": "ci-dummy-secret-key-32-chars-long",
        "env": "test",
    }


@pytest.fixture(autouse=True)
def _isolate_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both token env names absent so each test controls exactly one."""
    for name in _TOKEN_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("env_name", _TOKEN_ENV_NAMES)
def test_metrics_token_read_from_either_env_name(
    env_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``METRICS_ENDPOINT_TOKEN`` (canonical) AND ``ACTUS_METRICS_ENDPOINT_TOKEN``
    (runbook/CLI convention) must both populate ``metrics_endpoint_token``."""
    expected = f"tok-from-{env_name}"
    monkeypatch.setenv(env_name, expected)
    settings = Settings(**_required_env())
    assert settings.metrics_endpoint_token == expected


def test_metrics_token_defaults_empty_when_neither_set() -> None:
    """Neither env name set → token stays empty (endpoint stays 404/disabled)."""
    settings = Settings(**_required_env())
    assert settings.metrics_endpoint_token == ""
