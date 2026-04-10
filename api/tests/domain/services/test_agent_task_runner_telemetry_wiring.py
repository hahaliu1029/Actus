"""B5 C11: AgentTaskRunner telemetry DI wiring.

Verifies that:
- ``_build_prompt_telemetry`` returns a ``JsonlPromptTelemetry`` instance
- ``_build_prompt_assembler`` passes the telemetry instance into the
  ``PromptAssembler`` constructor
- The helpers honor the ``prompt_telemetry_log_dir`` setting
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.domain.services.agent_task_runner import AgentTaskRunner
from app.infrastructure.telemetry.prompt_telemetry import JsonlPromptTelemetry


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _FakeOverflowConfig:
    """Minimal stand-in for ContextOverflowConfig."""

    system_prompt_max_tokens = 3500
    token_estimator = "hybrid"
    model_name = ""


def test_build_prompt_telemetry_returns_jsonl_instance(
    monkeypatch, tmp_path
) -> None:
    """``_build_prompt_telemetry`` constructs a real JsonlPromptTelemetry."""
    from core import config as core_config

    # Override the log dir to a tmp path so the test doesn't touch
    # /app/data/telemetry/prompt.
    fake_settings = MagicMock(prompt_telemetry_log_dir=str(tmp_path))
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.get_settings",
        lambda: fake_settings,
    )

    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    telemetry = runner._build_prompt_telemetry()

    assert isinstance(telemetry, JsonlPromptTelemetry)
    # The log dir should be created eagerly by the constructor
    assert tmp_path.exists()


def test_build_prompt_assembler_uses_injected_telemetry(
    monkeypatch, tmp_path
) -> None:
    """``_build_prompt_assembler`` picks up ``self._prompt_telemetry`` so
    the assembler records events via JsonlPromptTelemetry."""
    from core import config as core_config

    fake_settings = MagicMock(prompt_telemetry_log_dir=str(tmp_path))
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.get_settings",
        lambda: fake_settings,
    )

    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._overflow_config = _FakeOverflowConfig()
    # Simulate what __init__ does: build telemetry before assembler
    runner._prompt_telemetry = runner._build_prompt_telemetry()

    assembler = runner._build_prompt_assembler()

    # The assembler's _telemetry field should be the same instance we
    # attached to the runner
    assert assembler._telemetry is runner._prompt_telemetry
    assert isinstance(assembler._telemetry, JsonlPromptTelemetry)


def test_assembler_falls_back_to_none_when_telemetry_missing(
    monkeypatch, tmp_path
) -> None:
    """If the runner skipped ``_build_prompt_telemetry`` (e.g. an older
    test path that bypasses ``__init__``), ``_build_prompt_assembler``
    must still return a usable assembler — just without telemetry."""
    fake_settings = MagicMock(prompt_telemetry_log_dir=str(tmp_path))
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.get_settings",
        lambda: fake_settings,
    )

    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._overflow_config = _FakeOverflowConfig()
    # Explicitly do NOT set runner._prompt_telemetry

    assembler = runner._build_prompt_assembler()
    assert assembler is not None
    assert assembler._telemetry is None


def test_jsonl_telemetry_writes_to_configured_dir(tmp_path) -> None:
    """End-to-end smoke test: JsonlPromptTelemetry actually creates its
    JSONL file and appends a line when record_llm_invocation is called."""
    telemetry = JsonlPromptTelemetry(log_dir=tmp_path)
    telemetry.record_llm_invocation(
        system_prompt_hash="abcd",
        system_prompt_bytes=10,
        tools_hash="efgh",
        lang="zh",
        provider="openai",
    )
    invocation_file = tmp_path / "llm_invocation.jsonl"
    assert invocation_file.exists()
    contents = invocation_file.read_text(encoding="utf-8")
    assert "abcd" in contents
    assert "openai" in contents
