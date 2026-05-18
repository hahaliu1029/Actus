"""Real LLM client fixture for permission SmartApprove eval.

Disabled by default — set ACTUS_RUN_SLOW_EVALS=1 to opt in.
Requires OPENAI_API_KEY (or equivalent provider key) in env.
"""

from __future__ import annotations

import os
from typing import Any

import pytest


@pytest.fixture(scope="session")
def real_summary_llm() -> Any:
    if os.environ.get("ACTUS_RUN_SLOW_EVALS") != "1":
        pytest.skip("set ACTUS_RUN_SLOW_EVALS=1 to run slow LLM-backed evals")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for permission SmartApprove evals")
    from app.infrastructure.external.llm import ActusChatModel
    return ActusChatModel(model="gpt-4o-mini")  # cheap default; override per repo conf


@pytest.fixture(scope="session")
def corpus_entries() -> list[dict]:
    import json
    from pathlib import Path
    corpus_dir = Path(__file__).parent / "corpus"
    entries: list[dict] = []
    for f in sorted(corpus_dir.glob("*.json")):
        entries.append(json.loads(f.read_text(encoding="utf-8")))
    return entries
