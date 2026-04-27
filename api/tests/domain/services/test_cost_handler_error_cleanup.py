"""B4 M0 post-audit: on_llm_error releases the pending entry.

Without this hook, a failed / cancelled LLM run leaves its ``_pending[run_id]``
entry allocated until LRU eviction. Over a crash loop this pressures the
LRU (10k cap) and masks real leaks.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.domain.services.cost_callback_handler import CostCallbackHandler

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _noop_persist(_record) -> None:
    return None


async def test_on_llm_error_pops_pending_entry() -> None:
    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=_noop_persist
    )
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="x")]],
        run_id=run_id,
        metadata={"langgraph_node": "n", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o"},
    )
    assert run_id in handler.pending_keys()

    await handler.on_llm_error(RuntimeError("boom"), run_id=run_id)
    assert run_id not in handler.pending_keys(), (
        "on_llm_error must pop the pending entry so the LRU doesn't leak "
        "on repeated error paths."
    )


async def test_on_llm_error_without_matching_start_is_noop() -> None:
    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=_noop_persist
    )
    await handler.on_llm_error(RuntimeError("no prior start"), run_id=uuid4())
    assert len(handler.pending_keys()) == 0
