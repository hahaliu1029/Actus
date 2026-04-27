"""B4 M0 Phase E (post-audit): real Actus adapter → CostCallbackHandler integration.

The audit flagged that ``BaseChatModel.dict()`` only surfaces ``_type`` — the
real ``ActusChatModel`` / ``ActusResponsesModel`` / ``ActusFallbackChatModel``
need to override ``_identifying_params`` so ``on_chat_model_start(**kwargs)
.invocation_params`` carries the real ``model`` name and
``ProviderProfile.provider_id``. This test drives the full path from the
adapter's ``ainvoke(config={"callbacks": [handler]})`` through the handler
and asserts the resulting CostRecord has the right attribution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.services.cost_callback_handler import CostCallbackHandler
from app.domain.services.provider_profiles import get_profile
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_chat_completion_with_usage(
    prompt_tokens: int = 1000, completion_tokens: int = 500
) -> SimpleNamespace:
    message = SimpleNamespace(
        role="assistant", content="hi", tool_calls=None
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(
        id="chatcmpl-x", choices=[choice], model="gpt-4o", usage=usage
    )


async def test_actus_chat_model_invocation_params_carry_model_and_provider() -> None:
    """Real adapter must surface ``model`` + ``provider_id`` via _identifying_params
    so on_chat_model_start.invocation_params is not just ``{_type: actus-chat}``."""
    captured: List[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    handler = CostCallbackHandler(
        session_id="sess-real",
        user_id="user-real",
        persister=persist,
    )

    profile = get_profile("openai_official")
    model = ActusChatModel(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model_name="gpt-4o",
        temperature=0.5,
        max_tokens=256,
        supports_response_format=True,
        profile=profile,
    )

    mock_resp = _make_chat_completion_with_usage(prompt_tokens=1000, completion_tokens=500)
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch.object(model, "_get_client", return_value=mock_client):
        await model.ainvoke(
            [HumanMessage(content="hi")],
            config={"callbacks": [handler]},
        )
    await handler.flush_pending()

    assert len(captured) == 1, f"expected 1 CostRecord, got {len(captured)}"
    rec = captured[0]
    assert rec.model == "gpt-4o", (
        f"model must come from ActusChatModel._identifying_params; got {rec.model!r}. "
        "If this is 'unknown', the _identifying_params override regressed."
    )
    assert rec.provider == "openai_official", (
        f"provider must be the ProviderProfile.provider_id; got {rec.provider!r}. "
        "If this is 'openai' (heuristic), the provider_id plumbing regressed."
    )
    assert rec.cost_status == CostStatus.ACTUAL
    assert rec.input_tokens == 1000
    assert rec.output_tokens == 500
