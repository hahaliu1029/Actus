from langchain_core.messages import AIMessage, HumanMessage
import pytest

from app.domain.services.provider_profiles import get_profile, infer_provider_from_base_url
from app.domain.services.provider_profiles._parse import parse_chat_completion_message, parse_chat_completion_stream_chunk
from app.domain.services.provider_profiles._rewrites import apply_outbound_rewrites
from app.domain.services.provider_profiles._wire import inject_reasoning_into_wire_entry


@pytest.mark.parametrize("path,expected", [
    ("/api/paas/v4", "glm_5_2"),
    ("/api/coding/paas/v4", "glm_5_2_coding"),
])
def test_glm_52_routes_without_changing_old_models(path, expected):
    url = "https://open.bigmodel.cn" + path
    assert infer_provider_from_base_url(url, model_name="glm-5.2") == expected
    assert infer_provider_from_base_url(url, model_name="glm-4.6v") == "glm"
    assert infer_provider_from_base_url(url, model_name="glm-5.20") == "glm"


@pytest.mark.parametrize("provider", ["glm_5_2", "glm_5_2_coding"])
def test_glm52_thinking_survives_inbound_and_tool_loop(provider):
    profile = get_profile(provider)
    assert profile.api_mode_fallback_enabled is False
    assert profile.emits_tool_calls_in_content is False
    assert profile.supports_vision is False
    for parse in (parse_chat_completion_message, parse_chat_completion_stream_chunk):
        assert parse({"reasoning_content": "reason"}, profile) == {"reasoning_content": "reason"}
    history = [HumanMessage(content="question"), AIMessage(content="", additional_kwargs={"reasoning_content": "reason"})]
    rewritten, _, _ = apply_outbound_rewrites(history, {}, profile, is_chat_completions_api=True)
    wire = inject_reasoning_into_wire_entry({"role": "assistant"}, rewritten[-1].additional_kwargs, profile, is_chat_completions_api=True)
    assert wire["reasoning_content"] == "reason"


@pytest.mark.parametrize("provider,preserved", [("glm_5_2", False), ("glm_5_2_coding", True)])
def test_cross_user_turn_preservation_matches_endpoint(provider, preserved):
    history = [HumanMessage(content="first"), AIMessage(content="answer", additional_kwargs={"reasoning_content": "original"}), HumanMessage(content="next")]
    rewritten, _, _ = apply_outbound_rewrites(history, {}, get_profile(provider), is_chat_completions_api=True)
    assert (rewritten[1].additional_kwargs.get("reasoning_content") == "original") is preserved
    assert history[1].additional_kwargs["reasoning_content"] == "original"
