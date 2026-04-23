"""T18: spec §3.6 C3 commitment scope — Memory.compact() deleted reasoning_content
is NOT reintroduced by A7 rewrites path.

Contract under test: once compact() has stripped ``reasoning_content`` from
persisted history, feeding that compacted history through
``apply_outbound_rewrites`` must never synthesize / refill the field — even
for a profile like Kimi that normally preserves cross-turn reasoning_content.
"""
from langchain_core.messages import AIMessage, HumanMessage

from app.domain.models.memory import Memory
from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._rewrites import apply_outbound_rewrites


def _compacted_history_as_lc_messages(raw: list[dict]) -> list:
    """Project compacted Memory.messages dicts into LangChain messages.

    Mirrors the real runtime projection when persisted history is fed back
    to an LLM adapter: assistant entries keep ``content`` but no longer carry
    ``reasoning_content`` (compact() already stripped it).
    """
    out: list = []
    for m in raw:
        role = m.get("role")
        if role == "user":
            out.append(HumanMessage(m.get("content") or ""))
        elif role == "assistant":
            # additional_kwargs intentionally omits reasoning_content — that
            # is the whole point of the C3 contract.
            out.append(AIMessage(content=m.get("content") or ""))
    return out


def test_memory_compact_strips_reasoning_and_a7_does_not_reintroduce() -> None:
    """A7 rewrites over *compacted* history must not reintroduce reasoning_content.

    Uses Kimi K2 (``reasoning_echo_across_user_turns=True``) intentionally:
    that profile WOULD preserve reasoning_content if it were present on the
    input AIMessage. A post-rewrite AIMessage whose ``additional_kwargs``
    contains ``reasoning_content`` would mean something upstream synthesized
    it — violating C3.
    """
    m = Memory()
    m.messages.append({
        "role": "assistant",
        "content": "a1",
        "reasoning_content": "think1",
    })
    m.messages.append({"role": "user", "content": "q2"})
    m.compact(keep_summary=False)

    # Stage 1: compact stripped reasoning_content from persisted dicts.
    for msg in m.messages:
        assert "reasoning_content" not in msg

    # Stage 2: feed the compacted history — not a fresh prompt — through A7
    # rewrites on a profile that would preserve reasoning_content if present.
    lc_messages = _compacted_history_as_lc_messages(m.messages)
    lc_messages.append(HumanMessage("q3"))

    kimi = get_profile("kimi_k2")
    assert kimi.reasoning_echo_across_user_turns is True  # would preserve if present
    rewritten, _, _ = apply_outbound_rewrites(
        lc_messages, {}, kimi, is_chat_completions_api=True,
    )

    # Every AI message post-rewrite must still lack reasoning_content.
    for rm in rewritten:
        if isinstance(rm, AIMessage):
            assert "reasoning_content" not in rm.additional_kwargs, (
                "A7 rewrites reintroduced reasoning_content on compacted "
                "history — violates spec §3.6 C3 contract."
            )


def test_a7_rewrites_do_not_reintroduce_reasoning_for_deepseek_either() -> None:
    """Same contract under DeepSeek Reasoner (``reasoning_echo_across_user_turns=False``).

    DeepSeek strips cross-turn reasoning_content actively, so absence post-
    rewrite is a weaker signal. This regression sibling pins that DeepSeek
    still doesn't *synthesize* a field that wasn't present in the compacted
    input.
    """
    compacted = [
        AIMessage(content="a1"),  # no reasoning_content — compact stripped it
        HumanMessage("q2"),
    ]
    deepseek = get_profile("deepseek_reasoner")
    rewritten, _, _ = apply_outbound_rewrites(
        compacted, {}, deepseek, is_chat_completions_api=True,
    )
    for rm in rewritten:
        if isinstance(rm, AIMessage):
            assert "reasoning_content" not in rm.additional_kwargs
