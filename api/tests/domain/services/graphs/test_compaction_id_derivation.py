"""Test compaction_id derivation: deterministic + content-sensitive.

Spec [R4-P1-1]: same input → same id (retry idempotent), graph-state advanced → different id.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.domain.services.graphs.compaction import (
    compute_messages_input_hash,
    derive_compaction_id,
)


def test_messages_input_hash_is_deterministic():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hi"), AIMessage(content="hello")]
    h1 = compute_messages_input_hash(msgs)
    h2 = compute_messages_input_hash(msgs)
    assert h1 == h2
    assert len(h1) == 16


def test_messages_input_hash_changes_with_added_message():
    base = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    longer = base + [AIMessage(content="reply")]
    assert compute_messages_input_hash(base) != compute_messages_input_hash(longer)


def test_messages_input_hash_handles_tool_messages_and_calls():
    msgs = [
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "f", "args": {"k": "v"}}],
        ),
        ToolMessage(content="result", tool_call_id="c1", name="f"),
    ]
    h = compute_messages_input_hash(msgs)
    assert len(h) == 16


def test_derive_compaction_id_same_inputs_same_id():
    a = derive_compaction_id(
        session_id="s1",
        messages_input_hash="aabbccddeeff0011",
        tokens_before_total=1000,
        tokens_after_total=500,
        messages_removed_total=10,
        summary="hello world summary",
    )
    b = derive_compaction_id(
        session_id="s1",
        messages_input_hash="aabbccddeeff0011",
        tokens_before_total=1000,
        tokens_after_total=500,
        messages_removed_total=10,
        summary="hello world summary",
    )
    assert a == b
    assert len(a) == 16


def test_derive_compaction_id_changes_with_input_hash():
    a = derive_compaction_id(
        session_id="s1",
        messages_input_hash="aabbccddeeff0011",
        tokens_before_total=1000,
        tokens_after_total=500,
        messages_removed_total=10,
        summary="x",
    )
    b = derive_compaction_id(
        session_id="s1",
        messages_input_hash="ffffffffffffffff",
        tokens_before_total=1000,
        tokens_after_total=500,
        messages_removed_total=10,
        summary="x",
    )
    assert a != b
