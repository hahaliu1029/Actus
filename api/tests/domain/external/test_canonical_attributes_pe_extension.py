"""PE-0 extends CANONICAL_ATTRIBUTES with 4 new attrs the engine emits."""

from app.domain.external.observability import CANONICAL_ATTRIBUTES


def test_pe_attrs_in_whitelist():
    required = {
        "decision_stage",        # policy_get | stage_p1_reader | stage_p2_smart | decision_final | session_mode_check
        "tool_source",           # native | skill | mcp | a2a
        "confirmation_id_hash",  # sha256(confirmation_id)[:16]
        "session_mode",          # SessionStatus.value
    }
    assert required.issubset(set(CANONICAL_ATTRIBUTES))


def test_existing_keys_not_dropped():
    # Sanity: previously-registered canonical keys are still present.
    expected = {
        "trace_id", "request_id", "session_id", "user_id_hash",
        "graph_node", "step_id", "tool_name", "tool_call_id",
        "tool_args_hash", "tool_args_size",
        "llm_provider", "model", "attempt_ix", "event_id",
        "decision_reason",
    }
    assert expected.issubset(set(CANONICAL_ATTRIBUTES))
