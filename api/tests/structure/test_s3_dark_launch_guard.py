"""S3 dark-launch structural guards (design §8).

Pin the 4 reachability blocks (F0.9) + the descoped mailbox plane (F0.10) so an
accidental future edit that enables nested spawning or re-touches the mailbox
envelope trips red here instead of silently shipping.
"""
from __future__ import annotations

from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[2]  # api/


def test_spawn_subagent_is_not_a_wired_tool():
    """Block 2 (F0.9): spawn_subagent must never be a live LangChain tool. It
    may appear ONLY in the defensive child_scope_gate HARD_BLOCKED set — never
    in a tool factory under domain/services/tools/."""
    tools_dir = API_ROOT / "app" / "domain" / "services" / "tools"
    leaks = [
        p.relative_to(API_ROOT).as_posix()
        for p in tools_dir.rglob("*.py")
        if "spawn_subagent" in p.read_text(encoding="utf-8")
    ]
    assert leaks == [], f"spawn_subagent leaked into a tool factory: {leaks}"


def test_spawn_subagent_stays_hard_blocked_for_children():
    """Forward-defensive: keep it hard-blocked so a future wiring can't bypass
    the child scope gate."""
    from app.domain.services.permission.child_scope_gate import (
        HARD_BLOCKED_FOR_CHILDREN,
    )

    assert "spawn_subagent" in HARD_BLOCKED_FOR_CHILDREN


def test_child_agent_runner_factory_wires_no_coordinator_subgraph():
    """Block 3 (F0.9): the child runner factory must not pull in the parallel-
    execution / main-graph dispatch surface (a child reaching the dispatch
    branch is what would make depth>1 spawn-capable)."""
    src = (
        API_ROOT
        / "app"
        / "application"
        / "services"
        / "child_agent_runner_factory.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("parallel_execution", "build_parallel", "create_main_graph"):
        assert forbidden not in src, f"child factory referenced {forbidden!r}"


def test_mailbox_envelope_has_no_root_session_id_field():
    """Descoped B3 guard (F0.10): the mailbox wire contract is extra='forbid';
    adding root_session_id would poison-skip ordinary depth-1 children on a
    rolling deploy. Pin the schema so an accidental B3 edit trips red."""
    from app.domain.models.mailbox_envelope import MailboxEnvelope

    assert "root_session_id" not in MailboxEnvelope.model_fields
    assert MailboxEnvelope.model_config.get("extra") == "forbid"


def test_max_subagent_depth_knob_absent_from_shipped_templates():
    """§4.4 / R3-C1: the depth knob must stay out of every shipped config
    template so no default deployment is affected by the le=2 tightening."""
    repo_root = API_ROOT.parent
    templates = [
        repo_root / ".env.example",
        API_ROOT / "config.yaml.example",
        repo_root / "docker-compose.yml",
    ]
    for path in templates:
        if not path.exists():
            continue
        assert "MAX_SUBAGENT_DEPTH" not in path.read_text(encoding="utf-8"), (
            f"{path.name} must not ship a max_subagent_depth override"
        )


def test_contributing_documents_dormant_depth_knob():
    """PR-4 doc gate: CONTRIBUTING explains config=2 is a dormant/forward-compat
    knob (NOT a production enable) + the deferred mailbox plane."""
    repo_root = API_ROOT.parent
    text = (repo_root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "max_subagent_depth" in text
    assert "dormant" in text.lower() or "forward-compat" in text.lower()
