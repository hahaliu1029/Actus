"""C4.1a PR-2 — SubagentRunModel ORM 结构测试（纯内存，无 DB）（spec §4 + §7 PR-2）。"""
from __future__ import annotations

from sqlalchemy import UniqueConstraint


def test_model_registered_in_metadata() -> None:
    # 注册进 __init__ 后 Base.metadata 才含新表（create_all/alembic discovery 依赖此）。
    from app.infrastructure.models import Base

    assert "subagent_runs" in Base.metadata.tables


def test_all_columns_present() -> None:
    from app.infrastructure.models.subagent_run_orm import SubagentRunModel

    cols = set(SubagentRunModel.__table__.columns.keys())
    expected = {
        "id",
        "runtime",
        "lifecycle_state",
        "terminal_outcome",
        "summary",
        "error_summary",
        "parent_session_id",
        "child_session_id",
        "source_ref",
        "cost_authoritative",
        "cost_total_input_tokens",
        "cost_total_output_tokens",
        "cost_total_usd",
        "cost_tool_call_count",
        "duration_seconds",
        "duration_source",
        "artifacts",
        "created_at",
    }
    # exact-match（非 subset）：ORM 若意外多一列（如 debug_payload）也必须 fail，
    # 锁死 ORM↔migration 18-列 parity（codex R2 P3）。
    assert cols == expected


def test_unique_constraint_on_child_session_id() -> None:
    from app.infrastructure.models.subagent_run_orm import SubagentRunModel

    unique_names = {
        c.name
        for c in SubagentRunModel.__table__.constraints
        if isinstance(c, UniqueConstraint)
    }
    assert "uq_subagent_runs_child_session_id" in unique_names


def test_parent_session_id_indexed() -> None:
    from app.infrastructure.models.subagent_run_orm import SubagentRunModel

    index_names = {ix.name for ix in SubagentRunModel.__table__.indexes}
    assert "ix_subagent_runs_parent_session_id" in index_names


def test_nullability_contract() -> None:
    from app.infrastructure.models.subagent_run_orm import SubagentRunModel

    cols = SubagentRunModel.__table__.columns
    # 两个 LOCAL seat 恒提供 → NOT NULL（查询键 / 恒有值）
    assert cols["parent_session_id"].nullable is False
    assert cols["runtime"].nullable is False
    assert cols["lifecycle_state"].nullable is False
    assert cols["duration_source"].nullable is False
    assert cols["cost_authoritative"].nullable is False
    # REMOTE 可 None / 非终态 None
    assert cols["child_session_id"].nullable is True
    assert cols["terminal_outcome"].nullable is True
    assert cols["cost_total_usd"].nullable is True
