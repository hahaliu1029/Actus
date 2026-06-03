"""R5b-1 DI wiring 单元测试：
``get_approval_state_writer`` / ``get_approval_state_reader`` factory 行为。

PE-4d1：legacy ``tool_approval_rules`` fallback 退役后，reader factory 变成
grants-only——既不读 ``_load_app_config``（曾用于读 ``legacy_rule_fallback``
开关），也不调 ``get_postgres``（曾用于构造 ``SessionLegacyRuleQuery``）。
本测试把这两个符号 monkeypatch 成会爆，固化"二者都不再被调用"这个契约。
Writer factory 路径纯函数，无状态断言。
"""

from __future__ import annotations

from app.application.services.approval_state_adapters import (
    UowApprovalGrantQuery,
)
from app.application.services.approval_state_writer import ApprovalStateWriter
from app.domain.services.approval_state_reader import ApprovalStateReader
from app.infrastructure.storage.postgres import get_uow
from app.interfaces import service_dependencies


def test_writer_factory_returns_writer_bound_to_get_uow() -> None:
    """Writer factory 返 ApprovalStateWriter，且 uow_factory 是线上 get_uow。"""
    writer = service_dependencies.get_approval_state_writer()
    assert isinstance(writer, ApprovalStateWriter)
    # 约定：DI 层注入 get_uow 作为 uow_factory（而非 adapter / wrapper）
    assert writer._uow_factory is get_uow


def test_reader_factory_builds_grants_only_reader(monkeypatch) -> None:
    """PE-4d1: Reader is grants-only — no SessionLegacyRuleQuery injection,
    no legacy_rule_fallback config read, no get_postgres call.

    Both ``_load_app_config`` (former legacy_rule_fallback read) and
    ``get_postgres`` (former SessionLegacyRuleQuery session_factory) are
    monkeypatched to raise: the grants-only factory must call neither, so a
    future re-introduction of either call fails this test loudly instead of
    silently resurrecting the legacy read path."""

    def _boom_config():
        raise AssertionError(
            "get_approval_state_reader must NOT read app config after PE-4d1 "
            "(legacy_rule_fallback retired)"
        )

    def _boom_postgres():
        raise AssertionError(
            "get_approval_state_reader must NOT call get_postgres after PE-4d1 "
            "(SessionLegacyRuleQuery retired)"
        )

    monkeypatch.setattr(service_dependencies, "_load_app_config", _boom_config)
    monkeypatch.setattr(service_dependencies, "get_postgres", _boom_postgres)

    reader = service_dependencies.get_approval_state_reader()
    assert isinstance(reader, ApprovalStateReader)
    assert isinstance(reader._query, UowApprovalGrantQuery)
    assert not hasattr(reader, "_legacy_rule_query")
