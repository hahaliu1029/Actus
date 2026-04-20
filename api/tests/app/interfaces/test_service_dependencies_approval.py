"""R5b-1 DI wiring 单元测试：
``get_approval_state_writer`` / ``get_approval_state_reader`` factory 行为。

不触发 ``_load_app_config`` 的真实文件 IO——通过 monkeypatch 替换为返回
一个最小 AppConfig 的 stub。Writer factory 路径纯函数，无状态断言。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.application.services.approval_state_adapters import (
    SessionLegacyRuleQuery,
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


class _StubAppConfig:
    """最小 stub：只暴露 agent_config.tool_confirmation.legacy_rule_fallback。"""

    def __init__(self, *, legacy_rule_fallback: bool) -> None:
        tool_conf = MagicMock()
        tool_conf.legacy_rule_fallback = legacy_rule_fallback
        agent = MagicMock()
        agent.tool_confirmation = tool_conf
        self.agent_config = agent


def test_reader_factory_with_legacy_fallback_enabled(monkeypatch) -> None:
    """legacy_rule_fallback=True → Reader 注入 SessionLegacyRuleQuery。"""
    monkeypatch.setattr(
        service_dependencies,
        "_load_app_config",
        lambda: _StubAppConfig(legacy_rule_fallback=True),
    )
    # 避免真去初始化 Postgres 单例——只需返回有 session_factory 属性的对象
    fake_pg = MagicMock()
    fake_pg.session_factory = MagicMock(name="fake_session_factory")
    monkeypatch.setattr(service_dependencies, "get_postgres", lambda: fake_pg)

    reader = service_dependencies.get_approval_state_reader()
    assert isinstance(reader, ApprovalStateReader)
    assert isinstance(reader._query, UowApprovalGrantQuery)
    assert isinstance(reader._legacy_rule_query, SessionLegacyRuleQuery)


def test_reader_factory_with_legacy_fallback_disabled(monkeypatch) -> None:
    """legacy_rule_fallback=False → Reader.legacy_rule_query is None。"""
    monkeypatch.setattr(
        service_dependencies,
        "_load_app_config",
        lambda: _StubAppConfig(legacy_rule_fallback=False),
    )
    # fallback 关闭时 get_postgres 不应被调用——给一个会爆的 stub 固化这个契约
    def _boom():
        raise AssertionError(
            "get_postgres should NOT be called when legacy_rule_fallback=False"
        )

    monkeypatch.setattr(service_dependencies, "get_postgres", _boom)

    reader = service_dependencies.get_approval_state_reader()
    assert isinstance(reader, ApprovalStateReader)
    assert isinstance(reader._query, UowApprovalGrantQuery)
    assert reader._legacy_rule_query is None
