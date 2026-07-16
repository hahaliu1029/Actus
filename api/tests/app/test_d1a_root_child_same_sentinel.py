"""D1a Task 13 — INV-D1-2 行为半：root/child 同源 AdmissionPort 实例（R1#21/#22）。

注入拓扑 AST 门证明**结构**（main.py `ChildRunnerSharedDeps(...)` 填 kwarg）；本文件
证明**行为**：真实 builder 路径拿到同一实例、child **不自建**。

- (a) root：真实 `AgentService(..., extension_admission_port=sentinel)` 构造后按身份
  持有该实例（`service._extension_admission_port is sentinel`）。AgentService 的
  `_create_task` 以 `extension_admission_port=getattr(self, "_extension_admission_port",
  None)` 无条件转发给 AgentTaskRunner；runner ctor 按身份存
  （`self._admission_port = extension_admission_port`）——该 runner ctor 的身份存储由
  (b) 直接锁定（root/child 共用同一 AgentTaskRunner ctor）。
- (b) child：真实 `_make_shared_child_runner_builder` + `ChildRunnerSharedDeps(
  extension_admission_port=sentinel)` → build child runner → `runner._admission_port
  is sentinel`（child 从 deps 拿同一实例，不新建 port）。
- (c) child 不自建：deps 缺省（None）时 `runner._admission_port is None`（child 绝不
  凭空造 port——off/legacy 向后兼容）。

harness 照抄 tests/interfaces/test_coordinator_composition_root.py 的 fake deps 集
与 builder 调用签名。
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.application.services.agent_service import AgentService
from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    MCPConfig,
    ToolRuntimeConfig,
)
from app.interfaces.service_dependencies import (
    ChildRunnerSharedDeps,
    _make_shared_child_runner_builder,
)
from tests.app.application.services.conftest import default_snapshot


class _DummyTaskClass:
    """AgentService.__init__ 仅存 `self._task_cls`——trivial class 足够。"""


def _child_deps(**overrides) -> ChildRunnerSharedDeps:
    base = dict(
        uow_factory=lambda: MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        file_storage=MagicMock(),
        search_engine=MagicMock(),
        checkpointer_pool=MagicMock(),
        execution_supervisor=MagicMock(),
        tool_runtime=ToolRuntimeConfig(),
    )
    base.update(overrides)
    return ChildRunnerSharedDeps(**base)


def _build_child_runner(deps: ChildRunnerSharedDeps):
    builder = _make_shared_child_runner_builder(resolve_child_runner_deps=lambda: deps)
    return builder(
        session_id="c1",
        tool_filter=frozenset({"file_write"}),
        mailbox_publisher=MagicMock(),
        terminal_envelope_publisher_disabled=True,
        sandbox_accessor=EagerSandboxAccessor(MagicMock()),
        browser_accessor=EagerBrowserAccessor(MagicMock()),
        user_id="u1",
        cost_callback_handler=MagicMock(),
    )


def test_root_agent_service_holds_admission_port_by_identity():
    """(a) root AgentService 按身份持有注入的 AdmissionPort 实例。"""
    sentinel = object()
    service = AgentService(
        uow_factory=lambda: MagicMock(),
        config_snapshot=default_snapshot(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        search_engine=object(),
        file_storage=object(),
        extension_admission_port=sentinel,
    )
    assert service._extension_admission_port is sentinel


def test_child_runner_inherits_same_admission_port_instance():
    """(b) child runner 从 ChildRunnerSharedDeps 拿到**同一** AdmissionPort 实例
    （admission 必须同源——非 B12 file_view child 自建 registry 模式）。"""
    sentinel = object()
    deps = _child_deps(extension_admission_port=sentinel)
    assert deps.extension_admission_port is sentinel
    runner = _build_child_runner(deps)
    assert runner._admission_port is sentinel


def test_child_runner_does_not_self_build_port_when_absent():
    """(c) deps 缺省（None）→ child runner `_admission_port is None`（child 不自建 port）。"""
    deps = _child_deps()  # extension_admission_port 缺省 = None
    assert deps.extension_admission_port is None
    runner = _build_child_runner(deps)
    assert runner._admission_port is None
