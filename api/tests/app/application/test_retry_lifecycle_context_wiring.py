"""C7 PR4 — retry_from_suspend → RetryLifecycleContext 透传 + §12-7(a) 旁路 pin。"""
import inspect
import re
from pathlib import Path

from app.application.services.agent_service import AgentService
from app.domain.models.lifecycle import RETRY_BUDGET_INITIAL


def test_resume_task_with_handoff_accepts_context_kwarg():
    sig = inspect.signature(AgentService._resume_task_with_handoff)
    param = sig.parameters.get("retry_lifecycle_context")
    assert param is not None and param.default is None and param.kind == param.KEYWORD_ONLY


def test_create_task_accepts_context_kwarg():
    sig = inspect.signature(AgentService._create_task)
    param = sig.parameters.get("retry_lifecycle_context")
    assert param is not None and param.default is None


def test_retry_from_suspend_builds_context_from_claimed_budget():
    # 源码 pin：claim 成功后以 claimed_retry_budget 构造 RetryLifecycleContext。
    # 裸子串 "retry_budget_remaining=claimed_retry_budget" 在既有代码
    # （agent_service.py:3938 supervisor.resume(...) kwarg）已出现——断言
    # 必须锚定到 RetryLifecycleContext ctor 内才有判别力（R3 review P3-3）。
    src = inspect.getsource(AgentService.retry_from_suspend)
    assert re.search(
        r"RetryLifecycleContext\((?s:.*?)retry_budget_remaining=claimed_retry_budget", src
    )


def test_create_task_derives_epoch_from_persistent_budget():
    # R10#A3 pin：epoch 从 session 持久列派生，禁止内存计数
    src = inspect.getsource(AgentService._create_task)
    assert "RETRY_BUDGET_INITIAL" in src
    assert "session.retry_budget_remaining" in src


def test_application_error_event_bypass_has_no_task_failed_emission():
    # §12-7(a) 负向 pin：application ErrorEvent 旁路（agent_service.py:2391,3122,3696）
    # 不产 task.failed——app 层不得出现 lifecycle 发射入口（documented gap，§1 局限 3）
    src = Path(inspect.getfile(AgentService)).read_text(encoding="utf-8")
    assert "build_lifecycle_event" not in src
    assert "_emit_task_lifecycle" not in src


def test_retry_budget_initial_is_three():
    assert RETRY_BUDGET_INITIAL == 3
