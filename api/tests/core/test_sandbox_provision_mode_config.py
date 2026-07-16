"""SPM Task 12/23 — sandbox_provision_mode 三态 config validator + off×coordinator
启动互斥校验的单元测试。

- config validator：PR-2 阶段 ``SANDBOX_PROVISION_MODE_ALLOWED == {"always",
  "on_demand"}``（Task 23 解锁 on_demand），故 on_demand 被接受、off 仍被 validator
  拒绝（off 在 PR-4 解锁）。
- exclusion：``check_sandbox_off_flag_exclusion`` 在 mode=="off" 时对三个 **env-only**
  coordinator flag helper（非 Settings 属性）做 fail-fast 互斥（SPM DD-6）。

构造方式照抄同目录 ``test_config_extension_governance_mode.py`` / ``test_sandbox_strict_caps_flag.py``
的 ``Settings(env="test", ...)`` 风格。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.main import check_sandbox_off_flag_exclusion
from core.config import Settings


class TestSandboxProvisionModeConfig:
    def test_default_always(self):
        assert Settings(env="test").sandbox_provision_mode == "always"

    def test_provision_timeout_default(self):
        # SPM §5.2e：provision 硬超时缺省 90s。
        assert Settings(env="test").sandbox_provision_timeout_seconds == 90

    def test_on_demand_accepted_after_unlock(self):
        """DD-22 阶段语义：PR-2（Task 23）解锁 on_demand → validator 接受，且沿用
        既有 normalize（case/whitespace 归一）。"""
        assert (
            Settings(env="test", sandbox_provision_mode="on_demand").sandbox_provision_mode
            == "on_demand"
        )
        # normalize 仍生效（与 always 同路径）。
        assert (
            Settings(env="test", sandbox_provision_mode=" On_Demand ").sandbox_provision_mode
            == "on_demand"
        )

    def test_off_rejected_while_not_allowed(self):
        """DD-22 阶段语义：PR-2 阶段 off 仍必须被拒（PR-4 解锁）。"""
        with pytest.raises(ValidationError):
            Settings(env="test", sandbox_provision_mode="off")

    def test_normalizes_case_whitespace(self):
        assert (
            Settings(env="test", sandbox_provision_mode=" Always ").sandbox_provision_mode
            == "always"
        )


# --- off × coordinator 启动互斥 -------------------------------------------

_C2_FLAGS = (
    "ACTUS_C2_COORDINATOR_ENABLED",
    "ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED",
    "ACTUS_C2_AGENT_TEAMS_ENABLED",
)


def _mk_settings(mode: str) -> Settings:
    """构造指定 provision_mode 的 Settings。

    PR-2 阶段 ``SANDBOX_PROVISION_MODE_ALLOWED == {"always", "on_demand"}``，validator
    仍会拒 "off"（PR-4 才解锁）。这里用 ``model_construct`` 绕过 validator 直接注入
    ``sandbox_provision_mode`` —— 本组测试验证的是 ``check_sandbox_off_flag_exclusion``
    的分支逻辑（只读 ``.sandbox_provision_mode``），validator 本身由
    ``TestSandboxProvisionModeConfig`` 覆盖，二者关注点分离。
    """
    return Settings.model_construct(sandbox_provision_mode=mode)


class TestOffCoordinatorExclusion:
    def test_off_with_coordinator_flag_fails_fast(self, monkeypatch):
        # flag 是 env-only helper（无缓存，每次读 os.environ）。hermetic：先清三键再设一，
        # 避免宿主/CI 环境已设的其它 C2 flag 干扰断言。
        for var in _C2_FLAGS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "1")
        with pytest.raises(RuntimeError, match="incompatible"):
            check_sandbox_off_flag_exclusion(_mk_settings("off"))

    def test_off_clean_passes(self, monkeypatch):
        for var in _C2_FLAGS:
            monkeypatch.delenv(var, raising=False)
        # 无冲突 flag → 不抛（None 返回）。
        check_sandbox_off_flag_exclusion(_mk_settings("off"))

    def test_always_with_flags_passes(self, monkeypatch):
        # mode != "off" → 提前 return，flag 状态无关（即使 coordinator flag 开着也放行）。
        monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "1")
        check_sandbox_off_flag_exclusion(_mk_settings("always"))
