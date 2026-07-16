"""SPM Task 12/23/32 — sandbox_provision_mode 三态 config validator + off×coordinator
启动互斥校验的单元测试。

- config validator：终态 ``SANDBOX_PROVISION_MODE_ALLOWED == {"always",
  "on_demand", "off"}``（Task 32 解锁 off，梯度收官）——三档全部被 validator 接受并
  按既有 normalize（strip + lower）归一；非法值（如 "lazy"、空串）仍被拒。
- exclusion：``check_sandbox_off_flag_exclusion`` 在 mode=="off" 时对三个 **env-only**
  coordinator flag helper（非 Settings 属性）做 fail-fast 互斥（SPM DD-6）。此组与
  ALLOWED 解锁正交（用 ``model_construct`` 绕过 validator），解锁后仍全绿。

构造方式照抄同目录 ``test_config_extension_governance_mode.py`` / ``test_sandbox_strict_caps_flag.py``
的 ``Settings(env="test", ...)`` 风格。
"""
from __future__ import annotations

import itertools

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

    def test_final_allowed_values_accept_three_reject_invalid(self):
        """DD-22 终态（Task 32）：``SANDBOX_PROVISION_MODE_ALLOWED`` 解锁为三值全集
        ``{"always", "on_demand", "off"}`` —— 三档全部被 validator 接受（含 case/
        whitespace normalize），非法值仍被拒。

        取代 PR-2 阶段的 ``test_off_rejected_while_not_allowed``（彼时 off 尚未进
        ALLOWED、被 validator fail-fast 拒）。off×coordinator 启动互斥
        （``TestOffCoordinatorExclusion``）与本解锁正交，仍全绿。"""
        # 三档全接受，且各自 round-trip 原值。
        for mode in ("always", "on_demand", "off"):
            assert (
                Settings(env="test", sandbox_provision_mode=mode).sandbox_provision_mode
                == mode
            )
        # off 也走既有 normalize（strip + lower）——大小写/空白变体归一为 "off"。
        assert (
            Settings(env="test", sandbox_provision_mode=" OFF ").sandbox_provision_mode
            == "off"
        )
        # 非法值仍拒（validator 语义不因解锁而放宽：空串、拼写错误、旧枚举名）。
        for bad in ("lazy", "", "disabled"):
            with pytest.raises(ValidationError):
                Settings(env="test", sandbox_provision_mode=bad)

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
    """构造指定 provision_mode 的 Settings（绕过 validator 直接注入）。

    这里用 ``model_construct`` 绕过 **所有** validator 直接注入
    ``sandbox_provision_mode`` —— 本组测试验证的是 ``check_sandbox_off_flag_exclusion``
    的分支逻辑（只读 ``.sandbox_provision_mode``），validator 本身（三档全接受、非法
    值拒）由 ``TestSandboxProvisionModeConfig`` 覆盖，二者关注点分离。终态下 off 已进
    ALLOWED，``model_construct`` 仍是把注入与 validation 关注点解耦的最短路径。
    """
    return Settings.model_construct(sandbox_provision_mode=mode)


# Task 29 — FULL off×coordinator exclusion matrix.
#
# Per-flag ``(env_value, is_truthy)`` options feeding a Cartesian product over
# the three env-only coordinator flags × mode {off, always}. Each flag exercises
# at least one truthy token ("1"/"true" — parsed by the shared
# ``_TRUTHY = {"true","1","yes","on"}`` in coordinator_feature_flag.py /
# coordinator_shell_mode_flag.py / agent_teams_flag.py) and one falsy form
# ("0" / "false" / unset). ``None`` means leave the var unset (a falsy path).
_MATRIX_FLAG_VALUES = {
    "ACTUS_C2_COORDINATOR_ENABLED": (("1", True), ("0", False)),
    "ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED": (("true", True), ("false", False)),
    "ACTUS_C2_AGENT_TEAMS_ENABLED": (("1", True), (None, False)),
}


def _off_coordinator_matrix_params():
    """Build the full mode × flag-value Cartesian as pytest params.

    2 modes × 2**3 flag combinations = 16 cases. ``env`` maps each flag to its
    cell value (``None`` → leave unset); ``expected_conflicts`` is the ordered
    subset of truthy flags (only meaningful under mode == "off").
    """
    flag_names = tuple(_MATRIX_FLAG_VALUES)
    per_flag = [_MATRIX_FLAG_VALUES[name] for name in flag_names]
    params = []
    for mode in ("off", "always"):
        for combo in itertools.product(*per_flag):
            env = {}
            truthy = []
            id_bits = [mode]
            for name, (value, is_truthy) in zip(flag_names, combo):
                env[name] = value
                short = name.removeprefix("ACTUS_C2_").removesuffix("_ENABLED")
                id_bits.append(f"{short}={'unset' if value is None else value}")
                if is_truthy:
                    truthy.append(name)
            params.append(
                pytest.param(mode, env, tuple(truthy), id="-".join(id_bits))
            )
    return params


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

    @pytest.mark.parametrize(
        "mode, env, expected_conflicts", _off_coordinator_matrix_params()
    )
    def test_off_coordinator_exclusion_full_truthy_falsy_matrix(
        self, monkeypatch, mode, env, expected_conflicts
    ):
        """Full matrix: off × any-truthy-flag → RuntimeError naming exactly the
        truthy flag(s); every other cell (mode == "always" regardless of flags,
        or off with all three flags falsy) returns None.

        env-only helpers read ``os.environ`` fresh on each call, so hermeticity
        requires clearing all three vars before seeding this cell.
        """
        for var in _C2_FLAGS:
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            if value is not None:
                monkeypatch.setenv(var, value)

        settings = _mk_settings(mode)

        if mode == "off" and expected_conflicts:
            with pytest.raises(RuntimeError, match="incompatible") as excinfo:
                check_sandbox_off_flag_exclusion(settings)
            message = str(excinfo.value)
            # 每个 truthy flag 都被点名 ...
            for name in expected_conflicts:
                assert name in message, (name, message)
            # ... 且 falsy flag 不会被误列（三名互不为子串，absence 断言无歧义）。
            for name in _C2_FLAGS:
                if name not in expected_conflicts:
                    assert name not in message, (name, message)
        else:
            # mode == "always" 在读 flag 前短路；off + 全 falsy 无冲突。二者均返回 None。
            assert check_sandbox_off_flag_exclusion(settings) is None
