"""D1a §3.6/§4.1/§4.2 判定内核（纯函数矩阵）。"""
from datetime import datetime, timedelta, timezone

from app.domain.services.extension_admission_logic import (
    REQUIRED_OBSERVED_CATEGORIES,
    decide_admitted,
    pin_presence,
    select_reason,
    should_persist_observation,
)

NOW = datetime(2026, 7, 11, tzinfo=timezone.utc)


class TestRequiredObserved:
    def test_per_kind_write_sets(self):
        # §5.2 per-kind pin/approve 写集
        assert REQUIRED_OBSERVED_CATEGORIES == {
            "mcp": frozenset({"surface", "config_fingerprint"}),
            "a2a": frozenset({"surface", "config_fingerprint"}),
            "skill": frozenset({"artifact"}),
            "plugin": frozenset({"artifact"}),
        }


class TestPinPresence:
    def test_matrix(self):
        assert pin_presence(None, None, 1) == "unpinned"
        assert pin_presence("h", 1, 1) == "pinned"
        assert pin_presence("h", 0, 1) == "pin_stale"   # R1#15：版本过期视同 unpinned（待 re-approve）


class TestShouldPersist:
    def _call(self, **kw):
        base = dict(stored_value="h1", new_value="h1",
                    last_observed_at=NOW - timedelta(seconds=10), now=NOW,
                    window_seconds=3600, stored_schema_version=1, current_schema_version=1)
        base.update(kw)
        return should_persist_observation(**base)

    def test_first_observation(self):
        assert self._call(stored_value=None) is True                     # ①首次

    def test_value_change(self):
        assert self._call(new_value="h2") is True                        # ②值变化

    def test_window_elapsed(self):
        assert self._call(last_observed_at=NOW - timedelta(seconds=3601)) is True   # ③超窗口

    def test_version_stale_forces_persist(self):
        # ④R44#1a：版本过期强制持久化——即使 digest 未变且窗口内（否则 refresh 假成功死锁）
        assert self._call(stored_schema_version=0, current_schema_version=1) is True

    def test_steady_state_no_write(self):
        # 窗口内匹配与稳态失配一视同仁=零写（R40#1——失配不是独立触发）
        assert self._call() is False

    def test_missing_last_observed_treated_as_elapsed(self):
        assert self._call(last_observed_at=None) is True


class TestSelectReason:
    def test_priority_unknown_first(self):
        assert select_reason(row_exists=False, status=None,
                             parent_blocked=False, detection=None) == "unknown"

    def test_administrative_over_detection(self):
        # R36#1/R37#1：行政类掩盖检测类——disabled 行的 drift 最终 reason=disabled
        assert select_reason(row_exists=True, status="disabled",
                             parent_blocked=False, detection="config_drift") == "disabled"
        assert select_reason(row_exists=True, status="quarantined",
                             parent_blocked=False, detection="pin_mismatch") == "quarantined"
        assert select_reason(row_exists=True, status="deleted",
                             parent_blocked=False, detection=None) == "deleted"

    def test_parent_blocked_is_administrative(self):
        assert select_reason(row_exists=True, status="active",
                             parent_blocked=True, detection="unpinned") == "parent_blocked"

    def test_detection_when_clean_admin(self):
        assert select_reason(row_exists=True, status="active",
                             parent_blocked=False, detection="pin_stale") == "pin_stale"

    def test_ok_when_nothing(self):
        assert select_reason(row_exists=True, status="active",
                             parent_blocked=False, detection=None) == "ok"


class TestDecideAdmitted:
    def test_neutral_always_admit(self):
        assert decide_admitted("shadow", "ok") is True
        assert decide_admitted("enforce", "ok") is True

    def test_detection_shadow_fail_open_enforce_closed(self):
        for r in ("unknown", "unpinned", "pin_stale", "config_drift",
                  "pin_mismatch", "registry_unavailable"):
            assert decide_admitted("shadow", r) is True, r
            assert decide_admitted("enforce", r) is False, r

    def test_administrative_blocked_both_modes(self):
        # §4.1 R32#1：行政/结构类 shadow 与 enforce 均强制（INV-D1-8 在 shadow 同样成立）
        for r in ("quarantined", "disabled", "deleted", "parent_blocked"):
            assert decide_admitted("shadow", r) is False, r
            assert decide_admitted("enforce", r) is False, r
