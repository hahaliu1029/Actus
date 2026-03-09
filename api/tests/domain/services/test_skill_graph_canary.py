"""灰度分桶函数测试。"""

from app.domain.services.flows.skill_graph_canary import is_skill_graph_enabled


def test_zero_percent_always_disabled():
    assert is_skill_graph_enabled("any-user", 0) is False


def test_100_percent_always_enabled():
    assert is_skill_graph_enabled("any-user", 100) is True


def test_negative_percent_disabled():
    assert is_skill_graph_enabled("user-1", -5) is False


def test_over_100_percent_enabled():
    assert is_skill_graph_enabled("user-1", 150) is True


def test_same_user_consistent():
    """同一 user_id 多次调用结果一致。"""
    results = [is_skill_graph_enabled("user-abc-123", 50) for _ in range(100)]
    assert len(set(results)) == 1  # 全部相同


def test_distribution_roughly_correct():
    """大量 user_id 的分布大致符合百分比。"""
    total = 10000
    enabled_count = sum(
        is_skill_graph_enabled(f"user-{i}", 30) for i in range(total)
    )
    ratio = enabled_count / total
    assert 0.20 < ratio < 0.40, f"ratio={ratio}, expected ~0.30"
