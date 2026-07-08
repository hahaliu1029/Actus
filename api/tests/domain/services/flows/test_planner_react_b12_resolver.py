"""B12 P1: PlannerReActFlow.__init__ 接受 file_view_image_resolver 参数。"""
import inspect


def test_planner_react_init_accepts_file_view_image_resolver() -> None:
    """R1#P2-3: 真 red→green——__init__ 签名必须含 file_view_image_resolver。
    （实现前 param 不存在 → FAIL；实现后 → PASS。完整 46-param 构造 unverified/CI,
    用 inspect.signature 精确锁新 API 而不需要构造。）"""
    from app.domain.services.flows.planner_react import PlannerReActFlow

    params = inspect.signature(PlannerReActFlow.__init__).parameters
    assert "file_view_image_resolver" in params
