from pydantic import BaseModel

from app.interfaces.schemas.base import Response


class LoginLikePayload(BaseModel):
    user: str
    token: str


def test_fail_response_can_be_used_with_typed_response_model() -> None:
    """失败响应不应因为 data 结构与业务成功模型不同而触发校验异常。"""
    response = Response[LoginLikePayload](code=401, msg="认证失败", data=None)

    assert response.code == 401
    assert response.msg == "认证失败"
    assert response.data is None


def test_response_fail_default_data_is_none() -> None:
    response = Response[LoginLikePayload].fail(code=500, msg="登录失败")

    assert response.code == 500
    assert response.msg == "登录失败"
    assert response.data is None
