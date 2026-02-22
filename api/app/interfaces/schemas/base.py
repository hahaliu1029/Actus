from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Response(BaseModel, Generic[T]):
    """基础API相应结构，继承自Pydantic的BaseModel，并使用泛型支持多种数据类型。"""

    code: int = 200  # 业务状态码，和HTTP状态码保持一致，200表示成功
    msg: str = "success"  # 相应消息提示
    data: Optional[T] = None  # 响应数据，类型为泛型T，可以为任意类型

    @staticmethod
    def success(data: Optional[T] = None, msg: str = "success") -> "Response[T]":
        """创建一个表示成功的响应对象。

        Args:
            data (Optional[T]): 响应数据，默认为None。
            msg (str): 响应消息提示，默认为"success"。

        Returns:
            Response[T]: 表示成功的响应对象。
        """
        return Response[T](code=200, msg=msg, data=data)

    @staticmethod
    def fail(
        code: int = 400, msg: str = "fail", data: Optional[Any] = None
    ) -> "Response[Any]":
        """创建一个表示失败的响应对象。

        Args:
            code (int): 业务状态码，默认为400。
            msg (str): 响应消息提示，默认为"fail"。

        Returns:
            Response[Any]: 表示失败的响应对象。
        """
        return Response[Any](code=code, msg=msg, data=data)
