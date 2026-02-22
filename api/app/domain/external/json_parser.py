from typing import Any, Dict, List, Optional, Protocol, Union


class JSONParser(Protocol):
    """JSON解析器，用于解析json字符串并修复"""

    async def invoke(
        self, text: str, default_value: Optional[Any] = None
    ) -> Union[Dict, List, Any]:
        """调用函数，用于将传递进来的文本进行解析并返回"""
        ...
