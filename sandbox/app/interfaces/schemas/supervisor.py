from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TimeoutRequest(BaseModel):
    """激活超时销毁请求"""

    minutes: Optional[int] = Field(default=None, gt=0, description="分钟数")

    @field_validator("minutes", mode="before")
    @classmethod
    def reject_boolean_minutes(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("minutes must be a positive integer")
        return value
