from __future__ import annotations
from datetime import datetime
from fnmatch import fnmatch
from uuid import uuid4
from pydantic import BaseModel, Field

class ToolApprovalRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    tool_name: str
    rule: str  # "always_allow" | "always_deny"
    command_pattern: str
    dir_pattern: str = ""  # empty = no directory constraint (NOT NULL sentinel)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True

    def matches(self, primary_arg: str, dir_arg: str | None) -> bool:
        if not fnmatch(primary_arg, self.command_pattern):
            return False
        if self.dir_pattern:  # non-empty = has constraint
            if dir_arg is None:
                return False
            if not fnmatch(dir_arg, self.dir_pattern):
                return False
        return True
