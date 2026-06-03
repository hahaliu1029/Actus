"""R5 CS4 Reader：grants 查询 + 优先级判定。

Reader 不直接依赖 SQLAlchemy。通过 ``ApprovalGrantQuery`` Protocol 抽象
grant 数据源（线上由 ``ApprovalGrantRepository`` 的 ``find_active_grants``
实现）。

优先级（design doc §Recommended Approach）：
  1. ``always_deny`` 任意命中 → ``deny``
  2. ``always_allow`` 任意命中 → ``allow``
  3. ``session_allow`` 精确 ``session_id + arg_digest`` 命中 → ``allow``
  4. ``session_deny`` **不** surface（审计保留，Reader 返 ``no_match``，等
     Phase 2 PermissionEngine retry policy 再开）
  5. 以上都 miss → ``no_match``

PE-4d1：旧 ``tool_approval_rules`` 表的 legacy fallback（former Priority 5）
已退役，Reader 只读 grants。
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol

from app.domain.models.approval_grant import ApprovalGrant

CheckResult = Literal["allow", "deny", "no_match"]


class ApprovalGrantQuery(Protocol):
    """Grant 数据源适配器协议。"""

    async def find_active_grants(
        self,
        user_id: str,
        session_id: Optional[str],
        tool_name: str,
    ) -> list[ApprovalGrant]: ...


class ApprovalStateReader:
    """CS4 Reader。

    Reader 自身不读环境变量，不读 config；只读 grants 数据源。
    """

    def __init__(
        self,
        query: ApprovalGrantQuery,
    ) -> None:
        self._query = query

    async def check(
        self,
        user_id: str,
        session_id: Optional[str],
        tool_name: str,
        arg_digest: str,
        primary_arg: str,
        dir_arg: Optional[str],
    ) -> CheckResult:
        grants = await self._query.find_active_grants(user_id, session_id, tool_name)

        # Priority 1: always_deny
        for g in grants:
            if (
                g.scope == "always"
                and g.effect == "deny"
                and g.matches(primary_arg, dir_arg)
            ):
                return "deny"

        # Priority 2: always_allow
        for g in grants:
            if (
                g.scope == "always"
                and g.effect == "approve"
                and g.matches(primary_arg, dir_arg)
            ):
                return "allow"

        # Priority 3: session_allow（精确 session + arg_digest 匹配）
        for g in grants:
            if (
                g.scope == "session"
                and g.effect == "approve"
                and g.session_id == session_id
                and g.arg_digest == arg_digest
            ):
                return "allow"

        # Priority 4: session_deny 不 surface（design doc §Open Questions 1）

        # PE-4d1: legacy tool_approval_rules fallback (former Priority 5) retired.
        return "no_match"
