"""R5 CS4 Reader：grants 查询 + 优先级判定 + 过渡期 legacy fallback。

Reader 不直接依赖 SQLAlchemy。通过两个 Protocol 抽象：
- ``ApprovalGrantQuery``：grant 数据源（线上由 ``ApprovalGrantRepository``
  的 ``find_active_grants`` 实现）
- ``LegacyRuleQuery``：旧 ``tool_approval_rules`` 表的 fallback 适配器

Phase 1 优先级（design doc §Recommended Approach）：
  1. ``always_deny`` 任意命中 → ``deny``
  2. ``always_allow`` 任意命中 → ``allow``
  3. ``session_allow`` 精确 ``session_id + arg_digest`` 命中 → ``allow``
  4. ``session_deny`` **不** surface（审计保留，Reader 返 ``no_match``，等
     Phase 2 PermissionEngine retry policy 再开）
  5. 以上都 miss 且 ``legacy_rule_fallback=True`` → 查旧表
  6. 以上都 miss → ``no_match``
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


class LegacyRuleQuery(Protocol):
    """过渡期旧规则表适配器协议。"""

    async def check(
        self,
        user_id: str,
        tool_name: str,
        primary_arg: str,
        dir_arg: Optional[str],
    ) -> CheckResult: ...


class ApprovalStateReader:
    """CS4 Reader。

    AppConfig 的 ``legacy_rule_fallback: bool`` 决定是否注入 ``legacy_rule_query``。
    Reader 自身不读环境变量，不读 config；由 DI 层根据 AppConfig 选择注入或不注入。
    """

    def __init__(
        self,
        query: ApprovalGrantQuery,
        legacy_rule_query: Optional[LegacyRuleQuery] = None,
    ) -> None:
        self._query = query
        self._legacy_rule_query = legacy_rule_query

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

        # Priority 5: 过渡期 legacy rules fallback
        if self._legacy_rule_query is not None:
            return await self._legacy_rule_query.check(
                user_id, tool_name, primary_arg, dir_arg
            )

        return "no_match"
