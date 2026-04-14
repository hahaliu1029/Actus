from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from app.domain.repositories.tool_approval_rule_repository import ToolApprovalRuleRepository

logger = logging.getLogger(__name__)


class ApprovalCache:
    """Check session-level (Redis) and always-level (DB) approval caches.
    Returns: 'allow', 'deny', or 'no_match'.

    Two construction modes:
    1. ``ApprovalCache(redis, rule_repo)`` — fixed repo (used in tests).
    2. ``ApprovalCache(redis, session_factory=factory)`` — creates a fresh
       DBToolApprovalRuleRepository per check() call (used in production).
    """

    def __init__(
        self,
        redis: "Redis",
        rule_repo: "ToolApprovalRuleRepository | None" = None,
        session_factory: Callable[[], Any] | None = None,
    ):
        self._redis = redis
        self._rule_repo = rule_repo
        self._session_factory = session_factory

    async def _get_rules(self, user_id: str, tool_name: str) -> list:
        """Fetch rules using either the fixed repo or a fresh session."""
        if self._rule_repo is not None:
            return await self._rule_repo.find_by_user_and_tool(user_id, tool_name)
        if self._session_factory is not None:
            from app.infrastructure.repositories.db_tool_approval_rule_repository import (
                DBToolApprovalRuleRepository,
            )
            async with self._session_factory() as session:
                repo = DBToolApprovalRuleRepository(session)
                return await repo.find_by_user_and_tool(user_id, tool_name)
        logger.warning(
            "ApprovalCache: no rule_repo or session_factory — always rules are disabled"
        )
        return []

    async def check(self, user_id: str, session_id: str, tool_name: str,
                    arg_digest: str, primary_arg: str, dir_arg: str | None) -> str:
        """Returns 'allow', 'deny', or 'no_match'."""
        # 1. Check always rules (DB) — deny takes precedence
        rules = await self._get_rules(user_id, tool_name)
        for rule in rules:
            if rule.rule == "always_deny" and rule.matches(primary_arg, dir_arg):
                return "deny"
        for rule in rules:
            if rule.rule == "always_allow" and rule.matches(primary_arg, dir_arg):
                return "allow"

        # 2. Check session cache (Redis)
        key = f"approval:{session_id}:{tool_name}:{arg_digest}"
        cached = await self._redis.get(key)
        if cached is not None:
            return "allow"

        return "no_match"

    async def write_session(self, session_id: str, tool_name: str, arg_digest: str, ttl: int = 86400) -> None:
        key = f"approval:{session_id}:{tool_name}:{arg_digest}"
        await self._redis.set(key, "1", ex=ttl)
