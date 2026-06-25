"""Domain protocol for the agent-team repository (spec §5)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.models.agent_team import TeamBundle


class TeamArtifactError(Exception):
    """A team package exists but is MALFORMED (bad YAML / schema violation).

    Distinct from "not found" (→ None). Defined in the DOMAIN [codex-R3-F1] (not
    the infra repo) so the domain expander/`_run_parallel_backend` can catch it
    for graceful fail-closed without importing infrastructure."""


class AgentTeamRepository(ABC):
    @abstractmethod
    async def get_by_slug(self, slug: str) -> Optional[TeamBundle]:
        """Return the team with this slug, or None if no such team package exists.

        Raises ``TeamArtifactError`` on a present-but-MALFORMED package (bad YAML /
        schema violation) so the caller can fail closed.
        """
        ...

    @abstractmethod
    async def list_all(self) -> list[TeamBundle]:
        ...
