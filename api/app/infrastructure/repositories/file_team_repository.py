"""File-system agent-team repository (spec §5). Mirrors FileSkillRepository:
asyncio.to_thread for blocking FS IO; a team lives at ``{root}/{slug}/`` with
meta.json (index) + TEAM.md (authoritative definition).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from app.domain.models.agent_team import TeamBundle
# [codex-R3-F1] TeamArtifactError is a DOMAIN exception (defined in the ABC module)
# so the domain expander can catch it; re-imported here so existing
# `from ...file_team_repository import TeamArtifactError` callers/tests still work.
from app.domain.repositories.agent_team_repository import (
    AgentTeamRepository,
    TeamArtifactError,
)
# Reuse the canonical frontmatter regex — do NOT duplicate the parser (§5).
from app.domain.services.skill_md_parser import _FRONTMATTER_RE


class FileTeamRepository(AgentTeamRepository):
    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)

    async def get_by_slug(self, slug: str) -> Optional[TeamBundle]:
        return await asyncio.to_thread(self._get_by_slug_sync, slug)

    async def list_all(self) -> list[TeamBundle]:
        return await asyncio.to_thread(self._list_all_sync)

    # ---- sync FS helpers (run in a thread) ------------------------------- #

    def _ensure_root(self) -> None:
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_team_dir(self, slug: str) -> Path:
        """Validate ``slug`` is a SINGLE safe path component and resolve it to a
        team directory contained in ``root_dir``. Fail-closed: any structurally
        unsafe slug (empty / ``.`` / ``..`` / absolute / containing a path
        separator or NUL) or a symlinked team dir raises ``TeamArtifactError``.

        ``slug`` is user-controlled (``Message.team_slug`` from the API in a
        later PR), so this is a security boundary, not a convenience check.
        """
        # Reject structurally-unsafe slugs BEFORE touching the filesystem.
        if not slug or slug in (".", ".."):
            raise TeamArtifactError(f"team {slug!r}: invalid slug")
        if "\x00" in slug:
            raise TeamArtifactError(f"team {slug!r}: invalid slug (NUL byte)")
        if Path(slug).is_absolute():
            raise TeamArtifactError(f"team {slug!r}: invalid slug (absolute path)")
        # Reject path separators explicitly for cross-platform safety. On POSIX a
        # backslash is a valid filename character, so ``Path(slug).name`` would
        # NOT flag ``a\\b`` — we reject both ``/`` and ``\\`` outright since no
        # legitimate team slug contains a separator.
        if "/" in slug or "\\" in slug:
            raise TeamArtifactError(f"team {slug!r}: invalid slug (path separator)")
        # Belt-and-suspenders: reject any residual multi-component / trailing-slash
        # form that survives the separator check above.
        if Path(slug).name != slug:
            raise TeamArtifactError(
                f"team {slug!r}: invalid slug (not a single path component)"
            )
        team_dir = self._root_dir / slug
        # A symlinked package dir is a containment-escape vector.
        if team_dir.is_symlink():
            raise TeamArtifactError(f"team {slug!r}: team dir is a symlink")
        return team_dir

    def _read_team_md(self, team_md: Path, slug: str) -> str:
        if not team_md.is_file():
            raise TeamArtifactError(f"team {slug!r}: TEAM.md is not a regular file")
        try:
            return team_md.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise TeamArtifactError(f"team {slug!r}: cannot read TEAM.md: {exc}") from exc

    def _get_by_slug_sync(self, slug: str) -> Optional[TeamBundle]:
        team_dir = self._resolve_team_dir(slug)
        team_md = team_dir / "TEAM.md"
        # Not-found contract: a structurally-VALID slug whose dir/TEAM.md simply
        # does not exist returns None (not raise).
        if not team_md.exists():
            return None
        # A symlinked TEAM.md is a containment-escape vector.
        if team_md.is_symlink():
            raise TeamArtifactError(f"team {slug!r}: TEAM.md is a symlink")
        return self._parse_team_md(self._read_team_md(team_md, slug), slug)

    def _list_all_sync(self) -> list[TeamBundle]:
        self._ensure_root()
        teams: list[TeamBundle] = []
        for child in sorted(self._root_dir.iterdir()):
            # Skip symlinked children (containment-escape vector) before treating
            # them as team dirs.
            if child.is_symlink():
                continue
            if not child.is_dir():
                continue
            team_md = child / "TEAM.md"
            if not team_md.exists():
                continue
            if team_md.is_symlink():
                continue
            teams.append(
                self._parse_team_md(self._read_team_md(team_md, child.name), child.name)
            )
        return teams

    def _parse_team_md(self, text: str, slug: str) -> TeamBundle:
        match = _FRONTMATTER_RE.match(text)
        if not match:
            raise TeamArtifactError(f"team {slug!r}: TEAM.md has no YAML frontmatter")
        try:
            data = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise TeamArtifactError(f"team {slug!r}: malformed TEAM.md YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise TeamArtifactError(f"team {slug!r}: TEAM.md frontmatter is not a mapping")
        try:
            team = TeamBundle.model_validate(data)
        except ValidationError as exc:
            raise TeamArtifactError(f"team {slug!r}: TEAM.md schema violation: {exc}") from exc
        # [codex-R4-F4] the directory slug is authoritative for lookup; reject a
        # frontmatter slug that disagrees, else get_by_slug("foo") could return a
        # team that self-identifies as "bar".
        if team.slug != slug:
            raise TeamArtifactError(
                f"team directory {slug!r} but TEAM.md frontmatter slug={team.slug!r}"
            )
        return team
