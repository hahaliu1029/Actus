from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.domain.models.skill import Skill
from app.domain.models.user_tool_preference import ToolType
from app.infrastructure.models.skill import SkillModel
from app.infrastructure.models.user_tool_preference import UserToolPreferenceModel
from app.infrastructure.storage.postgres import get_postgres
from sqlalchemy import delete

from core.config import get_settings


def _latest_snapshot(skills_root: Path) -> Path:
    snapshots_root = skills_root / "_migration" / "snapshots"
    if not snapshots_root.exists():
        raise RuntimeError("未找到迁移快照目录")
    snapshots = sorted(
        [path for path in snapshots_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    if not snapshots:
        raise RuntimeError("未找到可回滚快照")
    return snapshots[-1]


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


async def rollback() -> None:
    settings = get_settings()
    skills_root = Path(settings.skills_root_dir)
    snapshot = _latest_snapshot(skills_root)

    skills_data = json.loads((snapshot / "skills.json").read_text(encoding="utf-8"))
    prefs_data = json.loads(
        (snapshot / "skill_preferences.json").read_text(encoding="utf-8")
    )

    postgres = get_postgres()
    await postgres.init()

    async with postgres.session_factory() as session:
        await session.execute(delete(SkillModel))
        await session.execute(
            delete(UserToolPreferenceModel).where(
                UserToolPreferenceModel.tool_type == ToolType.SKILL.value
            )
        )

        for raw in skills_data:
            raw["created_at"] = _parse_datetime(raw["created_at"])
            raw["updated_at"] = _parse_datetime(raw["updated_at"])
            skill = Skill.model_validate(raw)
            session.add(SkillModel.from_domain(skill))

        for raw in prefs_data:
            session.add(
                UserToolPreferenceModel(
                    id=raw["id"],
                    user_id=raw["user_id"],
                    tool_type=raw["tool_type"],
                    tool_id=raw["tool_id"],
                    enabled=bool(raw["enabled"]),
                    created_at=_parse_datetime(raw["created_at"]),
                    updated_at=_parse_datetime(raw["updated_at"]),
                )
            )

        await session.commit()

    print(f"[rollback] restored snapshot={snapshot}")
    print(f"[rollback] skills={len(skills_data)} preferences={len(prefs_data)}")


if __name__ == "__main__":
    asyncio.run(rollback())
