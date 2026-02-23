from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.domain.models.skill import Skill, SkillSourceType, build_skill_key
from app.domain.models.user_tool_preference import ToolType
from app.infrastructure.models.user_tool_preference import UserToolPreferenceModel
from app.infrastructure.repositories.db_skill_repository import DBSkillRepository
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.storage.postgres import get_postgres
from sqlalchemy import select

from core.config import get_settings


def _serialize_skill(skill: Skill) -> dict:
    payload = skill.model_dump()
    payload["source_type"] = skill.source_type.value
    payload["runtime_type"] = skill.runtime_type.value
    payload["created_at"] = skill.created_at.isoformat()
    payload["updated_at"] = skill.updated_at.isoformat()
    return payload


async def migrate() -> None:
    settings = get_settings()
    skills_root = Path(settings.skills_root_dir)
    snapshot_root = (
        skills_root
        / "_migration"
        / "snapshots"
        / datetime.now().strftime("%Y%m%d%H%M%S")
    )
    snapshot_root.mkdir(parents=True, exist_ok=True)

    postgres = get_postgres()
    await postgres.init()

    mapping: dict[str, str] = {}

    async with postgres.session_factory() as session:
        db_repo = DBSkillRepository(session)
        fs_repo = FileSkillRepository(skills_root)

        skills = await db_repo.list()
        prefs_stmt = select(UserToolPreferenceModel).where(
            UserToolPreferenceModel.tool_type == ToolType.SKILL.value
        )
        prefs_result = await session.execute(prefs_stmt)
        prefs = list(prefs_result.scalars().all())

        (snapshot_root / "skills.json").write_text(
            json.dumps([_serialize_skill(skill) for skill in skills], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (snapshot_root / "skill_preferences.json").write_text(
            json.dumps(
                [
                    {
                        "id": pref.id,
                        "user_id": pref.user_id,
                        "tool_type": pref.tool_type,
                        "tool_id": pref.tool_id,
                        "enabled": pref.enabled,
                        "created_at": pref.created_at.isoformat(),
                        "updated_at": pref.updated_at.isoformat(),
                    }
                    for pref in prefs
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        for skill in skills:
            normalized_source = (
                skill.source_type
                if skill.source_type in {SkillSourceType.LOCAL, SkillSourceType.GITHUB}
                else SkillSourceType.LOCAL
            )
            skill_key = build_skill_key(skill.slug, normalized_source, skill.source_ref)
            mapping[skill.id] = skill_key
            migrated = skill.model_copy(
                deep=True,
                update={
                    "id": skill_key,
                    "source_type": normalized_source,
                },
            )
            await fs_repo.upsert(migrated)

        for pref in prefs:
            if pref.tool_id in mapping:
                pref.tool_id = mapping[pref.tool_id]

        await session.commit()

    report = {
        "skills_count": len(mapping),
        "mapped_preferences": len(mapping),
        "mapping": mapping,
    }
    (snapshot_root / "mapping.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[migrate] snapshot={snapshot_root}")
    print(f"[migrate] migrated_skills={len(mapping)}")


if __name__ == "__main__":
    asyncio.run(migrate())
