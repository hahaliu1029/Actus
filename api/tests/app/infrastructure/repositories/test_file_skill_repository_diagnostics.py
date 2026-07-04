"""B9 PR-0：坏 skill 目录不塌列表 + per-skill 诊断（spec §8/§13）。"""
import json

import pytest

from app.infrastructure.repositories.file_skill_repository import FileSkillRepository

GOOD_META = {
    "id": "good-skill", "slug": "good-skill", "name": "Good",
    "description": "", "version": "0.1.0",
    "source_type": "local", "source_ref": "ref",
    "runtime_type": "native", "enabled": True,
    "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00",
}
# ^ source_type/runtime_type 的合法枚举值以 domain/models/skill.py 为准，
#   写 fixture 前 grep 确认（复用 tests/app/infrastructure/repositories/test_file_skill_repository.py 既有 fixture 值最稳）。


def _write_skill(root, dirname, *, meta=GOOD_META, manifest=None, break_meta=False,
                 break_manifest=False, drop_meta=False, drop_manifest=False):
    d = root / dirname
    d.mkdir(parents=True)
    if not drop_meta:
        content = "{corrupt" if break_meta else json.dumps({**meta, "id": dirname, "slug": dirname})
        (d / "meta.json").write_text(content, encoding="utf-8")
    if not drop_manifest:
        content = "{corrupt" if break_manifest else json.dumps(manifest or {})
        (d / "manifest.json").write_text(content, encoding="utf-8")
    return d


@pytest.mark.anyio
async def test_corrupt_meta_does_not_collapse_list(tmp_path):
    _write_skill(tmp_path, "good-skill")
    _write_skill(tmp_path, "broken-meta", break_meta=True)
    repo = FileSkillRepository(tmp_path)
    skills = await repo.list()
    assert [s.id for s in skills] == ["good-skill"]     # 坏条目跳过，好条目保留


@pytest.mark.anyio
async def test_diagnostics_three_error_states(tmp_path):
    _write_skill(tmp_path, "good-skill")
    _write_skill(tmp_path, "broken-meta", break_meta=True)
    _write_skill(tmp_path, "broken-manifest", break_manifest=True)
    _write_skill(tmp_path, "no-meta", drop_meta=True)
    _write_skill(tmp_path, "no-manifest", drop_manifest=True)
    repo = FileSkillRepository(tmp_path)
    diags = {d.skill_key: d for d in await repo.list_with_diagnostics()}
    assert diags["good-skill"].ok and diags["good-skill"].skill is not None
    assert (diags["broken-meta"].error_code, diags["broken-meta"].relative_file) == ("parse_error", "meta.json")
    assert (diags["broken-manifest"].error_code, diags["broken-manifest"].relative_file) == ("parse_error", "manifest.json")
    assert (diags["no-meta"].error_code, diags["no-meta"].relative_file) == ("missing_meta", "meta.json")
    assert (diags["no-manifest"].error_code, diags["no-manifest"].relative_file) == ("missing_manifest", "manifest.json")
    for key in ("broken-meta", "broken-manifest", "no-meta", "no-manifest"):
        assert diags[key].skill is None


@pytest.mark.anyio
async def test_diagnostics_no_absolute_paths(tmp_path):
    """API 层不泄露绝对路径（spec §8-3）：diagnostic 只含目录名与相对文件名。"""
    _write_skill(tmp_path, "broken-meta", break_meta=True)
    repo = FileSkillRepository(tmp_path)
    (diag,) = await repo.list_with_diagnostics()
    assert str(tmp_path) not in diag.skill_key
    assert diag.relative_file in ("meta.json", "manifest.json")


@pytest.mark.anyio
async def test_diagnostics_manifest_non_dict_attributed_to_manifest(tmp_path):
    """manifest.json 是合法 JSON 但顶层非 dict（`[]`）→ 归 manifest.json，list() 跳过不塌。"""
    _write_skill(tmp_path, "good-skill")
    bad = tmp_path / "manifest-not-dict"
    bad.mkdir(parents=True)
    (bad / "meta.json").write_text(
        json.dumps({**GOOD_META, "id": "manifest-not-dict", "slug": "manifest-not-dict"}),
        encoding="utf-8",
    )
    (bad / "manifest.json").write_text("[]", encoding="utf-8")  # 合法 JSON，顶层是 list

    repo = FileSkillRepository(tmp_path)
    diags = {d.skill_key: d for d in await repo.list_with_diagnostics()}
    assert (diags["manifest-not-dict"].error_code, diags["manifest-not-dict"].relative_file) == (
        "parse_error",
        "manifest.json",
    )
    assert diags["manifest-not-dict"].skill is None
    # list() 跳过坏条目而不抛异常，好条目保留
    skills = await repo.list()
    assert [s.id for s in skills] == ["good-skill"]


@pytest.mark.anyio
async def test_diagnostics_both_invalid_json_attributed_to_meta(tmp_path):
    """meta.json 与 manifest.json 都是非法 JSON → 归 meta.json（meta 先检）。"""
    _write_skill(tmp_path, "both-broken", break_meta=True, break_manifest=True)
    repo = FileSkillRepository(tmp_path)
    (diag,) = await repo.list_with_diagnostics()
    assert (diag.error_code, diag.relative_file) == ("parse_error", "meta.json")
    assert diag.skill is None


@pytest.mark.anyio
async def test_diagnostics_construction_failure_attributed_to_meta(tmp_path):
    """双文件均为合法 dict，但 meta 枚举值非法（Skill 构造失败）→ 归 meta.json。"""
    _write_skill(
        tmp_path,
        "bad-enum",
        meta={**GOOD_META, "source_type": "not_a_real_enum"},
        manifest={},
    )
    repo = FileSkillRepository(tmp_path)
    (diag,) = await repo.list_with_diagnostics()
    assert (diag.error_code, diag.relative_file) == ("parse_error", "meta.json")
    assert diag.skill is None


@pytest.mark.anyio
async def test_list_enabled_skips_corrupt_json(tmp_path):
    """list_enabled() 与 list() 一样：损坏 JSON 的 skill 被跳过而非抛异常（走 _list_sync）。"""
    _write_skill(tmp_path, "good-skill")  # enabled=True by GOOD_META
    _write_skill(tmp_path, "broken-meta", break_meta=True)
    repo = FileSkillRepository(tmp_path)
    skills = await repo.list_enabled()
    assert [s.id for s in skills] == ["good-skill"]
