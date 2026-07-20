import json
import os
import textwrap

import pytest

from app.domain.models.agent_team import TeamBundle
from app.infrastructure.repositories.file_team_repository import (
    FileTeamRepository,
    TeamArtifactError,
)

_SYMLINKS_SUPPORTED = hasattr(os, "symlink")

_TEAM_MD = textwrap.dedent('''\
    ---
    name: "Code Migration Squad"
    slug: code-migration-squad
    description: "Maps a codebase then applies planned edits."
    version: "0.1.0"
    members:
      - role: explorer
        description: "Read-only: map the codebase."
        system_prompt: |
          You are a read-only code explorer.
        skills: [repo-map]
        default_phase: exploration
        shell_mode: false
      - role: implementer
        description: "Apply the planned edits."
        system_prompt: |
          You are an implementer.
        skills: [python-codegen]
        default_phase: write
        shell_mode: false
    ---

    # Code Migration Squad
    Informational notes (not injected in v1).
    ''')


def _write_team(root, slug, team_md=_TEAM_MD):
    d = root / slug
    d.mkdir(parents=True)
    (d / "TEAM.md").write_text(team_md, encoding="utf-8")
    (d / "meta.json").write_text(
        json.dumps({"id": slug, "slug": slug, "name": "Code Migration Squad",
                    "version": "0.1.0"}),
        encoding="utf-8",
    )
    return d


@pytest.mark.anyio
async def test_get_by_slug_parses_frontmatter(tmp_path):
    _write_team(tmp_path, "code-migration-squad")
    repo = FileTeamRepository(tmp_path)
    team = await repo.get_by_slug("code-migration-squad")
    assert isinstance(team, TeamBundle)
    assert team.slug == "code-migration-squad"
    assert [m.role for m in team.members] == ["explorer", "implementer"]
    assert team.members[0].skills == ("repo-map",)
    assert team.members[0].default_phase == "exploration"


@pytest.mark.anyio
async def test_get_by_slug_missing_returns_none(tmp_path):
    repo = FileTeamRepository(tmp_path)
    assert await repo.get_by_slug("nope") is None


@pytest.mark.anyio
async def test_list_all(tmp_path):
    _write_team(tmp_path, "code-migration-squad")
    repo = FileTeamRepository(tmp_path)
    teams = await repo.list_all()
    assert [t.slug for t in teams] == ["code-migration-squad"]


@pytest.mark.anyio
async def test_malformed_yaml_raises_team_artifact_error(tmp_path):
    _write_team(tmp_path, "broken", team_md="---\n: : not yaml :\n---\n")
    repo = FileTeamRepository(tmp_path)
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("broken")


@pytest.mark.anyio
async def test_schema_violation_raises_team_artifact_error(tmp_path):
    # Single-fault fixture: write to dir == frontmatter slug so the ONLY fault is
    # the duplicate-role schema violation (no dir-slug mismatch). Assert the branch.
    bad = _TEAM_MD.replace("role: implementer", "role: explorer")
    _write_team(tmp_path, "code-migration-squad", team_md=bad)
    repo = FileTeamRepository(tmp_path)
    with pytest.raises(TeamArtifactError, match="schema violation"):
        await repo.get_by_slug("code-migration-squad")


@pytest.mark.anyio
async def test_directory_slug_mismatch_raises(tmp_path):
    # [codex-R4-F4] directory name is authoritative; frontmatter slug must match.
    bad = _TEAM_MD.replace("slug: code-migration-squad", "slug: other-slug")
    _write_team(tmp_path, "code-migration-squad", team_md=bad)
    repo = FileTeamRepository(tmp_path)
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("code-migration-squad")


# ---- path-safety / fail-closed defenses (FIX 1 + FIX 2) ------------------ #


@pytest.mark.anyio
async def test_absolute_slug_rejected(tmp_path):
    # An absolute slug must be rejected (would escape root_dir entirely).
    repo = FileTeamRepository(tmp_path)
    abs_slug = os.path.abspath(os.sep + "etc")
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug(abs_slug)


@pytest.mark.anyio
async def test_parent_traversal_slug_rejected(tmp_path):
    # Place a VALID team exactly where a successful "../" traversal FROM root would
    # resolve, so absent the reject get_by_slug would actually load it — proving the
    # reject is real and nothing outside root is read. root/../outside_root/escape
    # resolves to tmp_path/outside_root/escape, so plant the fixture there.
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside_root"  # == root/../outside_root
    _write_team(outside, "escape")
    repo = FileTeamRepository(root)
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("../outside_root/escape")
    # Also a plain "../escape" form.
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("../escape")


@pytest.mark.anyio
async def test_backslash_slug_rejected(tmp_path):
    # A backslash is a valid filename char on POSIX, but no legitimate team slug
    # contains a separator; reject it explicitly (cross-platform hardening).
    repo = FileTeamRepository(tmp_path)
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("a\\b")


# A TEAM.md whose frontmatter slug == "evil" so the symlink tests are NON-VACUOUS:
# absent the symlink defense they would LOAD successfully (the dir-slug-mismatch
# guard would NOT fire), so a passing test genuinely proves the symlink reject.
_EVIL_TEAM_MD = _TEAM_MD.replace("slug: code-migration-squad", "slug: evil")


@pytest.mark.skipif(not _SYMLINKS_SUPPORTED, reason="symlinks unsupported on platform")
@pytest.mark.anyio
async def test_symlink_package_rejected(tmp_path):
    # A real, valid team dir OUTSIDE root; symlink root/evil -> it. Without the
    # symlink defense, get_by_slug("evil") would happily load it (frontmatter
    # slug == "evil" == lookup slug, so the dir-mismatch guard does NOT fire).
    real_dir = tmp_path / "real_team_outside"
    _write_team(real_dir, "evil", team_md=_EVIL_TEAM_MD)
    root = tmp_path / "root"
    root.mkdir()
    link = root / "evil"
    os.symlink(real_dir / "evil", link, target_is_directory=True)
    repo = FileTeamRepository(root)
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("evil")


@pytest.mark.skipif(not _SYMLINKS_SUPPORTED, reason="symlinks unsupported on platform")
@pytest.mark.anyio
async def test_symlink_team_md_rejected(tmp_path):
    # The package dir is real but TEAM.md is a symlink → containment-escape vector.
    # frontmatter slug == "evil" so, absent the symlink defense, it would LOAD.
    target = tmp_path / "secret.md"
    target.write_text(_EVIL_TEAM_MD, encoding="utf-8")
    root = tmp_path / "root"
    pkg = root / "evil"
    pkg.mkdir(parents=True)
    os.symlink(target, pkg / "TEAM.md")
    (pkg / "meta.json").write_text(
        json.dumps({"id": "evil", "slug": "evil", "name": "x", "version": "0.1.0"}),
        encoding="utf-8",
    )
    repo = FileTeamRepository(root)
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("evil")


@pytest.mark.anyio
async def test_team_md_is_directory_raises(tmp_path):
    # TEAM.md exists but is a DIRECTORY → read_text would leak a raw OSError.
    pkg = tmp_path / "x"
    (pkg / "TEAM.md").mkdir(parents=True)
    repo = FileTeamRepository(tmp_path)
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("x")


@pytest.mark.anyio
async def test_team_md_invalid_utf8_raises(tmp_path):
    # TEAM.md is non-UTF-8 → read_text(encoding="utf-8") would leak UnicodeDecodeError.
    pkg = tmp_path / "y"
    pkg.mkdir()
    (pkg / "TEAM.md").write_bytes(b"\xff\xfe\x00\x01not valid utf-8 \x80\x81")
    repo = FileTeamRepository(tmp_path)
    with pytest.raises(TeamArtifactError):
        await repo.get_by_slug("y")


@pytest.mark.anyio
async def test_valid_but_absent_slug_returns_none(tmp_path):
    # A structurally-VALID slug whose dir simply does not exist must return None.
    repo = FileTeamRepository(tmp_path)
    assert await repo.get_by_slug("totally-valid-but-absent") is None
