from __future__ import annotations

from pathlib import Path

import pytest

from app.application.errors.exceptions import ValidationError
from app.application.services.skill_source_loader import SkillSourceLoader
from app.domain.models.skill import SkillSourceType

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_load_local_skill_bundle_success(tmp_path: Path) -> None:
    skill_dir = tmp_path / "pptx"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: pptx\ndescription: deck helper\n---\n# PPTX\n",
        encoding="utf-8",
    )
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "references" / "guideline.md").write_text(
        "# guideline\n",
        encoding="utf-8",
    )

    loader = SkillSourceLoader()
    bundle = await loader.load(SkillSourceType.LOCAL, f"local:{skill_dir.as_posix()}")

    assert bundle.normalized_source_ref.startswith("local:")
    assert "SKILL.md" in bundle.files
    assert "references/guideline.md" in bundle.files
    assert bundle.skill_md.startswith("---")


async def test_load_local_skill_bundle_requires_skill_md(tmp_path: Path) -> None:
    skill_dir = tmp_path / "no-skill-md"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "README.md").write_text("# readme", encoding="utf-8")

    loader = SkillSourceLoader()
    with pytest.raises(ValidationError):
        await loader.load(SkillSourceType.LOCAL, skill_dir.as_posix())


async def test_github_source_ref_must_be_tree_url() -> None:
    loader = SkillSourceLoader()
    with pytest.raises(ValidationError):
        await loader.load(SkillSourceType.GITHUB, "https://github.com/anthropics/skills")


async def test_load_github_skill_bundle_with_mocked_http(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, status_code: int, payload=None, content: bytes = b"") -> None:
            self.status_code = status_code
            self._payload = payload
            self.content = content

        def json(self):
            return self._payload

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
            return None

        async def get(self, url: str, params=None):
            if url.endswith("/contents/skills/pptx") and params == {"ref": "main"}:
                return _FakeResponse(
                    200,
                    payload=[
                        {
                            "type": "file",
                            "path": "skills/pptx/SKILL.md",
                            "download_url": "https://raw.githubusercontent.com/owner/repo/main/skills/pptx/SKILL.md",
                        },
                        {
                            "type": "dir",
                            "path": "skills/pptx/references",
                        },
                    ],
                )
            if (
                url.endswith("/contents/skills/pptx/references")
                and params == {"ref": "main"}
            ):
                return _FakeResponse(
                    200,
                    payload=[
                        {
                            "type": "file",
                            "path": "skills/pptx/references/guide.md",
                            "download_url": "https://raw.githubusercontent.com/owner/repo/main/skills/pptx/references/guide.md",
                        },
                    ],
                )
            if url.endswith("/skills/pptx/SKILL.md"):
                return _FakeResponse(
                    200,
                    content=b"---\nname: PPTX\n---\n# PPTX\nSee [guide](references/guide.md)\n",
                )
            if url.endswith("/skills/pptx/references/guide.md"):
                return _FakeResponse(200, content=b"guide content")
            return _FakeResponse(404, payload={})

    monkeypatch.setattr(
        "app.application.services.skill_source_loader.httpx.AsyncClient",
        _FakeAsyncClient,
    )

    loader = SkillSourceLoader()
    bundle = await loader.load(
        SkillSourceType.GITHUB,
        "https://github.com/owner/repo/tree/main/skills/pptx",
    )

    assert bundle.normalized_source_ref == "https://github.com/owner/repo/tree/main/skills/pptx"
    assert "SKILL.md" in bundle.files
    assert "references/guide.md" in bundle.files
