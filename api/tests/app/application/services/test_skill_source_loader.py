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


async def test_github_source_ref_must_be_valid_github_url() -> None:
    loader = SkillSourceLoader()
    with pytest.raises(ValidationError):
        await loader.load(SkillSourceType.GITHUB, "https://example.com/anthropics/skills")


async def test_load_github_skill_bundle_from_repo_root_with_mocked_http(monkeypatch) -> None:
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
            if url.endswith("/repos/owner/repo"):
                return _FakeResponse(
                    200,
                    payload={"default_branch": "main"},
                )
            if url.endswith("/contents") and params == {"ref": "main"}:
                return _FakeResponse(
                    200,
                    payload=[
                        {
                            "type": "file",
                            "path": "SKILL.md",
                            "download_url": "https://raw.githubusercontent.com/owner/repo/main/SKILL.md",
                        },
                        {
                            "type": "dir",
                            "path": "references",
                        },
                    ],
                )
            if url.endswith("/contents/references") and params == {"ref": "main"}:
                return _FakeResponse(
                    200,
                    payload=[
                        {
                            "type": "file",
                            "path": "references/guide.md",
                            "download_url": "https://raw.githubusercontent.com/owner/repo/main/references/guide.md",
                        },
                    ],
                )
            if url.endswith("/main/SKILL.md"):
                return _FakeResponse(
                    200,
                    content=b"---\nname: RepoRoot\n---\n# Root Skill\nSee [guide](references/guide.md)\n",
                )
            if url.endswith("/main/references/guide.md"):
                return _FakeResponse(200, content=b"guide content")
            return _FakeResponse(404, payload={})

    monkeypatch.setattr(
        "app.application.services.skill_source_loader.httpx.AsyncClient",
        _FakeAsyncClient,
    )

    loader = SkillSourceLoader()
    bundle = await loader.load(
        SkillSourceType.GITHUB,
        "https://github.com/owner/repo",
    )

    assert bundle.normalized_source_ref == "https://github.com/owner/repo/tree/main"
    assert "SKILL.md" in bundle.files
    assert "references/guide.md" in bundle.files


async def test_load_github_repo_root_without_skill_md_should_raise(monkeypatch) -> None:
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
            if url.endswith("/repos/owner/repo"):
                return _FakeResponse(200, payload={"default_branch": "main"})
            if url.endswith("/contents") and params == {"ref": "main"}:
                return _FakeResponse(
                    200,
                    payload=[
                        {
                            "type": "file",
                            "path": "README.md",
                            "download_url": "https://raw.githubusercontent.com/owner/repo/main/README.md",
                        },
                    ],
                )
            if url.endswith("/main/README.md"):
                return _FakeResponse(200, content=b"# demo")
            return _FakeResponse(404, payload={})

    monkeypatch.setattr(
        "app.application.services.skill_source_loader.httpx.AsyncClient",
        _FakeAsyncClient,
    )

    loader = SkillSourceLoader()
    with pytest.raises(ValidationError) as exc_info:
        await loader.load(
            SkillSourceType.GITHUB,
            "https://github.com/owner/repo",
        )

    assert "根目录缺少 SKILL.md" in str(exc_info.value)


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
