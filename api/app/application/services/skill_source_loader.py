from __future__ import annotations

"""Load Skill bundles from GitHub URLs or local directories."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from app.application.errors.exceptions import ValidationError
from app.domain.models.skill import SkillSourceType

MAX_BUNDLE_FILE_COUNT = 200
MAX_BUNDLE_FILE_SIZE = 256 * 1024
MAX_BUNDLE_TOTAL_SIZE = 10 * 1024 * 1024

TEXT_INJECT_EXTENSIONS = {".md", ".txt", ".yaml", ".yml", ".json"}


@dataclass(slots=True)
class SkillBundleFile:
    path: str
    content: bytes
    size: int
    sha256: str
    is_text: bool


@dataclass(slots=True)
class SkillBundle:
    normalized_source_ref: str
    skill_md: str
    files: dict[str, SkillBundleFile]


class SkillSourceLoader:
    """Load skill source directory and return an in-memory bundle."""

    async def load(self, source_type: SkillSourceType, source_ref: str) -> SkillBundle:
        if source_type == SkillSourceType.LOCAL:
            return await self._load_from_local(source_ref)
        if source_type == SkillSourceType.GITHUB:
            return await self._load_from_github(source_ref)
        raise ValidationError(msg="source_type 仅支持 local 或 github")

    async def _load_from_local(self, source_ref: str) -> SkillBundle:
        source = (source_ref or "").strip()
        if not source:
            raise ValidationError(msg="source_ref 不能为空")

        raw_path = source[len("local:") :] if source.startswith("local:") else source
        skill_root = Path(raw_path).expanduser().resolve()
        if not skill_root.is_absolute():
            raise ValidationError(msg="local skill 路径必须是绝对路径")
        if not skill_root.exists() or not skill_root.is_dir():
            raise ValidationError(msg=f"local skill 目录不存在: {skill_root}")

        files: dict[str, SkillBundleFile] = {}
        total_size = 0

        for file_path in sorted(skill_root.rglob("*")):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(skill_root).as_posix()
            normalized_path = self._normalize_relative_path(relative_path)
            if not normalized_path:
                raise ValidationError(msg=f"检测到非法文件路径: {relative_path}")

            raw = file_path.read_bytes()
            total_size = self._validate_bundle_limits(
                file_count=len(files) + 1,
                total_size=total_size + len(raw),
                file_size=len(raw),
                file_path=normalized_path,
            )
            files[normalized_path] = SkillBundleFile(
                path=normalized_path,
                content=raw,
                size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                is_text=Path(normalized_path).suffix.lower() in TEXT_INJECT_EXTENSIONS,
            )

        if "SKILL.md" not in files:
            raise ValidationError(msg="目录中缺少 SKILL.md")

        skill_md = self._decode_utf8(files["SKILL.md"].content, "SKILL.md")
        return SkillBundle(
            normalized_source_ref=f"local:{skill_root.as_posix()}",
            skill_md=skill_md,
            files=files,
        )

    async def _load_from_github(self, source_ref: str) -> SkillBundle:
        owner, repo, parsed_ref, base_path = self._parse_github_source_ref(source_ref)
        files: dict[str, SkillBundleFile] = {}
        total_size = 0

        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Actus-SkillSourceLoader/1.0",
            },
        ) as client:
            ref = parsed_ref or await self._fetch_github_default_branch(
                client=client,
                owner=owner,
                repo=repo,
            )
            total_size = await self._fetch_github_dir(
                client=client,
                owner=owner,
                repo=repo,
                ref=ref,
                base_path=base_path,
                current_path=base_path,
                files=files,
                total_size=total_size,
            )

        if "SKILL.md" not in files:
            if not base_path:
                raise ValidationError(
                    msg=(
                        "GitHub 仓库根目录缺少 SKILL.md，请改用目录 tree URL，"
                        "例如 https://github.com/owner/repo/tree/main/skills/pptx"
                    )
                )
            raise ValidationError(msg="GitHub Skill 目录中缺少 SKILL.md")

        skill_md = self._decode_utf8(files["SKILL.md"].content, "SKILL.md")
        return SkillBundle(
            normalized_source_ref=self._build_github_source_ref(owner, repo, ref, base_path),
            skill_md=skill_md,
            files=files,
        )

    async def _fetch_github_dir(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        ref: str,
        base_path: str,
        current_path: str,
        files: dict[str, SkillBundleFile],
        total_size: int,
    ) -> int:
        if current_path:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{current_path}"
        else:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        response = await client.get(url, params={"ref": ref})

        if response.status_code == 404:
            raise ValidationError(msg=f"GitHub 目录不存在或无权限: {current_path}")
        if response.status_code >= 400:
            raise ValidationError(msg=f"GitHub 目录读取失败[{response.status_code}]")

        payload = response.json()
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValidationError(msg=f"GitHub 响应格式非法: {current_path}")

        for item in payload:
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type") or "")
            item_path = str(item.get("path") or "")
            if not item_path:
                continue

            if item_type == "dir":
                total_size = await self._fetch_github_dir(
                    client=client,
                    owner=owner,
                    repo=repo,
                    ref=ref,
                    base_path=base_path,
                    current_path=item_path,
                    files=files,
                    total_size=total_size,
                )
                continue

            if item_type != "file":
                continue

            if base_path and item_path.startswith(base_path):
                relative_path = item_path[len(base_path) :].lstrip("/")
            else:
                relative_path = item_path
            normalized_path = self._normalize_relative_path(relative_path)
            if not normalized_path:
                raise ValidationError(msg=f"检测到非法文件路径: {relative_path}")

            download_url = str(item.get("download_url") or "").strip()
            if not download_url:
                raise ValidationError(msg=f"无法下载文件: {item_path}")

            file_response = await client.get(download_url)
            if file_response.status_code >= 400:
                raise ValidationError(msg=f"下载文件失败[{file_response.status_code}]: {item_path}")

            raw = file_response.content
            total_size = self._validate_bundle_limits(
                file_count=len(files) + 1,
                total_size=total_size + len(raw),
                file_size=len(raw),
                file_path=normalized_path,
            )
            files[normalized_path] = SkillBundleFile(
                path=normalized_path,
                content=raw,
                size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                is_text=Path(normalized_path).suffix.lower() in TEXT_INJECT_EXTENSIONS,
            )

        return total_size

    @staticmethod
    def _parse_github_source_ref(source_ref: str) -> tuple[str, str, str | None, str]:
        raw = (source_ref or "").strip()
        parsed = urlparse(raw)

        if parsed.scheme != "https" or parsed.netloc not in {"github.com", "www.github.com"}:
            raise ValidationError(
                msg=(
                    "github source_ref 必须是 GitHub URL，例如 "
                    "https://github.com/owner/repo 或 "
                    "https://github.com/owner/repo/tree/main/skills/pptx"
                )
            )

        segments = [item for item in parsed.path.split("/") if item]
        if len(segments) < 2:
            raise ValidationError(
                msg="github source_ref 必须为 /owner/repo 或 /owner/repo/tree/{ref}/{path?} 形式"
            )

        owner = segments[0]
        repo = segments[1]
        if not owner or not repo:
            raise ValidationError(msg="github source_ref 缺少 owner/repo 信息")

        if len(segments) == 2:
            return owner, repo, None, ""

        if segments[2] != "tree":
            raise ValidationError(
                msg="github source_ref 必须为 /owner/repo 或 /owner/repo/tree/{ref}/{path?} 形式"
            )

        if len(segments) < 4:
            raise ValidationError(msg="github source_ref 缺少 ref 信息")

        ref = segments[3].strip()
        if not ref:
            raise ValidationError(msg="github source_ref 缺少 ref 信息")

        directory_path = "/".join(segments[4:])
        if not directory_path:
            return owner, repo, ref, ""

        normalized_path = SkillSourceLoader._normalize_relative_path(directory_path)
        if not normalized_path:
            raise ValidationError(msg="github source_ref 中 path 非法")

        return owner, repo, ref, normalized_path

    @staticmethod
    def _build_github_source_ref(owner: str, repo: str, ref: str, path: str) -> str:
        normalized_source_ref = f"https://github.com/{owner}/{repo}/tree/{ref}"
        if path:
            normalized_source_ref = (
                f"https://github.com/{owner}/{repo}/tree/{ref}/{path}"
            )
        return normalized_source_ref

    @staticmethod
    async def _fetch_github_default_branch(
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
    ) -> str:
        response = await client.get(f"https://api.github.com/repos/{owner}/{repo}")
        if response.status_code == 404:
            raise ValidationError(msg=f"GitHub 仓库不存在或无权限: {owner}/{repo}")
        if response.status_code >= 400:
            raise ValidationError(msg=f"GitHub 仓库读取失败[{response.status_code}]")

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValidationError(msg="GitHub 仓库信息响应格式非法")

        default_branch = str(payload.get("default_branch") or "").strip()
        if not default_branch:
            raise ValidationError(msg=f"GitHub 仓库默认分支缺失: {owner}/{repo}")
        return default_branch

    @staticmethod
    def _normalize_relative_path(path: str) -> str:
        parts = []
        for part in (path or "").replace("\\", "/").split("/"):
            token = part.strip()
            if not token or token == ".":
                continue
            if token == "..":
                if not parts:
                    return ""
                parts.pop()
                continue
            parts.append(token)
        return "/".join(parts)

    @staticmethod
    def _validate_bundle_limits(
        file_count: int,
        total_size: int,
        file_size: int,
        file_path: str,
    ) -> int:
        if file_count > MAX_BUNDLE_FILE_COUNT:
            raise ValidationError(msg=f"Skill 文件数量超过限制: {MAX_BUNDLE_FILE_COUNT}")
        if file_size > MAX_BUNDLE_FILE_SIZE:
            raise ValidationError(msg=f"Skill 文件过大[{file_path}]，限制 {MAX_BUNDLE_FILE_SIZE} bytes")
        if total_size > MAX_BUNDLE_TOTAL_SIZE:
            raise ValidationError(msg=f"Skill 总大小超过限制: {MAX_BUNDLE_TOTAL_SIZE} bytes")
        return total_size

    @staticmethod
    def _decode_utf8(content: bytes, path: str) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(msg=f"{path} 不是 UTF-8 文本") from exc
