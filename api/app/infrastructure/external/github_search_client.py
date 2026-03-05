"""GitHub Search API 客户端。"""

from __future__ import annotations

import base64
import logging
import re
from typing import Optional

import httpx

from app.domain.models.skill_creator import GitHubRepoInfo

logger = logging.getLogger(__name__)
GITHUB_API_BASE = "https://api.github.com"


class GitHubSearchClient:
    """封装 GitHub 仓库搜索与 README 摘要提取。"""

    def __init__(self, token: Optional[str] = None) -> None:
        self._headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.AsyncClient(base_url=GITHUB_API_BASE, timeout=30.0)

    async def search_repositories(
        self,
        keywords: list[str],
        language: str = "python",
        top_n: int = 5,
    ) -> list[GitHubRepoInfo]:
        """按关键词搜索仓库，按 stars 排序返回。"""
        query = " ".join(keyword for keyword in keywords if keyword.strip())
        if language:
            query = f"{query} language:{language}".strip()

        try:
            response = await self._http.get(
                "/search/repositories",
                params={"q": query, "sort": "stars", "per_page": top_n},
                headers=self._headers,
            )
            if response.status_code != 200:
                logger.warning("GitHub Search 失败: status=%s", response.status_code)
                return []

            items = response.json().get("items", [])
            if not isinstance(items, list):
                return []

            repos: list[GitHubRepoInfo] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                repos.append(
                    GitHubRepoInfo(
                        name=str(item.get("name") or ""),
                        full_name=str(item.get("full_name") or ""),
                        description=str(item.get("description") or ""),
                        stars=int(item.get("stargazers_count") or 0),
                        url=str(item.get("html_url") or ""),
                    )
                )
            return repos
        except Exception as exc:
            logger.warning("GitHub Search 异常: %s", exc)
            return []

    async def fetch_readme_summary(
        self,
        owner: str,
        repo: str,
        max_chars: int = 2000,
    ) -> str:
        """获取 README 摘要。"""
        try:
            response = await self._http.get(
                f"/repos/{owner}/{repo}/readme",
                headers=self._headers,
            )
            if response.status_code != 200:
                return ""
            readme = self._decode_readme(response.json())
            return readme[:max_chars]
        except Exception as exc:
            logger.warning("读取 README 失败(%s/%s): %s", owner, repo, exc)
            return ""

    @staticmethod
    def _decode_readme(data: dict) -> str:
        content = str(data.get("content") or "")
        encoding = str(data.get("encoding") or "base64").lower()
        if not content:
            return ""
        if encoding != "base64":
            return content
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            return ""

    async def research_keywords(
        self,
        keywords: list[str],
        language: str = "python",
        top_n: int = 3,
    ) -> list[GitHubRepoInfo]:
        """按多组关键词调研并去重。"""
        seen: set[str] = set()
        results: list[GitHubRepoInfo] = []

        for keyword in keywords:
            repos = await self.search_repositories([keyword], language=language, top_n=top_n)
            for repo in repos:
                if not repo.full_name or repo.full_name in seen:
                    continue
                seen.add(repo.full_name)

                owner, _, repo_name = repo.full_name.partition("/")
                if owner and repo_name:
                    repo.readme_summary = await self.fetch_readme_summary(owner, repo_name)
                repo.install_command = self._extract_install_command(repo.readme_summary)
                results.append(repo)

        return results

    @staticmethod
    def _extract_install_command(readme_summary: str) -> str:
        for line in (readme_summary or "").splitlines():
            candidate = line.strip().strip("`")
            if candidate.startswith("pip install"):
                return candidate
            if re.search(r"\bpython\s+-m\s+pip\s+install\b", candidate):
                return candidate
        return ""

    @staticmethod
    def format_research_report(repos: list[GitHubRepoInfo]) -> str:
        """将调研结果转成供 LLM 消费的文本。"""
        if not repos:
            return "未找到相关的 GitHub 仓库参考。"

        parts = ["## GitHub 调研结果", ""]
        for repo in repos:
            parts.append(f"### {repo.name} ({repo.full_name})")
            parts.append(f"- Stars: {repo.stars:,}")
            parts.append(f"- 描述: {repo.description}")
            if repo.install_command:
                parts.append(f"- 安装: `{repo.install_command}`")
            parts.append(f"- URL: {repo.url}")
            if repo.readme_summary:
                parts.append(f"- README 摘要:\n{repo.readme_summary[:1000]}")
            parts.append("")

        return "\n".join(parts).strip()

    async def close(self) -> None:
        if not self._http.is_closed:
            await self._http.aclose()
