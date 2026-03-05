from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models.skill_creator import GitHubRepoInfo
from app.infrastructure.external.github_search_client import GitHubSearchClient

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def client() -> GitHubSearchClient:
    return GitHubSearchClient(token=None)


@pytest.fixture
def client_with_token() -> GitHubSearchClient:
    return GitHubSearchClient(token="ghp_test_token")


class TestGitHubSearchClient:
    async def test_search_repositories_parses_response(self, client: GitHubSearchClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "name": "yt-dlp",
                    "full_name": "yt-dlp/yt-dlp",
                    "description": "A feature-rich video downloader",
                    "stargazers_count": 85000,
                    "html_url": "https://github.com/yt-dlp/yt-dlp",
                }
            ]
        }

        with patch.object(client._http, "get", new_callable=AsyncMock, return_value=mock_response):
            repos = await client.search_repositories(["bilibili download python"])

        assert len(repos) == 1
        assert repos[0].name == "yt-dlp"
        assert repos[0].stars == 85000

    async def test_search_with_token_includes_auth_header(
        self,
        client_with_token: GitHubSearchClient,
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}

        with patch.object(
            client_with_token._http,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_get:
            await client_with_token.search_repositories(["test"])

        headers = mock_get.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer ghp_test_token"

    async def test_search_gracefully_handles_api_error(self, client: GitHubSearchClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"message": "rate limit exceeded"}

        with patch.object(client._http, "get", new_callable=AsyncMock, return_value=mock_response):
            repos = await client.search_repositories(["test"])

        assert repos == []

    async def test_fetch_readme_summary_truncates(self, client: GitHubSearchClient) -> None:
        long_readme = "A" * 5000
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "", "encoding": "base64"}

        with patch.object(client, "_decode_readme", return_value=long_readme):
            with patch.object(
                client._http,
                "get",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                summary = await client.fetch_readme_summary("owner", "repo", max_chars=2000)

        assert len(summary) == 2000

    def test_format_research_report(self, client: GitHubSearchClient) -> None:
        repos = [
            GitHubRepoInfo(
                name="yt-dlp",
                full_name="yt-dlp/yt-dlp",
                description="Video downloader",
                stars=85000,
                url="https://github.com/yt-dlp/yt-dlp",
                readme_summary="A tool for downloading videos",
                install_command="pip install yt-dlp",
            )
        ]

        report = client.format_research_report(repos)
        assert "yt-dlp" in report
        assert "85,000" in report
        assert "pip install yt-dlp" in report
