"""B5 PR-S1-3 acceptance: ``RedactingFormatter`` happy-path patterns.

One ``test_*`` per v1 pattern (1 through 31) covers a positive fixture
(synthetic secret-shaped string fed through ``format``) and asserts the
output contains the truncated form rather than the raw secret. Pattern
#32 (credit card) is intentionally skipped — Actus does not handle
payments.

Test inputs use clearly-synthetic placeholders (``aaaaaaaa1234``,
``00000000`` runs) so a casual reader cannot mistake them for real
production tokens. The truncation rule keeps the first 6 and last 4
characters, so the assertions check: (a) the prefix bytes survive,
(b) the ``[REDACTED]`` mask appears, (c) the middle bytes do NOT
appear in the rendered output.
"""
from __future__ import annotations

import logging

from app.infrastructure.logging.redaction import RedactingFormatter


def _format(message: str) -> str:
    """Render ``message`` through a fresh ``RedactingFormatter``."""
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=message,
        args=None,
        exc_info=None,
    )
    return formatter.format(record)


def _assert_redacted(rendered: str, secret_middle: str) -> None:
    """Confirm the secret middle bytes are gone and the mask is present."""
    assert "[REDACTED]" in rendered, rendered
    assert secret_middle not in rendered, rendered


class TestApiKeys:
    def test_anthropic_key(self):
        rendered = _format("creds: sk-ant-aaaaaaaaaaaaaabbbbbbbbbbbbbb1234")
        _assert_redacted(rendered, "aaaaaaaaaaaa")
        assert "sk-ant" in rendered
        assert "1234" in rendered

    def test_openai_key(self):
        rendered = _format("got key sk-aaaaaaaaaaaaaaaaaaaa1234 from env")
        _assert_redacted(rendered, "aaaaaaaaaa")
        assert "sk-aaa" in rendered
        assert "1234" in rendered

    def test_github_pat(self):
        rendered = _format("token=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaXXXX")
        _assert_redacted(rendered, "aaaaaaaaaaaa")
        assert "ghp_aa" in rendered
        assert "XXXX" in rendered

    def test_github_oauth(self):
        rendered = _format("Bearer gho_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbYYYY")
        _assert_redacted(rendered, "bbbbbbbbbbbb")

    def test_github_app_user(self):
        rendered = _format(
            "X-GitHub-Token: ghu_ccccccccccccccccccccccccccccccccZZZZ"
        )
        _assert_redacted(rendered, "cccccccccccc")

    def test_github_app_server(self):
        rendered = _format(
            "server-token: ghs_ddddddddddddddddddddddddddddddddWWWW"
        )
        _assert_redacted(rendered, "dddddddddddd")

    def test_slack_bot(self):
        rendered = _format("slack: xoxb-1234-5678-aaaaaaaaaaaaaaaaaaaaaaaa")
        _assert_redacted(rendered, "aaaaaaaaaaaa")
        assert "xoxb" in rendered

    def test_google_api(self):
        # Pattern requires ≥35 chars after "AIza"; use 40 a's + suffix
        # so we have plenty of margin.
        rendered = _format(
            "gmaps key AIza" + "a" * 40 + "12 end"
        )
        _assert_redacted(rendered, "aaaaaaaaaaaa")
        assert "AIzaaa" in rendered

    def test_aws_access_key(self):
        rendered = _format("aws id: AKIAIOSFODNN7EXAMPLE end")
        _assert_redacted(rendered, "OSFODNN7")

    def test_aws_secret_key_in_context(self):
        rendered = _format(
            'aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        )
        _assert_redacted(rendered, "wJalrXUtnFEMI")
        assert "aws_secret_access_key" in rendered

    def test_stripe_key(self):
        rendered = _format("stripe sk_test_aaaaaaaaaaaaaaaaaaaaaaaaXXXX")
        _assert_redacted(rendered, "aaaaaaaaaaaa")
        assert "sk_tes" in rendered

    def test_telegram_bot(self):
        rendered = _format(
            "bot 1234567890:AAEhBOweikfkdkfkdkfkdkfkdkfkdkfkdkfk end"
        )
        _assert_redacted(rendered, "BOweikfkdk")

    def test_jwt(self):
        rendered = _format(
            "auth eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.SflKxwRJSMeKKF2QT4 over"
        )
        _assert_redacted(
            rendered, "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9"
        )


class TestHeaders:
    def test_bearer_token_redacted(self):
        rendered = _format("Authorization: Bearer abc123def456ghi789jkl0")
        _assert_redacted(rendered, "def456ghi")
        assert "Bearer" in rendered
        assert "Authorization" in rendered

    def test_generic_authorization_header(self):
        # Use a long Basic credential so the middle bytes are clearly
        # redacted (truncation rule keeps first 6 + last 4 only).
        rendered = _format(
            "Authorization: Basic dXNlcjpwYXNzd29yZGZvb2JhcgABCD"
        )
        assert "[REDACTED]" in rendered
        # The middle stretch of the base64 value is gone.
        assert "cjpwYXNzd29yZGZvb" not in rendered
        assert "Authorization" in rendered

    def test_cookie_session_redacted(self):
        rendered = _format(
            "Set-Cookie: session=abc123def456ghi789jkl0; HttpOnly; Path=/"
        )
        _assert_redacted(rendered, "def456ghi")
        assert "Set-Cookie" in rendered
        assert "session=" in rendered
        assert "HttpOnly" in rendered


class TestJsonShapes:
    def test_json_api_key(self):
        rendered = _format('{"apiKey": "secret-api-key-payload-12345678"}')
        _assert_redacted(rendered, "api-key-payload")
        assert '"apiKey"' in rendered

    def test_json_password(self):
        rendered = _format('{"password": "hunter2-very-secret-password"}')
        _assert_redacted(rendered, "very-secret")
        assert '"password"' in rendered

    def test_json_token(self):
        rendered = _format('{"token": "abc123def456ghi789jkl0XX"}')
        _assert_redacted(rendered, "def456ghi")

    def test_json_refresh_token(self):
        rendered = _format('{"refresh_token": "refresh-abc123def456ghi789jkl0"}')
        _assert_redacted(rendered, "def456ghi")

    def test_json_secret(self):
        rendered = _format('{"secret": "shhh-this-is-a-real-secret-12345"}')
        _assert_redacted(rendered, "real-secret")

    def test_json_client_secret(self):
        rendered = _format('{"client_secret": "oauth-client-secret-bytes-1234"}')
        _assert_redacted(rendered, "client-secret-bytes")

    def test_json_authorization(self):
        rendered = _format('{"authorization": "Bearer payload-12345-abcdef"}')
        assert "[REDACTED]" in rendered


class TestEnvAssignments:
    def test_env_secret_key(self):
        rendered = _format("ENV: SECRET_KEY=abc123def456ghi789jkl0")
        _assert_redacted(rendered, "def456ghi")
        assert "SECRET_KEY" in rendered

    def test_env_api_key(self):
        rendered = _format("API_KEY='leaked-into-traceback-12345678'")
        _assert_redacted(rendered, "into-traceback")

    def test_env_password(self):
        rendered = _format("PASSWORD=hunter2-very-secret-pw")
        _assert_redacted(rendered, "very-secret")

    def test_env_private_key(self):
        rendered = _format("PRIVATE_KEY=0123456789abcdef0123456789abcdef")
        _assert_redacted(rendered, "abcdef0123")

    def test_env_client_secret(self):
        rendered = _format("CLIENT_SECRET='oauth-client-bytes-aaaaaaaaa'")
        _assert_redacted(rendered, "client-bytes")


class TestPemBlocks:
    def test_rsa_private_key(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEAabcdef1234567890==\n"
            "ZXhhbXBsZS1mYWtlLWJsb2NrLW5vdC1yZWFs\n"
            "-----END RSA PRIVATE KEY-----"
        )
        rendered = _format(f"loaded {pem} done")
        _assert_redacted(rendered, "MIIEowIBAAKCAQEAabcdef1234567890")

    def test_openssh_private_key(self):
        pem = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "b3BlbnNzaC1rZXktYWFhYWFhYWFhYWFh\n"
            "-----END OPENSSH PRIVATE KEY-----"
        )
        rendered = _format(f"key={pem}")
        _assert_redacted(rendered, "b3BlbnNzaC1rZXktYWFhYWFhYWFh")


class TestDatabaseUrls:
    def test_postgres_url(self):
        rendered = _format(
            "DSN postgresql://app_user:supersecretpassword@db.host:5432/mydb done"
        )
        _assert_redacted(rendered, "supersecret")
        assert "app_user" in rendered
        assert "db.host" in rendered

    def test_postgres_asyncpg_url(self):
        rendered = _format(
            "url postgresql+asyncpg://u:passwordbytesover18chars@host/db"
        )
        _assert_redacted(rendered, "passwordbytes")

    def test_postgres_psycopg2_url(self):
        # Actus' startup path rewrites the asyncpg DSN to
        # ``postgresql+psycopg2://`` for Alembic; the redaction regex
        # must accept any driver suffix, not just ``asyncpg``.
        rendered = _format(
            "alembic dsn postgresql+psycopg2://app:syncpasswordover18chars@db:5432/app"
        )
        _assert_redacted(rendered, "sswordover18c")
        assert "app" in rendered
        assert "db:5432" in rendered

    def test_postgres_psycopg_url(self):
        # psycopg3 driver suffix.
        rendered = _format(
            "psycopg3 dsn postgresql+psycopg://u:psycopg3longpassword@db/main"
        )
        _assert_redacted(rendered, "ycopg3longpass")

    def test_mysql_url(self):
        rendered = _format("mysql://root:thisisamysqlsecret@db:3306/data")
        _assert_redacted(rendered, "amysqlsecret")

    def test_mongodb_url(self):
        rendered = _format(
            "mongodb://app:longmongopassbytes@cluster:27017/main"
        )
        _assert_redacted(rendered, "longmongopass")

    def test_redis_url(self):
        rendered = _format("redis://:longredispassbytesover@cache:6379/0")
        _assert_redacted(rendered, "longredispass")


class TestQueryParams:
    def test_url_query_api_key(self):
        rendered = _format(
            "request https://api.example.com/v1/foo?api_key=longapikeybytesover18chars&x=1"
        )
        _assert_redacted(rendered, "longapikeybytes")
        assert "api_key=" in rendered

    def test_url_query_token(self):
        rendered = _format(
            "GET /resource?token=abc123def456ghi789jkl0&format=json"
        )
        _assert_redacted(rendered, "def456ghi")
        assert "token=" in rendered
        assert "format=json" in rendered


class TestPhone:
    def test_e164_phone(self):
        rendered = _format("user phone +14155552671 contact")
        assert "+14155552671" not in rendered
        assert "[REDACTED]" in rendered


class TestTruncationBoundary:
    def test_short_token_fully_masked(self):
        rendered = _format("Authorization: Bearer short")
        assert "short" not in rendered
        assert "[REDACTED]" in rendered

    def test_long_token_keeps_prefix_and_suffix(self):
        secret = "abcdef1234567890123XYZQ"
        rendered = _format(f"Authorization: Bearer {secret}")
        assert "abcdef" in rendered
        assert "XYZQ" in rendered
        assert "[REDACTED]" in rendered
        assert "1234567890" not in rendered
