"""B9：公共 redact_text 与 RedactingFormatter 规则等价（INV-B9-2 / R16#2）。"""
import pytest

from app.infrastructure.logging.redaction import RedactingFormatter, redact_text

SECRET_FIXTURES = [
    "connect failed: Authorization: Bearer sk-abc123def456ghi789jkl012mno345",
    "http error url=redis://user:hunter2pass@host:9000/mcp",
    '{"api_key": "sk-proj-aaaabbbbccccdddd1111222233334444"}',
]


@pytest.mark.parametrize("raw", SECRET_FIXTURES)
def test_equivalent_to_formatter_rules(raw):
    """同一 secret fixture 双路径输出一致。"""
    assert redact_text(raw) == RedactingFormatter._redact(raw)


@pytest.mark.parametrize("raw", SECRET_FIXTURES)
def test_secret_material_removed(raw):
    out = redact_text(raw)
    for token in ("sk-abc123def456ghi789jkl012mno345", "hunter2pass",
                  "sk-proj-aaaabbbbccccdddd1111222233334444"):
        assert token not in out


def test_plain_text_passthrough():
    assert redact_text("连接MCP服务器[foo]超时(5s)") == "连接MCP服务器[foo]超时(5s)"
