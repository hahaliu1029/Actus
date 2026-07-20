"""Keep the API test suite on its single pytest-anyio async runner."""

from pathlib import Path
import re

import pytest


pytestmark = [pytest.mark.structure]

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_PYTEST_ASYNCIO_DECORATOR = re.compile(
    r"^\s*@pytest\.mark\.asyncio(?:\([^)]*\))?\s*$",
    re.MULTILINE,
)


def test_tests_do_not_use_pytest_asyncio_marker() -> None:
    offenders = [
        path.relative_to(_TESTS_ROOT).as_posix()
        for path in sorted(_TESTS_ROOT.rglob("test_*.py"))
        if _PYTEST_ASYNCIO_DECORATOR.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == [], (
        "API tests use pytest-anyio; replace @pytest.mark.asyncio with "
        f"@pytest.mark.anyio in: {offenders}"
    )
