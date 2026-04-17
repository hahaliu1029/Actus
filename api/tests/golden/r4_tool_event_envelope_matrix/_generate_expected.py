"""One-shot helper. Run once to bootstrap expected/*.json from fixtures/*.json.
DO NOT import in tests. DO NOT commit to pytest collection.

Usage: cd api && uv run python tests/golden/r4_tool_event_envelope_matrix/_generate_expected.py
"""
from __future__ import annotations

import json
from pathlib import Path

from app.application.services.tool_event_envelope_v1 import project_tool_event_to_envelope_v1
from app.domain.models.event import ToolEvent

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
EXPECTED = HERE / "expected"


def main() -> None:
    EXPECTED.mkdir(exist_ok=True)
    for src in sorted(FIXTURES.glob("*.json")):
        name = src.stem
        data = json.loads(src.read_text())
        evt = ToolEvent.model_validate(data)
        env = project_tool_event_to_envelope_v1(evt)
        out = env.model_dump(mode="json", by_alias=True)
        dst = EXPECTED / f"{name}_expected.json"
        dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
