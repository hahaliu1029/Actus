"""B9 catalog：加载校验 + 供应链字段 + 无 secret 字面量（spec §7/§13）。"""
import json
import re
from pathlib import Path

from app.interfaces.endpoints.runtime_extension_routes import load_catalog

CATALOG_PATH = Path("app/interfaces/data/mcp_catalog.json")


def test_catalog_loads_and_validates():
    items = load_catalog()
    assert 5 <= len(items) <= 8
    for it in items:
        assert it.source.startswith("https://")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", it.reviewed_at)
        assert it.config_template.get("transport") == it.transport     # 双处一致（R8#5）


def test_catalog_templates_have_no_real_secrets():
    raw = CATALOG_PATH.read_text(encoding="utf-8")
    assert "sk-" not in raw and "ghp_" not in raw
    # secrets 只允许 <YOUR_*> 占位形态
    for m in re.finditer(r'"(?:api_key|token|secret)[^"]*"\s*:\s*"([^"]*)"', raw):
        assert m.group(1).startswith("<YOUR_"), m.group(0)


def test_catalog_corrupt_file_degrades_empty(monkeypatch, tmp_path):
    bad = tmp_path / "mcp_catalog.json"
    bad.write_text("{corrupt", encoding="utf-8")
    import app.interfaces.endpoints.runtime_extension_routes as routes_mod
    monkeypatch.setattr(routes_mod, "CATALOG_PATH", bad)
    routes_mod.load_catalog.cache_clear()
    assert routes_mod.load_catalog() == []
    routes_mod.load_catalog.cache_clear()
