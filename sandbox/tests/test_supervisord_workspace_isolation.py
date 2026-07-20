from pathlib import Path


def test_chromium_profile_is_outside_scanned_workspace() -> None:
    config = (Path(__file__).parents[1] / "supervisord.conf").read_text()

    assert "--user-data-dir=/tmp/chromium" in config
    assert "environment=DISPLAY=:1,HOME=/tmp/chromium-home" in config
