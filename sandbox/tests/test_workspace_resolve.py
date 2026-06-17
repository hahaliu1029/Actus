"""Sandbox Workspace Isolation — resolve_in_workspace (Guard A) unit tests (PR-1)."""
import os

import pytest


def _patch_roots(monkeypatch, workspace, install="/sandbox"):
    """Point the helper at tmp_path-based roots and clear the settings cache."""
    from app.core import config as cfg

    monkeypatch.setenv("WORKSPACE_ROOT", workspace)
    monkeypatch.setenv("SERVICE_INSTALL_DIR", install)
    cfg.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _restore_settings_cache():
    yield
    from app.core import config as cfg
    cfg.get_settings.cache_clear()


def test_relative_anchors_under_workspace_root(monkeypatch, tmp_path):
    from app.core.workspace import resolve_in_workspace

    ws = str(tmp_path / "home")
    os.makedirs(ws)
    _patch_roots(monkeypatch, ws)
    assert resolve_in_workspace("out.txt") == os.path.join(ws, "out.txt")
    assert resolve_in_workspace("a/b/c.py") == os.path.join(ws, "a/b/c.py")


def test_absolute_passes_through_unchanged(monkeypatch, tmp_path):
    from app.core.workspace import resolve_in_workspace

    ws = str(tmp_path / "home")
    os.makedirs(ws)
    _patch_roots(monkeypatch, ws)
    for abs_path in ("/home/ubuntu/x", "/workspace/.memory/y", "/tmp/z"):
        assert resolve_in_workspace(abs_path) == abs_path


def test_empty_and_dot_resolve_to_root(monkeypatch, tmp_path):
    from app.core.workspace import resolve_in_workspace

    ws = str(tmp_path / "home")
    os.makedirs(ws)
    _patch_roots(monkeypatch, ws)
    assert resolve_in_workspace("") == ws
    assert resolve_in_workspace(".") == ws


def test_dotdot_escape_rejected(monkeypatch, tmp_path):
    from app.core.workspace import resolve_in_workspace, OutsideWorkspaceError

    ws = str(tmp_path / "home")
    os.makedirs(ws)
    _patch_roots(monkeypatch, ws)
    with pytest.raises(OutsideWorkspaceError):
        resolve_in_workspace("../etc/passwd")


def test_ancestor_symlink_to_outside_rejected(monkeypatch, tmp_path):
    """A relative path through an ancestor symlink pointing OUTSIDE the
    workspace is caught by the ancestor-realpath confinement."""
    from app.core.workspace import resolve_in_workspace, OutsideWorkspaceError

    ws = tmp_path / "home"
    ws.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    # /home/link -> /elsewhere ; writing link/f.txt would land in /elsewhere
    os.symlink(str(outside), str(ws / "link"))
    _patch_roots(monkeypatch, str(ws))
    with pytest.raises(OutsideWorkspaceError):
        resolve_in_workspace("link/f.txt")


def test_follow_final_rejects_final_symlink_to_outside(monkeypatch, tmp_path):
    """[follow_final=True] A dir_path that IS a final symlink to outside the
    workspace is caught only when the final component is resolved."""
    from app.core.workspace import resolve_in_workspace, OutsideWorkspaceError

    ws = tmp_path / "home"
    ws.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.symlink(str(outside), str(ws / "dlink"))
    _patch_roots(monkeypatch, str(ws))
    # follow_final=False: final stays lexical -> the link itself is "under" ws.
    assert resolve_in_workspace("dlink", follow_final=False) == os.path.join(str(ws), "dlink")
    # follow_final=True: final resolved -> /elsewhere -> rejected.
    with pytest.raises(OutsideWorkspaceError):
        resolve_in_workspace("dlink", follow_final=True)


def test_deep_new_path_does_not_raise_on_missing_parent(monkeypatch, tmp_path):
    """[R5] Use os.path.realpath (NOT Path.resolve(strict=True)) so a legitimate
    not-yet-created deep path resolves instead of raising."""
    from app.core.workspace import resolve_in_workspace

    ws = str(tmp_path / "home")
    os.makedirs(ws)
    _patch_roots(monkeypatch, ws)
    assert resolve_in_workspace("a/b/c/d.txt") == os.path.join(ws, "a/b/c/d.txt")


def test_deny_absolute_service_dir_write(monkeypatch, tmp_path):
    from app.core.workspace import deny_service_tree_write, ServiceTreeWriteDenied

    install = tmp_path / "sandbox"
    install.mkdir()
    (install / "app").mkdir()
    _patch_roots(monkeypatch, str(tmp_path / "home"), install=str(install))
    target = str(install / "app" / "main.py")
    with pytest.raises(ServiceTreeWriteDenied):
        deny_service_tree_write(target, follows_final_symlink=False)


def test_deny_does_not_false_positive_on_sibling_prefix(monkeypatch, tmp_path):
    """commonpath component containment — /sandbox-foo is NOT under /sandbox."""
    from app.core.workspace import deny_service_tree_write

    install = tmp_path / "sandbox"
    install.mkdir()
    sibling = tmp_path / "sandbox-foo"
    sibling.mkdir()
    _patch_roots(monkeypatch, str(tmp_path / "home"), install=str(install))
    # Must NOT raise.
    deny_service_tree_write(str(sibling / "x"), follows_final_symlink=False)


def test_deny_final_symlink_to_service_dir_for_append(monkeypatch, tmp_path):
    """A relative-anchored path whose FINAL component is a symlink to the
    service dir passes Guard A but append/sudo FOLLOW it -> Guard B with
    follows_final_symlink=True denies."""
    from app.core.workspace import deny_service_tree_write, ServiceTreeWriteDenied

    home = tmp_path / "home"
    home.mkdir()
    install = tmp_path / "sandbox"
    install.mkdir()
    (install / "app").mkdir()
    victim = install / "app" / "main.py"
    victim.write_text("SERVICE SOURCE")
    link = home / "link"
    os.symlink(str(victim), str(link))
    _patch_roots(monkeypatch, str(home), install=str(install))
    with pytest.raises(ServiceTreeWriteDenied):
        deny_service_tree_write(str(link), follows_final_symlink=True)


def test_deny_final_symlink_allowed_for_atomic_overwrite(monkeypatch, tmp_path):
    """Same final symlink, but atomic overwrite REPLACES the link in place
    (follows_final_symlink=False) -> the service file is untouched -> ALLOWED."""
    from app.core.workspace import deny_service_tree_write

    home = tmp_path / "home"
    home.mkdir()
    install = tmp_path / "sandbox"
    install.mkdir()
    (install / "app").mkdir()
    victim = install / "app" / "main.py"
    victim.write_text("SERVICE SOURCE")
    link = home / "link"
    os.symlink(str(victim), str(link))
    _patch_roots(monkeypatch, str(home), install=str(install))
    # Must NOT raise (the overwrite replaces the link node, not /sandbox/...).
    deny_service_tree_write(str(link), follows_final_symlink=False)


def test_is_within_workspace(monkeypatch, tmp_path):
    from app.core.workspace import is_within_workspace

    home = tmp_path / "home"
    home.mkdir()
    (home / "sub").mkdir()
    _patch_roots(monkeypatch, str(home))
    assert is_within_workspace(str(home / "sub" / "f.txt")) is True
    assert is_within_workspace(str(tmp_path / "elsewhere" / "f.txt")) is False
