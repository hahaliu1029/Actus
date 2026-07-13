"""D1a F7 范围修订：save 尽力原子（replace 优先 / {EBUSY,EXDEV} fallback truncate / 其余重抛）。"""
import errno
import os

import pytest

from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
)
from app.infrastructure.repositories.file_app_config_repository import (
    FileAppConfigRepository,
)


def minimal_app_config() -> AppConfig:
    """构造默认 AppConfig（镜像 _create_default_app_config_if_not_exists）。"""
    return AppConfig(
        llm_config=LLMConfig(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
    )


@pytest.fixture
def repo_factory():
    def _make(config_path) -> FileAppConfigRepository:
        return FileAppConfigRepository(str(config_path))

    return _make


def test_save_atomic_no_temp_left(tmp_path, repo_factory):
    repo = repo_factory(tmp_path / "config.yaml")
    repo.save(minimal_app_config())
    leftovers = [n for n in os.listdir(tmp_path) if n.startswith(".config.yaml.tmp-")]
    assert leftovers == []
    assert repo.load() is not None          # 内容可读


def test_fallback_on_ebusy_and_exdev(tmp_path, repo_factory, monkeypatch):
    repo = repo_factory(tmp_path / "config.yaml")
    for eno in (errno.EBUSY, errno.EXDEV):   # R52#1：两 errno 同分支
        def _boom(src, dst, _e=eno):
            raise OSError(_e, "mount")
        monkeypatch.setattr(os, "replace", _boom)
        repo.save(minimal_app_config())      # 不抛——fallback truncate 写
        assert repo.load() is not None
        assert not [n for n in os.listdir(tmp_path) if n.startswith(".config.yaml.tmp-")]
        monkeypatch.undo()


def test_other_errno_reraises(tmp_path, repo_factory, monkeypatch):
    repo = repo_factory(tmp_path / "config.yaml")
    def _boom(src, dst):
        raise OSError(errno.EACCES, "denied")
    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(Exception):           # 其余 errno 重抛（外层转 ServerRequestsError 亦可，断言非静默成功）
        repo.save(minimal_app_config())


def test_crash_before_replace_leaves_old_file_intact(tmp_path, repo_factory, monkeypatch):
    # R3#13/spec R50#2：temp 写完、replace 前崩溃 → 老文件完整可 load、无半截 YAML
    repo = repo_factory(tmp_path / "config.yaml")
    repo.save(minimal_app_config())          # 先落一份完整旧文件
    old_bytes = (tmp_path / "config.yaml").read_bytes()
    def _crash(src, dst):
        raise KeyboardInterrupt              # 模拟写后进程中断（不走 fallback 分支）
    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(KeyboardInterrupt):
        repo.save(minimal_app_config())
    assert (tmp_path / "config.yaml").read_bytes() == old_bytes
    assert repo.load() is not None


def test_save_preserves_existing_file_mode(tmp_path, repo_factory):
    """off-mode 修复：atomic replace 必须保留目标原有权限位（非 mkstemp 的 0600）。
    非 bind-mount（dev）部署——首写后 chmod 0644，二次 save 走 mkstemp+replace 不得
    把 config 降级到 0600（否则运维手设的可读权限被 D1a 原子写默默改掉）。"""
    cfg = tmp_path / "config.yaml"
    repo = repo_factory(cfg)
    repo.save(minimal_app_config())          # 首写建文件
    os.chmod(cfg, 0o644)                      # 模拟部署环境的 0644
    repo.save(minimal_app_config())          # 二次写走 mkstemp+replace
    mode = os.stat(cfg).st_mode & 0o777
    assert mode == 0o644, f"atomic replace 丢失原权限，实际 {oct(mode)}"


def test_first_write_uses_secure_default_mode(tmp_path, repo_factory):
    """首写（目标不存在）→ 保留 mkstemp 的 0600 稳健默认（config 含 secrets）。"""
    cfg = tmp_path / "config.yaml"
    repo = repo_factory(cfg)
    repo.save(minimal_app_config())
    mode = os.stat(cfg).st_mode & 0o777
    assert mode == 0o600, f"首写应为 0600，实际 {oct(mode)}"


def test_sweep_orphan_temps(tmp_path):
    from app.infrastructure.repositories.file_app_config_repository import (
        sweep_orphan_config_temps,
    )
    (tmp_path / ".config.yaml.tmp-abc").write_text("secret")
    (tmp_path / "config.yaml").write_text("keep: 1")
    removed = sweep_orphan_config_temps(str(tmp_path / "config.yaml"))
    assert removed == 1
    assert (tmp_path / "config.yaml").exists()
    assert not (tmp_path / ".config.yaml.tmp-abc").exists()
