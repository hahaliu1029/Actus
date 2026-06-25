from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """沙箱API服务基础配置信息"""

    log_level: str = "INFO"  # 日志等级
    server_timeout_minutes: int = 60  # 服务超时时间单位为分钟

    # Sandbox Workspace Isolation: RELATIVE file/shell paths anchor here
    # instead of the process CWD (/sandbox = the service install dir).
    workspace_root: str = "/home/ubuntu"
    # Protected service tree — writes/deletes resolving under here are denied.
    service_install_dir: str = "/sandbox"
    # Read-only memory MOUNT target INSIDE the sandbox container — mirrors the
    # api-side ``sandbox_memory_mount_target`` (api/core/config.py). The api
    # mounts the host memory tree read-only at THIS path (docker_sandbox.py),
    # which by default is ``/workspace/.memory`` — a DIFFERENT tree than the
    # scanned workspace root (/home/ubuntu). The snapshot walker excludes THIS
    # configured mount, NOT ``workspace_root/.memory`` (the latter is a child's
    # own legitimate write directory and must be captured).
    memory_mount_target: str = "/workspace/.memory"
    # Native MEMBER skill bundle root INSIDE the sandbox container — mirrors the
    # api-side ``skill_sandbox_bundle_root`` (api/core/config.py:108). A fresh
    # native member skill's foreground bundle sync writes its files here AFTER the
    # S2 PRE snapshot is taken; without exclusion they surface as spurious
    # ADD/MODIFY in the S2 patch manifest (R10-2). The snapshot walker excludes
    # THIS configured tree by realpath containment — exactly like the read-only
    # memory mount and the service-install tree — because ``.skills`` is never a
    # legitimate work-unit output. The api propagates its (possibly customized)
    # value into the container env (docker_sandbox.py) so this default stays in
    # sync; a basename/wrong-path prune would be fail-open.
    skill_sandbox_bundle_root: str = "/home/ubuntu/workspace/.skills"

    @field_validator(
        "workspace_root",
        "service_install_dir",
        "memory_mount_target",
        "skill_sandbox_bundle_root",
    )
    @classmethod
    def _must_be_absolute(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError(f"must be an absolute path (got {v!r})")
        return v

    # 使用pydantic v2提供的写法完成环境变量信息的声明
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
