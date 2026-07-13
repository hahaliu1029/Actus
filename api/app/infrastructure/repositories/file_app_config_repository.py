import errno
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import yaml
from app.application.errors.exceptions import ServerRequestsError
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
)
from app.domain.repositories.app_config_repository import AppConfigRepository
from filelock import FileLock

logger = logging.getLogger(__name__)

CONFIG_TMP_PREFIX = ".config.yaml.tmp-"


def sweep_orphan_config_temps(config_path: str) -> int:
    """§8.3-3 R51#3：清扫 config 目录孤儿 temp（SIGKILL 残留含 secrets，不过夜）。
    lifespan startup 四段全序的第①段调用（T17）。"""
    config_dir = os.path.dirname(str(config_path)) or "."
    removed = 0
    try:
        names = os.listdir(config_dir)
    except OSError:
        return 0
    for name in names:
        if name.startswith(CONFIG_TMP_PREFIX):
            try:
                os.remove(os.path.join(config_dir, name))
                removed += 1
            except OSError:
                pass
    return removed


class FileAppConfigRepository(AppConfigRepository):
    """基于本地文件的App配置数据仓库"""

    def __init__(self, config_path: str) -> None:
        """构造函数，完成文件配置仓库的相关信息初始化"""
        # 1.获取当前项目的根目录
        root_dir = Path.cwd()

        # 2.拼接配置文件路径并校验基础信息
        self._config_path = root_dir.joinpath(root_dir, config_path)
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self._config_path.with_suffix(".lock")  # 文件锁

    def _create_default_app_config_if_not_exists(self):
        """如果配置文件不存在，则使用默认配置并写入到本地文件"""
        if not self._config_path.exists():
            default_app_config = AppConfig(
                llm_config=LLMConfig(),
                agent_config=AgentConfig(),
                mcp_config=MCPConfig(),
                a2a_config=A2AConfig(),
            )
            self.save(default_app_config)

    def load(self) -> Optional[AppConfig]:
        """从本地yaml文件中加载应用配置"""
        # 1.创建默认配置确保文件存在
        self._create_default_app_config_if_not_exists()

        try:
            # 2.打开配置文件并加载为AppConfig
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return AppConfig.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"读取应用配置失败: {str(e)}")
            raise ServerRequestsError("读取应用配置失败，请稍后尝试")

    def save(self, app_config: AppConfig) -> None:
        """将app_config存储到本地yaml配置。

        D1a（spec §8.3-3，R50#2+R51#2+R52#1）：尽力原子——mkstemp 同目录+fsync →
        os.replace；生产单文件 bind-mount 拓扑（docker-compose.yml:139）replace 抛
        {EBUSY, EXDEV} → fallback FileLock 下 truncate 写（现状语义不劣化）；
        其余 errno 重抛。load-modify-save 读改写竞争仍 defer（spec §1.2）。
        """
        lock = FileLock(self._lock_file, timeout=5)
        try:
            with lock:
                data_to_dump = app_config.model_dump(mode="json")
                payload = yaml.dump(data_to_dump, allow_unicode=True, sort_keys=False)
                self._write_best_effort_atomic(payload)
        except TimeoutError:
            logger.error("无法获取配置文件")
            raise ServerRequestsError("写入配置文件失败，请稍后尝试")

    def _write_best_effort_atomic(self, payload: str) -> None:
        config_dir = os.path.dirname(str(self._config_path)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=CONFIG_TMP_PREFIX, dir=config_dir)  # mkstemp 默认 0600（temp 含 secrets）
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                # 权限保持（off-mode 行为等价，非 bind-mount dev 部署）：目标已存在 →
                # temp 继承其权限位，使 os.replace 不把 config 从 0644 降级到 mkstemp 的
                # 0600；首写（目标不存在）→ 保留 0600（secrets 文件稳健默认）。
                try:
                    existing_mode = os.stat(self._config_path).st_mode & 0o777
                except FileNotFoundError:
                    existing_mode = None
                if existing_mode is not None:
                    os.chmod(tmp_path, existing_mode)
                os.replace(tmp_path, self._config_path)
                tmp_path = None   # 已被原子消费
            except OSError as e:
                if e.errno not in (errno.EBUSY, errno.EXDEV):
                    raise
                with open(self._config_path, "w", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
