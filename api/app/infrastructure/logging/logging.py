import logging
import sys

from core.config import get_settings


def setup_logging() -> None:
    """设置应用程序的日志记录配置。

    根据应用程序的配置设置，初始化日志记录器。
    """
    settings = get_settings()

    # 创建日志记录器
    root_logger = logging.getLogger()

    # 设置日志级别
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # 4.日志输出格式定义
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 5.创建控制台处理器并设置格式
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # 6.将处理器添加到日志记录器
    root_logger.addHandler(console_handler)

    root_logger.info("日志记录器已初始化，日志级别: %s", settings.log_level)
