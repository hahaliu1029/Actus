import logging
import sys

from app.infrastructure.logging.redaction import (
    RedactingFormatter,
    clear_propagate_only_loggers,
    install_self_healing_logger_class,
    isolate_all_non_root_loggers,
)
from core.config import get_settings


def setup_logging() -> None:
    """设置应用程序的日志记录配置。

    根据应用程序的配置设置，初始化日志记录器。

    B5 PR-S1-3 升级：
    - root handler 使用 ``RedactingFormatter`` 替代 ``logging.Formatter``，
      所有经 root 流出的 LogRecord（含 ``formatException`` 渲染的
      traceback）会被 v1 secret pattern 集体脱敏后再打印
    - 调用 ``clear_propagate_only_loggers()`` 处理已存在但带自己 handler
      的第三方 logger（``uvicorn.access`` 是关键路径——uvicorn 在 server
      boot 阶段就建好该 logger，handler 直发 stdout 不经 root；不清掉，
      access log 里 ``?token=...`` 的 query 串就绕过 RedactingFormatter）
    - 调用 ``install_self_healing_logger_class()`` 让后续 late-imported
      lib（openai/httpx/anthropic/...）拿到的 logger 默认 propagate=True
      且 ``addHandler`` 为 NoOp，无法绕过 root
    - idempotent：函数可重复调用，先清空 root handler 再装新的，避免
      lifespan reload 或测试路径上 handler 堆叠
    """
    settings = get_settings()

    root_logger = logging.getLogger()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    formatter = RedactingFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Idempotent cleanup so re-invocations don't stack StreamHandlers.
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Q2 self-heal — order matters:
    # 1. ``install_self_healing_logger_class()`` first so any logger
    #    constructed AFTER this point comes up as the propagate-only
    #    subclass (covers names ``setup_logging`` doesn't know about
    #    and any late-imported library).
    # 2. ``clear_propagate_only_loggers()`` second; for each known
    #    name it both clears the existing handler list AND swaps
    #    ``__class__`` to the subclass — closing the bypass for SDKs
    #    that were imported before ``setup_logging`` ran (openai /
    #    httpx / anthropic / langchain / uvicorn.access, ...).
    # 3. ``isolate_all_non_root_loggers()`` last — registry-wide
    #    sweep that catches transitive deps not in the explicit list
    #    (``huggingface_hub`` and ``transformers`` were seen leaking
    #    via local StreamHandlers; ``transformers.propagate=False``
    #    additionally severed the root path entirely).
    install_self_healing_logger_class()
    clear_propagate_only_loggers()
    isolate_all_non_root_loggers()

    root_logger.info("日志记录器已初始化，日志级别: %s", settings.log_level)
