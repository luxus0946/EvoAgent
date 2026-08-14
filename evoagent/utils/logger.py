"""日志配置：统一格式，控制台 + 文件双输出。"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "evoagent",
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """初始化全局 logger。

    Args:
        name: logger 名称
        level: 日志级别
        log_file: 日志文件路径，None 时仅控制台输出

    Returns:
        配置完成的 logger 实例
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def new_log_file_path(output_dir: str | Path) -> str:
    """生成带时间戳的日志文件名。"""
    return str(
        Path(output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
