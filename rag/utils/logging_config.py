# -*- coding: utf-8 -*-
"""
统一日志配置模块

为项目中所有脚本提供一致的日志格式、级别和输出目标。
日志同时输出到控制台和文件，文件存储在 config.LOG_DIR 下。

Usage:
    from utils.logging_config import setup_logging
    _log = setup_logging(__name__)
"""

import logging
import sys
from pathlib import Path

import config


def setup_logging(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
    suppress_third_party: bool = True,
) -> logging.Logger:
    """
    统一配置并返回一个 logger 实例。

    Args:
        name: logger 名称 (通常传 __name__ 或模块名)
        log_file: 日志文件名 (不含路径)，默认使用 '{name}.log'
        level: 日志级别，默认 INFO
        suppress_third_party: 是否静默第三方库日志 (docling, RapidOCR, chromadb)

    Returns:
        配置好的 logging.Logger
    """
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 文件输出
    filename = log_file or f"{name.split('.')[-1]}.log"
    file_handler = logging.FileHandler(
        str(config.LOG_DIR / filename), mode="w", encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # 静默第三方库
    if suppress_third_party:
        for noisy_lib in ("docling", "RapidOCR", "chromadb", "openai", "httpx"):
            logging.getLogger(noisy_lib).setLevel(logging.ERROR)

    logger.propagate = False
    return logger
